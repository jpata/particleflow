import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import time
import os

# --- Model Definition ---

def get_activation(activation):
    if activation == "elu":
        return nn.ELU
    elif activation == "relu":
        return nn.ReLU
    elif activation == "gelu":
        return nn.GELU
    return nn.ReLU

class SimpleMultiheadAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__(embed_dim, num_heads, dropout=dropout, bias=True, batch_first=True)
        self.head_dim = int(embed_dim // num_heads)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(self, q, k, v, key_padding_mask=None):
        bs, seq_len, _ = q.size()
        head_dim = self.head_dim
        num_heads = self.num_heads

        # split stacked in_proj_weight, in_proj_bias to q, k, v matrices
        wq, wk, wv = torch.split(self.in_proj_weight, [self.embed_dim, self.embed_dim, self.embed_dim], dim=0)
        bq, bk, bv = torch.split(self.in_proj_bias, [self.embed_dim, self.embed_dim, self.embed_dim], dim=0)

        q = torch.matmul(q, wq.T) + bq
        k = torch.matmul(k, wk.T) + bk
        v = torch.matmul(v, wv.T) + bv

        q = q.reshape(bs, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.reshape(bs, -1, num_heads, head_dim).transpose(1, 2)
        v = v.reshape(bs, -1, num_heads, head_dim).transpose(1, 2)

        # Flash Attention is available on this device and we use it here.
        # Masking is handled by zeroing out padded elements before and after the attention block, 
        # as Flash Attention does not support arbitrary masks efficiently.
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, 
                dropout_p=self.dropout if self.training else 0.0
            )

        attn_output = attn_output.transpose(1, 2).reshape(bs, seq_len, num_heads * head_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, None

class PreLnSelfAttentionLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, width, dropout=0.1):
        super(PreLnSelfAttentionLayer, self).__init__()
        self.mha = SimpleMultiheadAttention(embedding_dim, num_heads, dropout=dropout)
        self.norm0 = nn.LayerNorm(embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.seq = nn.Sequential(
            nn.Linear(embedding_dim, width),
            nn.ELU(),
            nn.Linear(width, embedding_dim),
            nn.ELU()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x: [B, N, D]
        # mask: [B, N]
        if mask is not None:
            mask_ = mask.unsqueeze(-1)
        
        residual = x
        x_norm = self.norm0(x)
        
        q = x_norm
        if mask is not None:
            q = q * mask_
        
        mha_out, _ = self.mha(q, x_norm, x_norm)
        x = residual + mha_out
        
        residual = x
        x_norm = self.norm1(x)
        ffn_out = self.seq(x_norm)
        ffn_out = self.dropout(ffn_out)
        
        x = residual + ffn_out
        if mask is not None:
            x = x * mask_
        return x

def ffn(input_dim, output_dim, width, dropout=0.1):
    return nn.Sequential(
        nn.Linear(input_dim, width),
        nn.ELU(),
        nn.LayerNorm(width),
        nn.Dropout(dropout),
        nn.Linear(width, output_dim),
    )

class RegressionOutput(nn.Module):
    def __init__(self, embed_dim, width, elemtypes):
        super(RegressionOutput, self).__init__()
        self.elemtypes = elemtypes
        self.nn = ffn(embed_dim, len(elemtypes), width)

    def forward(self, X, x, orig_value):
        # X: [B, N, 25] (original features)
        # x: [B, N, D] (latent representation)
        # orig_value: [B, N, 1]
        
        nn_out = self.nn(x) # [B, N, num_elemtypes]
        
        # Create mask for each element type
        elemtype_mask = torch.stack([X[..., 0] == elemtype for elemtype in self.elemtypes], dim=-1) # [B, N, num_elemtypes]
        
        # Select the output corresponding to the element type
        res = torch.sum(elemtype_mask * nn_out, dim=-1, keepdim=True)
        
        return orig_value + res

class MLPF(nn.Module):
    def __init__(self, input_dim=25, num_classes=8, embedding_dim=128, width=128, num_convs=3, num_heads=8):
        super(MLPF, self).__init__()
        
        self.elemtypes = [1, 2, 3, 4, 5, 8, 9, 10, 11]
        num_types = len(self.elemtypes)
        
        # Input encoding
        self.nn0_id = ffn(input_dim, num_types * embedding_dim, width)
        self.nn0_reg = ffn(input_dim, num_types * embedding_dim, width)
        
        # Attention layers
        self.conv_id = nn.ModuleList([
            PreLnSelfAttentionLayer(embedding_dim, num_heads, width) for _ in range(num_convs)
        ])
        self.conv_reg = nn.ModuleList([
            PreLnSelfAttentionLayer(embedding_dim, num_heads, width) for _ in range(num_convs)
        ])
        
        # Final output heads
        self.nn_binary_particle = ffn(embedding_dim, 2, width)
        self.nn_pid = ffn(embedding_dim, num_classes, width)
        
        self.nn_pt = RegressionOutput(embedding_dim, width, self.elemtypes)
        self.nn_eta = RegressionOutput(embedding_dim, width, self.elemtypes)
        self.nn_sin_phi = RegressionOutput(embedding_dim, width, self.elemtypes)
        self.nn_cos_phi = RegressionOutput(embedding_dim, width, self.elemtypes)
        self.nn_energy = RegressionOutput(embedding_dim, width, self.elemtypes)

    def forward(self, X, mask):
        # X: [B, N, 25]
        # mask: [B, N]
        
        B, N, _ = X.shape
        num_types = len(self.elemtypes)
        
        # Split input encoding
        all_id = self.nn0_id(X).view(B, N, num_types, -1)
        all_reg = self.nn0_reg(X).view(B, N, num_types, -1)
        
        elemtype_mask = torch.stack([X[..., 0] == elemtype for elemtype in self.elemtypes], dim=-1)
        
        emb_id = torch.sum(all_id * elemtype_mask.unsqueeze(-1), dim=2)
        emb_reg = torch.sum(all_reg * elemtype_mask.unsqueeze(-1), dim=2)
        
        # Attention layers
        for conv in self.conv_id:
            emb_id = conv(emb_id, mask)
        for conv in self.conv_reg:
            emb_reg = conv(emb_reg, mask)
            
        # Outputs
        logits_binary = self.nn_binary_particle(emb_id)
        logits_pid = self.nn_pid(emb_id)
        
        preds_pt = self.nn_pt(X, emb_reg, X[..., 1:2])
        preds_eta = self.nn_eta(X, emb_reg, X[..., 2:3])
        preds_sin_phi = self.nn_sin_phi(X, emb_reg, X[..., 3:4])
        preds_cos_phi = self.nn_cos_phi(X, emb_reg, X[..., 4:5])
        preds_energy = self.nn_energy(X, emb_reg, X[..., 5:6])
        
        preds_momentum = torch.cat([preds_pt, preds_eta, preds_sin_phi, preds_cos_phi, preds_energy], dim=-1)
        
        return logits_binary, logits_pid, preds_momentum

# --- Loss Function ---

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction="none"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, x, y):
        # x: [B, C, N]
        # y: [B, N]
        log_p = F.log_softmax(x, dim=1)
        pt = torch.exp(torch.gather(log_p, 1, y.unsqueeze(1))).squeeze(1)
        loss = -((1 - pt) ** self.gamma) * torch.gather(log_p, 1, y.unsqueeze(1)).squeeze(1)
        
        if self.reduction == "mean":
            return loss.mean()
        return loss

def mlpf_loss(y, ypred, mask, X):
    """
    y: dict with "cls_id", "pt", "eta", "sin_phi", "cos_phi", "energy"
    ypred: (logits_binary, logits_pid, preds_momentum)
    mask: [B, N]
    X: [B, N, 25]
    """
    logits_binary, logits_pid, preds_momentum = ypred
    
    # y["cls_id"] is [B, N]
    npart = torch.sum(y["cls_id"] != 0)
    nelem = torch.sum(mask)
    
    # Binary loss
    loss_binary = F.cross_entropy(logits_binary.permute(0, 2, 1), (y["cls_id"] != 0).long(), reduction="none")
    
    # PID loss
    loss_obj_id = FocalLoss(gamma=2.0, reduction="none")
    loss_pid = loss_obj_id(logits_pid.permute(0, 2, 1), y["cls_id"].long())
    loss_pid[y["cls_id"] == 0] *= 0
    
    # Regression loss
    pred_pt = preds_momentum[..., 0]
    pred_eta = preds_momentum[..., 1]
    pred_sin_phi = preds_momentum[..., 2]
    pred_cos_phi = preds_momentum[..., 3]
    pred_energy = preds_momentum[..., 4]
    
    loss_pt = F.mse_loss(pred_pt, y["pt"], reduction="none")
    loss_eta = 1e-2 * F.mse_loss(pred_eta, y["eta"], reduction="none")
    loss_sin_phi = 1e-2 * F.mse_loss(pred_sin_phi, y["sin_phi"], reduction="none")
    loss_cos_phi = 1e-2 * F.mse_loss(pred_cos_phi, y["cos_phi"], reduction="none")
    loss_energy = F.mse_loss(pred_energy, y["energy"], reduction="none")
    
    # Weight regression loss by target pT
    sqrt_target_pt = torch.sqrt(torch.exp(y["pt"]) * X[:, :, 1])
    loss_pt *= sqrt_target_pt
    loss_energy *= sqrt_target_pt
    
    # Masking
    for l in [loss_pt, loss_eta, loss_sin_phi, loss_cos_phi, loss_energy]:
        l[y["cls_id"] == 0] *= 0
        l[mask == 0] *= 0
        
    loss_binary[mask == 0] *= 0
    loss_pid[mask == 0] *= 0
    
    tot_loss = (loss_binary.sum() / nelem + 
                loss_pid.sum() / nelem + 
                loss_pt.sum() / npart + 
                loss_eta.sum() / npart + 
                loss_sin_phi.sum() / npart + 
                loss_cos_phi.sum() / npart + 
                loss_energy.sum() / npart)
    
    return tot_loss

# --- Training Loop ---

def train(model, train_loader, optimizer, device, duration_seconds=120):
    model.train()
    start_time = time.time()
    num_steps = 0
    total_loss = 0
    
    print(f"Starting training for {duration_seconds} seconds...")
    
    while (time.time() - start_time) < duration_seconds:
        for batch in train_loader:
            if (time.time() - start_time) >= duration_seconds:
                break
                
            X = batch.X.to(device)
            mask = batch.mask.to(device)
            
            # Prepare targets
            y = {
                "cls_id": batch.ytarget[:, :, 0].to(device),
                "pt": batch.ytarget[:, :, 2].to(device),
                "eta": batch.ytarget[:, :, 3].to(device),
                "sin_phi": batch.ytarget[:, :, 4].to(device),
                "cos_phi": batch.ytarget[:, :, 5].to(device),
                "energy": batch.ytarget[:, :, 6].to(device),
            }
            
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                ypred = model(X, mask)
                loss = mlpf_loss(y, ypred, mask, X)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_steps += 1
            
            if num_steps % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Step {num_steps}, Loss: {loss.item():.4f}, Elapsed: {elapsed:.1f}s")
                
    return total_loss / num_steps if num_steps > 0 else 0, num_steps

if __name__ == "__main__":
    # This is for testing standalone, but usually will be called from eval.py or a script
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPF().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    print("Model initialized. Ready for training.")
