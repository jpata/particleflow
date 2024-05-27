import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import onnxscript
import onnx
from onnxconverter_common import float16
from onnxscript.function_libs.torch_lib.tensor_typing import TFloat
import onnxruntime as rt

opset_version = 17
custom_opset = onnxscript.values.Opset(domain="onnx-script", version=1)
msft_op = onnxscript.values.Opset("com.microsoft", 1)

model_state = torch.load(
    "experiments/pyg-cms_20240430_094836_751206/checkpoints/checkpoint-25-17.631161.pth", map_location=torch.device("cpu")
)

# these are from parameters/pyg/pyg-cms.yaml
# for explicit clarity, they are spliced in here
NUM_HEADS = 32
HEAD_DIM = 16
NUM_CONVS = 6
model_kwargs = {
    "input_dim": 55,
    "num_classes": 9,
    "pt_mode": "linear",
    "eta_mode": "linear",
    "sin_phi_mode": "linear",
    "cos_phi_mode": "linear",
    "energy_mode": "linear",
    "elemtypes_nonzero": [1, 4, 5, 6, 8, 9, 10, 11],
    "dropout_ff": 0.0,
    "dropout_conv_id_mha": 0.0,
    "dropout_conv_id_ff": 0.0,
    "dropout_conv_reg_mha": 0.0,
    "dropout_conv_reg_ff": 0.0,
    "activation": "relu",
    "num_convs": NUM_CONVS,
    "head_dim": HEAD_DIM,
    "num_heads": NUM_HEADS,
}


def get_activation(activation):
    if activation == "elu":
        act = nn.ELU
    elif activation == "relu":
        act = nn.ReLU
    elif activation == "relu6":
        act = nn.ReLU6
    elif activation == "leakyrelu":
        act = nn.LeakyReLU
    return act


def ffn(input_dim, output_dim, width, act, dropout):
    return nn.Sequential(
        nn.Linear(input_dim, width),
        act(),
        torch.nn.LayerNorm(width),
        nn.Dropout(dropout),
        nn.Linear(width, output_dim),
    )


class SimpleMultiheadAttention(nn.MultiheadAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(embed_dim, num_heads, dropout, bias=bias, batch_first=batch_first, **factory_kwargs)
        self.head_dim = int(embed_dim // num_heads)
        assert self.head_dim == HEAD_DIM
        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs), requires_grad=False)
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs), requires_grad=False)

        self.query_weight = self.in_proj_weight[: self.embed_dim]
        self.key_weight = self.in_proj_weight[self.embed_dim : 2 * self.embed_dim]
        self.value_weight = self.in_proj_weight[2 * self.embed_dim :]
        self.query_bias = self.in_proj_bias[: self.embed_dim]
        self.key_bias = self.in_proj_bias[self.embed_dim : 2 * self.embed_dim]
        self.value_bias = self.in_proj_bias[2 * self.embed_dim :]

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias, **factory_kwargs)
        self.onnx_export = False

    def _get_name(self):
        return "SimpleMultiheadAttention"

    def forward(self, query: Tensor, key: Tensor, value: Tensor, need_weights: bool = False) -> Tensor:

        bsz, tgt_len, embed_dim_to_check = query.size()
        # assert self.embed_dim == embed_dim_to_check
        # assert self.embed_dim == key.size()[-1]
        # assert self.embed_dim == value.size()[-1]

        q = query @ self.query_weight + self.query_bias
        k = key @ self.key_weight + self.key_bias
        v = value @ self.value_weight + self.value_bias

        # for torch internal scaled_dot_product_attention,
        # the embed_dim must be unpacked as follows to num_heads, head_dim
        # for com.microsoft.MultiheadAttention in onnx, this is not needed
        if not self.onnx_export:
            q = torch.reshape(q, [bsz, tgt_len, self.num_heads, self.head_dim])
            q = torch.permute(q, [0, 2, 1, 3])
            k = torch.reshape(k, [bsz, tgt_len, self.num_heads, self.head_dim])
            k = torch.permute(k, [0, 2, 1, 3])
            v = torch.reshape(v, [bsz, tgt_len, self.num_heads, self.head_dim])
            v = torch.permute(v, [0, 2, 1, 3])

        attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)

        if not self.onnx_export:
            attn_output = torch.permute(attn_output, [0, 2, 1, 3])
            attn_output = torch.reshape(attn_output, [bsz, tgt_len, self.num_heads * self.head_dim])
        # assert list(attn_output.size()) == [bsz, tgt_len, self.num_heads * self.head_dim]

        attn_output = self.out_proj(attn_output)

        return attn_output, None


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        activation="elu",
        embedding_dim=128,
        num_heads=2,
        width=128,
        dropout_mha=0.1,
        dropout_ff=0.1,
    ):
        super().__init__()

        self.act = get_activation(activation)
        self.mha = SimpleMultiheadAttention(embedding_dim, num_heads, dropout=dropout_mha, batch_first=True)
        self.norm0 = torch.nn.LayerNorm(embedding_dim)
        self.norm1 = torch.nn.LayerNorm(embedding_dim)
        self.seq = torch.nn.Sequential(
            nn.Linear(embedding_dim, width), self.act(), nn.Linear(width, embedding_dim), self.act()
        )
        self.dropout = torch.nn.Dropout(dropout_ff)

    def forward(self, x: torch.Tensor):
        mha_out = self.mha(x, x, x, need_weights=False)[0]

        x = x + mha_out
        x = self.norm0(x)
        x = x + self.seq(x)
        x = self.norm1(x)
        x = self.dropout(x)
        return x


class RegressionOutput(nn.Module):
    def __init__(self, mode, embed_dim, width, act, dropout, elemtypes):
        super().__init__()
        self.mode = mode
        self.elemtypes = elemtypes
        self.nn = ffn(embed_dim, 2, width, act, dropout)

    def forward(self, elems, x, orig_value):
        nn_out = self.nn(x)
        nn_out = orig_value * nn_out[..., 0:1] + nn_out[..., 1:2]
        return nn_out


class MLPF(nn.Module):
    def __init__(
        self,
        input_dim=34,
        num_classes=8,
        embedding_dim=128,
        width=128,
        num_convs=2,
        dropout_ff=0.0,
        activation="elu",
        layernorm=True,
        pt_mode="linear",
        eta_mode="linear",
        sin_phi_mode="linear",
        cos_phi_mode="linear",
        energy_mode="linear",
        # element types which actually exist in the dataset
        elemtypes_nonzero=[1, 4, 5, 6, 8, 9, 10, 11],
        # self-attention specific parameters
        num_heads=16,
        head_dim=16,
        dropout_conv_reg_mha=0.0,
        dropout_conv_reg_ff=0.0,
        dropout_conv_id_mha=0.0,
        dropout_conv_id_ff=0.0,
    ):
        super().__init__()

        self.act = get_activation(activation)

        self.input_dim = input_dim
        self.num_convs = num_convs

        self.elemtypes_nonzero = elemtypes_nonzero

        embedding_dim = num_heads * head_dim
        width = num_heads * head_dim

        # embedding of the inputs
        self.nn0_id = ffn(self.input_dim, embedding_dim, width, self.act, dropout_ff)
        self.nn0_reg = ffn(self.input_dim, embedding_dim, width, self.act, dropout_ff)

        self.conv_id = nn.ModuleList()
        self.conv_reg = nn.ModuleList()

        for i in range(num_convs):
            self.conv_id.append(
                SelfAttentionLayer(
                    activation=activation,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    width=width,
                    dropout_mha=dropout_conv_id_mha,
                    dropout_ff=dropout_conv_id_ff,
                )
            )
            self.conv_reg.append(
                SelfAttentionLayer(
                    activation=activation,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    width=width,
                    dropout_mha=dropout_conv_reg_mha,
                    dropout_ff=dropout_conv_reg_ff,
                )
            )

        decoding_dim = self.input_dim + embedding_dim

        # DNN that acts on the node level to predict the PID
        self.nn_id = ffn(decoding_dim, num_classes, width, self.act, dropout_ff)

        # elementwise DNN for node momentum regression
        embed_dim = decoding_dim + num_classes
        self.nn_pt = RegressionOutput(pt_mode, embed_dim, width, self.act, dropout_ff, self.elemtypes_nonzero)
        self.nn_eta = RegressionOutput(eta_mode, embed_dim, width, self.act, dropout_ff, self.elemtypes_nonzero)
        self.nn_sin_phi = RegressionOutput(sin_phi_mode, embed_dim, width, self.act, dropout_ff, self.elemtypes_nonzero)
        self.nn_cos_phi = RegressionOutput(cos_phi_mode, embed_dim, width, self.act, dropout_ff, self.elemtypes_nonzero)
        self.nn_energy = RegressionOutput(energy_mode, embed_dim, width, self.act, dropout_ff, self.elemtypes_nonzero)

    # @torch.compile
    def forward(self, X_features):
        Xfeat_normed = X_features

        embeddings_id, embeddings_reg = [], []
        embedding_id = self.nn0_id(Xfeat_normed)
        embedding_reg = self.nn0_reg(Xfeat_normed)

        for num, conv in enumerate(self.conv_id):
            conv_input = embedding_id if num == 0 else embeddings_id[-1]
            out_padded = conv(conv_input)
            embeddings_id.append(out_padded)
        for num, conv in enumerate(self.conv_reg):
            conv_input = embedding_reg if num == 0 else embeddings_reg[-1]
            out_padded = conv(conv_input)
            embeddings_reg.append(out_padded)

        final_embedding_id = torch.cat([Xfeat_normed] + [embeddings_id[-1]], axis=-1)
        preds_id = self.nn_id(final_embedding_id)

        final_embedding_id = torch.cat([Xfeat_normed] + [embeddings_id[-1]], axis=-1)
        final_embedding_reg = torch.cat([Xfeat_normed] + [embeddings_reg[-1]] + [preds_id], axis=-1)

        # The PFElement feature order in X_features defined in fcc/postprocessing.py
        preds_pt = self.nn_pt(X_features, final_embedding_reg, X_features[..., 1:2])
        preds_eta = self.nn_eta(X_features, final_embedding_reg, X_features[..., 2:3])
        preds_sin_phi = self.nn_sin_phi(X_features, final_embedding_reg, X_features[..., 3:4])
        preds_cos_phi = self.nn_cos_phi(X_features, final_embedding_reg, X_features[..., 4:5])
        preds_energy = self.nn_energy(X_features, final_embedding_reg, X_features[..., 5:6])
        preds_momentum = torch.cat([preds_pt, preds_eta, preds_sin_phi, preds_cos_phi, preds_energy], axis=-1)

        return preds_id, preds_momentum


@onnxscript.script(custom_opset)
def SDPA(
    query: TFloat,
    key: TFloat,
    value: TFloat,
) -> TFloat:

    # unlike pytorch scaled_dot_product_attention,
    # the input here is (bsz, seq_len, num_head*head_dim)
    output, _, _ = msft_op.MultiHeadAttention(query, key, value, num_heads=NUM_HEADS)

    return output


# setType API provides shape/type to ONNX shape/type inference
def custom_scaled_dot_product_attention(
    g, query: TFloat, key: TFloat, value: TFloat, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
):
    return g.onnxscript_op(SDPA, query, key, value).setType(query.type())


print("registering custom op for scaled_dot_product_attention")
torch.onnx.register_custom_op_symbolic(
    symbolic_name="aten::scaled_dot_product_attention",
    symbolic_fn=custom_scaled_dot_product_attention,
    opset_version=opset_version,
)

if __name__ == "__main__":
    model_fp32 = MLPF(**model_kwargs)
    model_fp32.eval()
    model_fp32.load_state_dict(model_state["model_state_dict"])

    # (batch, seq, feat)
    dummy_features = torch.randn(1, 256, 55).float()
    pred = model_fp32(dummy_features)

    # inform model that the sdpa happens with 3D (batch, seq_len, num_heads*head_dim) data as is required for ONNX
    for conv in model_fp32.conv_id + model_fp32.conv_reg:
        conv.mha.onnx_export = True

    torch.onnx.export(
        model_fp32,
        dummy_features,
        "test_fp32.onnx",
        opset_version=opset_version,
        verbose=False,
        input_names=[
            "Xfeat_normed",
        ],
        output_names=["id", "momentum"],
        dynamic_axes={
            "Xfeat_normed": {0: "num_batch", 1: "num_elements"},
            "id": {0: "num_batch", 1: "num_elements"},
            "momentum": {0: "num_batch", 1: "num_elements"},
        },
    )

    # check that fp32 model gets the same result between pytorch and ONNX
    sess_options = rt.SessionOptions()
    onnx_sess = rt.InferenceSession("test_fp32.onnx", sess_options, providers=["CPUExecutionProvider"])
    pred_onnx = onnx_sess.run(None, {"Xfeat_normed": dummy_features.numpy()})
    torch.testing.assert_close(pred[0], torch.tensor(pred_onnx[0]), rtol=1e-3, atol=1e-3)
    print("fp32 closes")

    # convert to fp16 (needed for com.microsoft.MultiheadAttention to trigger flash attention)
    model = onnx.load("test_fp32.onnx")
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, "test_fp16.onnx")

    # check that fp16 model gets the same result between pytorch and onnx
    sess_options = rt.SessionOptions()
    onnx_sess = rt.InferenceSession("test_fp16.onnx", sess_options, providers=["CPUExecutionProvider"])
    pred_onnx = onnx_sess.run(None, {"Xfeat_normed": np.float16(dummy_features.numpy())})
    torch.testing.assert_close(pred[0], torch.tensor(pred_onnx[0]).float(), rtol=0.1, atol=0.1)
    print("fp16 closes")

    # dynamo does not aten::scaled_dot_product_attention, so our op replacement doesn't work
    # export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
    # onnx_program = torch.onnx.dynamo_export(torch.compile(model_fp32), dummy_features,
    #     export_options=export_options)
    # onnx_program.save("test_fp32_dynamo.onnx")
