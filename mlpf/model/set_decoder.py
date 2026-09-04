import torch
from torch import nn
from torch.nn import functional as F


class ParticleSetDecoderLayer(nn.Module):
    """Pre-norm particle-query decoder layer.

    Cross-attention is evaluated event by event on only the valid input embeddings.
    With ``need_weights=False``, PyTorch dispatches through scaled-dot-product
    attention and can select a fused FlashAttention kernel on CUDA without retaining
    the slot-by-input attention matrix.
    """

    def __init__(self, embedding_dim, num_heads, ffn_dim, dropout=0.0):
        super().__init__()
        self.query_norm = nn.LayerNorm(embedding_dim)
        self.memory_norm = nn.LayerNorm(embedding_dim)
        self.cross_attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(embedding_dim)
        self.self_attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, slots, memory, memory_mask):
        cross_queries = self.query_norm(slots)
        normalized_memory = self.memory_norm(memory)
        cross_outputs = []
        for event_idx in range(memory.shape[0]):
            event_memory = normalized_memory[event_idx : event_idx + 1, memory_mask[event_idx]]
            if event_memory.shape[1] == 0:
                cross_outputs.append(torch.zeros_like(cross_queries[event_idx : event_idx + 1]))
                continue
            event_output, _ = self.cross_attention(
                cross_queries[event_idx : event_idx + 1],
                event_memory,
                event_memory,
                need_weights=False,
            )
            cross_outputs.append(event_output)
        slots = slots + torch.cat(cross_outputs, dim=0)

        normalized_slots = self.self_norm(slots)
        self_output, _ = self.self_attention(normalized_slots, normalized_slots, normalized_slots, need_weights=False)
        slots = slots + self_output
        return slots + self.ffn(self.ffn_norm(slots))


class ParticleSetDecoder(nn.Module):
    """Decode a fixed bank of learned queries into an unordered particle set."""

    def __init__(self, embedding_dim, num_classes, config):
        super().__init__()
        if embedding_dim % config.num_heads != 0:
            raise ValueError(f"Set decoder embedding_dim={embedding_dim} must be divisible by num_heads={config.num_heads}")

        self.num_slots = config.num_slots
        self.queries = nn.Parameter(torch.empty(1, config.num_slots, embedding_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        ffn_dim = int(config.ffn_multiplier * embedding_dim)
        self.layers = nn.ModuleList(
            ParticleSetDecoderLayer(embedding_dim, config.num_heads, ffn_dim, config.dropout) for _ in range(config.num_layers)
        )
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.presence_head = nn.Linear(embedding_dim, 2)
        self.pid_head = nn.Linear(embedding_dim, num_classes)
        self.momentum_head = nn.Linear(embedding_dim, 5)

    def forward(self, memory, memory_mask):
        slots = self.queries.expand(memory.shape[0], -1, -1)
        for layer in self.layers:
            slots = layer(slots, memory, memory_mask.bool())
        slots = self.output_norm(slots)

        presence = self.presence_head(slots)
        pid = self.pid_head(slots)
        momentum = self.momentum_head(slots)
        phi_direction = F.normalize(momentum[..., 2:4], dim=-1, eps=1e-6)
        momentum = torch.cat([momentum[..., :2], phi_direction, momentum[..., 4:5]], dim=-1)
        pileup = torch.zeros_like(presence)
        return presence, pid, momentum, pileup
