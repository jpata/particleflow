import math

import torch
from torch import nn
from torch.nn import functional as F


def _wrapped_delta_phi(left, right):
    return torch.remainder(left - right + math.pi, 2.0 * math.pi) - math.pi


class ParticleSetDecoderLayer(nn.Module):
    """Pre-norm particle-query decoder layer with optional local cross-attention."""

    def __init__(self, embedding_dim, num_heads, ffn_dim, dropout=0.0):
        super().__init__()
        self.query_norm = nn.LayerNorm(embedding_dim)
        self.memory_norm = nn.LayerNorm(embedding_dim)
        self.cross_attention = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.self_norm = nn.LayerNorm(embedding_dim)
        self.self_attention = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        slots,
        memory,
        memory_mask,
        query_references=None,
        query_reference_mask=None,
        memory_positions=None,
        local_attention_radius=None,
    ):
        cross_queries = self.query_norm(slots)
        normalized_memory = self.memory_norm(memory)
        cross_outputs = []
        for event_idx in range(memory.shape[0]):
            valid_memory = memory_mask[event_idx]
            event_memory = normalized_memory[event_idx : event_idx + 1, valid_memory]
            if event_memory.shape[1] == 0:
                cross_outputs.append(
                    torch.zeros_like(cross_queries[event_idx : event_idx + 1])
                )
                continue

            attention_mask = None
            if local_attention_radius is not None:
                event_positions = memory_positions[event_idx, valid_memory]
                reference = query_references[event_idx]
                delta_eta = reference[:, None, 0] - event_positions[None, :, 0]
                delta_phi = _wrapped_delta_phi(
                    reference[:, None, 1], event_positions[None, :, 1]
                )
                attention_mask = (
                    delta_eta.square() + delta_phi.square() > local_attention_radius**2
                )
                if query_reference_mask is not None:
                    attention_mask[~query_reference_mask[event_idx]] = False

                # A sparse or malformed event must never produce a fully masked
                # attention row. Fall back to its nearest valid input.
                fully_masked = attention_mask.all(dim=1)
                if fully_masked.any():
                    distance = delta_eta.square() + delta_phi.square()
                    nearest = distance[fully_masked].argmin(dim=1)
                    attention_mask[fully_masked] = True
                    attention_mask[fully_masked, nearest] = False

            event_output, _ = self.cross_attention(
                cross_queries[event_idx : event_idx + 1],
                event_memory,
                event_memory,
                attn_mask=attention_mask,
                need_weights=False,
            )
            cross_outputs.append(event_output)
        slots = slots + torch.cat(cross_outputs, dim=0)

        normalized_slots = self.self_norm(slots)
        self_output, _ = self.self_attention(
            normalized_slots, normalized_slots, normalized_slots, need_weights=False
        )
        slots = slots + self_output
        return slots + self.ffn(self.ffn_norm(slots))


class ParticleSetDecoder(nn.Module):
    """Decode learned or detector-seeded queries into an unordered particle set."""

    def __init__(self, embedding_dim, num_classes, config):
        super().__init__()
        if embedding_dim % config.num_heads != 0:
            raise ValueError(
                f"Set decoder embedding_dim={embedding_dim} must be divisible by num_heads={config.num_heads}"
            )

        self.num_slots = config.num_slots
        self.query_init = getattr(config.query_init, "value", config.query_init)
        self.local_attention_radius = config.local_attention_radius
        self.tracker_query_fraction = config.tracker_query_fraction
        self.use_auxiliary_losses = config.auxiliary_loss_weight > 0
        self.queries = nn.Parameter(torch.empty(1, config.num_slots, embedding_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        ffn_dim = int(config.ffn_multiplier * embedding_dim)
        self.layers = nn.ModuleList(
            ParticleSetDecoderLayer(
                embedding_dim, config.num_heads, ffn_dim, config.dropout
            )
            for _ in range(config.num_layers)
        )
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.presence_head = nn.Linear(embedding_dim, 2)
        self.pid_head = nn.Linear(embedding_dim, num_classes)

        if self.query_init == "input-conditioned":
            self.seed_projection = nn.Linear(embedding_dim, embedding_dim)
            self.reference_embedding = nn.Sequential(
                nn.Linear(3, embedding_dim),
                nn.GELU(),
                nn.Linear(embedding_dim, embedding_dim),
            )
            self.reference_delta_heads = nn.ModuleList(
                nn.Linear(embedding_dim, 2) for _ in self.layers
            )
            self.scale_head = nn.Linear(embedding_dim, 2)
            self.momentum_head = None
        else:
            self.seed_projection = None
            self.reference_embedding = None
            self.reference_delta_heads = None
            self.scale_head = None
            self.momentum_head = nn.Linear(embedding_dim, 5)

        # Populated on every forward pass. The main four-tensor return signature
        # remains unchanged for inference and elementwise compatibility.
        self.auxiliary_outputs = []

    @staticmethod
    def _take_topk(scores, candidates, count):
        count = min(count, int(candidates.sum().item()))
        if count == 0:
            return torch.empty(0, dtype=torch.long, device=scores.device)
        ranked = scores.masked_fill(~candidates, -torch.inf)
        return torch.topk(ranked, count, sorted=True).indices

    def _input_conditioned_queries(self, memory, memory_mask, input_features):
        if input_features is None or input_features.shape[-1] < 6:
            raise ValueError(
                "Input-conditioned set queries require raw input features through energy"
            )

        batch_size = memory.shape[0]
        slots = self.queries.expand(batch_size, -1, -1).clone()
        references = memory.new_zeros(
            (batch_size, self.num_slots, 2), dtype=torch.float32
        )
        reference_mask = torch.zeros(
            (batch_size, self.num_slots), dtype=torch.bool, device=memory.device
        )
        num_tracker_slots = round(self.num_slots * self.tracker_query_fraction)

        element_type = input_features[..., 0]
        proposal_score = torch.log1p(
            input_features[..., 1].float().abs()
        ) + torch.log1p(input_features[..., 5].float().abs())
        proposal_score = torch.nan_to_num(
            proposal_score, nan=-torch.inf, posinf=1.0e6, neginf=-torch.inf
        )
        input_eta = torch.nan_to_num(
            input_features[..., 2].float(), nan=0.0, posinf=10.0, neginf=-10.0
        ).clamp(-10.0, 10.0)
        input_phi = torch.atan2(
            input_features[..., 3].float(), input_features[..., 4].float()
        )

        for event_idx in range(batch_size):
            valid = memory_mask[event_idx].bool()
            chosen_mask = torch.zeros_like(valid)
            tracker = valid & (element_type[event_idx] == 1)
            calorimeter = valid & (element_type[event_idx] == 2)

            tracker_indices = self._take_topk(
                proposal_score[event_idx], tracker, num_tracker_slots
            )
            chosen_mask[tracker_indices] = True
            calo_slots = self.num_slots - len(tracker_indices)
            calo_indices = self._take_topk(
                proposal_score[event_idx], calorimeter & ~chosen_mask, calo_slots
            )
            chosen_mask[calo_indices] = True
            selected = torch.cat([tracker_indices, calo_indices])

            remaining = self.num_slots - len(selected)
            if remaining:
                fallback = self._take_topk(
                    proposal_score[event_idx], valid & ~chosen_mask, remaining
                )
                selected = torch.cat([selected, fallback])

            num_selected = len(selected)
            if num_selected == 0:
                continue
            reference = torch.stack(
                [input_eta[event_idx, selected], input_phi[event_idx, selected]], dim=-1
            )
            position_features = torch.stack(
                [
                    reference[:, 0],
                    torch.sin(reference[:, 1]),
                    torch.cos(reference[:, 1]),
                ],
                dim=-1,
            )
            slots[event_idx, :num_selected] = (
                slots[event_idx, :num_selected]
                + self.seed_projection(memory[event_idx, selected])
                + self.reference_embedding(position_features).to(memory.dtype)
            )
            references[event_idx, :num_selected] = reference
            reference_mask[event_idx, :num_selected] = True
        return slots, references, reference_mask

    def _predict(self, slots, references=None):
        normalized_slots = self.output_norm(slots)
        presence = self.presence_head(normalized_slots)
        pid = self.pid_head(normalized_slots)
        if references is None:
            momentum = self.momentum_head(normalized_slots)
            phi_direction = F.normalize(momentum[..., 2:4], dim=-1, eps=1e-6)
            momentum = torch.cat(
                [momentum[..., :2], phi_direction, momentum[..., 4:5]], dim=-1
            )
        else:
            scales = self.scale_head(normalized_slots)
            momentum = torch.stack(
                [
                    scales[..., 0],
                    references[..., 0],
                    torch.sin(references[..., 1]),
                    torch.cos(references[..., 1]),
                    scales[..., 1],
                ],
                dim=-1,
            )
        pileup = torch.zeros_like(presence)
        return presence, pid, momentum, pileup

    def forward(self, memory, memory_mask, input_features=None):
        memory_mask = memory_mask.bool()
        references = reference_mask = memory_positions = None
        if self.query_init == "input-conditioned":
            slots, references, reference_mask = self._input_conditioned_queries(
                memory, memory_mask, input_features
            )
            memory_positions = torch.stack(
                [
                    torch.nan_to_num(
                        input_features[..., 2].float(),
                        nan=0.0,
                        posinf=10.0,
                        neginf=-10.0,
                    ).clamp(-10.0, 10.0),
                    torch.atan2(
                        input_features[..., 3].float(), input_features[..., 4].float()
                    ),
                ],
                dim=-1,
            )
        else:
            slots = self.queries.expand(memory.shape[0], -1, -1)

        outputs = []
        for layer_index, layer in enumerate(self.layers):
            slots = layer(
                slots,
                memory,
                memory_mask,
                query_references=references,
                query_reference_mask=reference_mask,
                memory_positions=memory_positions,
                local_attention_radius=self.local_attention_radius,
            )
            if references is not None:
                normalized_slots = self.output_norm(slots)
                delta = torch.tanh(
                    self.reference_delta_heads[layer_index](normalized_slots)
                )
                step_size = self.local_attention_radius or 1.0
                eta = references[..., 0] + step_size * delta[..., 0]
                phi = references[..., 1] + step_size * delta[..., 1]
                references = torch.stack(
                    [eta, torch.atan2(torch.sin(phi), torch.cos(phi))], dim=-1
                )
            is_final_layer = layer_index == len(self.layers) - 1
            if self.use_auxiliary_losses or is_final_layer:
                outputs.append(self._predict(slots, references))

        self.auxiliary_outputs = outputs[:-1] if self.use_auxiliary_losses else []
        return outputs[-1]
