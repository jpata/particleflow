import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import overload

import torch
from torch import nn
from torch.nn import functional as F

try:
    from flash_attn import flash_attn_varlen_func
except (ImportError, OSError):
    flash_attn_varlen_func = None


PredictionTensors = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class ParticleSetDecoderOutput(Sequence[torch.Tensor]):
    """Primary predictions plus optional intermediate decoder predictions.

    The sequence interface preserves the existing four-tensor model output API,
    while auxiliary predictions remain explicit data from the same forward pass.
    """

    predictions: PredictionTensors
    auxiliary_predictions: tuple[PredictionTensors, ...] = ()

    @overload
    def __getitem__(self, index: int) -> torch.Tensor: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[torch.Tensor, ...]: ...

    def __getitem__(self, index: int | slice) -> torch.Tensor | tuple[torch.Tensor, ...]:
        return self.predictions[index]

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter(self.predictions)

    def __len__(self) -> int:
        return len(self.predictions)


def _wrapped_delta_phi(left, right):
    return torch.remainder(left - right + math.pi, 2.0 * math.pi) - math.pi


def _streaming_attention(query, key, value, dropout_p=0.0, query_chunk_size=128, key_chunk_size=512):
    """Small-device fallback with bounded attention workspace and autograd support."""
    if key.shape[-2] == 0:
        return torch.zeros_like(query)
    scale = query.shape[-1] ** -0.5
    outputs = []
    for query_chunk in query.split(query_chunk_size, dim=-2):
        maximum = query_chunk.new_full((*query_chunk.shape[:-1], 1), -torch.inf, dtype=torch.float32)
        denominator = query_chunk.new_zeros((*query_chunk.shape[:-1], 1), dtype=torch.float32)
        numerator = query_chunk.new_zeros(query_chunk.shape, dtype=torch.float32)
        for key_chunk, value_chunk in zip(key.split(key_chunk_size, dim=-2), value.split(key_chunk_size, dim=-2)):
            scores = torch.matmul(query_chunk.float(), key_chunk.float().transpose(-1, -2)) * scale
            new_maximum = torch.maximum(maximum, scores.amax(dim=-1, keepdim=True))
            old_scale = torch.exp(maximum - new_maximum)
            probabilities = torch.exp(scores - new_maximum)
            denominator = denominator * old_scale + probabilities.sum(dim=-1, keepdim=True)
            numerator = numerator * old_scale + torch.matmul(
                F.dropout(probabilities, p=dropout_p, training=dropout_p > 0), value_chunk.float()
            )
            maximum = new_maximum
        outputs.append((numerator / denominator.clamp_min(1.0e-12)).to(query.dtype))
    return torch.cat(outputs, dim=-2)


@torch.compiler.disable
def _flash_attn_varlen_eager(*args, **kwargs):
    # ROCm's custom op has a fake registration incompatible with torch.compile.
    return flash_attn_varlen_func(*args, **kwargs)


def _flash_multihead_attention(module, queries, memory, memory_mask=None):
    """Cross/self attention without a query-by-memory allocation.

    GPU execution requires variable-length FlashAttention. The streaming fallback
    keeps CPU tests and inference bounded too; it is not the production kernel.
    """
    batch_size, num_queries, embedding_dim = queries.shape
    num_heads = module.num_heads
    head_dim = embedding_dim // num_heads
    q_weight, k_weight, v_weight = module.in_proj_weight.chunk(3, dim=0)
    q_bias, k_bias, v_bias = module.in_proj_bias.chunk(3, dim=0) if module.in_proj_bias is not None else (None,) * 3
    q = F.linear(queries, q_weight, q_bias).reshape(batch_size, num_queries, num_heads, head_dim)
    k = F.linear(memory, k_weight, k_bias).reshape(batch_size, memory.shape[1], num_heads, head_dim)
    v = F.linear(memory, v_weight, v_bias).reshape(batch_size, memory.shape[1], num_heads, head_dim)
    if memory_mask is None:
        memory_mask = torch.ones(memory.shape[:2], dtype=torch.bool, device=memory.device)
    else:
        memory_mask = memory_mask.bool()
    empty_events = ~memory_mask.any(dim=1)

    if queries.is_cuda:
        if flash_attn_varlen_func is None:
            raise RuntimeError("Set decoder attention on GPU requires flash_attn_varlen_func")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("Set decoder FlashAttention requires float16 or bfloat16; enable mixed precision")
        if memory.shape[1] == 0:
            attention_output = torch.zeros_like(q)
        else:
            # FlashAttention requires a nonempty key sequence for every event.
            safe_mask = memory_mask.clone()
            safe_mask[empty_events, 0] = True
            lengths = safe_mask.sum(dim=1, dtype=torch.int32)
            key_offsets = F.pad(lengths.cumsum(dim=0, dtype=torch.int32), (1, 0))
            query_offsets = torch.arange(batch_size + 1, dtype=torch.int32, device=queries.device) * num_queries
            flash_attn_varlen = _flash_attn_varlen_eager if torch.version.hip is not None else flash_attn_varlen_func
            attention_output = flash_attn_varlen(
                q.contiguous().reshape(-1, num_heads, head_dim),
                k[safe_mask].contiguous(),
                v[safe_mask].contiguous(),
                query_offsets,
                key_offsets,
                num_queries,
                int(lengths.max().item()),
                dropout_p=module.dropout if module.training else 0.0,
                causal=False,
            ).reshape(batch_size, num_queries, num_heads, head_dim)
            attention_output = attention_output.masked_fill(empty_events[:, None, None, None], 0)
    else:
        attention_output = torch.stack(
            [
                _streaming_attention(
                    q[event_idx].transpose(0, 1),
                    k[event_idx, memory_mask[event_idx]].transpose(0, 1),
                    v[event_idx, memory_mask[event_idx]].transpose(0, 1),
                    dropout_p=module.dropout if module.training else 0.0,
                ).transpose(0, 1)
                for event_idx in range(batch_size)
            ]
        )
    projected = F.linear(attention_output.reshape(batch_size, num_queries, embedding_dim), module.out_proj.weight, module.out_proj.bias)
    return projected.masked_fill(empty_events[:, None, None], 0)


class ParticleSetDecoderLayer(nn.Module):
    """Pre-norm particle-query decoder layer using global FlashAttention."""

    def __init__(self, embedding_dim, num_heads, ffn_dim, dropout=0.0):
        super().__init__()
        self.query_norm = nn.LayerNorm(embedding_dim)
        self.memory_norm = nn.LayerNorm(embedding_dim)
        # Keep the standard parameter layout so existing decoder checkpoints load.
        # Attention itself runs through _flash_multihead_attention.
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

    def forward(
        self,
        slots,
        memory,
        memory_mask,
    ):
        cross_queries = self.query_norm(slots)
        normalized_memory = self.memory_norm(memory)
        slots = slots + _flash_multihead_attention(self.cross_attention, cross_queries, normalized_memory, memory_mask)

        normalized_slots = self.self_norm(slots)
        slots = slots + _flash_multihead_attention(self.self_attention, normalized_slots, normalized_slots)
        return slots + self.ffn(self.ffn_norm(slots))


def _phi_sector(phi, num_sectors):
    normalized_phi = torch.remainder(torch.nan_to_num(phi.float(), nan=0.0) + math.pi, 2.0 * math.pi)
    return torch.floor(normalized_phi * (num_sectors / (2.0 * math.pi))).long().remainder(num_sectors)


class SectorizedParticleSetDecoderLayer(ParticleSetDecoderLayer):
    """Cross- and self-attention on periodic phi sectors without pairwise masks."""

    def forward(
        self,
        slots,
        memory,
        memory_mask,
        query_references=None,
        query_reference_mask=None,
        memory_positions=None,
        num_sectors=32,
        sector_neighbors=1,
    ):
        cross_queries = self.query_norm(slots)
        normalized_memory = self.memory_norm(memory)
        cross_outputs = []
        slot_sectors = _phi_sector(query_references[..., 1], num_sectors)
        if query_reference_mask is not None:
            fallback_sectors = torch.arange(slots.shape[1], device=slots.device).remainder(num_sectors)
            slot_sectors = torch.where(query_reference_mask, slot_sectors, fallback_sectors.unsqueeze(0))
        hit_sectors = _phi_sector(memory_positions[..., 1], num_sectors)

        for event_idx in range(memory.shape[0]):
            sector_indices = []
            sector_outputs = []
            valid_hits = torch.nonzero(memory_mask[event_idx], as_tuple=True)[0]
            event_hit_sectors = hit_sectors[event_idx, valid_hits]
            for sector in range(num_sectors):
                slot_indices = torch.nonzero(slot_sectors[event_idx] == sector, as_tuple=True)[0]
                if slot_indices.numel() == 0 or valid_hits.numel() == 0:
                    continue
                sector_distance = (event_hit_sectors - sector).remainder(num_sectors)
                nearby = (sector_distance <= sector_neighbors) | (sector_distance >= num_sectors - sector_neighbors)
                hit_indices = valid_hits[nearby]
                if hit_indices.numel() == 0:
                    sector_center = -math.pi + (sector + 0.5) * (2.0 * math.pi / num_sectors)
                    nearest = _wrapped_delta_phi(memory_positions[event_idx, valid_hits, 1], sector_center).abs().argmin()
                    hit_indices = valid_hits[nearest : nearest + 1]
                event_queries = cross_queries[event_idx : event_idx + 1, slot_indices]
                event_memory = normalized_memory[event_idx : event_idx + 1, hit_indices]
                sector_output = _flash_multihead_attention(self.cross_attention, event_queries, event_memory)
                sector_indices.append(slot_indices)
                sector_outputs.append(sector_output.squeeze(0))
            if sector_outputs:
                cross_outputs.append(torch.cat(sector_outputs)[torch.argsort(torch.cat(sector_indices))])
            else:
                cross_outputs.append(torch.zeros_like(cross_queries[event_idx]))
        slots = slots + torch.stack(cross_outputs)

        normalized_slots = self.self_norm(slots)
        self_outputs = []
        for event_idx in range(slots.shape[0]):
            sector_indices = []
            sector_outputs = []
            for sector in range(num_sectors):
                slot_indices = torch.nonzero(slot_sectors[event_idx] == sector, as_tuple=True)[0]
                if slot_indices.numel() == 0:
                    continue
                sector_slots = normalized_slots[event_idx : event_idx + 1, slot_indices]
                sector_output = _flash_multihead_attention(self.self_attention, sector_slots, sector_slots)
                sector_indices.append(slot_indices)
                sector_outputs.append(sector_output.squeeze(0))
            self_outputs.append(torch.cat(sector_outputs)[torch.argsort(torch.cat(sector_indices))])
        slots = slots + torch.stack(self_outputs)
        return slots + self.ffn(self.ffn_norm(slots))


class ParticleSetDecoder(nn.Module):
    """Decode learned or detector-seeded queries into an unordered particle set."""

    AGGREGATE_FEATURE_DIM = 8

    def __init__(self, embedding_dim, num_classes, config):
        super().__init__()
        if embedding_dim % config.num_heads != 0:
            raise ValueError(f"Set decoder embedding_dim={embedding_dim} must be divisible by num_heads={config.num_heads}")

        self.num_slots = config.num_slots
        self.query_init = getattr(config.query_init, "value", config.query_init)
        # Retain the historical config name for checkpoint/scenario compatibility.
        # Global attention no longer uses it as a radius mask.
        self.reference_update_scale = config.local_attention_radius
        self.attention_mode = config.attention_mode
        self.num_sectors = config.num_sectors
        self.sector_neighbors = config.sector_neighbors
        self.tracker_query_fraction = config.tracker_query_fraction
        self.proposal_mode = config.proposal_mode
        self.proposal_grid_size = config.proposal_grid_size
        self.use_aggregate_anchors = config.use_aggregate_anchors
        self.aggregate_grid_size = config.aggregate_grid_size
        self.use_auxiliary_losses = config.auxiliary_loss_weight > 0
        self.queries = nn.Parameter(torch.empty(1, config.num_slots, embedding_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        ffn_dim = int(config.ffn_multiplier * embedding_dim)
        layer_type = {
            "global-flash": ParticleSetDecoderLayer,
            "sectorized": SectorizedParticleSetDecoderLayer,
        }[self.attention_mode]
        self.layers = nn.ModuleList(layer_type(embedding_dim, config.num_heads, ffn_dim, config.dropout) for _ in range(config.num_layers))
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
            self.reference_delta_heads = nn.ModuleList(nn.Linear(embedding_dim, 2) for _ in self.layers)
            self.scale_head = nn.Linear(embedding_dim, 2)
            self.aggregate_projection = (
                nn.Sequential(
                    nn.Linear(self.AGGREGATE_FEATURE_DIM, embedding_dim),
                    nn.GELU(),
                    nn.Linear(embedding_dim, embedding_dim),
                )
                if self.use_aggregate_anchors
                else None
            )
            self.momentum_head = None
        else:
            self.seed_projection = None
            self.reference_embedding = None
            self.reference_delta_heads = None
            self.scale_head = None
            self.aggregate_projection = None
            self.momentum_head = nn.Linear(embedding_dim, 5)

    @staticmethod
    def _take_topk(scores, candidates, count):
        count = min(count, int(candidates.sum().item()))
        if count == 0:
            return torch.empty(0, dtype=torch.long, device=scores.device)
        ranked = scores.masked_fill(~candidates, -torch.inf)
        return torch.topk(ranked, count, sorted=True).indices

    @staticmethod
    def _angular_grid_summary(input_features, valid, grid_size):
        """Return per-element cell summaries and one representative per occupied cell.

        The implementation uses sorting and scatter reductions rather than
        materializing pairwise hit distances. Tracker and calorimeter hits use
        disjoint cell keys.
        """

        device = input_features.device
        num_elements = input_features.shape[0]
        summaries = input_features.new_zeros((num_elements, ParticleSetDecoder.AGGREGATE_FEATURE_DIM), dtype=torch.float32)
        cell_rank = input_features.new_full((num_elements,), -torch.inf, dtype=torch.float32)
        representative = torch.zeros(num_elements, dtype=torch.bool, device=device)
        valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        if valid_indices.numel() == 0:
            return summaries, cell_rank, representative

        features = input_features[valid_indices].float()
        element_type = features[:, 0]
        pt = features[:, 1].abs()
        eta = torch.nan_to_num(features[:, 2], nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        phi = torch.atan2(features[:, 3], features[:, 4])
        energy = features[:, 5].abs()
        radius = torch.linalg.vector_norm(features[:, 6:9], dim=-1)

        num_phi_bins = math.ceil(2.0 * math.pi / grid_size)
        num_eta_bins = math.ceil(20.0 / grid_size)
        eta_bin = torch.floor((eta + 10.0) / grid_size).to(torch.int64).clamp_(0, num_eta_bins - 1)
        phi_bin = torch.floor((phi + math.pi) / grid_size).to(torch.int64).clamp_(0, num_phi_bins - 1)
        type_index = (element_type == 2).to(torch.int64)
        keys = (type_index * num_eta_bins + eta_bin) * num_phi_bins + phi_bin
        occupied_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
        num_cells = occupied_keys.shape[0]

        is_calo = element_type == 2
        weight = torch.where(is_calo, energy.clamp_min(1.0e-8), torch.ones_like(energy))
        values = torch.stack(
            [
                torch.ones_like(weight),
                pt,
                energy,
                weight,
                weight * eta,
                weight * torch.sin(phi),
                weight * torch.cos(phi),
            ],
            dim=-1,
        )
        reductions = torch.zeros((num_cells, values.shape[-1]), dtype=torch.float32, device=device)
        reductions.scatter_add_(0, inverse.unsqueeze(-1).expand_as(values), values)
        local = reductions[inverse]
        centroid_eta = local[:, 4] / local[:, 3].clamp_min(1.0e-8)
        centroid_phi = torch.atan2(local[:, 5], local[:, 6])
        local_summary = torch.stack(
            [
                torch.log1p(local[:, 0]),
                torch.log1p(local[:, 1]),
                torch.log1p(local[:, 2]),
                centroid_eta,
                torch.sin(centroid_phi),
                torch.cos(centroid_phi),
                (~is_calo).to(torch.float32),
                is_calo.to(torch.float32),
            ],
            dim=-1,
        )
        summaries[valid_indices] = local_summary

        # Rank calorimeter cells by total energy and tracker cells by occupancy.
        local_rank = torch.where(is_calo, torch.log1p(local[:, 2]), torch.log1p(local[:, 0]))
        cell_rank[valid_indices] = local_rank

        # Use the highest-energy calorimeter hit and the innermost tracker hit
        # as a deterministic representative of each occupied cell.
        element_rank = torch.where(is_calo, torch.log1p(energy), -radius)
        max_rank = torch.full((num_cells,), -torch.inf, dtype=torch.float32, device=device)
        max_rank.scatter_reduce_(0, inverse, element_rank, reduce="amax", include_self=True)
        local_position = torch.arange(valid_indices.numel(), device=device)
        candidate_position = torch.where(element_rank == max_rank[inverse], local_position, valid_indices.numel())
        representative_position = torch.full((num_cells,), valid_indices.numel(), dtype=torch.long, device=device)
        representative_position.scatter_reduce_(0, inverse, candidate_position, reduce="amin", include_self=True)
        representative[valid_indices[representative_position]] = True
        return summaries, cell_rank, representative

    def _input_conditioned_queries(self, memory, memory_mask, input_features):
        if input_features is None or input_features.shape[-1] < 6:
            raise ValueError("Input-conditioned set queries require raw input features through energy")
        if (self.proposal_mode == "grid-diverse" or self.use_aggregate_anchors) and input_features.shape[-1] < 9:
            raise ValueError("Grid-diverse proposals and aggregate anchors require raw hit positions")

        batch_size = memory.shape[0]
        slots = self.queries.expand(batch_size, -1, -1).clone()
        references = memory.new_zeros((batch_size, self.num_slots, 2), dtype=torch.float32)
        reference_mask = torch.zeros((batch_size, self.num_slots), dtype=torch.bool, device=memory.device)
        scale_anchors = memory.new_zeros((batch_size, self.num_slots, 2), dtype=torch.float32)
        scale_anchor_mask = torch.zeros((batch_size, self.num_slots), dtype=torch.bool, device=memory.device)
        num_tracker_slots = round(self.num_slots * self.tracker_query_fraction)

        element_type = input_features[..., 0]
        proposal_score = torch.log1p(input_features[..., 1].float().abs()) + torch.log1p(input_features[..., 5].float().abs())
        proposal_score = torch.nan_to_num(proposal_score, nan=-torch.inf, posinf=1.0e6, neginf=-torch.inf)
        input_eta = torch.nan_to_num(input_features[..., 2].float(), nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        input_phi = torch.atan2(input_features[..., 3].float(), input_features[..., 4].float())

        for event_idx in range(batch_size):
            valid = memory_mask[event_idx].bool()
            chosen_mask = torch.zeros_like(valid)
            tracker = valid & (element_type[event_idx] == 1)
            calorimeter = valid & (element_type[event_idx] == 2)

            proposal_candidates = valid
            proposal_scores = proposal_score[event_idx]
            proposal_summary = None
            if self.proposal_mode == "grid-diverse":
                proposal_summary, proposal_scores, proposal_candidates = self._angular_grid_summary(
                    input_features[event_idx], valid, self.proposal_grid_size
                )

            tracker_indices = self._take_topk(proposal_scores, tracker & proposal_candidates, num_tracker_slots)
            chosen_mask[tracker_indices] = True
            calo_slots = self.num_slots - len(tracker_indices)
            calo_indices = self._take_topk(proposal_scores, calorimeter & proposal_candidates & ~chosen_mask, calo_slots)
            chosen_mask[calo_indices] = True
            selected = torch.cat([tracker_indices, calo_indices])

            remaining = self.num_slots - len(selected)
            if remaining and self.proposal_mode == "grid-diverse":
                extra_representatives = self._take_topk(proposal_scores, proposal_candidates & ~chosen_mask, remaining)
                chosen_mask[extra_representatives] = True
                selected = torch.cat([selected, extra_representatives])
                remaining = self.num_slots - len(selected)
            if remaining:
                fallback = self._take_topk(proposal_score[event_idx], valid & ~chosen_mask, remaining)
                selected = torch.cat([selected, fallback])

            num_selected = len(selected)
            if num_selected == 0:
                continue
            selected_eta = input_eta[event_idx, selected]
            selected_phi = input_phi[event_idx, selected]
            selected_summary = None
            if self.use_aggregate_anchors:
                if proposal_summary is not None and self.aggregate_grid_size == self.proposal_grid_size:
                    summary = proposal_summary
                else:
                    summary, _, _ = self._angular_grid_summary(input_features[event_idx], valid, self.aggregate_grid_size)
                selected_summary = summary[selected]
                selected_is_calo = element_type[event_idx, selected] == 2
                centroid_eta = selected_summary[:, 3]
                centroid_phi = torch.atan2(selected_summary[:, 4], selected_summary[:, 5])
                selected_eta = torch.where(selected_is_calo, centroid_eta, selected_eta)
                selected_phi = torch.where(selected_is_calo, centroid_phi, selected_phi)
                log_energy = selected_summary[:, 2]
                log_pt = torch.log(torch.expm1(log_energy).clamp_min(1.0e-8) / torch.cosh(selected_eta).clamp_min(1.0))
                scale_anchors[event_idx, :num_selected] = torch.stack([log_pt, log_energy], dim=-1)
                scale_anchor_mask[event_idx, :num_selected] = selected_is_calo

            reference = torch.stack([selected_eta, selected_phi], dim=-1)
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
            if self.aggregate_projection is not None:
                slots[event_idx, :num_selected] = slots[event_idx, :num_selected] + self.aggregate_projection(selected_summary.to(memory.dtype))
            references[event_idx, :num_selected] = reference
            reference_mask[event_idx, :num_selected] = True
        return slots, references, reference_mask, scale_anchors, scale_anchor_mask

    def _predict(self, slots, references=None, scale_anchors=None, scale_anchor_mask=None):
        normalized_slots = self.output_norm(slots)
        presence = self.presence_head(normalized_slots)
        pid = self.pid_head(normalized_slots)
        if references is None:
            momentum = self.momentum_head(normalized_slots)
            phi_direction = F.normalize(momentum[..., 2:4], dim=-1, eps=1e-6)
            momentum = torch.cat([momentum[..., :2], phi_direction, momentum[..., 4:5]], dim=-1)
        else:
            scales = self.scale_head(normalized_slots)
            if scale_anchors is not None:
                scales = torch.where(scale_anchor_mask.unsqueeze(-1), scales + scale_anchors.to(scales.dtype), scales)
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
        scale_anchors = scale_anchor_mask = None
        if self.query_init == "input-conditioned":
            slots, references, reference_mask, scale_anchors, scale_anchor_mask = self._input_conditioned_queries(memory, memory_mask, input_features)
            memory_positions = torch.stack(
                [
                    torch.nan_to_num(
                        input_features[..., 2].float(),
                        nan=0.0,
                        posinf=10.0,
                        neginf=-10.0,
                    ).clamp(-10.0, 10.0),
                    torch.atan2(input_features[..., 3].float(), input_features[..., 4].float()),
                ],
                dim=-1,
            )
        else:
            slots = self.queries.expand(memory.shape[0], -1, -1)

        outputs = []
        for layer_index, layer in enumerate(self.layers):
            if self.attention_mode == "sectorized":
                slots = layer(
                    slots, memory, memory_mask, references, reference_mask, memory_positions,
                    num_sectors=self.num_sectors, sector_neighbors=self.sector_neighbors,
                )
            else:
                slots = layer(slots, memory, memory_mask)
            if references is not None:
                normalized_slots = self.output_norm(slots)
                delta = torch.tanh(self.reference_delta_heads[layer_index](normalized_slots))
                step_size = self.reference_update_scale or 1.0
                eta = references[..., 0] + step_size * delta[..., 0]
                phi = references[..., 1] + step_size * delta[..., 1]
                references = torch.stack([eta, torch.atan2(torch.sin(phi), torch.cos(phi))], dim=-1)
            is_final_layer = layer_index == len(self.layers) - 1
            if self.use_auxiliary_losses or is_final_layer:
                outputs.append(self._predict(slots, references, scale_anchors, scale_anchor_mask))

        auxiliary_predictions = tuple(outputs[:-1]) if self.use_auxiliary_losses else ()
        return ParticleSetDecoderOutput(
            predictions=outputs[-1],
            auxiliary_predictions=auxiliary_predictions,
        )
