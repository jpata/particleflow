"""Sparse association and tracker-only geometric pooling for particle-set readout.

Positions in the CLD raw-hit schema are millimetres (columns 6:9). Geometry
and reductions stay in FP32. Candidate selection is discrete; selected scores,
background competition, and weighted pooling remain differentiable.
"""

import torch
from torch import nn
from torch.nn import functional as F


RADIAL_BINS = 4
GEOMETRY_DIM = 10
TRACKER_SUMMARY_DIM = RADIAL_BINS * (GEOMETRY_DIM + 1)


def association_weights(hit_keys, slot_keys, background, topk, temperature, sparse):
    """Return slot indices and weights; omitted probability belongs to background.

    Sparse normalization uses only the selected top-k slots and background. It
    is a truncated softmax, not an approximation preserving full-softmax mass.
    The caller chunks hits, bounding temporary candidate scores by chunk x slots.
    """
    with torch.autocast(device_type=hit_keys.device.type, enabled=False):
        hit_keys, slot_keys, background = hit_keys.float(), slot_keys.float(), background.float()
        with torch.no_grad():
            scores = 10.0 * (hit_keys @ slot_keys.T)
            if not sparse:
                winner = torch.cat((scores, background), dim=-1).argmax(-1)
                return winner.clamp_max(len(slot_keys) - 1)[:, None], (winner < len(slot_keys)).float()[:, None]
            destinations = scores.topk(topk, dim=-1).indices
        # Recompute just selected affinities, so autograd never retains H x S.
        selected = (hit_keys[:, None, :] * slot_keys[destinations]).sum(-1) * 10.0
        logits = torch.cat((selected, background), dim=-1) / temperature
        weights = logits.softmax(-1)[:, :-1]
        return destinations, weights


def pool_hit_associations(memory, memory_mask, features, grouping, num_slots, chunk_size, topk, temperature, sparse, tracker_summary=False):
    """Detector-wise weighted means and optional tracker radial geometry summaries."""
    hit_keys, slot_keys, background = grouping
    width = memory.shape[-1]
    all_summaries, all_tracker = [], []
    with torch.autocast(device_type=memory.device.type, enabled=False):
        for event in range(len(memory)):
            # A connected zero also keeps empty-event DDP parameter usage valid.
            zero = (memory[event].float().sum() + hit_keys[event].sum() + slot_keys[event].sum() + background[event].sum()) * 0
            pooled = torch.zeros(num_slots * 2, width, device=memory.device) + zero
            counts = torch.zeros(num_slots * 2, device=memory.device) + zero
            energy = torch.zeros_like(counts) + zero
            geom = torch.zeros(num_slots * RADIAL_BINS, GEOMETRY_DIM, device=memory.device) + zero
            geom_counts = torch.zeros(num_slots * RADIAL_BINS, device=memory.device) + zero
            indices = torch.nonzero(memory_mask[event], as_tuple=True)[0]
            for idx in indices.split(chunk_size):
                if not len(idx):
                    continue
                dest, weights = association_weights(hit_keys[event, idx], slot_keys[event], background[event, idx], topk, temperature, sparse)
                detector = (features[event, idx, 0] == 2).long()
                emb = memory[event, idx].float()
                deposit = features[event, idx, 5].float().clamp_min(0)
                if tracker_summary:
                    xyz = features[event, idx, 6:9].float() * 0.001
                    tracker = (features[event, idx, 0] == 1) & torch.isfinite(xyz).all(-1)
                    xyz = torch.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)
                    x, y, z = xyz.unbind(-1)
                    radius2 = x.square() + y.square()
                    radius = radius2.clamp_min(1e-12).sqrt()
                    # Regularized conformal coordinates supplement Cartesian moments.
                    u, v = 0.1 * x / radius2.clamp_min(1e-6), 0.1 * y / radius2.clamp_min(1e-6)
                    geometry = torch.stack((x, y, z, radius, u, v, x.square(), y.square(), x * y, z * radius), -1)
                    radial_bin = torch.bucketize(radius, radius.new_tensor([0.1, 0.3, 0.7]))
                for k in range(weights.shape[1]):
                    w = weights[:, k]
                    flat = dest[:, k] * 2 + detector
                    pooled = pooled.index_add(0, flat, emb * w[:, None])
                    counts = counts.index_add(0, flat, w)
                    energy = energy.index_add(0, flat, deposit * w)
                    if tracker_summary:
                        gflat = dest[:, k] * RADIAL_BINS + radial_bin
                        gw = w * tracker.float()
                        geom = geom.index_add(0, gflat, geometry * gw[:, None])
                        geom_counts = geom_counts.index_add(0, gflat, gw)
            counts = counts.view(num_slots, 2)
            # Clamp at one to suppress nearly empty slots, preserving background gating.
            mean = pooled.view(num_slots, 2, width) / counts.clamp_min(1)[..., None]
            all_summaries.append(torch.cat((mean[:, 0], mean[:, 1], counts.log1p(), energy.view(num_slots, 2)[:, 1:2].log1p()), -1))
            if tracker_summary:
                gmean = geom / geom_counts.clamp_min(1)[:, None]
                radial = torch.cat((gmean, geom_counts.log1p()[:, None]), -1).view(num_slots, -1)
                all_tracker.append(torch.cat((mean[:, 0], radial), -1))
        return torch.stack(all_summaries), torch.stack(all_tracker) if tracker_summary else None


class TrackerMomentumReadout(nn.Module):
    """Bounded log-pT residual from tracker embeddings and radial moments only."""

    def __init__(self, embedding_dim, max_log_correction):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_log_correction = max_log_correction
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.hidden = nn.Linear(embedding_dim + TRACKER_SUMMARY_DIM, embedding_dim)
        self.output = nn.Linear(embedding_dim, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, summary):
        with torch.autocast(device_type=summary.device.type, enabled=False):
            embedding = self.embedding_norm(summary[..., : self.embedding_dim].float())
            geometry = summary[..., self.embedding_dim :].float()
            inputs = torch.cat((embedding, geometry), -1)
            correction = self.max_log_correction * self.output(F.gelu(self.hidden(inputs))).squeeze(-1).tanh()
            # Continuous support gate; exactly zero for slots without tracker hits.
            radial_counts = geometry.view(*geometry.shape[:-1], RADIAL_BINS, GEOMETRY_DIM + 1)[..., -1].expm1().sum(-1)
            return correction * radial_counts.clamp(0, 1)
