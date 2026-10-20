"""Study-only oracle pooling, kept out of the model/inference API.

The reductions mirror production pool_hit_associations; parity tests protect
against drift. Only association selection is replaced with fixed memberships.
"""

import torch

from mlpf.model.tracker_momentum_readout import (
    RADIAL_BINS, GEOMETRY_DIM, pool_hit_associations as model_pool_hit_associations,
)


def pool_hit_associations(*args, fixed_associations=None, **kwargs):
    """Delegate learned memberships unchanged; handle oracle memberships locally."""
    if fixed_associations is None:
        return model_pool_hit_associations(*args, **kwargs)
    return pool_fixed_associations(*args, fixed_associations=fixed_associations, **kwargs)


def pool_fixed_associations(memory, memory_mask, features, grouping, num_slots, chunk_size, topk, temperature, sparse, tracker_summary=False, *, fixed_associations):
    """Pool diagnostic memberships with the same reductions as production pooling.

    The unused candidate-selection arguments retain the production call shape.
    Membership is supplied by the study, never by the model training path.
    """
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
                dest, weights = (value[event, idx].detach() for value in fixed_associations)
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
