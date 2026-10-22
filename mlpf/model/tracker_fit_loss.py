"""Training-only circle supervision for learned tracker ownership.

Candidate selection and quality decisions are detached; continuous ownership
weights and the normalized double-precision least-squares solve are trainable.
No truth-selected hits enter the predicted fit or inference readout.
"""

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from mlpf.conf import Y_FEATURES


def weighted_circle(xy, weights):
    """Return centre/radius in metres, and detached conditioning decision."""
    xy, weights = xy.double(), weights.double()
    # Fixed coordinate normalization avoids weight-dependent coordinate collapse.
    mean = xy.mean(0)
    scale = (xy - mean).square().sum(-1).mean().sqrt().clamp_min(1e-6)
    uv = (xy - mean) / scale
    design = torch.cat((2 * uv, torch.ones_like(uv[:, :1])), -1)
    weights = weights / weights.sum().clamp_min(1e-12)
    gram = design.T @ (weights[:, None] * design)
    rhs = design.T @ (weights * uv.square().sum(-1))
    pars = torch.linalg.solve(gram + 1e-10 * torch.eye(3, device=xy.device, dtype=xy.dtype), rhs)
    radius = (pars[:2].square().sum() + pars[2]).clamp_min(1e-12).sqrt() * scale
    with torch.no_grad():
        eigen = torch.linalg.eigvalsh(gram)
        valid = (eigen[0] > 1e-7 * eigen[-1]) & radius.isfinite() & (radius > 0.01)
        valid &= weights.square().sum().reciprocal() >= 4
    return mean + scale * pars[:2], radius, valid


def _supported(features):
    if len(features) < 4:
        return False
    rho = features[:, 6:8].double().norm(dim=-1) * 0.001
    return bool((rho.max() - rho.min() >= 0.1) & (torch.unique(features[:, 12:15], dim=0).shape[0] >= 3))


def tracker_fit_losses(grouping, batch, matches, magnetic_field=2.0, max_hits=64, chunk_size=1024):
    """Equal-object auxiliary losses on Hungarian-matched hadrons and muons.

    Geometry targets come from detached truth-owned circles (supervision only).
    Curvature targets come directly from truth pT. Candidates are capped highest
    probability hard-owned tracker hits. Both losses retain ownership CE.
    """
    keys, slots, background = grouping
    zero = keys.sum() * 0 + slots.sum() * 0 + background.sum() * 0
    geometry, curvature = [], []
    candidates = 0
    pn_index = Y_FEATURES.index("particle_number")
    for event, (slot_indices, target_indices) in enumerate(matches):
        mask = batch.mask[event].bool() & (batch.X[event, :, 0] == 1)
        mask &= torch.isfinite(batch.X[event, :, 6:9]).all(-1)
        indices = mask.nonzero(as_tuple=True)[0]
        if len(indices) < 4:
            continue
        x = batch.X[event, indices]
        with torch.no_grad(), torch.autocast(device_type=keys.device.type, enabled=False):
            owners, confidence = [], []
            for idx in indices.split(chunk_size):
                logits = torch.cat((10 * keys[event, idx].float() @ slots[event].float().T, background[event, idx].float()), -1)
                prob, owner = logits.softmax(-1).max(-1)
                owners.append(owner)
                confidence.append(prob)
            owners, confidence = torch.cat(owners), torch.cat(confidence)
        targets = batch.ytarget_set[event, batch.target_mask[event]]
        for slot, target_index in zip(slot_indices, target_indices):
            target = targets[target_index]
            if int(target[0]) not in (1, 5):  # charged hadrons and muons; no electrons
                continue
            selected = (owners == slot).nonzero(as_tuple=True)[0]
            if len(selected) > max_hits:
                selected = selected[confidence[selected].topk(max_hits).indices]
            features = x[selected]
            if not _supported(features):
                continue
            candidates += 1
            idx = indices[selected]

            def ownership(k, s, bg, slot_index):
                logits = torch.cat((10 * k.float() @ s.float().T, bg.float()), -1)
                return logits.softmax(-1)[:, slot_index]

            # Discard candidate-by-slot logits in the backward graph.
            args = (keys[event, idx], slots[event], background[event, idx], slot)
            with torch.autocast(device_type=keys.device.type, enabled=False):
                if torch.is_grad_enabled() and any(a.requires_grad for a in args[:3]):
                    weights = checkpoint(ownership, *args, use_reentrant=True)
                else:
                    weights = ownership(*args)
                xy = features[:, 6:8].double() * 0.001
                centre, radius, valid = weighted_circle(xy, weights)
                if bool(valid):
                    inverse_pt = 1 / (0.299792458 * magnetic_field * radius)
                    truth_inverse_pt = (-target[2].double()).exp()
                    # Relative curvature error with a floor above 50 GeV. Avoid
                    # conversion to pT, whose derivative diverges for straight tracks.
                    error = (inverse_pt - truth_inverse_pt) / truth_inverse_pt.clamp_min(0.02)
                    curvature.append(F.smooth_l1_loss(error, torch.zeros_like(error), beta=0.2))
                truth_mask = batch.ytarget[event, indices, pn_index] == target[pn_index]
                truth_features = x[truth_mask]
                if not _supported(truth_features):
                    continue
                with torch.no_grad():
                    truth_xy = truth_features[:, 6:8].double() * 0.001
                    truth_centre, truth_radius, truth_valid = weighted_circle(truth_xy, torch.ones(len(truth_xy), device=xy.device))
                    truth_residual = ((truth_xy - truth_centre).norm(dim=-1) - truth_radius).abs()
                    # Reject bad truth circles (interactions/looping/mixed trajectories).
                    truth_valid &= truth_residual.square().mean().sqrt() < 0.001
                if bool(truth_valid):
                    residual = ((xy - truth_centre).norm(dim=-1) - truth_radius) / 0.001
                    per_hit = F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=1.0, reduction="none")
                    geometry.append((weights.double() * per_hit).sum() / weights.double().sum().clamp_min(1e-12))
    return {
        "TrackerGeometry": torch.stack(geometry).mean().float() if geometry else zero,
        "TrackerCurvature": torch.stack(curvature).mean().float() if curvature else zero,
        "TrackerGeometryCount": zero.detach().new_tensor(len(geometry)),
        "TrackerCurvatureCount": zero.detach().new_tensor(len(curvature)),
        "TrackerCandidateCount": zero.detach().new_tensor(candidates),
    }
