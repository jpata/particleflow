"""Robust transverse-circle summaries of predicted tracker groups.

Input positions are mm. The detached fit provides a pT anchor for reliable
multi-surface groups; the learned decoder retains a residual correction.
"""

import torch

SUMMARY_DIM = 9


@torch.no_grad()
def tracker_group_geometry(features, groups, num_slots, magnetic_field):
    device = features.device
    summary = torch.zeros(num_slots, SUMMARY_DIM, device=device, dtype=torch.float32)
    log_pt = torch.zeros(num_slots, device=device, dtype=torch.float32)
    valid_fit = torch.zeros(num_slots, device=device, dtype=torch.bool)
    if len(features) == 0:
        return summary, log_pt, valid_fit
    # Double precision matters for the very small sagitta of high-pT tracks.
    xyz = features[:, 6:9].double() * 0.001
    finite = xyz.isfinite().all(-1)
    xyz, groups, features = xyz[finite], groups[finite], features[finite]
    if len(xyz) == 0:
        return summary, log_pt, valid_fit

    def sum_by_group(value):
        result = torch.zeros((num_slots, *value.shape[1:]), device=device, dtype=value.dtype)
        return result.index_add_(0, groups, value)

    count = sum_by_group(torch.ones(len(xyz), device=device, dtype=torch.float64))
    mean = sum_by_group(xyz[:, :2]) / count.clamp_min(1)[:, None]
    centered = xyz[:, :2] - mean[groups]
    scale = (sum_by_group(centered.square().sum(-1)) / count.clamp_min(1)).sqrt().clamp_min(1e-6)
    xy = centered / scale[groups, None]
    design = torch.cat((2 * xy, torch.ones_like(xy[:, :1])), dim=-1)
    rhs = xy.square().sum(-1)
    weights = torch.ones_like(rhs)
    eye = torch.eye(3, device=device, dtype=torch.float64)
    for _ in range(3):
        gram = sum_by_group(weights[:, None, None] * design[:, :, None] * design[:, None, :])
        vector = sum_by_group(weights[:, None] * design * rhs[:, None])
        parameters = torch.linalg.solve(gram + 1e-10 * count.clamp_min(1)[:, None, None] * eye, vector)
        radius_norm = (parameters[:, :2].square().sum(-1) + parameters[:, 2]).clamp_min(0).sqrt()
        residual = ((xy - parameters[groups, :2]).norm(dim=-1) - radius_norm[groups]) * scale[groups]
        rms = (sum_by_group(residual.square()) / count.clamp_min(1)).sqrt()
        # Huber IRLS weights; detach the geometric estimator from model gradients.
        cutoff = (2 * rms[groups]).clamp_min(1e-5)
        weights = (cutoff / residual.abs().clamp_min(1e-12)).clamp_max(1)
    radius = radius_norm * scale
    pt = 0.299792458 * magnetic_field * radius  # GeV, tesla, metres
    rho = xyz[:, :2].norm(dim=-1)

    def span(value):
        lower = torch.full((num_slots,), float("inf"), device=device, dtype=torch.float64)
        upper = torch.full_like(lower, -float("inf"))
        lower.scatter_reduce_(0, groups, value, reduce="amin", include_self=True)
        upper.scatter_reduce_(0, groups, value, reduce="amax", include_self=True)
        return torch.where(count > 0, upper - lower, 0)

    rho_span, z_span = span(rho), span(xyz[:, 2])
    surfaces = torch.unique(torch.cat((groups[:, None].double(), features[:, 12:15].double()), dim=1), dim=0)
    surface_count = torch.bincount(surfaces[:, 0].long(), minlength=num_slots)
    eigenvalues = torch.linalg.eigvalsh(gram)
    condition = eigenvalues[:, 0].clamp_min(0) / eigenvalues[:, -1].clamp_min(1e-12)
    relative_residual = rms / rho_span.clamp_min(1e-6)
    valid_fit = (count >= 4) & (surface_count >= 3) & (rho_span >= 0.1) & (condition > 1e-7)
    valid_fit &= (relative_residual < 0.02) & pt.isfinite() & (pt > 0.1) & (pt < 1e4)
    log_pt = torch.where(valid_fit, pt.clamp_min(1e-8).log(), 0).float()
    summary = torch.stack(
        (
            count.log1p(),
            surface_count.double().log1p(),
            rho_span,
            z_span,
            scale,
            log_pt.double(),
            relative_residual.clamp_max(10),
            condition.clamp_min(1e-12).log10(),
            valid_fit.double(),
        ),
        dim=-1,
    )
    summary = torch.where((count > 0)[:, None], summary, 0)
    return torch.nan_to_num(summary).float(), log_pt, valid_fit
