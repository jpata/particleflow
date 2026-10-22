"""Additive reconstruction diagnostics; see docs/reconstruction_diagnostics.md.

Keys do not depend on batch contents, including empty classes on DDP ranks.
"""

import math

import torch

from mlpf.conf import EDM4HEP, Y_FEATURES

ETA_BINS = (("abs_eta_lt1", 0.0, 1.0), ("abs_eta_1to2", 1.0, 2.0), ("abs_eta_ge2", 2.0, math.inf))
PT_BINS = (("pt_lt1", 0.0, 1.0), ("pt_1to10", 1.0, 10.0), ("pt_ge10", 10.0, math.inf))
COVERAGE_BINS = (("surfaces_1to3", 1, 4), ("surfaces_4to7", 4, 8), ("surfaces_ge8", 8, math.inf))
DENSITY_BINS = (("neighbors_0", 0, 1), ("neighbors_1to2", 1, 3), ("neighbors_ge3", 3, math.inf))


def _add(metrics, key, total, count):
    metrics[key][0] += float(total)
    metrics[key][1] += float(count)


def _scopes(num_classes):
    return ["all", *(f"class_{i}" for i in range(1, num_classes)), *(name for name, _, _ in ETA_BINS)]


def empty_momentum_metrics(num_classes):
    names = []
    for scope in _scopes(num_classes):
        root = f"momentum/{scope}"
        names.extend(f"{root}/{name}" for name in ("target_samples", "efficiency"))
        for population in ("assigned", "accepted"):
            names.extend(
                f"{root}/{population}/{name}"
                for name in (
                    "samples",
                    "pt_relative_bias",
                    "pt_relative_abs_mean",
                    "pt_relative_residual_rms",
                    "energy_relative_abs_mean",
                    "delta_r_mean",
                    "pz_bias",
                    "pz_abs_mean",
                    "pz_residual_rms",
                    "pt_relative_gt0p5_fraction",
                    "pz_abs_gt10_fraction",
                )
            )
        names.extend(
            f"{root}/event/{name}"
            for name in (
                "samples",
                "px_bias",
                "py_bias",
                "pz_bias",
                "pz_abs_mean",
                "pz_residual_rms",
                "p3_error_mean",
                "pt_vector_error_mean",
                "energy_bias",
            )
        )
    return {name: [0.0, 0.0] for name in names}


def _scope_mask(collection, scope):
    if scope == "all":
        return torch.ones_like(collection["pt"], dtype=torch.bool)
    if scope.startswith("class_"):
        return collection["cls_id"] == int(scope.split("_")[1])
    _, lower, upper = next(item for item in ETA_BINS if item[0] == scope)
    return (collection["eta"].abs() >= lower) & (collection["eta"].abs() < upper)


def _p3(collection):
    pt, phi, eta = (collection[key].double() for key in ("pt", "phi", "eta"))
    return torch.stack((pt * phi.cos(), pt * phi.sin(), pt * eta.sinh()), dim=-1)


def accumulate_momentum_metrics(metrics, targets, predictions, pred_indices, target_indices, accepted, matched_dr, num_classes):
    """Reuse standard Hungarian pairs, including pairs failing acceptance cuts."""
    target_p3, pred_p3 = _p3(targets), _p3(predictions)
    pt_residual = (predictions["pt"][pred_indices].double() - targets["pt"][target_indices]) / targets["pt"][target_indices].clamp_min(1e-8)
    energy_residual = (predictions["energy"][pred_indices].double() - targets["energy"][target_indices]) / targets["energy"][
        target_indices
    ].clamp_min(1e-8)
    pz_residual = pred_p3[pred_indices, 2] - target_p3[target_indices, 2]
    for scope in _scopes(num_classes):
        root = f"momentum/{scope}"
        truth_mask, pred_mask = _scope_mask(targets, scope), _scope_mask(predictions, scope)
        assigned_mask = truth_mask[target_indices]
        _add(metrics, f"{root}/target_samples", truth_mask.sum(), 1)
        _add(metrics, f"{root}/efficiency", (assigned_mask & accepted).sum(), truth_mask.sum())
        for population, mask in (("assigned", assigned_mask), ("accepted", assigned_mask & accepted)):
            prefix = f"{root}/{population}"
            n = mask.sum()
            _add(metrics, f"{prefix}/samples", n, 1)
            values = {
                "pt_relative_bias": pt_residual,
                "pt_relative_abs_mean": pt_residual.abs(),
                "pt_relative_residual_rms": pt_residual.square(),
                "energy_relative_abs_mean": energy_residual.abs(),
                "delta_r_mean": matched_dr,
                "pz_bias": pz_residual,
                "pz_abs_mean": pz_residual.abs(),
                "pz_residual_rms": pz_residual.square(),
                "pt_relative_gt0p5_fraction": pt_residual.abs() > 0.5,
                "pz_abs_gt10_fraction": pz_residual.abs() > 10.0,
            }
            for name, value in values.items():
                _add(metrics, f"{prefix}/{name}", value[mask].sum(), n)
        # Include unmatched objects. Truth/prediction use their own class and eta;
        # migrations and PID errors therefore contribute to event closure.
        residual = pred_p3[pred_mask].sum(0) - target_p3[truth_mask].sum(0)
        energy_delta = predictions["energy"][pred_mask].double().sum() - targets["energy"][truth_mask].double().sum()
        for name, value in {
            "samples": 1,
            "px_bias": residual[0],
            "py_bias": residual[1],
            "pz_bias": residual[2],
            "pz_abs_mean": residual[2].abs(),
            "pz_residual_rms": residual[2].square(),
            "p3_error_mean": residual.norm(),
            "pt_vector_error_mean": residual[:2].norm(),
            "energy_bias": energy_delta,
        }.items():
            _add(metrics, f"{root}/event/{name}", value, 1)


def _group_scopes():
    return ["all", *(name for bins in (PT_BINS, ETA_BINS, COVERAGE_BINS, DENSITY_BINS) for name, _, _ in bins)]


def _empty_grouping_metrics():
    names = []
    for detector in ("tracker", "calorimeter"):
        names.extend(
            f"{detector}/{name}"
            for name in (
                "hit_samples",
                "owned_hit_samples",
                "purity",
                "completeness",
                "background_recall",
                "background_assignment_fraction",
                "energy_purity",
                "energy_completeness",
                "merge_fraction",
            )
        )
        for scope in _group_scopes():
            names.extend(
                f"{detector}/{scope}/{name}"
                for name in (
                    "object_samples",
                    "object_purity",
                    "object_completeness",
                    "best_slot_completeness",
                    "split_fraction",
                    "merged_slot_fraction",
                    "energy_completeness",
                )
            )
    return {name: [0.0, 0.0] for name in names}


@torch.no_grad()
def compute_hit_grouping_metrics(grouping, batch, matches, chunk_size=1024):
    """Score actual all-slot hard grouping aligned by training Hungarian matches.

    Reconstruct scores in chunks; retain only winners and a slot-by-target
    contingency table, never a full hit-by-slot map.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    metrics = _empty_grouping_metrics()
    hit_keys, slot_keys, background = grouping
    slots = slot_keys.shape[1]
    pn_index = Y_FEATURES.index("particle_number")
    features = EDM4HEP.HitFeatures.get_names()
    for event, (matched_slots, matched_targets) in enumerate(matches):
        valid = batch.mask[event].bool()
        indices = valid.nonzero().flatten()
        winners = []
        for chunk in indices.split(chunk_size):
            score = 10.0 * (hit_keys[event, chunk].float() @ slot_keys[event].float().T)
            winners.append(torch.cat((score, background[event, chunk].float()), dim=-1).argmax(-1).cpu())
        winner = torch.cat(winners) if winners else torch.empty(0, dtype=torch.long)
        x = batch.X[event, valid].detach().float().cpu()
        hit_pn = batch.ytarget[event, valid, pn_index].long().cpu()
        target = batch.ytarget_set[event, batch.target_mask[event].bool()].detach().float().cpu()
        target_pn = target[:, pn_index].long()
        n = len(target)
        truth = torch.full_like(hit_pn, n)
        if n:
            sorted_pn, order = target_pn.sort()
            location = torch.searchsorted(sorted_pn.contiguous(), hit_pn.contiguous()).clamp_max(n - 1)
            owned = (hit_pn > 0) & (sorted_pn[location] == hit_pn)
            truth[owned] = order[location[owned]]
        slot_for_target = torch.full((n,), -1, dtype=torch.long)
        slot_for_target[matched_targets.cpu()] = matched_slots.cpu()
        pt = target[:, Y_FEATURES.index("pt")].clamp(-20, 20).exp()
        eta = target[:, Y_FEATURES.index("eta")]
        phi = torch.atan2(target[:, Y_FEATURES.index("sin_phi")], target[:, Y_FEATURES.index("cos_phi")])
        dphi = phi[:, None] - phi[None, :]
        dr = torch.hypot(eta[:, None] - eta[None, :], torch.atan2(dphi.sin(), dphi.cos()))
        neighbors = ((dr < 0.1) & ~torch.eye(n, dtype=torch.bool)).sum(1)
        for kind, detector in ((1, "tracker"), (2, "calorimeter")):
            mask = x[:, 0] == kind
            flat = winner[mask] * (n + 1) + truth[mask]
            table = torch.bincount(flat, minlength=(slots + 1) * (n + 1)).reshape(slots + 1, n + 1).double()
            energy = x[mask, features.index("energy")].double().clamp_min(0)
            energy_table = torch.bincount(flat, weights=energy, minlength=(slots + 1) * (n + 1)).reshape(slots + 1, n + 1)
            true_count = table[:, :n].sum(0)
            slot_count = table[:slots].sum(1)
            safe_slot = slot_for_target.clamp_min(0)
            target_index = torch.arange(n)
            overlap = table[safe_slot, target_index] * (slot_for_target >= 0)
            assigned = slot_count[safe_slot] * (slot_for_target >= 0)
            true_energy = energy_table[:, :n].sum(0)
            overlap_energy = energy_table[safe_slot, target_index] * (slot_for_target >= 0)
            # Require >=2 hits and >=10% contribution for significant fragments.
            split = ((table[:slots, :n] >= 2) & (table[:slots, :n] >= 0.1 * true_count)).sum(0) >= 2
            merge = ((table[:slots, :n] >= 2) & (table[:slots, :n] >= 0.1 * slot_count[:, None])).sum(1) >= 2
            for name, total, count in (
                ("hit_samples", table.sum(), 1),
                ("owned_hit_samples", true_count.sum(), 1),
                ("purity", overlap.sum(), slot_count.sum()),
                ("completeness", overlap.sum(), true_count.sum()),
                ("background_recall", table[slots, n], table[:, n].sum()),
                ("background_assignment_fraction", table[slots].sum(), table.sum()),
                ("merge_fraction", merge.sum(), (slot_count > 0).sum()),
            ):
                _add(metrics, f"{detector}/{name}", total, count)
            if kind == 2:
                _add(metrics, f"{detector}/energy_purity", overlap_energy.sum(), energy_table[:slots].sum())
                _add(metrics, f"{detector}/energy_completeness", overlap_energy.sum(), true_energy.sum())
            coverage = torch.zeros(n, dtype=torch.long)
            surface_columns = [features.index(key) for key in ("system", "side", "layer")]
            for target_id in range(n):
                surfaces = x[mask & (truth == target_id)][:, surface_columns]
                coverage[target_id] = len(torch.unique(surfaces, dim=0))
            scopes = {"all": true_count > 0}
            for bins, value in ((PT_BINS, pt), (ETA_BINS, eta.abs()), (COVERAGE_BINS, coverage), (DENSITY_BINS, neighbors)):
                scopes.update({name: (true_count > 0) & (value >= low) & (value < high) for name, low, high in bins})
            for scope, selected in scopes.items():
                prefix = f"{detector}/{scope}"
                count = selected.sum()
                _add(metrics, f"{prefix}/object_samples", count, 1)
                for name, value in {
                    "object_purity": overlap / assigned.clamp_min(1),
                    "object_completeness": overlap / true_count.clamp_min(1),
                    "best_slot_completeness": table[:slots, :n].max(0).values / true_count.clamp_min(1),
                    "split_fraction": split,
                    "merged_slot_fraction": merge[safe_slot] & (slot_for_target >= 0),
                }.items():
                    _add(metrics, f"{prefix}/{name}", value[selected].sum(), count)
                if kind == 2:
                    positive = selected & (true_energy > 0)
                    _add(metrics, f"{prefix}/energy_completeness", (overlap_energy[positive] / true_energy[positive]).sum(), positive.sum())
    return {key: tuple(value) for key, value in metrics.items()}
