"""Scheme-independent particle metrics for validation.

Matching minimizes a fixed combination of delta-R and absolute log-pT ratio. A
match is accepted for efficiency/purity when delta-R < 0.1 and relative pT error
< 0.5. Neither model's native target association or training cost is used.
"""

import math

import torch
from scipy.optimize import linear_sum_assignment

from mlpf.conf import OutputMode
from mlpf.model.utils import unpack_target


MATCH_DR = 0.1
MATCH_LOG_PT = math.log(2.0)
MATCH_REL_PT = 0.5
STRICT_DR = 0.05
STRICT_REL_PT = 0.2


def validation_particle_collections(batch, predictions, output_mode):
    """Return physical target/prediction collections with independent masks."""

    if output_mode == OutputMode.SET:
        targets = unpack_target(batch.ytarget_set.to(torch.float32), None)
        targets["pt"] = torch.exp(targets["pt"].clamp(-20.0, 20.0))
        targets["energy"] = torch.exp(targets["energy"].clamp(-20.0, 20.0))
        target_mask = batch.target_mask.bool()
        prediction_mask = predictions["cls_id"] != 0
    else:
        targets = unpack_target(batch.ytarget.to(torch.float32), None)
        if batch.ytarget_pt_orig is not None and batch.ytarget_e_orig is not None:
            targets["pt"] = batch.ytarget_pt_orig.to(torch.float32)
            targets["energy"] = batch.ytarget_e_orig.to(torch.float32)
        else:
            # Legacy/custom collaters may omit the cached absolute values. The
            # elementwise targets store log(target/input), so reconstruct them
            # without requiring the input dataset to be regenerated.
            targets["pt"] = torch.exp(targets["pt"].clamp(-20.0, 20.0)) * batch.X[..., 1].to(torch.float32)
            targets["energy"] = torch.exp(targets["energy"].clamp(-20.0, 20.0)) * batch.X[..., 5].to(torch.float32)
        target_mask = batch.mask.bool() & (targets["cls_id"] != 0)
        prediction_mask = batch.mask.bool() & (predictions["cls_id"] != 0)

    return targets, target_mask, predictions, prediction_mask


def _empty_metrics(num_classes):
    names = [
        "count/target_mean",
        "count/prediction_mean",
        "count/bias_mean",
        "count/mae",
        "matching/target_coverage_dr0p05",
        "matching/target_coverage_dr0p10",
        "matching/target_coverage_rel_pt0p20",
        "matching/target_coverage_rel_pt0p50",
        "matching/efficiency",
        "matching/purity",
        "matching/f1",
        "matching/duplicate_fraction",
        "matched/pid_accuracy",
        "matched/delta_r_mean",
        "matched/delta_eta_abs_mean",
        "matched/delta_phi_abs_mean",
        "matched/pt_relative_abs_mean",
        "matched/energy_relative_abs_mean",
        "event/energy_response_mean",
        "event/energy_relative_abs_error",
        "event/scalar_pt_response_mean",
        "event/scalar_pt_relative_abs_error",
        "event/vector_pt_closure",
        "event/met_abs_error",
    ]
    for class_id in range(1, num_classes):
        names.extend(
            [
                f"class_{class_id}/efficiency",
                f"class_{class_id}/pid_accuracy",
            ]
        )
    return {name: [0.0, 0.0] for name in names}


def _add(metrics, name, total, count):
    metrics[name][0] += float(total)
    metrics[name][1] += float(count)


def _clean_kinematics(collection, mask):
    selected = {}
    limits = {
        "pt": (0.0, 1.0e6),
        "eta": (-10.0, 10.0),
        "energy": (0.0, 1.0e6),
        "sin_phi": (-1.0, 1.0),
        "cos_phi": (-1.0, 1.0),
    }
    for name, (minimum, maximum) in limits.items():
        value = collection[name][mask].detach().to(device="cpu", dtype=torch.float32)
        selected[name] = torch.nan_to_num(value, nan=0.0, posinf=maximum, neginf=minimum).clamp(minimum, maximum)
    selected["cls_id"] = collection["cls_id"][mask].detach().to(device="cpu", dtype=torch.long)
    selected["phi"] = torch.atan2(selected["sin_phi"], selected["cos_phi"])
    return selected


def _pairwise_geometry(targets, predictions):
    delta_eta = predictions["eta"][:, None] - targets["eta"][None, :]
    delta_phi = predictions["phi"][:, None] - targets["phi"][None, :]
    delta_phi = torch.remainder(delta_phi + math.pi, 2.0 * math.pi) - math.pi
    delta_r = torch.sqrt(delta_eta.square() + delta_phi.square())
    log_pt_ratio = torch.abs(torch.log(predictions["pt"].clamp_min(1.0e-8))[:, None] - torch.log(targets["pt"].clamp_min(1.0e-8))[None, :])
    relative_pt = torch.abs(predictions["pt"][:, None] - targets["pt"][None, :]) / targets["pt"].clamp_min(1.0e-8)[None, :]
    return delta_eta, delta_phi, delta_r, log_pt_ratio, relative_pt


def _accumulate_event_metrics(metrics, targets, predictions, num_classes):
    num_targets = len(targets["pt"])
    num_predictions = len(predictions["pt"])
    _add(metrics, "count/target_mean", num_targets, 1)
    _add(metrics, "count/prediction_mean", num_predictions, 1)
    _add(metrics, "count/bias_mean", num_predictions - num_targets, 1)
    _add(metrics, "count/mae", abs(num_predictions - num_targets), 1)

    matched_prediction_indices = torch.empty(0, dtype=torch.long)
    matched_target_indices = torch.empty(0, dtype=torch.long)
    delta_eta = delta_phi = delta_r = log_pt_ratio = pairwise_relative_pt = None
    if num_targets and num_predictions:
        delta_eta, delta_phi, delta_r, log_pt_ratio, pairwise_relative_pt = _pairwise_geometry(targets, predictions)
        cost = (delta_r / MATCH_DR).square() + (log_pt_ratio / MATCH_LOG_PT).square()
        prediction_indices, target_indices = linear_sum_assignment(cost.numpy())
        matched_prediction_indices = torch.as_tensor(prediction_indices, dtype=torch.long)
        matched_target_indices = torch.as_tensor(target_indices, dtype=torch.long)

    matched_dr = delta_r[matched_prediction_indices, matched_target_indices] if delta_r is not None else torch.empty(0)
    relative_pt = pairwise_relative_pt[matched_prediction_indices, matched_target_indices] if pairwise_relative_pt is not None else torch.empty(0)
    accepted = (matched_dr < MATCH_DR) & (relative_pt < MATCH_REL_PT)
    num_accepted = int(accepted.sum())

    _add(
        metrics,
        "matching/target_coverage_dr0p05",
        int((matched_dr < STRICT_DR).sum()),
        num_targets,
    )
    _add(
        metrics,
        "matching/target_coverage_dr0p10",
        int((matched_dr < MATCH_DR).sum()),
        num_targets,
    )

    _add(
        metrics,
        "matching/target_coverage_rel_pt0p20",
        int((relative_pt < STRICT_REL_PT).sum()),
        num_targets,
    )
    _add(
        metrics,
        "matching/target_coverage_rel_pt0p50",
        int((relative_pt < MATCH_REL_PT).sum()),
        num_targets,
    )
    _add(metrics, "matching/efficiency", num_accepted, num_targets)
    _add(metrics, "matching/purity", num_accepted, num_predictions)
    _add(metrics, "matching/f1", 2 * num_accepted, num_targets + num_predictions)

    if delta_r is not None:
        close_to_any_target = ((delta_r < MATCH_DR) & (pairwise_relative_pt < MATCH_REL_PT)).any(dim=1)
        num_duplicates = max(int(close_to_any_target.sum()) - num_accepted, 0)
    else:
        num_duplicates = 0
    _add(metrics, "matching/duplicate_fraction", num_duplicates, num_predictions)

    accepted_prediction_indices = matched_prediction_indices[accepted]
    accepted_target_indices = matched_target_indices[accepted]
    if num_accepted:
        accepted_target_pt = targets["pt"][accepted_target_indices]
        accepted_prediction_pt = predictions["pt"][accepted_prediction_indices]
        accepted_target_energy = targets["energy"][accepted_target_indices]
        accepted_prediction_energy = predictions["energy"][accepted_prediction_indices]
        accepted_relative_pt = torch.abs(accepted_prediction_pt - accepted_target_pt) / accepted_target_pt.clamp_min(1.0e-8)
        accepted_relative_energy = torch.abs(accepted_prediction_energy - accepted_target_energy) / accepted_target_energy.clamp_min(1.0e-8)
        pid_correct = predictions["cls_id"][accepted_prediction_indices] == targets["cls_id"][accepted_target_indices]

        _add(metrics, "matched/pid_accuracy", pid_correct.sum(), num_accepted)
        _add(metrics, "matched/delta_r_mean", matched_dr[accepted].sum(), num_accepted)
        _add(
            metrics,
            "matched/delta_eta_abs_mean",
            delta_eta[matched_prediction_indices, matched_target_indices][accepted].abs().sum(),
            num_accepted,
        )
        _add(
            metrics,
            "matched/delta_phi_abs_mean",
            delta_phi[matched_prediction_indices, matched_target_indices][accepted].abs().sum(),
            num_accepted,
        )
        _add(
            metrics,
            "matched/pt_relative_abs_mean",
            accepted_relative_pt.sum(),
            num_accepted,
        )
        _add(
            metrics,
            "matched/energy_relative_abs_mean",
            accepted_relative_energy.sum(),
            num_accepted,
        )
    else:
        pid_correct = torch.empty(0, dtype=torch.bool)

    for class_id in range(1, num_classes):
        class_target_count = int((targets["cls_id"] == class_id).sum())
        if num_accepted:
            accepted_in_class = targets["cls_id"][accepted_target_indices] == class_id
            class_accepted_count = int(accepted_in_class.sum())
            class_pid_correct = int((accepted_in_class & pid_correct).sum())
        else:
            class_accepted_count = 0
            class_pid_correct = 0
        _add(
            metrics,
            f"class_{class_id}/efficiency",
            class_accepted_count,
            class_target_count,
        )
        _add(
            metrics,
            f"class_{class_id}/pid_accuracy",
            class_pid_correct,
            class_accepted_count,
        )

    target_energy = targets["energy"].to(torch.float64).sum()
    prediction_energy = predictions["energy"].to(torch.float64).sum()
    target_scalar_pt = targets["pt"].to(torch.float64).sum()
    prediction_scalar_pt = predictions["pt"].to(torch.float64).sum()
    energy_denominator = target_energy.clamp_min(1.0e-8)
    pt_denominator = target_scalar_pt.clamp_min(1.0e-8)
    energy_residual = (prediction_energy - target_energy) / energy_denominator
    scalar_pt_residual = (prediction_scalar_pt - target_scalar_pt) / pt_denominator
    _add(metrics, "event/energy_response_mean", prediction_energy / energy_denominator, 1)
    _add(metrics, "event/energy_relative_abs_error", energy_residual.abs(), 1)
    _add(
        metrics,
        "event/scalar_pt_response_mean",
        prediction_scalar_pt / pt_denominator,
        1,
    )
    _add(metrics, "event/scalar_pt_relative_abs_error", scalar_pt_residual.abs(), 1)

    target_px = (targets["pt"].to(torch.float64) * torch.cos(targets["phi"].to(torch.float64))).sum()
    target_py = (targets["pt"].to(torch.float64) * torch.sin(targets["phi"].to(torch.float64))).sum()
    prediction_px = (predictions["pt"].to(torch.float64) * torch.cos(predictions["phi"].to(torch.float64))).sum()
    prediction_py = (predictions["pt"].to(torch.float64) * torch.sin(predictions["phi"].to(torch.float64))).sum()
    vector_pt_error = torch.hypot(prediction_px - target_px, prediction_py - target_py)
    target_met = torch.hypot(target_px, target_py)
    prediction_met = torch.hypot(prediction_px, prediction_py)
    _add(metrics, "event/vector_pt_closure", vector_pt_error / pt_denominator, 1)
    _add(metrics, "event/met_abs_error", torch.abs(prediction_met - target_met), 1)


def compute_validation_particle_metrics(targets, target_mask, predictions, prediction_mask, num_classes):
    """Compute additive, scheme-independent particle metrics for one batch."""

    metrics = _empty_metrics(num_classes)
    for event_idx in range(target_mask.shape[0]):
        event_targets = _clean_kinematics(
            {name: value[event_idx] for name, value in targets.items()},
            target_mask[event_idx],
        )
        event_predictions = _clean_kinematics(
            {name: value[event_idx] for name, value in predictions.items()},
            prediction_mask[event_idx],
        )
        _accumulate_event_metrics(metrics, event_targets, event_predictions, num_classes)
    return {name: tuple(values) for name, values in metrics.items()}
