import math
from dataclasses import dataclass

import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

from mlpf.logger import _logger
from mlpf.model.losses import LOSS_TASKS, REGRESSION_FEATURES


@dataclass(frozen=True)
class SetMatcherWeights:
    presence: float = 1.0
    pid: float = 1.0
    geometry: float = 1.0
    pt: float = 1.0
    energy: float = 0.0
    dr_scale: float = 0.1
    log_pt_scale: float = math.log(2.0)
    log_energy_scale: float = math.log(2.0)


def _pairwise_matching_cost(target, prediction, weights):
    """Return the [num_slots, num_targets] detached matching cost."""

    target_cls = target["cls_id"].long()
    presence_cost = -F.log_softmax(prediction["cls_binary"].float(), dim=-1)[:, 1:2]
    pid_cost = -F.log_softmax(prediction["cls_id_onehot"].float(), dim=-1)[:, target_cls]

    pred_phi = torch.atan2(prediction["sin_phi"].float(), prediction["cos_phi"].float())
    target_phi = torch.atan2(target["sin_phi"].float(), target["cos_phi"].float())
    delta_phi = pred_phi[:, None] - target_phi[None, :]
    delta_phi = torch.atan2(torch.sin(delta_phi), torch.cos(delta_phi))
    delta_eta = prediction["eta"].float()[:, None] - target["eta"].float()[None, :]
    delta_r = torch.sqrt(delta_eta.square() + delta_phi.square() + 1e-12)

    log_pt_cost = torch.abs(prediction["pt"].float()[:, None] - target["pt"].float()[None, :])
    log_energy_cost = torch.abs(prediction["energy"].float()[:, None] - target["energy"].float()[None, :])

    return (
        weights.presence * presence_cost
        + weights.pid * pid_cost
        + weights.geometry * delta_r / weights.dr_scale
        + weights.pt * log_pt_cost / weights.log_pt_scale
        + weights.energy * log_energy_cost / weights.log_energy_scale
    ).detach()


def hungarian_match(targets, predictions, target_mask, weights=None):
    """Match particle slots to targets independently for each event."""

    weights = weights or SetMatcherWeights()
    matches = []
    num_slots = predictions["cls_binary"].shape[1]
    for event_idx in range(predictions["cls_binary"].shape[0]):
        valid = target_mask[event_idx].bool()
        num_targets = int(valid.sum().item())
        if num_targets > num_slots:
            raise ValueError(f"Event {event_idx} has {num_targets} targets but the decoder has only {num_slots} slots")
        if num_targets == 0:
            empty = torch.empty(0, dtype=torch.long, device=predictions["cls_binary"].device)
            matches.append((empty, empty))
            continue

        event_targets = {key: value[event_idx][valid] for key, value in targets.items()}
        event_predictions = {key: value[event_idx] for key, value in predictions.items()}
        cost = _pairwise_matching_cost(event_targets, event_predictions, weights)
        slot_indices, target_indices = linear_sum_assignment(cost.float().cpu().numpy())
        matches.append(
            (
                torch.as_tensor(slot_indices, dtype=torch.long, device=cost.device),
                torch.as_tensor(target_indices, dtype=torch.long, device=cost.device),
            )
        )
    return matches


def set_event_loss(
    targets,
    predictions,
    target_mask,
    regression_weights,
    matcher_weights=None,
    no_object_weight=1.0,
    cardinality_loss_weight=0.0,
    matches=None,
):
    """Permutation-invariant particle-set loss for a padded event batch."""

    if matches is None:
        matches = hungarian_match(targets, predictions, target_mask, matcher_weights)
    device = predictions["cls_binary"].device
    presence_targets = torch.zeros(predictions["cls_binary"].shape[:2], dtype=torch.long, device=device)

    matched_predictions = {key: [] for key in ("cls_id_onehot", *REGRESSION_FEATURES)}
    matched_targets = {key: [] for key in ("cls_id", *REGRESSION_FEATURES)}
    for event_idx, (slot_indices, target_indices) in enumerate(matches):
        if len(slot_indices) == 0:
            continue
        presence_targets[event_idx, slot_indices] = 1
        valid_targets = target_mask[event_idx].bool()
        for key in matched_predictions:
            matched_predictions[key].append(predictions[key][event_idx, slot_indices])
        for key in matched_targets:
            matched_targets[key].append(targets[key][event_idx, valid_targets][target_indices])

    presence_class_weights = predictions["cls_binary"].new_tensor([no_object_weight, 1.0])
    losses = {
        "Classification_binary": F.cross_entropy(
            predictions["cls_binary"].reshape(-1, 2),
            presence_targets.reshape(-1),
            weight=presence_class_weights,
        )
    }
    if cardinality_loss_weight > 0:
        predicted_count = F.softmax(predictions["cls_binary"].float(), dim=-1)[..., 1].sum(dim=1)
        target_count = target_mask.sum(dim=1).to(dtype=predicted_count.dtype)
        losses["Cardinality"] = cardinality_loss_weight * F.smooth_l1_loss(predicted_count, target_count)

    num_matched = int(presence_targets.sum().item())
    if num_matched == 0:
        # Keep every set-output head in the autograd graph even for a batch with
        # no target particles. This produces zero gradients for PID and momentum
        # rather than making their parameters unused under DDP.
        output_keys = ("cls_binary", "cls_id_onehot", *REGRESSION_FEATURES)
        zero = sum(predictions[key].sum() * 0.0 for key in output_keys)
        losses["Classification"] = zero
        for feature in REGRESSION_FEATURES:
            losses[f"Regression_{feature}"] = zero
        return losses, matches

    matched_predictions = {key: torch.cat(value, dim=0) for key, value in matched_predictions.items()}
    matched_targets = {key: torch.cat(value, dim=0) for key, value in matched_targets.items()}
    losses["Classification"] = F.cross_entropy(matched_predictions["cls_id_onehot"], matched_targets["cls_id"])

    sqrt_target_pt = torch.sqrt(torch.exp(matched_targets["pt"].float()).clamp_min(1e-6))
    for feature in REGRESSION_FEATURES:
        prediction = torch.nan_to_num(matched_predictions[feature].float())
        per_particle = regression_weights[feature] * F.mse_loss(prediction, matched_targets[feature].float(), reduction="none")
        losses[f"Regression_{feature}"] = (per_particle * sqrt_target_pt).sum() / num_matched
    return losses, matches


def set_mlpf_loss(
    targets,
    predictions,
    batch,
    regression_weights,
    task_loss_weighter=None,
    matcher_weights=None,
    no_object_weight=1.0,
    cardinality_loss_weight=0.0,
    auxiliary_predictions=None,
    auxiliary_loss_weight=0.0,
):
    """Compute the set-prediction objective with the standard task names."""

    if batch.target_mask is None:
        raise ValueError("Set prediction requires batch.ytarget_set and batch.target_mask")

    effective_regression_weights = regression_weights if task_loss_weighter is None else {feature: 1.0 for feature in REGRESSION_FEATURES}
    losses, matches = set_event_loss(
        targets,
        predictions,
        batch.target_mask,
        effective_regression_weights,
        matcher_weights=matcher_weights,
        no_object_weight=no_object_weight,
        cardinality_loss_weight=cardinality_loss_weight,
    )
    task_losses = {task: losses[task] for task in LOSS_TASKS}
    if task_loss_weighter is None:
        loss_opt = sum(task_losses.values())
        diagnostics = None
    else:
        # Keep the same task names so the existing one-time calibration can be
        # evaluated for set mode rather than introducing a second mechanism.
        loss_opt, diagnostics = task_loss_weighter(task_losses)

    if "Cardinality" in losses:
        loss_opt = loss_opt + losses["Cardinality"]

    if auxiliary_predictions and auxiliary_loss_weight > 0:
        auxiliary_losses = []
        for auxiliary_prediction in auxiliary_predictions:
            layer_losses, _ = set_event_loss(
                targets,
                auxiliary_prediction,
                batch.target_mask,
                effective_regression_weights,
                matcher_weights=matcher_weights,
                no_object_weight=no_object_weight,
                cardinality_loss_weight=cardinality_loss_weight,
                matches=matches,
            )
            auxiliary_losses.append(sum(layer_losses.values()))
        losses["Auxiliary"] = auxiliary_loss_weight * torch.stack(auxiliary_losses).mean()
        loss_opt = loss_opt + losses["Auxiliary"]

    losses["Total"] = loss_opt
    if not torch.isfinite(loss_opt):
        _logger.error(predictions)
        _logger.error(losses)
        raise RuntimeError("Set-prediction loss became non-finite")

    detached_losses = {key: value.detach() for key, value in losses.items()}
    if diagnostics is not None:
        diagnostics = {name: {task: value.detach() for task, value in values.items()} for name, values in diagnostics.items()}
    return loss_opt, detached_losses, diagnostics
