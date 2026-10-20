import math
from dataclasses import dataclass

import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from mlpf.conf import Y_FEATURES
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


def _nonfinite_field_summary(values, keys):
    failures = []
    for key in keys:
        tensor = values[key].detach().float().cpu()
        bad = ~torch.isfinite(tensor)
        if bad.any():
            first = tuple(bad.nonzero()[0].tolist())
            failures.append(f"{key}: count={int(bad.sum())}, first_index={first}, value={tensor[first].item()}")
    return "; ".join(failures) if failures else "none"


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
        cost_cpu = cost.float().cpu()
        bad_cost = ~torch.isfinite(cost_cpu)
        if bad_cost.any():
            slot_idx, target_idx = bad_cost.nonzero()[0].tolist()
            target_row = valid.nonzero()[target_idx].item()
            prediction_fields = _nonfinite_field_summary(
                event_predictions, ("cls_binary", "cls_id_onehot", "eta", "sin_phi", "cos_phi", "pt", "energy")
            )
            target_fields = _nonfinite_field_summary(event_targets, ("eta", "sin_phi", "cos_phi", "pt", "energy"))
            raise ValueError(
                f"Non-finite Hungarian matching cost in event {event_idx}: "
                f"{int(bad_cost.sum())} invalid entries; first at slot {slot_idx}, "
                f"target {target_idx} (padded row {target_row}), value={cost_cpu[slot_idx, target_idx].item()}; "
                f"prediction fields: {prediction_fields}; target fields: {target_fields}"
            )
        slot_indices, target_indices = linear_sum_assignment(cost_cpu.numpy())
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
    momentum_loss="mse",
    momentum_huber_delta=0.2,
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
        target = matched_targets[feature].float()
        if feature in ("pt", "energy") and momentum_loss == "huber":
            # Match MSE's quadratic scale for small residuals while limiting
            # the influence of large log-momentum residuals.
            error = 2.0 * F.huber_loss(prediction, target, reduction="none", delta=momentum_huber_delta)
        else:
            error = F.mse_loss(prediction, target, reduction="none")
        per_particle = regression_weights[feature] * error
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
    momentum_loss="mse",
    momentum_huber_delta=0.2,
    hit_grouping=None,
    hit_grouping_loss_weight=0.0,
    hit_grouping_chunk_size=1024,
    hit_grouping_background_weight=0.1,
    hit_grouping_detector_balanced=False,
    hit_grouping_tracker_particle_balanced=False,
    tracker_overlap_loss_weight=0.0,
    tracker_geometry_loss_weight=0.0,
    tracker_curvature_loss_weight=0.0,
    tracker_loss_max_hits=64,
    tracker_group_magnetic_field=2.0,
    event_consistency_loss_weight=0.0,
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
        momentum_loss=momentum_loss,
        momentum_huber_delta=momentum_huber_delta,
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

    if hit_grouping is not None:
        grouping_loss = hit_grouping_loss(
            hit_grouping, batch, matches, chunk_size=hit_grouping_chunk_size,
            background_weight=hit_grouping_background_weight,
            detector_balanced=hit_grouping_detector_balanced,
            tracker_particle_balanced=hit_grouping_tracker_particle_balanced,
        )
        losses["HitGrouping"] = hit_grouping_loss_weight * grouping_loss
        loss_opt = loss_opt + losses["HitGrouping"]

    if tracker_overlap_loss_weight > 0:
        if hit_grouping is None:
            raise ValueError("tracker overlap loss requires learned hit grouping")
        overlap = tracker_overlap_loss(hit_grouping, batch, matches, chunk_size=hit_grouping_chunk_size)
        losses["TrackerOverlap"] = tracker_overlap_loss_weight * overlap["loss"]
        loss_opt = loss_opt + losses["TrackerOverlap"]
        losses["TrackerOverlapCount"] = overlap["count"]
        losses["TrackerSoftPurity"] = overlap["purity"]
        losses["TrackerSoftCompleteness"] = overlap["completeness"]

    if tracker_geometry_loss_weight > 0 or tracker_curvature_loss_weight > 0:
        from mlpf.model.tracker_fit_loss import tracker_fit_losses

        if hit_grouping is None:
            raise ValueError("Tracker fit supervision requires learned hit grouping")
        fit_losses = tracker_fit_losses(
            hit_grouping, batch, matches, magnetic_field=tracker_group_magnetic_field,
            max_hits=tracker_loss_max_hits, chunk_size=hit_grouping_chunk_size,
        )
        for name, weight in (("TrackerGeometry", tracker_geometry_loss_weight), ("TrackerCurvature", tracker_curvature_loss_weight)):
            losses[name] = weight * fit_losses[name]
            loss_opt = loss_opt + losses[name]
        for name in ("TrackerGeometryCount", "TrackerCurvatureCount", "TrackerCandidateCount"):
            losses[name] = fit_losses[name]

    if event_consistency_loss_weight > 0:
        losses["EventConsistency"] = event_consistency_loss_weight * event_consistency_loss(targets, predictions, batch.target_mask)
        loss_opt = loss_opt + losses["EventConsistency"]

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
                momentum_loss=momentum_loss,
                momentum_huber_delta=momentum_huber_delta,
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


def _grouping_targets(batch, event_idx, match, num_slots):
    """Full-hit labels and charged truth slots, independent of model selection."""
    slot_indices, target_indices = match
    pn_index = Y_FEATURES.index("particle_number")
    targets = batch.ytarget_set[event_idx, batch.target_mask[event_idx]]
    matched_pn = targets[target_indices, pn_index].long()
    sorted_pn, order = matched_pn.sort()
    sorted_slots = slot_indices[order]
    hit_pn = batch.ytarget[event_idx, batch.mask[event_idx].bool(), pn_index].long()
    labels = torch.full_like(hit_pn, num_slots)
    if sorted_pn.numel():
        position = torch.searchsorted(sorted_pn.contiguous(), hit_pn.contiguous()).clamp_max(sorted_pn.numel() - 1)
        owned = (hit_pn > 0) & (sorted_pn[position] == hit_pn)
        labels[owned] = sorted_slots[position[owned]]
    cls = targets[target_indices, 0].long()
    charged = torch.zeros(num_slots, dtype=torch.bool, device=labels.device)
    charged[slot_indices] = (cls == 1) | (cls == 4) | (cls == 5)
    return labels, charged


def hit_grouping_loss(grouping, batch, matches, chunk_size=1024, background_weight=0.1, detector_balanced=False,
                      tracker_particle_balanced=False):
    """Supervise event-local hit ownership without a dense hit-pair matrix."""
    hit_keys, slot_keys, background = grouping
    num_slots = slot_keys.shape[1]
    pn_index = Y_FEATURES.index("particle_number")
    total = hit_keys.sum() * 0.0 + slot_keys.sum() * 0.0 + background.sum() * 0.0
    weight_sum = hit_keys.new_zeros(())
    tracker_counts = []
    tracker_total = hit_keys.new_zeros((), dtype=torch.float32)
    track_total = hit_keys.new_zeros((), dtype=torch.float32)
    if tracker_particle_balanced:
        for event_idx, match in enumerate(matches):
            labels, charged = _grouping_targets(batch, event_idx, match, num_slots)
            detector = batch.X[event_idx, batch.mask[event_idx].bool(), 0]
            selected = (detector == 1) & (labels < num_slots) & charged[labels.clamp_max(num_slots - 1)]
            counts = torch.bincount(labels[selected], minlength=num_slots)
            tracker_counts.append((counts, selected))
            tracker_total += selected.sum()
            track_total += (counts > 0).sum()
    detector_denominators = None
    if detector_balanced:
        # Compute denominators over the batch before chunking. Include background
        # with its usual weight, and average only detector types present here.
        detector_denominators = hit_keys.new_zeros(2)
        for event_idx, (_, target_indices) in enumerate(matches):
            mask = batch.mask[event_idx].bool()
            pn = batch.ytarget[event_idx, mask, pn_index].long()
            target_pn = batch.ytarget_set[event_idx, batch.target_mask[event_idx], pn_index].long()[target_indices]
            owned = (pn > 0) & torch.isin(pn, target_pn)
            weights = torch.where(owned, 1.0, background_weight)
            detector = batch.X[event_idx, mask, 0]
            for index, kind in enumerate((1, 2)):
                detector_denominators[index] += weights[detector == kind].sum()
    for event_idx, (slot_indices, target_indices) in enumerate(matches):
        hit_mask = batch.mask[event_idx].bool()
        labels, _ = _grouping_targets(batch, event_idx, (slot_indices, target_indices), num_slots)
        weight_sum = weight_sum + torch.where(labels == num_slots, background_weight, 1.0).sum()
        event_hit_keys = hit_keys[event_idx, hit_mask]
        event_background = background[event_idx, hit_mask]
        event_slot_keys = slot_keys[event_idx]
        hit_weights = torch.where(labels == num_slots, background_weight, 1.0)
        if tracker_particle_balanced:
            counts, selected = tracker_counts[event_idx]
            # Redistribute the original charged-tracker foreground weight, keeping
            # its total unchanged. Calorimeter/background weights stay unchanged.
            per_track = tracker_total / track_total.clamp_min(1)
            hit_weights[selected] = per_track / counts[labels[selected]].clamp_min(1)
        if detector_denominators is not None:
            detector = batch.X[event_idx, hit_mask, 0]
            normalization = torch.zeros_like(hit_weights)
            for index, kind in enumerate((1, 2)):
                normalization[detector == kind] = detector_denominators[index].clamp_min(1e-8).reciprocal()
            hit_weights = hit_weights * normalization / (detector_denominators > 0).sum().clamp_min(1)
        for keys_chunk, background_chunk, labels_chunk, weights_chunk in zip(
            event_hit_keys.split(chunk_size), event_background.split(chunk_size), labels.split(chunk_size), hit_weights.split(chunk_size)
        ):

            def chunk_loss(keys, slots, bg, targets, weights):
                scores = torch.cat([10.0 * (keys @ slots.T), bg], dim=-1)
                per_hit = F.cross_entropy(scores, targets, reduction="none")
                return (per_hit * weights).sum()

            if torch.is_grad_enabled() and (keys_chunk.requires_grad or event_slot_keys.requires_grad or background_chunk.requires_grad):
                # Reentrant checkpointing discards the chunk's CE activations;
                # non-reentrant checkpointing retains them and grows as hits * slots.
                part = checkpoint(
                    chunk_loss,
                    keys_chunk,
                    event_slot_keys,
                    background_chunk,
                    labels_chunk,
                    weights_chunk,
                    use_reentrant=True,
                )
            else:
                part = chunk_loss(keys_chunk, event_slot_keys, background_chunk, labels_chunk, weights_chunk)
            total = total + part
    return total if detector_balanced else total / weight_sum.clamp_min(1.0)


def tracker_overlap_loss(grouping, batch, matches, chunk_size=1024):
    """Mean charged-track soft Dice over full slot-plus-background probabilities.

    All valid tracker hits contribute to predicted mass, including background and
    hits belonging to other particles. Eligibility depends only on truth support.
    Chunk checkpointing retains O(chunks * slots) reductions, not O(hits * slots)
    softmax activations. Reductions and probabilities remain FP32 under autocast.
    """
    hit_keys, slot_keys, background = grouping
    num_slots = slot_keys.shape[1]
    with torch.autocast(device_type=hit_keys.device.type, enabled=False):
        zero = (hit_keys.float().sum() + slot_keys.float().sum() + background.float().sum()) * 0
        total, purity, completeness = zero, zero, zero
        count = zero.detach().clone()
        for event_idx, match in enumerate(matches):
            labels, charged = _grouping_targets(batch, event_idx, match, num_slots)
            tracker = batch.X[event_idx, batch.mask[event_idx].bool(), 0] == 1
            truth_labels = labels[tracker]
            truth_count = torch.bincount(truth_labels, minlength=num_slots + 1)[:num_slots].float()
            eligible = charged & (truth_count > 0)
            keys = hit_keys[event_idx, batch.mask[event_idx].bool()][tracker]
            bg = background[event_idx, batch.mask[event_idx].bool()][tracker]
            tp = zero + torch.zeros(num_slots, device=hit_keys.device)
            mass = zero + torch.zeros_like(tp)

            def chunk_sums(k, s, b, lab):
                with torch.autocast(device_type=k.device.type, enabled=False):
                    probabilities = torch.cat([10.0 * (k.float() @ s.float().T), b.float()], -1).softmax(-1)[:, :num_slots]
                    owned = lab < num_slots
                    correct = probabilities[owned].gather(1, lab[owned, None]).squeeze(1)
                    true_mass = probabilities.new_zeros(num_slots).index_add(0, lab[owned], correct)
                    return true_mass, probabilities.sum(0)

            for kc, bc, lc in zip(keys.split(chunk_size), bg.split(chunk_size), truth_labels.split(chunk_size)):
                if torch.is_grad_enabled() and any(v.requires_grad for v in (kc, slot_keys, bc)):
                    tpc, mc = checkpoint(chunk_sums, kc, slot_keys[event_idx], bc, lc, use_reentrant=True)
                else:
                    tpc, mc = chunk_sums(kc, slot_keys[event_idx], bc, lc)
                tp, mass = tp + tpc, mass + mc
            dice = (2 * tp + 1e-6) / (truth_count + mass + 1e-6)
            total = total + (1 - dice[eligible]).sum()
            purity = purity + (tp / mass.clamp_min(1e-8))[eligible].sum()
            completeness = completeness + (tp / truth_count.clamp_min(1))[eligible].sum()
            count = count + eligible.sum()
        denominator = count.clamp_min(1)
        return {"loss": total / denominator, "count": count,
                "purity": purity / denominator, "completeness": completeness / denominator}


def event_consistency_loss(targets, predictions, target_mask):
    """Soft-presence visible momentum/energy closure relative to target sums.

    Each residual is divided by the true scalar momentum or energy sum (floor
    1 GeV). The event momentum need not be zero. Only final slots participate.
    """

    def physical(collection):
        pt = collection["pt"].float().clamp(-20, 20).exp()
        eta = collection["eta"].float().clamp(-10, 10)
        phi = torch.atan2(collection["sin_phi"].float(), collection["cos_phi"].float())
        p3 = torch.stack((pt * phi.cos(), pt * phi.sin(), pt * eta.sinh()), dim=-1)
        energy = collection["energy"].float().clamp(-20, 20).exp()
        return p3, energy

    truth_p, truth_e = physical(targets)
    pred_p, pred_e = physical(predictions)
    mask = target_mask.float()
    presence = predictions["cls_binary"].float().softmax(-1)[..., 1]
    p_scale = (truth_p.norm(dim=-1) * mask).sum(1).clamp_min(1)
    e_scale = (truth_e * mask).sum(1).clamp_min(1)
    dp = ((pred_p * presence[..., None]).sum(1) - (truth_p * mask[..., None]).sum(1)) / p_scale[:, None]
    de = ((pred_e * presence).sum(1) - (truth_e * mask).sum(1)) / e_scale
    return F.smooth_l1_loss(dp, torch.zeros_like(dp)) + F.smooth_l1_loss(de, torch.zeros_like(de))
