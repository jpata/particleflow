import math
from dataclasses import dataclass

import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

from mlpf.logger import _logger
from mlpf.conf import Y_FEATURES
from mlpf.model.losses import LOSS_TASKS, REGRESSION_FEATURES


PARTICLE_NUMBER_INDEX = Y_FEATURES.index("particle_number")


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


def query_origin_contrastive_loss(
    query_embeddings,
    memory_embeddings,
    input_particle_numbers,
    target_particle_numbers,
    input_mask,
    target_mask,
    matches,
    temperature=0.1,
):
    """Align queries to truth-linked hit groups with a symmetric contrastive loss.

    ``particle_number`` is used only as an event-local grouping label. Hit
    embeddings are accumulated directly into target-particle prototypes, avoiding
    the O(num_queries * num_hits) ownership tensor that a dense mask loss would
    require. The particle-to-query direction includes every query as a negative,
    so duplicate queries aligned to the same particle are explicitly penalized.
    """

    if temperature <= 0:
        raise ValueError("query-origin temperature must be positive")
    if query_embeddings is None or memory_embeddings is None:
        raise ValueError("query-origin loss requires decoder query and memory embeddings")

    zero = query_embeddings.reshape(-1)[0].float() * 0.0 + memory_embeddings.reshape(-1)[0].float() * 0.0
    # Perform reductions and similarities in FP32 even under BF16 autocast.
    with torch.autocast(device_type=query_embeddings.device.type, enabled=False):
        batch_size, num_inputs, embedding_dim = memory_embeddings.shape
        num_targets = target_particle_numbers.shape[1]
        if batch_size == 0 or num_inputs == 0 or num_targets == 0:
            return zero

        hit_numbers = input_particle_numbers.long()
        target_numbers = target_particle_numbers.long()
        valid_hits = input_mask.bool() & (hit_numbers > 0)
        valid_numbered_targets = target_mask.bool() & (target_numbers > 0)
        if not valid_hits.any() or not valid_numbered_targets.any():
            return zero

        # particle_number is event-local. A composite key lets one searchsorted
        # map every hit in the batch to its compact target row without a Python
        # event loop or a dense query-by-hit ownership tensor.
        max_particle_number = torch.maximum(
            hit_numbers.masked_fill(~valid_hits, 0).max(),
            target_numbers.masked_fill(~valid_numbered_targets, 0).max(),
        )
        key_stride = max_particle_number + 1
        event_offsets = torch.arange(batch_size, device=query_embeddings.device, dtype=torch.long) * key_stride
        hit_keys = (hit_numbers + event_offsets[:, None])[valid_hits]
        target_keys = (target_numbers + event_offsets[:, None])[valid_numbered_targets]
        target_flat_indices = torch.arange(
            batch_size * num_targets, device=query_embeddings.device, dtype=torch.long
        ).reshape(batch_size, num_targets)[valid_numbered_targets]

        sorted_target_keys, target_key_order = torch.sort(target_keys)
        sorted_target_flat_indices = target_flat_indices[target_key_order]
        positions = torch.searchsorted(sorted_target_keys, hit_keys)
        in_range = positions < len(sorted_target_keys)
        safe_positions = positions.clamp_max(len(sorted_target_keys) - 1)
        associated = in_range & (sorted_target_keys[safe_positions] == hit_keys)
        if not associated.any():
            return zero

        group_flat_indices = sorted_target_flat_indices[safe_positions[associated]]
        # Select associated hits before promoting BF16 activations to FP32 so a
        # padded batch never acquires a full-size FP32 memory copy.
        valid_memory = memory_embeddings[valid_hits][associated].float()
        prototype_sums = valid_memory.new_zeros((batch_size * num_targets, embedding_dim))
        prototype_sums.index_add_(0, group_flat_indices, valid_memory)
        prototype_counts = valid_memory.new_zeros(batch_size * num_targets)
        prototype_counts.index_add_(
            0,
            group_flat_indices,
            torch.ones_like(group_flat_indices, dtype=valid_memory.dtype),
        )
        prototype_sums = prototype_sums.reshape(batch_size, num_targets, embedding_dim)
        prototype_counts = prototype_counts.reshape(batch_size, num_targets)
        valid_prototypes = prototype_counts > 0
        prototypes = prototype_sums / prototype_counts.clamp_min(1.0)[..., None]
        prototypes = F.normalize(prototypes, dim=-1, eps=1.0e-6)
        queries = F.normalize(query_embeddings.float(), dim=-1, eps=1.0e-6)
        similarities = torch.bmm(queries, prototypes.transpose(1, 2)) / temperature
        similarities = similarities.masked_fill(~valid_prototypes[:, None, :], torch.finfo(similarities.dtype).min)

        pair_batches = []
        pair_slots = []
        pair_targets = []
        for event_idx, (slot_indices, target_indices) in enumerate(matches):
            if len(slot_indices):
                # Hungarian target indices address the compact valid-target view,
                # whereas the batched prototype tensor retains padded positions.
                valid_target_positions = torch.nonzero(target_mask[event_idx], as_tuple=False).squeeze(1)
                pair_batches.append(torch.full_like(slot_indices, event_idx))
                pair_slots.append(slot_indices)
                pair_targets.append(valid_target_positions[target_indices])
        if not pair_slots:
            return zero

        pair_batches = torch.cat(pair_batches)
        pair_slots = torch.cat(pair_slots)
        pair_targets = torch.cat(pair_targets)
        has_prototype = valid_prototypes[pair_batches, pair_targets]
        pair_batches = pair_batches[has_prototype]
        pair_slots = pair_slots[has_prototype]
        pair_targets = pair_targets[has_prototype]
        if len(pair_slots) == 0:
            return zero

        # Query -> particle aligns each matched query with its originating hit
        # group. Particle -> query makes that group select exactly one query;
        # all unmatched and duplicate queries participate as negatives.
        query_to_particle = F.cross_entropy(
            similarities[pair_batches, pair_slots],
            pair_targets,
            reduction="sum",
        )
        particle_to_query = F.cross_entropy(
            similarities[pair_batches, :, pair_targets],
            pair_slots,
            reduction="sum",
        )
        return (query_to_particle + particle_to_query) / (2 * len(pair_slots))


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
    query_embeddings=None,
    memory_embeddings=None,
    query_origin_loss_weight=0.0,
    query_origin_temperature=0.1,
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

    if query_origin_loss_weight > 0:
        if batch.ytarget is None:
            raise ValueError("query-origin loss requires per-hit ytarget particle_number labels")
        if "particle_number" not in targets:
            raise ValueError("query-origin loss requires particle_number in set targets")
        origin_loss = query_origin_contrastive_loss(
            query_embeddings,
            memory_embeddings,
            batch.ytarget[..., PARTICLE_NUMBER_INDEX],
            targets["particle_number"],
            batch.mask,
            batch.target_mask,
            matches,
            temperature=query_origin_temperature,
        )
        losses["Query_origin"] = query_origin_loss_weight * origin_loss
        loss_opt = loss_opt + losses["Query_origin"]

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
