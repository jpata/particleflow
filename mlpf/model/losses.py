from typing import Optional

import torch
from torch.nn import functional as F
from torch import Tensor, nn

from mlpf.logger import _logger


REGRESSION_FEATURES = ("pt", "eta", "sin_phi", "cos_phi", "energy")


def _mask_no_target_regression(y, ypred):
    """Return copies with regression values zeroed where no target particle exists."""
    is_no_target = y["cls_id"] == 0

    def mask_values(values):
        masked = dict(values)
        for key in REGRESSION_FEATURES:
            if key in masked:
                masked[key] = torch.where(is_no_target, torch.zeros_like(masked[key]), masked[key])
        return masked

    return mask_values(y), mask_values(ypred)


def sliced_wasserstein_loss(y_pred, y_true, num_projections=200):
    # create normalized random basis vectors
    theta = torch.randn(num_projections, y_true.shape[-1]).to(device=y_true.device)
    theta = theta / torch.sqrt(torch.sum(theta**2, dim=1, keepdims=True))

    # project the features with the random basis
    A = torch.matmul(y_true, torch.transpose(theta, -1, -2))
    B = torch.matmul(y_pred, torch.transpose(theta, -1, -2))

    A_sorted = torch.sort(A, dim=-2).values
    B_sorted = torch.sort(B, dim=-2).values

    ret = torch.sqrt(torch.sum(torch.pow(A_sorted - B_sorted, 2), dim=[-1, -2]))
    return ret


def _weighted_sum(values, weights=None):
    if weights is None:
        return values.sum()
    return (values * weights).sum()


def _weighted_count(mask, weights=None):
    if weights is None:
        return mask.sum()
    return weights[mask].sum()


def classification_loss(y, ypred, element_weights=None):
    """Compute per-element particle-presence and particle-ID losses."""
    cls_id = y["cls_id"]
    num_elements = cls_id.numel()
    is_particle = cls_id != 0

    binary_per_element = F.cross_entropy(ypred["cls_binary"], is_particle.long(), reduction="none")
    if element_weights is None:
        binary_denominator = torch.as_tensor(float(num_elements), dtype=binary_per_element.dtype, device=binary_per_element.device)
    else:
        binary_denominator = element_weights.sum()
    binary = 10.0 * _weighted_sum(binary_per_element, element_weights) / binary_denominator.clamp_min(1.0)

    pid_per_element = FocalLoss(gamma=2.0, reduction="none")(ypred["cls_id_onehot"], cls_id)
    pid_per_element = torch.where(is_particle, pid_per_element, torch.zeros_like(pid_per_element))
    pid = _weighted_sum(pid_per_element, element_weights) / binary_denominator.clamp_min(1.0)

    return {
        "Classification_binary": binary,
        "Classification": pid,
    }


def regression_loss(y, ypred, input_pt, regression_weights, element_weights=None):
    """Compute per-particle kinematic losses for flattened event elements."""
    is_particle = y["cls_id"] != 0
    num_particles = _weighted_count(is_particle, element_weights).clamp_min(1.0)
    sqrt_target_pt = torch.sqrt(torch.clamp(torch.exp(y["pt"]) * input_pt, min=1e-6))

    losses = {}
    for feature in REGRESSION_FEATURES:
        weight = regression_weights[feature]
        prediction = torch.nan_to_num(ypred[feature])
        per_element = weight * F.mse_loss(prediction, y[feature], reduction="none")
        per_element = torch.where(is_particle, per_element, torch.zeros_like(per_element))
        if feature in {"pt", "energy"}:
            per_element = per_element * sqrt_target_pt
        losses[f"Regression_{feature}"] = _weighted_sum(per_element, element_weights) / num_particles

    return losses


def particle_loss(y, ypred, input_pt, regression_weights, element_weights=None):
    """Compute classification and regression losses over flattened particles."""
    losses = classification_loss(y, ypred, element_weights=element_weights)
    losses.update(regression_loss(y, ypred, input_pt, regression_weights, element_weights=element_weights))
    return losses


def hit_particle_clustering_loss_raw(embeddings, y, batch, clustering_config):
    """Supervise hit embeddings to cluster by target particle_number.

    The objective is centroid-based to avoid O(N^2) hit-pair losses on raw-hit
    events. It applies only to hit-input events when input_type_id is present.
    """
    if "particle_number" not in y:
        return embeddings.sum() * 0.0

    margin = float(clustering_config.margin)
    max_particles_per_event = int(clustering_config.max_particles_per_event)
    min_elements_per_particle = int(clustering_config.min_elements_per_particle)

    embeddings = F.normalize(embeddings.to(torch.float32), dim=-1)
    cls_id = y["cls_id"]
    particle_number = y["particle_number"].to(torch.long)
    event_losses = []

    for iev in range(embeddings.shape[0]):
        if batch.input_type_id is not None and int(batch.input_type_id[iev].detach().item()) != 1:
            continue

        valid = batch.mask[iev].bool() & (cls_id[iev] != 0) & (particle_number[iev] > 0)
        if not valid.any():
            continue

        event_embeddings = embeddings[iev][valid]
        event_particle_number = particle_number[iev][valid]
        unique_particles = torch.unique(event_particle_number)
        if unique_particles.numel() > max_particles_per_event:
            unique_particles = unique_particles[:max_particles_per_event]

        centroids = []
        pull_terms = []
        for particle_id in unique_particles:
            particle_mask = event_particle_number == particle_id
            if particle_mask.sum() < min_elements_per_particle:
                continue
            particle_embeddings = event_embeddings[particle_mask]
            centroid = F.normalize(particle_embeddings.mean(dim=0, keepdim=True), dim=-1).squeeze(0)
            centroids.append(centroid)
            pull_terms.append(((particle_embeddings - centroid) ** 2).sum(dim=-1).mean())

        if not pull_terms:
            continue

        pull_loss = torch.stack(pull_terms).mean()
        if len(centroids) > 1:
            centroids_tensor = torch.stack(centroids, dim=0)
            distances = torch.pdist(centroids_tensor, p=2)
            push_loss = torch.relu(margin - distances).pow(2).mean()
            event_losses.append(pull_loss + push_loss)
        else:
            event_losses.append(pull_loss)

    if not event_losses:
        return embeddings.sum() * 0.0
    return torch.stack(event_losses).mean()


def hit_particle_clustering_loss(embeddings, y, batch, clustering_config):
    weight = float(clustering_config.weight)
    if weight <= 0.0:
        return embeddings.sum() * 0.0
    return weight * hit_particle_clustering_loss_raw(embeddings, y, batch, clustering_config)


def _input_type_element_weights(batch, input_type_loss_weights):
    if input_type_loss_weights is None or batch.input_type_id is None:
        return None

    unknown_weight = float(input_type_loss_weights.get("unknown", 1.0))
    event_weights = torch.full(
        batch.input_type_id.shape,
        unknown_weight,
        dtype=batch.X.dtype,
        device=batch.X.device,
    )
    hit_weight = torch.as_tensor(float(input_type_loss_weights.get("hits", 1.0)), dtype=batch.X.dtype, device=batch.X.device)
    pf_weight = torch.as_tensor(float(input_type_loss_weights.get("pf", 1.0)), dtype=batch.X.dtype, device=batch.X.device)
    event_weights = torch.where(batch.input_type_id == 1, hit_weight, event_weights)
    event_weights = torch.where(batch.input_type_id == 2, pf_weight, event_weights)
    return event_weights.unsqueeze(-1).expand_as(batch.mask)[batch.mask.bool()]


def event_loss(y, ypred, batch, regression_weights, input_type_loss_weights=None):
    """Compute losses for complete padded event batches.

    The standard loss currently contains only independent particle terms.
    Event-level terms comparing particle collections can be added here.
    """
    y, ypred = _mask_no_target_regression(y, ypred)
    valid = batch.mask.bool()

    particle_targets = {
        "cls_id": y["cls_id"][valid],
        **{feature: y[feature][valid] for feature in REGRESSION_FEATURES},
    }
    particle_predictions = {
        "cls_binary": ypred["cls_binary"][valid],
        "cls_id_onehot": ypred["cls_id_onehot"][valid],
        **{feature: ypred[feature][valid] for feature in REGRESSION_FEATURES},
    }
    input_pt = batch.X[..., 1][valid]
    element_weights = _input_type_element_weights(batch, input_type_loss_weights)

    return particle_loss(particle_targets, particle_predictions, input_pt, regression_weights, element_weights=element_weights)


def mlpf_loss(y, ypred, batch, regression_weights, input_type_loss_weights=None):
    """Compute the standard MLPF objective for a batch of events."""
    loss = event_loss(y, ypred, batch, regression_weights, input_type_loss_weights=input_type_loss_weights)

    loss_opt = sum(loss.values())
    loss["Total"] = loss_opt
    if torch.isnan(loss_opt):
        _logger.error(ypred)
        _logger.error(loss)
        raise Exception("Loss became NaN")

    # store these separately but detached
    for k in loss.keys():
        loss[k] = loss[k].detach()

    return loss_opt, loss


# from https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py
class FocalLoss(nn.Module):
    """Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self, alpha: Optional[Tensor] = None, gamma: float = 0.0, reduction: str = "mean", ignore_index: int = -100):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ("mean", "sum", "none"):
            raise ValueError('Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(weight=alpha, reduction="none")

    def __repr__(self):
        arg_keys = ["alpha", "gamma", "reduction"]
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f"{k}={v!r}" for k, v in zip(arg_keys, arg_vals)]
        arg_str = ", ".join(arg_strs)
        return f"{type(self).__name__}({arg_str})"

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        # this is slow due to indexing
        # all_rows = torch.arange(len(x))
        # log_pt = log_p[all_rows, y]
        log_pt = torch.gather(log_p, 1, y.unsqueeze(dim=-1)).squeeze(dim=-1)

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
