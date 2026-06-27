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


def classification_loss(y, ypred):
    """Compute per-element particle-presence and particle-ID losses."""
    cls_id = y["cls_id"]
    num_elements = cls_id.numel()
    is_particle = cls_id != 0

    binary = 10.0 * F.cross_entropy(ypred["cls_binary"], is_particle.long())

    pid_per_element = FocalLoss(gamma=2.0, reduction="none")(ypred["cls_id_onehot"], cls_id)
    pid_per_element = torch.where(is_particle, pid_per_element, torch.zeros_like(pid_per_element))
    pid = pid_per_element.sum() / num_elements

    return {
        "Classification_binary": binary,
        "Classification": pid,
    }


def regression_loss(y, ypred, input_pt, regression_weights):
    """Compute per-particle kinematic losses for flattened event elements."""
    is_particle = y["cls_id"] != 0
    num_particles = is_particle.sum().clamp_min(1)
    sqrt_target_pt = torch.sqrt(torch.clamp(torch.exp(y["pt"]) * input_pt, min=1e-6))

    losses = {}
    for feature in REGRESSION_FEATURES:
        weight = regression_weights[feature]
        prediction = torch.nan_to_num(ypred[feature])
        per_element = weight * F.mse_loss(prediction, y[feature], reduction="none")
        per_element = torch.where(is_particle, per_element, torch.zeros_like(per_element))
        if feature in {"pt", "energy"}:
            per_element = per_element * sqrt_target_pt
        losses[f"Regression_{feature}"] = per_element.sum() / num_particles

    return losses


def particle_loss(y, ypred, input_pt, regression_weights):
    """Compute classification and regression losses over flattened particles."""
    losses = classification_loss(y, ypred)
    losses.update(regression_loss(y, ypred, input_pt, regression_weights))
    return losses


def event_loss(y, ypred, batch, regression_weights):
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

    return particle_loss(particle_targets, particle_predictions, input_pt, regression_weights)


def mlpf_loss(y, ypred, batch, regression_weights):
    """Compute the standard MLPF objective for a batch of events."""
    loss = event_loss(y, ypred, batch, regression_weights)

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
