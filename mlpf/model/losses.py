from typing import Optional, Tuple, Union

import torch
from torch.nn import functional as F
from torch import Tensor, nn
from torch_scatter import scatter_max, scatter_add

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
<<<<<<< HEAD
    y, ypred = _mask_no_target_regression(y, ypred)
    valid = batch.mask.bool()
=======
    Args
        y [dict]: relevant keys are "cls_id, momentum, charge, particle_number"
        ypred [dict]: relevant keys are "cls_id_onehot, momentum, charge, oc_beta, oc_coords"
        batch [PFBatch]: the MLPF inputs
    """
    loss = {}
    loss_obj_id = FocalLoss(gamma=2.0, reduction="none")
>>>>>>> d27cb2de (add clustering ground truth)

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

<<<<<<< HEAD
    loss_opt = sum(loss.values())
    loss["Total"] = loss_opt
=======
    # Object Condensation loss
    # Flatten across batch and sequence length, but only for non-padded elements
    mask_flat = batch.mask.view(-1).bool()
    beta_flat = ypred["oc_beta"].view(-1)[mask_flat]
    coords_flat = ypred["oc_coords"].view(-1, 3)[mask_flat]
    particle_number_flat = y["particle_number"].view(-1)[mask_flat]

    # Create batch index for flattened elements
    batch_idx = (
        torch.arange(batch.mask.shape[0], device=batch.mask.device).unsqueeze(1).repeat(1, batch.mask.shape[1]).view(-1)[mask_flat].long()
    )

    l_v, l_beta = calc_LV_Lbeta(beta_flat, coords_flat, particle_number_flat.long(), batch_idx)
    loss["OC_V"] = l_v
    loss["OC_beta"] = l_beta

    # compare the particle type, only for cases where there was a true particle
    loss_pid_classification = loss_obj_id(ypred["cls_id_onehot"], y["cls_id"]).reshape(y["cls_id"].shape)
    loss_pid_classification[y["cls_id"] == 0] *= 0

    # compare particle "PU-ness", only for cases where there was a true particle
    # loss_pu = torch.nn.functional.cross_entropy(ypred["ispu"], y["ispu"].long(), reduction="none")
    # loss_pu = loss_obj_id(ypred["ispu"], y["ispu"].long()).reshape(y["cls_id"].shape)
    # loss_pu[y["cls_id"] == 0] *= 0

    # do not compute PU loss if no PU samples in this batch
    # if y["ispu"].long().sum() == 0:
    #     loss_pu *= 0

    # compare particle momentum, only for cases where there was a true particle
    loss_regression_pt = torch.nn.functional.mse_loss(ypred["pt"], y["pt"], reduction="none")
    loss_regression_eta = 1e-2 * torch.nn.functional.mse_loss(ypred["eta"], y["eta"], reduction="none")
    loss_regression_sin_phi = 1e-2 * torch.nn.functional.mse_loss(ypred["sin_phi"], y["sin_phi"], reduction="none")
    loss_regression_cos_phi = 1e-2 * torch.nn.functional.mse_loss(ypred["cos_phi"], y["cos_phi"], reduction="none")
    loss_regression_energy = torch.nn.functional.mse_loss(ypred["energy"], y["energy"], reduction="none")

    loss_regression_pt[y["cls_id"] == 0] *= 0
    loss_regression_eta[y["cls_id"] == 0] *= 0
    loss_regression_sin_phi[y["cls_id"] == 0] *= 0
    loss_regression_cos_phi[y["cls_id"] == 0] *= 0
    loss_regression_energy[y["cls_id"] == 0] *= 0

    # set the loss to 0 on padded elements in the batch
    loss_binary_classification[batch.mask == 0] *= 0
    loss_pid_classification[batch.mask == 0] *= 0
    # loss_pu[batch.mask == 0] *= 0
    loss_regression_pt[batch.mask == 0] *= 0
    loss_regression_eta[batch.mask == 0] *= 0
    loss_regression_sin_phi[batch.mask == 0] *= 0
    loss_regression_cos_phi[batch.mask == 0] *= 0
    loss_regression_energy[batch.mask == 0] *= 0

    # add weight based on target pt
    sqrt_target_pt = torch.sqrt(torch.exp(y["pt"]) * batch.X[:, :, 1])
    loss_regression_pt *= sqrt_target_pt
    loss_regression_energy *= sqrt_target_pt

    # average over all target particles
    loss["Regression_pt"] = loss_regression_pt.sum() / npart
    loss["Regression_eta"] = loss_regression_eta.sum() / npart
    loss["Regression_sin_phi"] = loss_regression_sin_phi.sum() / npart
    loss["Regression_cos_phi"] = loss_regression_cos_phi.sum() / npart
    loss["Regression_energy"] = loss_regression_energy.sum() / npart

    # average over all elements that were not padded
    loss["Classification_binary"] = loss_binary_classification.sum() / nelem
    loss["Classification"] = loss_pid_classification.sum() / nelem
    # loss["ispu"] = loss_pu.sum() / nelem

    # compute predicted pt from model output
    # pred_pt = torch.unsqueeze(torch.exp(ypred["pt"]) * batch.X[..., 1], dim=-1) * msk_pred_particle
    # pred_px = pred_pt * torch.unsqueeze(ypred["cos_phi"].detach(), dim=-1) * msk_pred_particle
    # pred_py = pred_pt * torch.unsqueeze(ypred["sin_phi"].detach(), dim=-1) * msk_pred_particle

    # compute MET, sum across particle axis in event
    # pred_met = torch.sqrt(torch.sum(pred_px, dim=-2) ** 2 + torch.sum(pred_py, dim=-2) ** 2).detach()
    # loss["MET"] = torch.nn.functional.huber_loss(pred_met.squeeze(dim=-1), batch.genmet).mean()

    # was_input_pred = torch.concat([torch.softmax(ypred["cls_binary"].transpose(1, 2), dim=-1), ypred["momentum"]], dim=-1) * batch.mask.unsqueeze(
    #     dim=-1
    # )
    # was_input_true = torch.concat([torch.nn.functional.one_hot((y["cls_id"] != 0).to(torch.long)), y["momentum"]], dim=-1) * batch.mask.unsqueeze(
    #     dim=-1
    # )

    # standardize Wasserstein loss
    # std = was_input_true[batch.mask].std(dim=0)
    # loss["Sliced_Wasserstein_Loss"] = sliced_wasserstein_loss(was_input_pred / std, was_input_true / std).mean()

    # this is the final loss to be optimized
    loss["Total"] = (
        loss["Classification_binary"]
        + loss["Classification"]
        + loss["OC_V"]
        + loss["OC_beta"]
        # + loss["ispu"]
        + loss["Regression_pt"]
        + loss["Regression_eta"]
        + loss["Regression_sin_phi"]
        + loss["Regression_cos_phi"]
        + loss["Regression_energy"]
    )
    loss_opt = loss["Total"]
>>>>>>> d27cb2de (add clustering ground truth)
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
