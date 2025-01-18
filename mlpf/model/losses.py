from typing import Optional

import torch
from torch.nn import functional as F
from torch import Tensor, nn

from mlpf.model.logger import _logger


def sliced_wasserstein_loss(y_pred, y_true, num_projections=200):
    # create normalized random basis vectors
    theta = torch.randn(num_projections, y_true.shape[-1]).to(device=y_true.device)
    theta = theta / torch.sqrt(torch.sum(theta**2, axis=1, keepdims=True))

    # project the features with the random basis
    A = torch.matmul(y_true, torch.transpose(theta, -1, -2))
    B = torch.matmul(y_pred, torch.transpose(theta, -1, -2))

    A_sorted = torch.sort(A, axis=-2).values
    B_sorted = torch.sort(B, axis=-2).values

    ret = torch.sqrt(torch.sum(torch.pow(A_sorted - B_sorted, 2), axis=[-1, -2]))
    return ret


def mlpf_loss(y, ypred, batch):
    """
    Args
        y [dict]: relevant keys are "cls_id, momentum, charge"
        ypred [dict]: relevant keys are "cls_id_onehot, momentum, charge"
        batch [PFBatch]: the MLPF inputs
    """
    loss = {}
    loss_obj_id = FocalLoss(gamma=2.0, reduction="none")

    msk_pred_particle = torch.unsqueeze((ypred["cls_id"] != 0).to(dtype=torch.float32), axis=-1)
    msk_true_particle = torch.unsqueeze((y["cls_id"] != 0).to(dtype=torch.float32), axis=-1)
    nelem = torch.sum(batch.mask)
    npart = torch.sum(y["cls_id"] != 0)

    ypred["momentum"] = ypred["momentum"] * msk_true_particle
    y["momentum"] = y["momentum"] * msk_true_particle

    # in case of the 3D-padded mode, pytorch expects (batch, num_classes, ...)
    ypred["cls_binary"] = ypred["cls_binary"].permute((0, 2, 1))
    ypred["cls_id_onehot"] = ypred["cls_id_onehot"].permute((0, 2, 1))
    ypred["ispu"] = ypred["ispu"].permute((0, 2, 1))

    # binary loss for particle / no-particle classification
    # loss_binary_classification = loss_obj_id(ypred["cls_binary"], (y["cls_id"] != 0).long()).reshape(y["cls_id"].shape)
    loss_binary_classification = 10 * torch.nn.functional.cross_entropy(ypred["cls_binary"], (y["cls_id"] != 0).long(), reduction="none")

    # compare the particle type, only for cases where there was a true particle
    loss_pid_classification = loss_obj_id(ypred["cls_id_onehot"], y["cls_id"]).reshape(y["cls_id"].shape)
    loss_pid_classification[y["cls_id"] == 0] *= 0

    # compare particle "PU-ness", only for cases where there was a true particle
    loss_pu = torch.nn.functional.binary_cross_entropy_with_logits(torch.squeeze(ypred["ispu"], dim=1), y["ispu"], reduction="none")
    loss_pu[y["cls_id"] == 0] *= 0

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
    loss_pu[batch.mask == 0] *= 0
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
    loss["ispu"] = loss_pu.sum() / nelem

    # compute predicted pt from model output
    pred_pt = torch.unsqueeze(torch.exp(ypred["pt"]) * batch.X[..., 1], axis=-1) * msk_pred_particle
    pred_px = pred_pt * torch.unsqueeze(ypred["cos_phi"].detach(), axis=-1) * msk_pred_particle
    pred_py = pred_pt * torch.unsqueeze(ypred["sin_phi"].detach(), axis=-1) * msk_pred_particle
    # pred_pz = pred_pt * torch.unsqueeze(torch.sinh(ypred["eta"].detach()), axis=-1) * msk_pred_particle
    # pred_mass2 = pred_e**2 - pred_pt**2 - pred_pz**2

    # compute MET, sum across particle axis in event
    pred_met = torch.sqrt(torch.sum(pred_px, axis=-2) ** 2 + torch.sum(pred_py, axis=-2) ** 2).detach()
    loss["MET"] = torch.nn.functional.huber_loss(pred_met.squeeze(dim=-1), batch.genmet).mean()

    was_input_pred = torch.concat([torch.softmax(ypred["cls_binary"].transpose(1, 2), axis=-1), ypred["momentum"]], axis=-1) * batch.mask.unsqueeze(
        axis=-1
    )
    was_input_true = torch.concat([torch.nn.functional.one_hot((y["cls_id"] != 0).to(torch.long)), y["momentum"]], axis=-1) * batch.mask.unsqueeze(
        axis=-1
    )

    # standardize Wasserstein loss
    std = was_input_true[batch.mask].std(axis=0)
    loss["Sliced_Wasserstein_Loss"] = sliced_wasserstein_loss(was_input_pred / std, was_input_true / std).mean()

    # this is the final loss to be optimized
    loss["Total"] = (
        loss["Classification_binary"]
        + loss["Classification"]
        + loss["ispu"]
        + loss["Regression_pt"]
        + loss["Regression_eta"]
        + loss["Regression_sin_phi"]
        + loss["Regression_cos_phi"]
        + loss["Regression_energy"]
    )
    loss_opt = loss["Total"]
    if torch.isnan(loss_opt):
        _logger.error(ypred)
        _logger.error(sqrt_target_pt)
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
        log_pt = torch.gather(log_p, 1, y.unsqueeze(axis=-1)).squeeze(axis=-1)

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
