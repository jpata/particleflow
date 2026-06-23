from typing import Optional

import torch
from torch.nn import functional as F
from torch import Tensor, nn

from mlpf.logger import _logger


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


def mlpf_loss_standard(y, ypred, batch, mask_flat, npart, nelem, loss_obj_id):
    loss = {}
    # Predictions from model are (B, N, C)
    ypred_cls_binary_flat = ypred["cls_binary"].reshape(-1, 2)
    ypred_cls_id_flat = ypred["cls_id_onehot"].reshape(-1, ypred["cls_id_onehot"].shape[-1])
    y_cls_id_flat = y["cls_id"].view(-1)

    # binary loss for particle / no-particle classification
    loss_binary_classification = 10.0 * torch.nn.functional.cross_entropy(ypred_cls_binary_flat, (y_cls_id_flat != 0).long(), reduction="none")
    loss_binary_classification[~mask_flat] *= 0
    loss["Classification_binary"] = loss_binary_classification.sum() / nelem

    # compare the particle type, only for cases where there was a true particle
    loss_pid_classification = loss_obj_id(ypred_cls_id_flat, y_cls_id_flat)
    loss_pid_classification[y_cls_id_flat == 0] *= 0
    loss_pid_classification[~mask_flat] *= 0
    loss["Classification"] = loss_pid_classification.sum() / nelem

    # compare particle momentum, only for cases where there was a true particle
    ypred_pt_flat = ypred["pt"].view(-1)
    y_pt_flat = y["pt"].view(-1)
    ypred_eta_flat = ypred["eta"].view(-1)
    y_eta_flat = y["eta"].view(-1)
    ypred_sin_phi_flat = ypred["sin_phi"].view(-1)
    y_sin_phi_flat = y["sin_phi"].view(-1)
    ypred_cos_phi_flat = ypred["cos_phi"].view(-1)
    y_cos_phi_flat = y["cos_phi"].view(-1)
    ypred_energy_flat = ypred["energy"].view(-1)
    y_energy_flat = y["energy"].view(-1)

    loss_regression_pt = torch.nn.functional.mse_loss(ypred_pt_flat, y_pt_flat, reduction="none")
    loss_regression_eta = 1e-2 * torch.nn.functional.mse_loss(ypred_eta_flat, y_eta_flat, reduction="none")
    loss_regression_sin_phi = 1e-2 * torch.nn.functional.mse_loss(ypred_sin_phi_flat, y_sin_phi_flat, reduction="none")
    loss_regression_cos_phi = 1e-2 * torch.nn.functional.mse_loss(ypred_cos_phi_flat, y_cos_phi_flat, reduction="none")
    loss_regression_energy = torch.nn.functional.mse_loss(ypred_energy_flat, y_energy_flat, reduction="none")

    loss_regression_pt[y_cls_id_flat == 0] *= 0
    loss_regression_eta[y_cls_id_flat == 0] *= 0
    loss_regression_sin_phi[y_cls_id_flat == 0] *= 0
    loss_regression_cos_phi[y_cls_id_flat == 0] *= 0
    loss_regression_energy[y_cls_id_flat == 0] *= 0

    loss_regression_pt[~mask_flat] *= 0
    loss_regression_eta[~mask_flat] *= 0
    loss_regression_sin_phi[~mask_flat] *= 0
    loss_regression_cos_phi[~mask_flat] *= 0
    loss_regression_energy[~mask_flat] *= 0

    # add weight based on target pt
    # ensure input to sqrt is positive
    sqrt_target_pt = torch.sqrt(torch.clamp(torch.exp(y_pt_flat) * batch.X.view(-1, batch.X.shape[-1])[:, 1], min=1e-6))
    loss_regression_pt *= sqrt_target_pt
    loss_regression_energy *= sqrt_target_pt

    if npart > 0:
        loss["Regression_pt"] = loss_regression_pt.sum() / npart
        loss["Regression_eta"] = loss_regression_eta.sum() / npart
        loss["Regression_sin_phi"] = loss_regression_sin_phi.sum() / npart
        loss["Regression_cos_phi"] = loss_regression_cos_phi.sum() / npart
        loss["Regression_energy"] = loss_regression_energy.sum() / npart
    else:
        loss["Regression_pt"] = torch.tensor(0.0, device=batch.X.device)
        loss["Regression_eta"] = torch.tensor(0.0, device=batch.X.device)
        loss["Regression_sin_phi"] = torch.tensor(0.0, device=batch.X.device)
        loss["Regression_cos_phi"] = torch.tensor(0.0, device=batch.X.device)
        loss["Regression_energy"] = torch.tensor(0.0, device=batch.X.device)

    return loss


def mlpf_loss(y, ypred, batch):
    """
    Args
        y [dict]: relevant keys are "cls_id, momentum, charge, particle_number"
        ypred [dict]: relevant keys are "cls_id_onehot, momentum, charge, oc_beta, oc_coords"
        batch [PFBatch]: the MLPF inputs
    """
    loss_obj_id = FocalLoss(gamma=2.0, reduction="none")

    is_no_target = y["cls_id"] == 0
    for d in [ypred, y]:
        for key in ["pt", "eta", "sin_phi", "cos_phi", "energy", "phi"]:
            if key in d:
                d[key] = torch.where(is_no_target, torch.zeros_like(d[key]), d[key])
        if "p4" in d:
            p4 = d["p4"].clone()
            for i in range(4):
                p4[..., i] = torch.where(is_no_target, torch.zeros_like(p4[..., i]), p4[..., i])
            d["p4"] = p4
        if "momentum" in d:
            momentum = d["momentum"].clone()
            for i in range(5):
                momentum[..., i] = torch.where(is_no_target, torch.zeros_like(momentum[..., i]), momentum[..., i])
            d["momentum"] = momentum

    msk_true_particle = torch.unsqueeze((y["cls_id"] != 0).to(dtype=torch.float32), dim=-1)
    npart = torch.sum(y["cls_id"] != 0)

    # Flatten across batch and sequence length, but only for non-padded elements
    mask_flat = batch.mask.view(-1).bool()
    nelem = torch.sum(mask_flat)

    # Create local copies of momentum to avoid in-place modification of model outputs
    momentum_pred = ypred["momentum"] * msk_true_particle
    momentum_true = y["momentum"] * msk_true_particle

    # Also extract individual regression components
    ypred_momentum_unpacked = {
        "pt": torch.nan_to_num(ypred["pt"]),
        "eta": torch.nan_to_num(ypred["eta"]),
        "sin_phi": torch.nan_to_num(ypred["sin_phi"]),
        "cos_phi": torch.nan_to_num(ypred["cos_phi"]),
        "energy": torch.nan_to_num(ypred["energy"]),
        "momentum": torch.nan_to_num(momentum_pred),
    }

    ypred_combined = {**ypred, **ypred_momentum_unpacked}
    y_combined = {**y, "momentum": momentum_true}

    loss = mlpf_loss_standard(y_combined, ypred_combined, batch, mask_flat, npart, nelem, loss_obj_id)

    # this is the final loss to be optimized
    loss["Total"] = (
        loss["Classification_binary"]
        + loss["Classification"]
        + loss["Regression_pt"]
        + loss["Regression_eta"]
        + loss["Regression_sin_phi"]
        + loss["Regression_cos_phi"]
        + loss["Regression_energy"]
    )
    loss_opt = loss["Total"]
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
