from typing import Optional, Tuple

import torch
from torch.nn import functional as F
from torch import Tensor, nn
from torch_scatter import scatter_max, scatter_add

from mlpf.logger import _logger


def scatter_count(input: torch.Tensor):
    return scatter_add(torch.ones_like(input, dtype=torch.long), input.long())


def scatter_counts_to_indices(input: torch.LongTensor) -> torch.LongTensor:
    return torch.repeat_interleave(torch.arange(input.size(0), device=input.device), input).long()


def batch_cluster_indices(cluster_id: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.LongTensor, torch.LongTensor]:
    device = cluster_id.device
    assert cluster_id.device == batch.device
    n_clusters_per_event = scatter_max(cluster_id, batch, dim=-1)[0] + 1
    offset_values_nozero = n_clusters_per_event[:-1].cumsum(dim=-1)
    offset_values = torch.cat((torch.zeros(1, device=device), offset_values_nozero))
    offset = torch.gather(offset_values, 0, batch).long()
    return offset + cluster_id, n_clusters_per_event


def get_inter_event_norms_mask(batch: torch.LongTensor, nclusters_per_event: torch.LongTensor):
    device = batch.device
    batch_expanded_as_ones = (batch == torch.arange(batch.max() + 1, dtype=torch.long, device=device).unsqueeze(-1)).long()
    return batch_expanded_as_ones.repeat_interleave(nclusters_per_event, dim=0).T


def calc_LV_Lbeta(
    beta: torch.Tensor,
    cluster_space_coords: torch.Tensor,
    cluster_index_per_event: torch.Tensor,
    batch: torch.Tensor,
    qmin: float = 0.1,
    s_B: float = 1.0,
    noise_cluster_index: int = 0,
):
    """
    Calculates the L_V and L_beta object condensation losses.
    Note: Compared to the HitPF reference implementation, this version:
    1. Omits 'use_average_cc_pos' and 'frac_combinations' heuristics to focus on core OC math.
    2. Omits the 'L_alpha_coordinates' term as detector-space coordinate regression is handled separately in MLPF.
    3. Uses dynamic batch-size normalization for L_beta_noise instead of a hardcoded constant.
    """
    # device = beta.device
    beta = torch.nan_to_num(beta, nan=0.0)
    if torch.isnan(cluster_space_coords).any():
        _logger.error(f"NaN in cluster_space_coords: {cluster_space_coords}")

    cluster_index, n_clusters_per_event = batch_cluster_indices(cluster_index_per_event, batch)
    # n_clusters = n_clusters_per_event.sum()
    n_hits, cluster_space_dim = cluster_space_coords.size()
    batch_size = batch.max() + 1

    # batch_cluster = scatter_counts_to_indices(n_clusters_per_event)

    is_noise = cluster_index_per_event == noise_cluster_index
    is_sig = ~is_noise
    # n_hits_sig = is_sig.sum()

    is_object = scatter_max(is_sig.long(), cluster_index)[0].bool()

    object_index_per_event = cluster_index_per_event[is_sig] - 1
    object_index, n_objects_per_event = batch_cluster_indices(object_index_per_event, batch[is_sig])
    # n_hits_per_object = scatter_count(object_index)
    # batch_object = batch_cluster[is_object]
    # n_objects = is_object.sum()

    # L_V term
    # Clip beta more aggressively to avoid large arctanh values, especially in bfloat16
    # 1e-2 is safer for bfloat16 precision
    q = (beta.clip(0.0, 1 - 1e-2).arctanh() / 1.01) ** 2 + qmin
    if torch.isnan(q).any():
        _logger.error(f"NaN in q detected!")
    if torch.isinf(q).any():
        _logger.error(f"Inf in q detected! max beta: {beta.max()}")
        # Defensive clamp to avoid propagation of Inf
        q = torch.clamp(q, max=1e6)

    q_alpha, index_alpha = scatter_max(q[is_sig], object_index)
    if torch.isinf(q_alpha).any():
        _logger.error(f"Inf in q_alpha detected!")
    if torch.isnan(q_alpha).any():
        _logger.error(f"NaN in q_alpha detected!")

    x_alpha = cluster_space_coords[is_sig][index_alpha]

    M = torch.nn.functional.one_hot(cluster_index).long()
    M_inv = get_inter_event_norms_mask(batch, n_clusters_per_event) - M

    M = M[:, is_object]
    M_inv = M_inv[:, is_object]

    # Check coords range
    if torch.isnan(cluster_space_coords).any():
        _logger.error("NaN in cluster_space_coords detected!")
    coords_max = torch.max(torch.abs(cluster_space_coords))
    if coords_max > 1e3:
        _logger.warning(f"Large coords detected: {coords_max}")

    norms = torch.sum(
        torch.square(cluster_space_coords.unsqueeze(1) - x_alpha.unsqueeze(0)),
        dim=-1,
    )
    if torch.isnan(norms).any():
        _logger.error(f"NaN in norms detected!")
    if torch.isinf(norms).any():
        _logger.error(f"Inf in norms detected!")

    N_k = torch.sum(M, dim=0)
    norms_att = norms[is_sig]
    # Use a more stable log calculation
    # log(exp(1) * x / 2 + 1)
    norms_att = torch.log(torch.exp(torch.tensor([1.0], device=norms_att.device)) * norms_att / 2.0 + 1.0)
    norms_att *= M[is_sig]
    if torch.isnan(norms_att).any():
        _logger.error(f"NaN in norms_att detected!")

    V_attractive = (q[is_sig]).unsqueeze(-1) * q_alpha.unsqueeze(0) * norms_att
    V_attractive = V_attractive.sum(dim=0)
    V_attractive = V_attractive.view(-1) / (N_k.view(-1) + 1e-3)
    L_V_attractive = torch.mean(V_attractive)
    if torch.isnan(L_V_attractive):
        _logger.error(f"NaN in L_V_attractive!")

    norms_rep = torch.relu(1.0 - torch.sqrt(norms + 1e-6)) * M_inv
    V_repulsive = q.unsqueeze(1) * q_alpha.unsqueeze(0) * norms_rep

    L_V_repulsive = V_repulsive.sum(dim=0)
    number_of_repulsive_terms_per_object = torch.sum(M_inv, dim=0)
    # Check if number_of_repulsive_terms_per_object is 0
    if (number_of_repulsive_terms_per_object == 0).any():
         _logger.warning(f"Some objects have 0 repulsive terms")

    L_V_repulsive = L_V_repulsive.view(-1) / (number_of_repulsive_terms_per_object.view(-1) + 1e-3)
    L_V_repulsive = torch.mean(L_V_repulsive)
    if torch.isnan(L_V_repulsive):
        _logger.error(f"NaN in L_V_repulsive!")
        # If we still have NaN, it might be Inf * 0.0. Let's try to clamp V_repulsive
        L_V_repulsive = torch.mean(torch.nan_to_num(V_repulsive, nan=0.0).sum(dim=0) / (number_of_repulsive_terms_per_object.view(-1) + 1e-3))
        _logger.info(f"Corrected L_V_repulsive to {L_V_repulsive}")

    L_V = L_V_attractive + L_V_repulsive



    n_noise_hits_per_event = scatter_count(batch[is_noise])
    n_noise_hits_per_event[n_noise_hits_per_event == 0] = 1
    # Note: L_beta_noise is normalized by batch_size here, whereas HitPF uses a hardcoded constant (4)
    L_beta_noise = s_B * ((scatter_add(beta[is_noise], batch[is_noise])) / n_noise_hits_per_event).sum() / batch_size

    beta_per_object_c = scatter_add(beta[is_sig], object_index)
    beta_alpha = beta[is_sig][index_alpha]
    L_beta_sig = torch.mean(1 - beta_alpha + 1 - torch.clip(beta_per_object_c, 0, 1))

    L_beta = L_beta_noise + L_beta_sig

    return L_V, L_beta


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


def mlpf_loss(y, ypred, batch):
    """
    Args
        y [dict]: relevant keys are "cls_id, momentum, charge, particle_number"
        ypred [dict]: relevant keys are "cls_id_onehot, momentum, charge, oc_beta, oc_coords"
        batch [PFBatch]: the MLPF inputs
    """
    loss = {}
    loss_obj_id = FocalLoss(gamma=2.0, reduction="none")

    # msk_pred_particle = torch.unsqueeze((ypred["cls_id"] != 0).to(dtype=torch.float32), dim=-1)
    msk_true_particle = torch.unsqueeze((y["cls_id"] != 0).to(dtype=torch.float32), dim=-1)
    nelem = torch.sum(batch.mask)
    npart = torch.sum(y["cls_id"] != 0)

    ypred["momentum"] = ypred["momentum"] * msk_true_particle
    y["momentum"] = y["momentum"] * msk_true_particle

    # in case of the 3D-padded mode, pytorch expects (batch, num_classes, ...)
    ypred["cls_binary"] = ypred["cls_binary"].permute((0, 2, 1))
    ypred["cls_id_onehot"] = ypred["cls_id_onehot"].permute((0, 2, 1))
    # ypred["ispu"] = ypred["ispu"].permute((0, 2, 1))

    # binary loss for particle / no-particle classification
    # loss_binary_classification = 10.0 * loss_obj_id(ypred["cls_binary"], (y["cls_id"] != 0).long()).reshape(y["cls_id"].shape)
    loss_binary_classification = 10.0 * torch.nn.functional.cross_entropy(ypred["cls_binary"], (y["cls_id"] != 0).long(), reduction="none")

    # Object Condensation loss
    # Flatten across batch and sequence length, but only for non-padded elements
    mask_flat = batch.mask.view(-1).bool()
    beta_flat = ypred["oc_beta"].view(-1)[mask_flat]
    coords_flat = ypred["oc_coords"].view(-1, 3)[mask_flat]
    particle_number_flat = y["particle_number"].view(-1)[mask_flat]

    # Create batch index for flattened elements
    batch_idx = torch.arange(batch.mask.shape[0], device=batch.mask.device).unsqueeze(1).repeat(1, batch.mask.shape[1]).view(-1)[mask_flat].long()

    l_v, l_beta = calc_LV_Lbeta(beta_flat, coords_flat, particle_number_flat.long(), batch_idx)
    loss["OC_V"] = 1e-3 * l_v
    loss["OC_beta"] = 1e-3 * l_beta

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
