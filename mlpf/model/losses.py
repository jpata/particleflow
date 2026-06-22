from typing import Optional, Tuple

import torch
from torch.nn import functional as F
from torch import Tensor, nn
from torch_scatter import scatter_max, scatter_add

from mlpf.logger import _logger
from mlpf.conf import LossType


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
    qmin: float = 1.0,
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
    cluster_space_coords = torch.nan_to_num(cluster_space_coords, nan=0.0)
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
    # Ensure beta is float32 to safely compute arctanh near 1.0 without overflowing bfloat16
    beta_fp32 = beta.to(torch.float32)
    # 1e-4 matches HitPF reference and provides higher q values
    q = (beta_fp32.clip(0.0, 1 - 1e-4).arctanh() / 1.01) ** 2 + qmin
    q = q.to(beta.dtype)
    if torch.isnan(q).any():
        _logger.error(f"NaN in q detected! num NaNs: {torch.isnan(q).sum()}")
    if torch.isinf(q).any():
        _logger.error(f"Inf in q detected! max beta: {beta.max()}")
        # Defensive clamp to avoid propagation of Inf
        q = torch.clamp(q, max=1e6)

    q_alpha, index_alpha = scatter_max(q[is_sig], object_index)
    # Filter out empty groups (those that don't have any hits in is_sig)
    # For empty groups, scatter_max returns argmax = input.size(0)
    mask_valid = index_alpha < q[is_sig].size(0)
    q_alpha = q_alpha[mask_valid]
    index_alpha = index_alpha[mask_valid]

    if torch.isinf(q_alpha).any():
        _logger.error(f"Inf in q_alpha detected! num infs: {torch.isinf(q_alpha).sum()}")
    if torch.isnan(q_alpha).any():
        _logger.error(f"NaN in q_alpha detected! num NaNs: {torch.isnan(q_alpha).sum()}")

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
        _logger.error(f"NaN in norms detected! num NaNs: {torch.isnan(norms).sum()}")
    if torch.isinf(norms).any():
        _logger.error(f"Inf in norms detected! num infs: {torch.isinf(norms).sum()}")

    N_k = torch.sum(M, dim=0)
    norms_att = norms[is_sig]
    # Use a more stable log calculation
    # log(exp(1) * x / 2 + 1)
    norms_att = torch.log(torch.exp(torch.tensor([1.0], device=norms_att.device)) * norms_att / 2.0 + 1.0)
    norms_att *= M[is_sig]
    if torch.isnan(norms_att).any():
        _logger.error(f"NaN in norms_att detected! num NaNs: {torch.isnan(norms_att).sum()}")

    V_attractive = (q[is_sig]).unsqueeze(-1) * q_alpha.unsqueeze(0) * norms_att
    V_attractive = V_attractive.sum(dim=0)
    V_attractive = V_attractive.view(-1) / (N_k.view(-1) + 1e-3)
    if V_attractive.numel() > 0:
        L_V_attractive = torch.mean(V_attractive)
    else:
        L_V_attractive = torch.tensor(0.0, device=beta.device)

    if torch.isnan(L_V_attractive):
        _logger.error("NaN in L_V_attractive!")

    norms_rep = torch.relu(1.0 - torch.sqrt(norms + 1e-6)) * M_inv
    V_repulsive = q.unsqueeze(1) * q_alpha.unsqueeze(0) * norms_rep

    L_V_repulsive = V_repulsive.sum(dim=0)
    number_of_repulsive_terms_per_object = torch.sum(M_inv, dim=0)
    # Check if number_of_repulsive_terms_per_object is 0
    if (number_of_repulsive_terms_per_object == 0).any():
        _logger.warning("Some objects have 0 repulsive terms")

    L_V_repulsive = L_V_repulsive.view(-1) / (number_of_repulsive_terms_per_object.view(-1) + 1e-3)
    if L_V_repulsive.numel() > 0:
        L_V_repulsive = torch.mean(L_V_repulsive)
    else:
        L_V_repulsive = torch.tensor(0.0, device=beta.device)

    if torch.isnan(L_V_repulsive):
        _logger.error("NaN in L_V_repulsive!")
        # If we still have NaN, it might be Inf * 0.0. Let's try to clamp V_repulsive
        L_V_repulsive = torch.mean(torch.nan_to_num(V_repulsive, nan=0.0).sum(dim=0) / (number_of_repulsive_terms_per_object.view(-1) + 1e-3))
        _logger.info("Corrected L_V_repulsive to {L_V_repulsive}")

    L_V = L_V_attractive + L_V_repulsive

    n_noise_hits_per_event = scatter_count(batch[is_noise])
    n_noise_hits_per_event[n_noise_hits_per_event == 0] = 1
    # Note: L_beta_noise is normalized by batch_size here, whereas HitPF uses a hardcoded constant (4)
    L_beta_noise = s_B * ((scatter_add(beta[is_noise], batch[is_noise])) / n_noise_hits_per_event).sum() / batch_size

    beta_per_object_c = scatter_add(beta[is_sig], object_index)
    if index_alpha.numel() > 0:
        L_beta_sig = torch.mean(1 - beta[is_sig][index_alpha] + 1 - torch.clip(beta_per_object_c[mask_valid], 0, 1))
    else:
        L_beta_sig = torch.tensor(0.0, device=beta.device)

    L_beta = L_beta_noise + L_beta_sig

    # Map index_alpha back to the indices of the input tensors
    sig_indices = torch.nonzero(is_sig).view(-1)
    absolute_index_alpha = sig_indices[index_alpha]

    # Calculate additional metrics
    with torch.no_grad():
        if index_alpha.numel() > 0:
            beta_alpha = beta[is_sig][index_alpha]
            # avg_dist_att: mean distance of hits to their true particle's centroid
            avg_dist_att = torch.sqrt(norms[is_sig][M[is_sig].bool()] + 1e-6).mean()
            # avg_dist_rep: mean distance of hits to other particles' centroids
            avg_dist_rep = torch.sqrt(norms + 1e-6)[M_inv.bool()].mean()
            metrics = {
                "beta_alpha_mean": beta_alpha.mean(),
                "frac_found_05": (beta_alpha > 0.5).float().mean(),
                "frac_found_09": (beta_alpha > 0.9).float().mean(),
                "avg_dist_att": avg_dist_att,
                "avg_dist_rep": avg_dist_rep,
                "oc_coords_mean": torch.abs(cluster_space_coords).mean(),
                "oc_coords_std": cluster_space_coords.std(),
                "oc_coords_max": torch.abs(cluster_space_coords).max(),
                "beta_noise_mean": beta[is_noise].mean() if is_noise.any() else torch.tensor(0.0, device=beta.device),
                "n_objects_true": torch.tensor(float(index_alpha.numel()), device=beta.device),
            }
        else:
            metrics = {
                "beta_alpha_mean": torch.tensor(0.0, device=beta.device),
                "frac_found_05": torch.tensor(0.0, device=beta.device),
                "frac_found_09": torch.tensor(0.0, device=beta.device),
                "avg_dist_att": torch.tensor(0.0, device=beta.device),
                "avg_dist_rep": torch.tensor(0.0, device=beta.device),
                "oc_coords_mean": torch.abs(cluster_space_coords).mean(),
                "oc_coords_std": cluster_space_coords.std(),
                "oc_coords_max": torch.abs(cluster_space_coords).max(),
                "beta_noise_mean": beta[is_noise].mean() if is_noise.any() else torch.tensor(0.0, device=beta.device),
                "n_objects_true": torch.tensor(0.0, device=beta.device),
            }

    return L_V, L_beta, absolute_index_alpha, metrics


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

    loss["OC_V"] = torch.tensor(0.0, device=batch.X.device)
    loss["OC_beta"] = torch.tensor(0.0, device=batch.X.device)

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


def mlpf_loss_object_condensation(y, ypred, batch, mask_flat, npart, nelem, loss_obj_id):
    loss = {}
    beta_flat = torch.nan_to_num(ypred["oc_beta"].view(-1)[mask_flat], nan=0.0)
    coords_flat = torch.nan_to_num(ypred["oc_coords"].view(-1, 3)[mask_flat], nan=0.0)
    particle_number_flat = y["particle_number"].view(-1)[mask_flat]
    batch_idx = torch.arange(batch.mask.shape[0], device=batch.mask.device).unsqueeze(1).repeat(1, batch.mask.shape[1]).view(-1)[mask_flat].long()

    l_v, l_beta, cp_indices, metrics = calc_LV_Lbeta(beta_flat, coords_flat, particle_number_flat.long(), batch_idx)
    loss["OC_V"] = 1.0 * l_v
    loss["OC_beta"] = 1.0 * l_beta
    for k, v in metrics.items():
        loss[f"OC_{k}"] = v

    loss["Classification_binary"] = torch.tensor(0.0, device=batch.X.device)

    # Use unique particle indices across the batch to aggregate hits belonging to the same particle
    particle_index, _ = batch_cluster_indices(particle_number_flat.long(), batch_idx)
    is_sig_hit = particle_number_flat > 0
    # Map to signal-only particles (excluding noise cluster index 0)
    is_particle = scatter_max(is_sig_hit.long(), particle_index)[0].bool()

    # Aggregate predictions across hits in each true particle, weighted by predicted beta
    # This provides a denser gradient signal than single-hit (CP) regression
    # Detach weights to prevent regression gradients from causing beta collapse.
    weights = beta_flat.detach()

    def aggregate(values, indices, weights):
        # Flatten values if they are multi-dimensional (e.g. PID one-hot)
        if values.ndim > 1:
            weights = weights.unsqueeze(-1)
        num = scatter_add(values * weights, indices, dim=0)
        den = scatter_add(weights, indices, dim=0) + 1e-6
        return num / den

    # Determine true physical quantities for the hits
    X_hit_pt_flat = batch.X.view(-1, batch.X.shape[-1])[:, 1][mask_flat]
    X_hit_e_flat = batch.X.view(-1, batch.X.shape[-1])[:, 5][mask_flat]

    target_pt_flat = torch.exp(y["pt"].view(-1)[mask_flat]) * X_hit_pt_flat
    target_e_flat = torch.exp(y["energy"].view(-1)[mask_flat]) * X_hit_e_flat

    ypred_pt_flat = torch.exp(ypred["pt"].view(-1)[mask_flat]) * X_hit_pt_flat
    ypred_e_flat = torch.exp(ypred["energy"].view(-1)[mask_flat]) * X_hit_e_flat

    if is_particle.any():
        # Aggregate true physical properties (mean across the identical values)
        num_target_pt = scatter_add(target_pt_flat, particle_index, dim=0)
        den_target_pt = scatter_add(torch.ones_like(target_pt_flat), particle_index, dim=0) + 1e-6
        target_pt_phys_agg = (num_target_pt / den_target_pt)[is_particle]

        num_target_e = scatter_add(target_e_flat, particle_index, dim=0)
        den_target_e = scatter_add(torch.ones_like(target_e_flat), particle_index, dim=0) + 1e-6
        target_e_phys_agg = (num_target_e / den_target_e)[is_particle]

        # Aggregate predicted physical properties
        ypred_pt_phys_agg = aggregate(ypred_pt_flat, particle_index, weights)[is_particle]
        ypred_e_phys_agg = aggregate(ypred_e_flat, particle_index, weights)[is_particle]

        # Compute regression losses in log scale to match standard MLPF relative scaling
        loss["Regression_pt"] = torch.nn.functional.mse_loss(
            torch.log(ypred_pt_phys_agg + 1e-6), torch.log(target_pt_phys_agg + 1e-6), reduction="none"
        )
        loss["Regression_energy"] = torch.nn.functional.mse_loss(
            torch.log(ypred_e_phys_agg + 1e-6), torch.log(target_e_phys_agg + 1e-6), reduction="none"
        )

        sqrt_target_pt_particle = torch.sqrt(torch.clamp(target_pt_phys_agg, min=1e-6))
        loss["Regression_pt"] = (loss["Regression_pt"] * sqrt_target_pt_particle).mean()
        loss["Regression_energy"] = (loss["Regression_energy"] * sqrt_target_pt_particle).mean()

        # Helper for calculating aggregated regression losses for absolute target quantities (eta, sin_phi, cos_phi)
        def get_agg_regression_loss(key, weight=1.0):
            ypred_agg = aggregate(ypred[key].view(-1)[mask_flat], particle_index, weights)[is_particle]
            # Use scatter_mean custom logic to avoid negative clipping with scatter_max
            y_flat = y[key].view(-1)[mask_flat]
            num = scatter_add(y_flat, particle_index, dim=0)
            den = scatter_add(torch.ones_like(y_flat), particle_index, dim=0) + 1e-6
            y_agg = (num / den)[is_particle]
            _l = weight * torch.nn.functional.mse_loss(ypred_agg, y_agg, reduction="none")
            return _l.mean()

        # Regression losses for absolute values
        loss["Regression_eta"] = get_agg_regression_loss("eta", weight=1e-2)
        loss["Regression_sin_phi"] = get_agg_regression_loss("sin_phi", weight=1e-2)
        loss["Regression_cos_phi"] = get_agg_regression_loss("cos_phi", weight=1e-2)

        # Aggregated PID classification
        ypred_cls_id_flat = ypred["cls_id_onehot"].reshape(-1, ypred["cls_id_onehot"].shape[-1])
        ypred_pid_agg = aggregate(ypred_cls_id_flat[mask_flat], particle_index, weights)[is_particle]
        # For truth PID, we take the mean or max (since they are identical, mean avoids any indexing issue)
        y_pid_flat = y["cls_id"].view(-1)[mask_flat].float()
        num_pid = scatter_add(y_pid_flat, particle_index, dim=0)
        den_pid = scatter_add(torch.ones_like(y_pid_flat), particle_index, dim=0) + 1e-6
        y_pid_agg = (num_pid / den_pid)[is_particle].long()
        loss["Classification"] = loss_obj_id(ypred_pid_agg, y_pid_agg).mean()
    else:
        loss["Regression_pt"] = torch.tensor(0.0, device=batch.X.device)
        loss["Regression_eta"] = torch.tensor(0.0, device=batch.X.device)
        loss["Regression_sin_phi"] = torch.tensor(0.0, device=batch.X.device)
        loss["Regression_cos_phi"] = torch.tensor(0.0, device=batch.X.device)
        loss["Regression_energy"] = torch.tensor(0.0, device=batch.X.device)
        loss["Classification"] = torch.tensor(0.0, device=batch.X.device)

    return loss


def mlpf_loss(y, ypred, batch, loss_mode=LossType.STANDARD):
    """
    Args
        y [dict]: relevant keys are "cls_id, momentum, charge, particle_number"
        ypred [dict]: relevant keys are "cls_id_onehot, momentum, charge, oc_beta, oc_coords"
        batch [PFBatch]: the MLPF inputs
        loss_mode [LossType]: whether to use standard or object condensation loss
    """
    loss_obj_id = FocalLoss(gamma=2.0, reduction="none")

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

    if loss_mode == LossType.STANDARD:
        loss = mlpf_loss_standard(y_combined, ypred_combined, batch, mask_flat, npart, nelem, loss_obj_id)
    elif loss_mode == LossType.OBJECT_CONDENSATION:
        loss = mlpf_loss_object_condensation(y_combined, ypred_combined, batch, mask_flat, npart, nelem, loss_obj_id)
    else:
        raise ValueError(f"Unknown loss_mode {loss_mode}")

    # this is the final loss to be optimized
    loss["Total"] = (
        loss["Classification_binary"]
        + loss["Classification"]
        + loss["OC_V"]
        + loss["OC_beta"]
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
