"""Geometry-conditioned region masking and input-derived prediction targets.

The catalogue is fixed before training. Query selection reads only hits on
non-target surfaces; no hidden coordinates/counts/energies enter the predictor.
"""

import json
import math
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import nn
from torch.nn import functional as F


class RegionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    geometry_path: str
    tracker_surface_fraction: float = Field(default=.25, gt=0, lt=1)
    calo_surface_fraction: float = Field(default=.4, gt=0, lt=1)
    tracker_queries: int = Field(default=128, ge=1)
    calo_queries: int = Field(default=512, ge=1)
    activity_weight_power: float = Field(default=.5, ge=0, le=1)
    predictor_dim: int = Field(default=128, ge=8)
    occupancy_weight: float = Field(default=1., ge=0)
    count_weight: float = Field(default=.1, ge=0)
    energy_weight: float = Field(default=1., ge=0)
    latent_weight: float = Field(default=.1, ge=0)

    @model_validator(mode="after")
    def nonzero_objective(self):
        if self.occupancy_weight + self.count_weight + self.energy_weight + self.latent_weight == 0:
            raise ValueError("At least one region objective weight must be positive")
        return self


def hit_addresses(x, eta_width, phi_bins, fine_bins):
    """Fixed angular bins on (type, subdetector, system, side, layer) surfaces."""
    surface = x[:, [0, 10, 12, 13, 14]].long()
    eta = x[:, 2].float() / eta_width
    phi = torch.remainder(torch.atan2(x[:, 3].float(), x[:, 4].float()) + math.pi, 2 * math.pi) * phi_bins / (2 * math.pi)
    eta_bin, phi_bin = eta.floor().long(), phi.floor().long().clamp_max(phi_bins - 1)
    fine_eta = ((eta - eta_bin) * fine_bins).long().clamp(0, fine_bins - 1)
    fine_phi = ((phi - phi_bin) * fine_bins).long().clamp(0, fine_bins - 1)
    return torch.cat([surface, eta_bin[:, None], phi_bin[:, None]], 1), fine_eta * fine_bins + fine_phi


def row_lookup(rows, catalogue):
    """Map integer row tuples to catalogue indices (-1 if absent)."""
    _, inverse = torch.unique(torch.cat([catalogue, rows]), dim=0, return_inverse=True)
    lookup = torch.full((int(inverse.max()) + 1,), -1, dtype=torch.long, device=rows.device)
    lookup[inverse[:len(catalogue)]] = torch.arange(len(catalogue), device=rows.device)
    return lookup[inverse[len(catalogue):]]


def query_features(regions, eta_width, phi_bins):
    """Catalogue bin centers, never centroids of event hits."""
    eta = (regions[:, 5].float() + .5) * eta_width
    phi = (regions[:, 6].float() + .5) * (2 * math.pi / phi_bins) - math.pi
    return torch.cat([
        F.one_hot(regions[:, 0] - 1, 2).float(), F.one_hot(regions[:, 1], 4).float(),
        regions[:, 2:3].float() / 32, regions[:, 3:4].float(), regions[:, 4:5].float() / 64,
        eta[:, None] / 5, phi.sin()[:, None], phi.cos()[:, None],
    ], 1)


class RegionSampler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        data = json.loads(Path(config.geometry_path).read_text())
        if data["schema"] != "edm4hep-hit-regions-v1":
            raise ValueError("Unsupported region geometry schema")
        self.eta_width, self.phi_bins, self.fine_bins = data["eta_width"], data["phi_bins"], data["fine_bins"]
        regions = torch.tensor(data["regions"], dtype=torch.long)
        valid_bins = torch.tensor(data["valid_bins"], dtype=torch.bool)
        if regions.ndim != 2 or regions.shape[1] != 7 or len(regions) == 0:
            raise ValueError("Region catalogue must contain seven-column region addresses")
        if valid_bins.shape != (len(regions), self.fine_bins**2) or not valid_bins.any(1).all():
            raise ValueError("Invalid empirical geometry coverage")
        if len(torch.unique(regions, dim=0)) != len(regions) or not torch.isin(regions[:, 0], torch.tensor([1, 2])).all():
            raise ValueError("Catalogue must have unique raw-hit regions")
        surfaces, surface_index = torch.unique(regions[:, :5], dim=0, return_inverse=True)
        self.register_buffer("regions", regions)
        self.register_buffer("valid_bins", valid_bins)
        self.register_buffer("surfaces", surfaces)
        self.register_buffer("surface_index", surface_index)
        self.register_buffer("features", query_features(regions, self.eta_width, self.phi_bins))
        self.num_queries = config.tracker_queries + config.calo_queries

    def select_queries(self, x, generator=None):
        """Only non-target surfaces provide activity used to choose angular bins."""
        address, fine = hit_addresses(x, self.eta_width, self.phi_bins, self.fine_bins)
        hit_surface = row_lookup(address[:, :5], self.surfaces)
        selected = []
        for kind, fraction, quota in [(1, self.config.tracker_surface_fraction, self.config.tracker_queries),
                                      (2, self.config.calo_surface_fraction, self.config.calo_queries)]:
            surfaces = torch.where(self.surfaces[:, 0] == kind)[0]
            if len(surfaces) < 2:
                selected.append(torch.empty(0, dtype=torch.long, device=x.device))
                continue
            count = min(len(surfaces) - 1, max(1, round(len(surfaces) * fraction)))
            target_surfaces = surfaces[torch.randperm(len(surfaces), device=x.device, generator=generator)[:count]]
            visible = (x[:, 0] == kind) & ~torch.isin(hit_surface, target_surfaces)
            # Queries use angular footprints from other surfaces; hidden rows cannot change them.
            footprint, activity = torch.unique(address[visible, 5:7], dim=0, return_counts=True)
            if not len(footprint):
                selected.append(torch.empty(0, dtype=torch.long, device=x.device))
                continue
            angular_index = row_lookup(self.regions[:, 5:7], footprint)
            candidates = torch.where(torch.isin(self.surface_index, target_surfaces) & (angular_index >= 0))[0]
            if len(candidates):
                weights = activity[angular_index[candidates]].float().pow(self.config.activity_weight_power)
                order = torch.multinomial(weights, min(quota, len(candidates)), replacement=False, generator=generator)
                candidates = candidates[order]
            selected.append(candidates)
        indices = torch.cat(selected)
        return indices, address, fine

    @torch.no_grad()
    def forward(self, X, mask, generator=None):
        batch, _, _ = X.shape
        q, cells = self.num_queries, self.fine_bins**2
        features = X.new_zeros(batch, q, 12)
        types = torch.zeros(batch, q, dtype=torch.long, device=X.device)
        valid = torch.zeros(batch, q, cells, dtype=torch.bool, device=X.device)
        counts = torch.zeros(batch, q, cells, device=X.device)
        energy = torch.zeros_like(counts)
        views, maps, fine_maps, removed, coverage = [], [], [], [], []
        for event in range(batch):
            original = torch.where(mask[event])[0]
            x = X[event, original]
            indices, address, fine = self.select_queries(x, generator)
            regions = self.regions[indices]
            n = len(indices)
            mapping = row_lookup(address, regions) if n else torch.full((len(x),), -1, dtype=torch.long, device=X.device)
            hidden = mapping >= 0
            if hidden.all():
                raise ValueError("Region masking removed every hit; check surface metadata")
            features[event, :n] = self.features[indices]
            types[event, :n] = regions[:, 0]
            valid[event, :n] = self.valid_bins[indices]
            bins = mapping[hidden] * cells + fine[hidden]
            counts[event].view(-1).scatter_add_(0, bins, torch.ones(len(bins), device=X.device))
            energy[event].view(-1).scatter_add_(0, bins, x[hidden, 5].float().clamp_min(0))
            full_map = torch.full_like(mask[event], -1, dtype=torch.long)
            full_fine = torch.zeros_like(full_map)
            full_map[original] = mapping
            full_fine[original] = fine
            maps.append(full_map)
            fine_maps.append(full_fine)
            views.append(x[~hidden])
            removed.append(torch.stack([(hidden & (x[:, 0] == kind)).sum() / (x[:, 0] == kind).sum().clamp_min(1) for kind in [1, 2]]))
            coverage.append((valid[event].view(-1)[bins].float().mean() if len(bins) else X.new_tensor(1.)))
        view = nn.utils.rnn.pad_sequence(views, batch_first=True)
        return dict(view=view, mask=view[..., 0] != 0, features=features, types=types, valid=valid,
                    counts=counts, energy=energy, hit_to_query=torch.stack(maps), fine_index=torch.stack(fine_maps),
                    removed=torch.stack(removed), coverage=torch.stack(coverage))


class RegionPredictor(nn.Module):
    """An auxiliary cross-attention head; the MLPF backbone is unchanged."""
    def __init__(self, dimension, config, cells):
        super().__init__()
        width = config.predictor_dim
        self.query = nn.Sequential(nn.Linear(12, width), nn.GELU(), nn.Linear(width, width))
        self.context = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, width))
        self.key, self.value = nn.Linear(width, width), nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        self.output = nn.Linear(width, 3 * cells)
        self.latent = nn.Linear(width, dimension)
        self.cells = cells

    def forward(self, embeddings, mask, features):
        q, context = self.query(features), self.context(embeddings)
        # One attention head keeps this compatible with arbitrary predictor widths.
        attended = F.scaled_dot_product_attention(q[:, None], self.key(context)[:, None], self.value(context)[:, None], attn_mask=mask[:, None, None, :])[:, 0]
        hidden = self.norm(q + attended)
        hidden = hidden + self.ff(hidden)
        raw = self.output(hidden).float().reshape(*hidden.shape[:2], 3, self.cells)
        return raw[:, :, 0], raw[:, :, 1].clamp(-10, 10), raw[:, :, 2], self.latent(hidden)


def region_mean(values, types, valid):
    """Equal bins within type, equal present types within event, equal active events."""
    terms, present = [], []
    for kind in [1, 2]:
        selected = valid & (types == kind).unsqueeze(-1)
        denominator = selected.sum((1, 2))
        terms.append((values * selected).sum((1, 2)) / denominator.clamp_min(1))
        present.append(denominator > 0)
    weights = torch.stack(present, 1)
    per_event = torch.stack(terms, 1).sum(1) / weights.sum(1).clamp_min(1)
    return per_event.sum() / weights.any(1).sum().clamp_min(1)


def measurement_loss(prediction, sample, config):
    occupancy, log_count, log_energy, _ = prediction
    counts, energy, valid, types = sample["counts"], sample["energy"], sample["valid"], sample["types"]
    occupied = (counts > 0).float()
    occ_loss = region_mean(F.binary_cross_entropy_with_logits(occupancy, occupied, reduction="none"), types, valid)
    count_loss = region_mean(F.poisson_nll_loss(log_count, counts, log_input=True, full=True, reduction="none"), types, valid)
    # Predict log1p of observed deposited energy, including zeros; never MC particle energy.
    energy_loss = region_mean(F.smooth_l1_loss(log_energy, energy.log1p(), reduction="none"), types, valid & (types == 2).unsqueeze(-1))
    loss = config.occupancy_weight * occ_loss + config.count_weight * count_loss + config.energy_weight * energy_loss
    metrics = dict(occupancy_loss=occ_loss.detach(), count_loss=count_loss.detach(), energy_loss=energy_loss.detach(),
                   occupied_fraction=region_mean(occupied, types, valid).detach(),
                   occupancy_accuracy=region_mean(((occupancy > 0) == occupied.bool()).float(), types, valid).detach(),
                   tracker_removed_fraction=sample["removed"][:, 0].mean(), calo_removed_fraction=sample["removed"][:, 1].mean(),
                   catalogue_target_coverage=sample["coverage"].mean(), valid_queries=valid.any(-1).sum(1).float().mean())
    with torch.no_grad():
        positive = (occupancy > 0).float()
        true_positive = region_mean(positive * occupied, types, valid)
        predicted_rate = region_mean(positive, types, valid)
        target_rate = region_mean(occupied, types, valid)
        metrics.update(occupancy_precision=true_positive / predicted_rate.clamp_min(1e-12),
                       occupancy_recall=true_positive / target_rate.clamp_min(1e-12),
                       occupancy_f1=2 * true_positive / (predicted_rate + target_rate).clamp_min(1e-12))
    return loss, metrics


def pool_teacher(embeddings, sample):
    mapping = sample["hit_to_query"]
    batch, queries = sample["types"].shape
    pooled = embeddings.new_zeros(batch, queries, embeddings.shape[-1], dtype=torch.float32)
    count = embeddings.new_zeros(batch, queries, dtype=torch.float32)
    for event in range(batch):
        present = mapping[event] >= 0
        indices = mapping[event, present]
        pooled[event].index_add_(0, indices, embeddings[event, present].float())
        count[event].scatter_add_(0, indices, torch.ones(len(indices), device=embeddings.device))
    return pooled / count.clamp_min(1)[..., None], count > 0
