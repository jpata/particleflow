"""Input-only tracker geometry features; no truth or learned hit assignment."""

import math

import torch
from torch import nn


class TrackerDoubletFeatures(nn.Module):
    """Bounded cross-surface neighbors in a periodic eta/phi grid.

    At most 16 candidates per bin, sampled across radial order, and nine bins
    per hit. Query chunks cap temporary memory independently of event occupancy.
    The closest angular candidate in each radial direction supplies a prompt
    origin-plus-two-hits curvature estimate, not a reconstructed track.
    """

    BIN_SIZE = 0.1
    NUM_PHI_BINS = math.ceil(2 * math.pi / BIN_SIZE)
    NUM_ETA_BINS = 100
    CANDIDATES_PER_BIN = 16
    QUERY_CHUNK_SIZE = 2048
    OUTPUT_FEATURE_NAMES = tuple(
        f"tracker_doublet_{direction}_{name}"
        for direction in ("inner", "outer")
        for name in ("valid", "curvature_asinh", "rz_slope_asinh", "radial_span_norm", "delta_r", "candidate_count_log")
    )
    NUM_OUTPUT_FEATURES = len(OUTPUT_FEATURE_NAMES)

    def _candidate_table(self, keys, rho, eta, phi, surface):
        # Canonical tie breakers preserve permutation invariance even on fixed-r
        # detector surfaces. Exact duplicates have identical doublet geometry.
        order = torch.arange(len(keys), device=keys.device)
        for values in (surface, phi, eta, rho, keys):
            order = order[torch.argsort(values[order], stable=True)]
        unique, counts = torch.unique_consecutive(keys[order], return_counts=True)
        starts = counts.cumsum(0) - counts
        ranks = torch.arange(self.CANDIDATES_PER_BIN, device=keys.device)[None]
        offsets = torch.where(
            counts[:, None] > self.CANDIDATES_PER_BIN,
            torch.div(ranks * (counts[:, None] - 1), self.CANDIDATES_PER_BIN - 1, rounding_mode="floor"),
            ranks,
        )
        indices = order[(starts[:, None] + offsets).clamp(max=len(order) - 1)]
        return unique, indices, ranks < counts[:, None]

    def _neighbors(self, query, event, eta_bin, phi_bin, keys, table, table_valid, xyz, rho, eta, phi, surface):
        n = len(query)
        best_score = torch.full((n, 2), torch.inf, device=xyz.device)
        best_index = query[:, None].expand(-1, 2).clone()
        candidate_count = torch.zeros((n, 2), device=xyz.device)
        for de in (-1, 0, 1):
            eb = eta_bin[query] + de
            for dp in (-1, 0, 1):
                pb = (phi_bin[query] + dp) % self.NUM_PHI_BINS
                lookup = (event[query] * self.NUM_ETA_BINS + eb) * self.NUM_PHI_BINS + pb
                bins = torch.searchsorted(keys, lookup).clamp(max=len(keys) - 1)
                exists = (eb >= 0) & (eb < self.NUM_ETA_BINS) & (keys[bins] == lookup)
                candidates = table[bins]
                dr = rho[candidates] - rho[query, None]
                deta = eta[candidates] - eta[query, None]
                dphi = phi[candidates] - phi[query, None]
                dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))
                score = deta.square() + dphi.square()
                eligible = exists[:, None] & table_valid[bins] & (surface[candidates] != surface[query, None]) & (score <= 0.15**2)
                for direction in range(2):
                    use = eligible & ((dr <= -5.0) if direction == 0 else (dr >= 5.0))
                    candidate_count[:, direction] += use.sum(-1)
                    local_score, column = torch.where(use, score, torch.inf).min(-1)
                    index = candidates.gather(1, column[:, None]).squeeze(1)
                    better = local_score < best_score[:, direction]
                    best_score[:, direction] = torch.minimum(best_score[:, direction], local_score)
                    best_index[:, direction] = torch.where(better, index, best_index[:, direction])

        delta = xyz[best_index] - xyz[query, None]
        dr = rho[best_index] - rho[query, None]
        chord = torch.linalg.vector_norm(delta[..., :2], dim=-1)
        # Orient both inner and outer pairs from the smaller to larger radius.
        cross = xyz[query, None, 0] * xyz[best_index, 1] - xyz[query, None, 1] * xyz[best_index, 0]
        curvature = 2000.0 * cross * dr.sign() / (rho[query, None] * rho[best_index] * chord).clamp_min(1e-8)
        valid = torch.isfinite(best_score)
        slope = delta[..., 2] / torch.where(valid, dr, torch.ones_like(dr))
        features = torch.stack(
            [
                valid,
                torch.asinh(curvature),
                torch.asinh(slope),
                dr.abs() / 3000.0,
                torch.sqrt(torch.where(valid, best_score, 0.0)),
                torch.log1p(candidate_count),
            ],
            dim=-1,
        )
        return torch.where(valid[..., None], features, 0.0).reshape(n, -1)

    @torch.no_grad()
    def forward(self, features, mask=None):
        batch, length, _ = features.shape
        flat = features.float().reshape(-1, features.shape[-1])
        valid = (flat[:, 0] == 1) & (flat[:, 12] >= 1) & (flat[:, 12] <= 6) & torch.isfinite(flat).all(-1)
        if mask is not None:
            valid &= mask.reshape(-1)
        valid &= flat[:, 6:8].square().sum(-1) > 1e-8
        indices = torch.nonzero(valid, as_tuple=True)[0]
        output = flat.new_zeros((batch * length, self.NUM_OUTPUT_FEATURES))
        if len(indices) == 0:
            return output.reshape(batch, length, -1).to(features.dtype)
        x = flat[indices]
        xyz = x[:, 6:9]
        rho = torch.linalg.vector_norm(xyz[:, :2], dim=-1)
        eta = x[:, 2].clamp(-5, 5 - 1e-6)
        phi = torch.atan2(x[:, 3], x[:, 4])
        event = torch.div(indices, length, rounding_mode="floor")
        eta_bin = torch.floor((eta + 5) / self.BIN_SIZE).long().clamp(0, self.NUM_ETA_BINS - 1)
        phi_bin = torch.floor((phi + math.pi) * (self.NUM_PHI_BINS / (2 * math.pi))).long() % self.NUM_PHI_BINS
        surface = (x[:, 12].long() * 3 + x[:, 13].long() + 1) * 64 + x[:, 14].long()
        keys = (event * self.NUM_ETA_BINS + eta_bin) * self.NUM_PHI_BINS + phi_bin
        keys, table, table_valid = self._candidate_table(keys, rho, eta, phi, surface)
        for query in torch.arange(len(indices), device=x.device).split(self.QUERY_CHUNK_SIZE):
            output[indices[query]] = self._neighbors(query, event, eta_bin, phi_bin, keys, table, table_valid, xyz, rho, eta, phi, surface)
        return torch.nan_to_num(output).reshape(batch, length, -1).to(features.dtype)
