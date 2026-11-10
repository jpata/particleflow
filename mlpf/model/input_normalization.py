"""Per-dataset standardization of the track/cluster input columns.

Each column is clipped to CLIP_BOUNDS, transformed, either x -> log1p(x / scale) or x -> x depending
on LOG1P_SCALES, then standardized with the mean and std of its transformed values in the dataset
version it comes from. These statistics are committed in input_normalization_stats/<dataset>_<version>.yaml,
written by scripts/compute_input_normalization_stats.py with the name and transform of each column.
"""

import torch
import torch.nn as nn

from mlpf.conf import ELEMENT_TYPES, SOURCE_IDS, load_stats


def build_normalization_tables(input_dim, versions):
    """[num source ids, 3 element types, input_dim] clip bounds, log1p flag and scale, mean and inverse std of each column.

    versions: {dataset: version of its statistics}. Other datasets are left unchanged.
    """
    shape = (max(SOURCE_IDS.values()) + 1, 3, input_dim)
    low = torch.full(shape, -torch.inf)
    high = torch.full(shape, torch.inf)
    is_log1p = torch.zeros(shape, dtype=torch.bool)
    scale = torch.ones(shape)
    mean = torch.zeros(shape)
    inv_std = torch.ones(shape)
    for dataset, version in versions.items():
        stats = load_stats(dataset, version)
        for elemtype, name in ELEMENT_TYPES.items():
            for column in stats[name]:
                index = (SOURCE_IDS[dataset], elemtype, column["column"])
                if column["clip"] is not None:
                    low[index], high[index] = column["clip"]
                if column["log1p_scale"] is not None:
                    is_log1p[index], scale[index] = True, column["log1p_scale"]
                mean[index] = column["mean"]
                inv_std[index] = 1.0 / column["std"] if column["std"] > 0 else 0.0
    return low, high, is_log1p, scale, mean, inv_std


class InputNormalization(nn.Module):
    """Normalize [B, N, input_dim] features with the statistics of each event's dataset."""

    def __init__(self, input_dim, versions):
        super().__init__()
        for name, table in zip(("low", "high", "is_log1p", "scale", "mean", "inv_std"), build_normalization_tables(input_dim, versions)):
            self.register_buffer(name, table, persistent=False)

    def forward(self, X_features, source_id):
        if source_id is None:
            raise ValueError("Input normalization needs each event's source_id")
        source = source_id.to(device=X_features.device, dtype=torch.long).view(-1, 1).expand(X_features.shape[:2])
        index = (source, X_features[..., 0].long().clamp(min=0, max=2))
        x = torch.clamp(X_features.to(torch.float32), self.low[index], self.high[index])
        x = torch.where(self.is_log1p[index], torch.log1p(x / self.scale[index]), x)
        return ((x - self.mean[index]) * self.inv_std[index]).to(X_features.dtype)
