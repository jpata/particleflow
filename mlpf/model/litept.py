import torch
import torch.nn as nn
from mlpf.logger import _logger

import sys
import os

# add the LitePT submodule to the path
litept_path = os.path.join(os.getcwd(), "LitePT")
if litept_path not in sys.path:
    sys.path.append(litept_path)

try:
    from litept.model import LitePT
except ImportError:
    LitePT = None


class LitePTLayer(nn.Module):
    def __init__(self, name, litept_config, embedding_dim):
        super(LitePTLayer, self).__init__()
        self.name = name
        self.embedding_dim = embedding_dim
        if LitePT is None:
            raise ImportError("LitePT is not available")

        # We need to remove MLPF-specific keys from litept_config before passing to LitePT
        config = litept_config.copy()
        self.coord_indices = config.pop("coord_indices", [2, 3, 4])
        self.grid_size = config.pop("grid_size", 0.01)

        # Remove keys that LitePT constructor doesn't expect
        for key in ["num_convs", "conv_type", "embedding_dim", "width", "activation", "dropout_ff"]:
            if key in config:
                config.pop(key)

        self.litept = LitePT(in_channels=embedding_dim, **config)

        # Check output dimension and add projection if necessary
        litept_out_dim = config["enc_channels"][-1] if config.get("enc_mode") else config["dec_channels"][0]
        if litept_out_dim != embedding_dim:
            self.output_proj = nn.Linear(litept_out_dim, embedding_dim)
        else:
            self.output_proj = nn.Identity()

    def forward(self, x, mask, X_features):
        B, S, D = x.shape
        mask_flat = mask.view(-1).bool()

        # Flatten and filter
        x_flat = x.view(-1, D)[mask_flat]
        coords_flat = X_features[..., self.coord_indices].reshape(-1, len(self.coord_indices))[mask_flat]
        batch = torch.arange(B, device=x.device).repeat_interleave(S)[mask_flat]

        _logger.debug(
            f"LitePTLayer {self.name} forward: B={B}, S={S}, D={D}, "
            + f"n_valid={x_flat.shape[0]}, "
            + f"coords_min={coords_flat.min(0)[0].tolist() if x_flat.shape[0] > 0 else 'N/A'}, "
            + f"coords_max={coords_flat.max(0)[0].tolist() if x_flat.shape[0] > 0 else 'N/A'}"
        )

        data = {
            "feat": x_flat.to(torch.float32),
            "coord": coords_flat.to(torch.float32),
            "batch": batch,
            "grid_size": self.grid_size,
        }

        # spconv can fail with bfloat16 during evaluation (tuner issue)
        # we force float32 for the LitePT part
        with torch.autocast(device_type=x.device.type, enabled=False):
            out = self.litept(data)

        feat = self.output_proj(out.feat).to(x.dtype)

        _logger.debug(f"LitePTLayer {self.name} output: feat_shape={feat.shape}")

        # out.feat shape is [N_valid, D_out]
        D_out = feat.shape[-1]
        res = torch.zeros(B, S, D_out, device=x.device, dtype=feat.dtype)
        res.view(-1, D_out)[mask_flat] = feat
        return res
