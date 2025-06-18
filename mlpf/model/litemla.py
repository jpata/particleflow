import torch
from typing import Any, Optional
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from functools import partial
from inspect import signature
import math # Added for math.pi
from typing import Any, Callable, Optional
from mlpf.model.ElementBinner import ElementBinner # Assuming ElementBinner is accessible

def val2tuple(x: list | tuple | Any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)

def val2list(x: list | tuple | Any, repeat_time=1) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]

def build_kwargs_from_config(config: dict, target_func: Callable) -> dict[str, Any]:
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key in config:
        if key in valid_keys:
            kwargs[key] = config[key]
    return kwargs


def get_same_padding(kernel_size: int | tuple[int, ...]) -> int | tuple[int, ...]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2

REGISTERED_NORM_DICT: dict[str, type] = {
    "bn2d": nn.BatchNorm2d,
    "ln": nn.LayerNorm,
}
        
def build_norm(name="bn2d", num_features=None, **kwargs) -> Optional[nn.Module]:
    if name in ["ln", "ln2d", "trms2d"]:
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        args = build_kwargs_from_config(kwargs, norm_cls)
        return norm_cls(**args)
    else:
        return None

# register activation function here
REGISTERED_ACT_DICT: dict[str, type] = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
    "silu": nn.SiLU,
    "gelu": partial(nn.GELU, approximate="tanh"),
}


def build_act(name: str, **kwargs) -> Optional[nn.Module]:
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        args = build_kwargs_from_config(kwargs, act_cls)
        return act_cls(**args)
    else:
        return None


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        dropout=0,
        norm="bn2d",
        act_func="relu",
    ):
        super(ConvLayer, self).__init__()

        padding = get_same_padding(kernel_size)
        padding *= dilation

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class LiteMLA(nn.Module):
    r"""Lightweight multi-scale linear attention"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: Optional[int] = None,
        heads_ratio: float = 1.0,
        dim=8,
        use_bias=False,
        norm=(None, "bn2d"),
        act_func=(None, None),
        kernel_func="relu",
        scales: tuple[int, ...] = (5,),
        eps=1.0e-15,
        # ElementBinner parameters with defaults
        eta_bin_edges: list[float] = list(np.linspace(-5, 5, 10)),
        phi_bin_edges: list[float] = list(np.linspace(-math.pi, math.pi, 10)),
        max_elems_per_bin: int = 10,
        bin_agg_mlp_layer_dims: Optional[list[int]] = [32,32],
    ):
        super(LiteMLA, self).__init__()
        self.eps = eps
        heads = int(in_channels // dim * heads_ratio) if heads is None else heads

        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.binner = ElementBinner(
            eta_bin_edges=eta_bin_edges,
            phi_bin_edges=phi_bin_edges,
            max_elems_per_bin=max_elems_per_bin,
        )

        self.bin_pooling_mlp = None
        qkv_input_channels = in_channels # This is F_in_orig (feature dim of elements)

        if bin_agg_mlp_layer_dims and len(bin_agg_mlp_layer_dims) > 0:
            mlp_layers = []
            current_mlp_in_dim = self.binner.max_elems_per_bin * in_channels
            
            for i, layer_out_dim in enumerate(bin_agg_mlp_layer_dims):
                mlp_layers.append(nn.Linear(current_mlp_in_dim, layer_out_dim))
                if i < len(bin_agg_mlp_layer_dims) - 1: # Add ReLU for all but last linear layer
                    mlp_layers.append(nn.ReLU()) 
                current_mlp_in_dim = layer_out_dim
            self.bin_pooling_mlp = nn.Sequential(*mlp_layers)
            qkv_input_channels = current_mlp_in_dim # This is bin_agg_mlp_layer_dims[-1]

        # Calculate heads based on the actual input dimension to the QKV projection
        effective_in_channels_for_heads = qkv_input_channels
        _heads = heads 
        if _heads is None:
            _heads = int(effective_in_channels_for_heads / dim * heads_ratio)
            _heads = max(1, _heads) 
        
        total_dim = _heads * dim

        self.dim = dim
        self.qkv = ConvLayer(
            qkv_input_channels, # Use the potentially modified input channel size for QKV
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        scale,
                        padding=get_same_padding(scale),
                        groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        self.kernel_func = build_act(kernel_func, inplace=False)

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )
        
    @torch.autocast(device_type="cuda", enabled=False) # Keep for relu_linear_att if needed for specific dtype
    def relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(qkv.size())

        if qkv.dtype == torch.float16:
            qkv = qkv.float()

        qkv = torch.reshape(
            qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        q, k, v = (
            qkv[:, :, 0 : self.dim],
            qkv[:, :, self.dim : 2 * self.dim],
            qkv[:, :, 2 * self.dim :],
        )

        # lightweight linear attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        # linear matmul
        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)
        vk = torch.matmul(v, trans_k)
        out = torch.matmul(vk, q)
        if out.dtype == torch.bfloat16:
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)

        out = torch.reshape(out, (B, -1, H, W))
        return out
        
    def forward(self, x: torch.Tensor, x_coords: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor: # Output is now (B, N_eta, N_phi, F_out)
        """
        Args:
            x (torch.Tensor): Input features, shape (batch, num_elems, input_feature_dim).
            x_coords (torch.Tensor): Coordinates for binning (eta, sin_phi, cos_phi),
                                     shape (batch, num_elems, 3).
            mask (torch.Tensor, optional): Original mask for x, shape (batch, num_elems).
                                           If None, all elements are considered valid.

        Returns:
            torch.Tensor: Processed features, unbinned to original element shape (batch, num_elems, out_channels).
        """
        B_orig, N_orig, F_in_orig = x.shape
        device_orig = x.device
        dtype_orig = x.dtype

        original_mask = mask if mask is not None else torch.ones(B_orig, N_orig, dtype=torch.bool, device=device_orig)
        
        binned_x, binned_mask, (final_batch_indices_kept, final_element_indices_in_N_kept) = self.binner(x, x_coords, original_mask)
        # binned_x shape: (B, N_eta, N_phi, N_per_bin, F_in)
        # binned_mask shape: (B, N_eta, N_phi, N_per_bin)

        B, N_eta, N_phi, N_per_bin, F_in_binned = binned_x.shape # F_in_binned is F_in_orig

        if self.bin_pooling_mlp is not None:
            # Learnable aggregation using MLP
            # Apply mask to zero out features of non-existent elements before flattening
            masked_binned_x = binned_x * binned_mask.unsqueeze(-1)  # (B, N_eta, N_phi, N_per_bin, F_in_binned)
            
            # Flatten the N_per_bin and F_in_binned dimensions
            # Input to MLP is (B, N_eta, N_phi, N_per_bin * F_in_binned)
            flattened_bin_features = masked_binned_x.reshape(
                B, N_eta, N_phi, self.binner.max_elems_per_bin * F_in_binned
            )
            
            # Reshape for MLP: (B * N_eta * N_phi, N_per_bin * F_in_binned)
            num_spatial_bins = B * N_eta * N_phi
            reshaped_for_mlp = flattened_bin_features.reshape(num_spatial_bins, self.binner.max_elems_per_bin * F_in_binned)
            
            aggregated_bin_features_flat = self.bin_pooling_mlp(reshaped_for_mlp) # (B*N_eta*N_phi, mlp_output_dim)
            
            # Reshape back: (B, N_eta, N_phi, mlp_output_dim)
            # mlp_output_dim is self.bin_agg_mlp_layer_dims[-1]
            aggregated_bin_features = aggregated_bin_features_flat.reshape(
                B, N_eta, N_phi, -1 
            )
        else:
            # Original sum/mean aggregation
            binned_mask_expanded = binned_mask.unsqueeze(-1)
            masked_binned_x = binned_x * binned_mask_expanded
            sum_features_per_bin = masked_binned_x.sum(dim=3) 
            
            count_elements_per_bin = binned_mask.sum(dim=3).unsqueeze(-1)
            aggregated_bin_features = sum_features_per_bin / (count_elements_per_bin + self.eps)
            
        # Aggregate features within bins; MLA core then processes this (eta,phi) grid of aggregated bin features.
        # Reshape for convolutional processing: (B, N_eta, N_phi, F_in) -> (B, F_in, N_eta, N_phi)
        spatial_input_to_mla = aggregated_bin_features.permute(0, 3, 1, 2).contiguous()

        # Core MLA operations
        qkv = self.qkv(spatial_input_to_mla) # Input: (B, F_in, N_eta, N_phi)

        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        qkv_aggregated = torch.cat(multi_scale_qkv, dim=1)

        # relu_linear_att performs attention over the N_eta, N_phi spatial dimensions
        out_attention = self.relu_linear_att(qkv_aggregated).to(qkv_aggregated.dtype)
        # out_attention shape: (B, total_dim * (1 + len(scales)), N_eta, N_phi)

        processed_bin_conv_features = self.proj(out_attention) # (B, out_channels, N_eta, N_phi)

        # Permute to (B, N_eta, N_phi, out_channels) for easier unbinning
        processed_bin_features_map = processed_bin_conv_features.permute(0, 2, 3, 1).contiguous()

        # --- Unbin the processed per-bin features back to original elements ---
        # Re-calculate eta and phi bin indices for all original elements (respecting x_coords, not original_mask yet)
        eta_vals_orig = x_coords[..., 0]       # (B_orig, N_orig)
        sin_phi_vals_orig = x_coords[..., 1]   # (B_orig, N_orig)
        cos_phi_vals_orig = x_coords[..., 2]   # (B_orig, N_orig)
        phi_vals_orig = torch.atan2(sin_phi_vals_orig, cos_phi_vals_orig) # (B_orig, N_orig)

        eta_bin_indices_orig = torch.searchsorted(self.binner.eta_search_edges, eta_vals_orig.contiguous(), right=False)
        phi_bin_indices_orig = torch.searchsorted(self.binner.phi_search_edges, phi_vals_orig.contiguous(), right=False)

        batch_indices_for_gathering = torch.arange(B_orig, device=device_orig).view(B_orig, 1).expand_as(eta_bin_indices_orig)

        # Create an empty tensor for the unbinned output
        unbinned_output = torch.zeros(B_orig, N_orig, self.proj.conv.out_channels, device=device_orig, dtype=dtype_orig)

        # Gather features for original elements based on their bin assignment
        # This effectively assigns the feature of the bin to all elements belonging to that bin
        unbinned_output = processed_bin_features_map[batch_indices_for_gathering, eta_bin_indices_orig, phi_bin_indices_orig]

        # Apply the original element mask
        unbinned_output = unbinned_output * original_mask.unsqueeze(-1)
        
        return unbinned_output