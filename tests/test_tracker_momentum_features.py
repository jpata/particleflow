import math

import pytest
import torch

from mlpf.conf import MLPFConfig
from mlpf.model.mlpf import HitFeatureEngineering, TrackerNeighborhoodFeatures
from mlpf.model.tracker_features import TrackerDoubletFeatures


def circle_hits(radius=20000.0, rotation=0.02, n=8, z_slope=0.0, z0=0.0):
    """Circle through the origin; transverse positions and lengths in mm."""
    rho = torch.linspace(200.0, 1000.0, n)
    phi = rotation + torch.asin(rho / (2 * radius))
    x = torch.zeros(1, n, 15)
    x[0, :, 0] = 1
    x[0, :, 1] = 1
    x[0, :, 3] = phi.sin()
    x[0, :, 4] = phi.cos()
    x[0, :, 5] = 1
    x[0, :, 6] = rho * phi.cos()
    x[0, :, 7] = rho * phi.sin()
    x[0, :, 8] = z_slope * rho + z0
    x[0, :, 2] = torch.asinh(x[0, :, 8] / rho)
    x[0, :, 10] = 3
    x[0, :, 12] = 3
    x[0, :, 14] = torch.arange(n)
    return x


def test_prompt_curvature_and_longitudinal_moments():
    x = circle_hits(z_slope=0.35, z0=1.0)
    block = TrackerNeighborhoodFeatures(momentum=True)
    y = block(x)
    idx = {name: i for i, name in enumerate(block.OUTPUT_FEATURE_NAMES)}
    for name, expected in [
        ("curvature_asinh", math.asinh(1 / 20)),
        ("rz_slope_asinh", math.asinh(0.35)),
        ("rz_intercept_norm", 1 / 3000),
        ("curvature_valid", 1.0),
        ("rz_valid", 1.0),
    ]:
        torch.testing.assert_close(y[0, :, idx[f"tracker_momentum_large_{name}"]], torch.full((8,), expected), atol=2e-5, rtol=2e-3)
    assert y[0, :, idx["tracker_momentum_large_rz_residual_norm"]].max() < 1e-4


def test_smooth_conformal_features_preserve_inner_hit_information():
    x = circle_hits(n=2)
    x[0, :, 6:9] = torch.tensor([[16.0, 0, 0], [32.0, 0, 0]])
    block = TrackerNeighborhoodFeatures(momentum=True)
    y = block(x)
    index = block.OUTPUT_FEATURE_NAMES.index("tracker_conformal_u_asinh")
    torch.testing.assert_close(y[0, :, index], torch.asinh(torch.tensor([100 / 16, 100 / 32])))
    assert y[0, 0, index] != y[0, 1, index]


@pytest.mark.parametrize("rotation", [0.02, math.pi - 0.02])
@pytest.mark.parametrize("radius", [20000.0, -20000.0])
def test_doublet_curvature_sign_longitudinal_slope_and_phi_wrap(rotation, radius):
    x = circle_hits(radius=radius, rotation=rotation, z_slope=0.35, z0=1.0)
    block = TrackerDoubletFeatures()
    y = block(x)
    idx = {name: i for i, name in enumerate(block.OUTPUT_FEATURE_NAMES)}
    for direction, selected in [("inner", slice(1, None)), ("outer", slice(None, -1))]:
        valid = y[0, selected, idx[f"tracker_doublet_{direction}_valid"]]
        assert (valid == 1).all()
        torch.testing.assert_close(
            y[0, selected, idx[f"tracker_doublet_{direction}_curvature_asinh"]],
            torch.full_like(valid, math.asinh(1000 / radius)),
            atol=2e-5,
            rtol=2e-3,
        )
        torch.testing.assert_close(
            y[0, selected, idx[f"tracker_doublet_{direction}_rz_slope_asinh"]],
            torch.full_like(valid, math.asinh(0.35)),
            atol=2e-5,
            rtol=2e-3,
        )


@pytest.mark.parametrize("block", [TrackerNeighborhoodFeatures(momentum=True), TrackerDoubletFeatures()])
def test_isolation_padding_permutation_and_degenerate_hits(block):
    x = circle_hits(n=40)
    # Exercise candidate caps and equal-radius tie breaking.
    x[0, 10:20, 6:9] = x[0, :10, 6:9]
    x[0, 10:20, 2:5] = x[0, :10, 2:5]
    baseline = block(x)
    order = torch.randperm(x.shape[1])
    torch.testing.assert_close(block(x[:, order])[:, order.argsort()], baseline, atol=5e-4, rtol=2e-3)
    second = circle_hits(radius=-20000.0, n=40)
    torch.testing.assert_close(block(torch.cat([x, second]))[:1], baseline, atol=5e-4, rtol=2e-3)
    padded = torch.nn.functional.pad(x, (0, 0, 0, 3))
    padded[0, -1] = x[0, 0]
    padded[0, -1, 0] = 2  # Calorimeter inputs never enter these summaries.
    output = block(padded, padded[..., 0] != 0)
    torch.testing.assert_close(output[:, :40], baseline, atol=5e-4, rtol=2e-3)
    assert (output[:, 40:] == 0).all()
    assert torch.isfinite(block(torch.zeros_like(x))).all()
    single = block(x[:, :1])
    for i, name in enumerate(block.OUTPUT_FEATURE_NAMES):
        if name.endswith("_valid"):
            assert single[..., i].item() == 0


def test_doublets_do_not_pair_hits_on_same_surface():
    x = circle_hits()
    x[..., 14] = 1
    assert (TrackerDoubletFeatures()(x) == 0).all()


def test_feature_flags_and_dimensions():
    with pytest.raises(ValueError, match="tracker_momentum requires"):
        HitFeatureEngineering(tracker_momentum=True, tracker_neighborhood=False)
    base = HitFeatureEngineering()
    extended = HitFeatureEngineering(tracker_momentum=True, tracker_doublets=True)
    assert extended.num_output_features == base.num_output_features + 36
    x = circle_hits()
    assert extended(x).shape[-1] == x.shape[-1] + extended.num_output_features
    with pytest.raises(ValueError, match="tracker_momentum requires"):
        MLPFConfig.model_validate(
            {
                "dataset": "cld_hits",
                "data_dir": "/tmp",
                "conv_type": "heptv2",
                "model": {"type": "heptv2", "heptv2": {}, "hit_feature_engineering": {"tracker_momentum": True}},
            }
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_features_match_cpu_in_bfloat16_autocast():
    x = circle_hits(n=40)
    block = HitFeatureEngineering(tracker_momentum=True, tracker_doublets=True)
    expected = block(x)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = block.cuda()(x.cuda()).cpu()
    # Legacy circular-width features subtract nearly equal FP32 moments.
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=5e-4)
    added = [
        15 + i
        for i, name in enumerate(block.output_feature_names)
        if name.startswith(("tracker_momentum_", "tracker_conformal_", "tracker_doublet_"))
    ]
    torch.testing.assert_close(actual[..., added], expected[..., added], rtol=2e-3, atol=2e-5)


@pytest.mark.parametrize("radius", [20000.0, 1000000.0])
def test_curvature_reliability_and_high_momentum(radius):
    x = circle_hits(radius=radius)
    block = TrackerNeighborhoodFeatures(momentum=True)
    idx = {name: i for i, name in enumerate(block.OUTPUT_FEATURE_NAMES)}
    clean = block(x)
    expected = math.asinh(1000 / radius)
    torch.testing.assert_close(clean[0, :, idx["tracker_momentum_large_curvature_asinh"]], torch.full((8,), expected), atol=1e-6, rtol=2e-3)
    # A transverse perturbation degrades the line-quality measurement.
    perturbed = x.clone()
    perturbed[0, 3, 7] += 2.0
    noisy = block(perturbed)
    quality_index = idx["tracker_momentum_large_relative_transverse_rms"]
    assert noisy[0, :, quality_index].mean() > clean[0, :, quality_index].mean() + 1e-4
    x[..., 14] = 0
    assert (block(x)[..., idx["tracker_momentum_large_curvature_valid"]] == 0).all()
