import torch

from mlpf.conf import EDM4HEP, MLPFConfig
from mlpf.model.mlpf import CalorimeterNeighborhoodFeatures, HitFeatureEngineering, MLPF, TrackerNeighborhoodFeatures


def make_hit_input():
    features = torch.zeros(1, 3, len(EDM4HEP.HitFeatures.get_names()))

    # Tracker hit at (3, 4, 12) mm with a 1 ns timestamp.
    features[0, 0, 0] = 1
    features[0, 0, 6:10] = torch.tensor([3.0, 4.0, 12.0, 1.0])
    features[0, 0, 10] = 3

    # ECAL hit with the same transverse position.
    features[0, 1, 0] = 2
    features[0, 1, 6:10] = torch.tensor([3.0, 4.0, 12.0, 1.0])
    features[0, 1, 10] = 0

    mask = torch.tensor([[True, True, False]])
    return features, mask


def test_hit_feature_engineering_values_and_padding():
    features, mask = make_hit_input()
    output = HitFeatureEngineering()(features, mask)
    engineered = output[..., features.shape[-1] :]

    assert output.shape == (1, 3, features.shape[-1] + HitFeatureEngineering().num_output_features)
    torch.testing.assert_close(output[..., : features.shape[-1]], features)
    torch.testing.assert_close(engineered[0, 0, 3:7], torch.tensor([5.0 / 3000.0, 13.0 / 3000.0, 5.0 / 13.0, 12.0 / 13.0]))
    torch.testing.assert_close(engineered[0, 0, 8:10], torch.tensor([120.0, 160.0]).clamp(max=4.0))
    torch.testing.assert_close(engineered[0, 1, 8:10], torch.zeros(2))
    torch.testing.assert_close(engineered[0, 0, 11:15], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(engineered[0, 1, 11:15], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(engineered[0, 2], torch.zeros(HitFeatureEngineering().num_output_features))


def test_multiscale_calorimeter_reductions():
    features = torch.zeros(1, 4, len(EDM4HEP.HitFeatures.get_names()))
    mask = torch.ones(1, 4, dtype=torch.bool)

    # Two hits share the finest angular bin and span ECAL/HCAL and depth.
    features[0, 0, [0, 2, 4, 5, 6, 9, 10]] = torch.tensor([2.0, 0.0, 1.0, 2.0, 2000.0, 7.0, 0.0])
    features[0, 1, [0, 2, 3, 4, 5, 6, 9, 10]] = torch.tensor([2.0, 0.01, -0.01, 1.0, 3.0, 2300.0, 8.0, 1.0])
    # A distant ECAL hit and a tracker hit must not contribute to that bin.
    features[0, 2, [0, 2, 4, 5, 6, 10]] = torch.tensor([2.0, 1.0, 1.0, 5.0, 2000.0, 0.0])
    features[0, 3, [0, 2, 4, 5, 6, 10]] = torch.tensor([1.0, 0.0, 1.0, 100.0, 2000.0, 3.0])

    layer = CalorimeterNeighborhoodFeatures()
    output = layer(features, mask)
    indices = {name: index for index, name in enumerate(layer.OUTPUT_FEATURE_NAMES)}

    torch.testing.assert_close(output[0, 0, indices["calo_small_count_log"]], torch.log1p(torch.tensor(2.0)))
    torch.testing.assert_close(output[0, 0, indices["calo_small_energy_sum_log"]], torch.log1p(torch.tensor(5.0)))
    torch.testing.assert_close(output[0, 0, indices["calo_small_hit_energy_fraction"]], torch.tensor(0.4))
    torch.testing.assert_close(output[0, 1, indices["calo_small_hit_energy_fraction"]], torch.tensor(0.6))
    torch.testing.assert_close(output[0, 0, indices["calo_small_ecal_energy_fraction"]], torch.tensor(0.4))
    torch.testing.assert_close(output[0, 0, indices["calo_small_hcal_energy_fraction"]], torch.tensor(0.6))
    torch.testing.assert_close(output[0, 0, indices["calo_small_early_energy_fraction"]], torch.tensor(0.4))
    torch.testing.assert_close(output[0, 0, indices["calo_small_is_energy_max"]], torch.tensor(0.0))
    torch.testing.assert_close(output[0, 1, indices["calo_small_is_energy_max"]], torch.tensor(1.0))
    torch.testing.assert_close(output[0, 3], torch.zeros(layer.NUM_OUTPUT_FEATURES))


def test_tracker_surface_and_cross_layer_tracklet_reductions():
    features = torch.zeros(1, 6, len(EDM4HEP.HitFeatures.get_names()))
    mask = torch.tensor([[True, True, True, True, True, False]])

    # Four tracker hits lie in one projective angular bin. The first two also
    # share an exact detector surface, while the other two extend the tracklet
    # through the inner and outer tracker systems.
    radii = torch.tensor([100.0, 100.0, 400.0, 1000.0])
    phis = torch.tensor([0.0, 0.001, 0.0, 0.0])
    systems = torch.tensor([1.0, 1.0, 3.0, 5.0])
    for index in range(4):
        features[0, index, 0] = 1
        features[0, index, 2] = 0
        features[0, index, 3] = torch.sin(phis[index])
        features[0, index, 4] = torch.cos(phis[index])
        features[0, index, 6] = radii[index] * torch.cos(phis[index])
        features[0, index, 7] = radii[index] * torch.sin(phis[index])
        features[0, index, 9] = radii[index] / TrackerNeighborhoodFeatures.SPEED_OF_LIGHT_MM_PER_NS
        features[0, index, 10] = 3
        features[0, index, 12] = systems[index]
        features[0, index, 13] = 0
        features[0, index, 14] = 0

    # A calorimeter hit and padding must not enter tracker reductions.
    features[0, 4, 0] = 2
    features[0, 4, 4] = 1
    features[0, 4, 6] = 100.0
    features[0, 4, 10] = 0
    features[0, 4, 12] = 20

    layer = TrackerNeighborhoodFeatures()
    output = layer(features, mask)
    indices = {name: index for index, name in enumerate(layer.OUTPUT_FEATURE_NAMES)}

    assert output.shape == (1, 6, layer.NUM_OUTPUT_FEATURES)
    torch.testing.assert_close(output[0, 0, indices["tracker_surface_small_count_log"]], torch.log1p(torch.tensor(2.0)))
    torch.testing.assert_close(output[0, 2, indices["tracker_surface_small_count_log"]], torch.log1p(torch.tensor(1.0)))
    torch.testing.assert_close(output[0, 0, indices["tracker_surface_small_is_isolated"]], torch.tensor(0.0))
    torch.testing.assert_close(output[0, 2, indices["tracker_surface_small_is_isolated"]], torch.tensor(1.0))
    torch.testing.assert_close(output[0, 0, indices["tracker_tracklet_small_count_log"]], torch.log1p(torch.tensor(4.0)))
    torch.testing.assert_close(
        output[0, 0, indices["tracker_tracklet_small_distinct_surface_count_log"]], torch.log1p(torch.tensor(3.0))
    )
    torch.testing.assert_close(output[0, 0, indices["tracker_tracklet_small_path_span"]], torch.tensor(0.3))
    torch.testing.assert_close(output[0, 2, indices["tracker_tracklet_small_path_rank"]], torch.tensor(1.0 / 3.0))
    torch.testing.assert_close(output[0, 0, indices["tracker_tracklet_small_vxd_fraction"]], torch.tensor(0.5))
    torch.testing.assert_close(output[0, 0, indices["tracker_tracklet_small_inner_fraction"]], torch.tensor(0.25))
    torch.testing.assert_close(output[0, 0, indices["tracker_tracklet_small_outer_fraction"]], torch.tensor(0.25))
    assert output[0, 0, indices["tracker_tracklet_small_conformal_linearity"]] > 0.99
    torch.testing.assert_close(output[0, 4], torch.zeros(layer.NUM_OUTPUT_FEATURES))
    torch.testing.assert_close(output[0, 5], torch.zeros(layer.NUM_OUTPUT_FEATURES))


def make_model_config(dataset, hit_feature_engineering=None, input_dim=None):
    model = {
        "type": "attention",
        "input_encoding": "split",
        "attention": {"num_convs": 1, "num_heads": 2, "head_dim": 4, "attention_type": "simple"},
    }
    if hit_feature_engineering is not None:
        model["hit_feature_engineering"] = hit_feature_engineering
    return MLPFConfig.model_validate(
        {
            "dataset": dataset,
            "data_dir": "/tmp",
            "model": model,
            "conv_type": "attention",
            "input_dim": input_dim,
        }
    )


def test_hit_model_engineers_features_before_input_encoder():
    config = make_model_config("cld_hits")
    model = MLPF(config).eval()
    features, mask = make_hit_input()
    features[..., 1] = 1.0
    features[..., 5] = 1.0
    encoder_inputs = []
    hook = model.nn0.register_forward_pre_hook(lambda _module, inputs: encoder_inputs.append(inputs[0]))

    with torch.no_grad():
        outputs = model(features, mask)
    hook.remove()

    assert model.raw_input_dim == config.input_dim == len(EDM4HEP.HitFeatures.get_names())
    assert model.input_dim == config.input_dim + model.feature_engineering.num_output_features
    assert encoder_inputs[0].shape[-1] == model.input_dim
    assert all(output.shape[1] == features.shape[1] for output in outputs)


def test_non_hit_model_does_not_engineer_features():
    config = make_model_config("cld")
    model = MLPF(config)

    assert not model.uses_hit_feature_engineering
    assert model.raw_input_dim == model.input_dim == config.input_dim


def test_hit_model_reconstructs_legacy_elemtype_from_subdetector():
    config = make_model_config("cld_hits", hit_feature_engineering={"enabled": False})
    model = MLPF(config)
    features, mask = make_hit_input()
    features[..., 0] = 2  # TFDS 3.2.0 postprocessing mistake

    corrected = model._engineer_input_features(features, mask)

    torch.testing.assert_close(corrected[0, :2, 0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(features[..., 0], torch.full_like(features[..., 0], 2))


def test_hit_feature_blocks_are_independently_toggleable():
    features, mask = make_hit_input()
    cases = [
        (HitFeatureEngineering(tracker_neighborhood=False, calorimeter_neighborhood=False), 15),
        (HitFeatureEngineering(geometry=False, calorimeter_neighborhood=False), 53),
        (HitFeatureEngineering(geometry=False, tracker_neighborhood=False), 71),
        (HitFeatureEngineering(geometry=False, tracker_neighborhood=False, calorimeter_neighborhood=False), 0),
    ]

    for layer, expected_features in cases:
        output = layer(features, mask)
        assert layer.num_output_features == expected_features
        assert len(layer.output_feature_names) == expected_features
        assert output.shape[-1] == features.shape[-1] + expected_features
        if expected_features == 0:
            torch.testing.assert_close(output, features)


def test_geometry_only_model_supports_legacy_checkpoint_input_dimension():
    config = make_model_config(
        "cld_hits",
        hit_feature_engineering={"tracker_neighborhood": False, "calorimeter_neighborhood": False},
        input_dim=12,
    )
    model = MLPF(config)

    assert model.uses_hit_feature_engineering
    assert model.raw_input_dim == 12
    assert model.input_dim == 27
    assert model.nn0[0].in_features == 27


def test_hit_feature_engineering_can_be_disabled():
    config = make_model_config("cld_hits", hit_feature_engineering={"enabled": False})
    model = MLPF(config)

    assert not model.uses_hit_feature_engineering
    assert model.raw_input_dim == model.input_dim == config.input_dim
