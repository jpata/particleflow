import torch

from mlpf.conf import MLPFConfig
from mlpf.model.mlpf import CalorimeterNeighborhoodFeatures, HitFeatureEngineering, MLPF


def make_hit_input():
    features = torch.zeros(1, 3, 12)

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

    assert output.shape == (1, 3, 12 + HitFeatureEngineering().num_output_features)
    torch.testing.assert_close(output[..., :12], features)
    torch.testing.assert_close(engineered[0, 0, 3:7], torch.tensor([5.0 / 3000.0, 13.0 / 3000.0, 5.0 / 13.0, 12.0 / 13.0]))
    torch.testing.assert_close(engineered[0, 0, 8:10], torch.tensor([120.0, 160.0]).clamp(max=4.0))
    torch.testing.assert_close(engineered[0, 1, 8:10], torch.zeros(2))
    torch.testing.assert_close(engineered[0, 0, 11:15], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(engineered[0, 1, 11:15], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(engineered[0, 2], torch.zeros(HitFeatureEngineering().num_output_features))


def test_multiscale_calorimeter_reductions():
    features = torch.zeros(1, 4, 12)
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


def make_model_config(dataset):
    return MLPFConfig.model_validate(
        {
            "dataset": dataset,
            "data_dir": "/tmp",
            "model": {
                "type": "attention",
                "input_encoding": "split",
                "attention": {"num_convs": 1, "num_heads": 2, "head_dim": 4, "attention_type": "simple"},
            },
            "conv_type": "attention",
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

    assert model.raw_input_dim == config.input_dim == 12
    assert model.input_dim == 12 + model.feature_engineering.num_output_features
    assert encoder_inputs[0].shape[-1] == model.input_dim
    assert all(output.shape[1] == features.shape[1] for output in outputs)


def test_non_hit_model_does_not_engineer_features():
    config = make_model_config("cld")
    model = MLPF(config)

    assert not model.uses_hit_feature_engineering
    assert model.raw_input_dim == model.input_dim == config.input_dim
