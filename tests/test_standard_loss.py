"""
Spec: Validates 'mlpf_loss' with 'LossType.STANDARD'. Tests classification (PID) and regression (momentum/energy) heads. Assertions: Verifies loss components are correctly masked for noise elements, checks stability with extreme/NaN values, and confirms that 'Regression_pt' and other components are zero for perfectly matching predictions.
"""

import torch
from mlpf.conf import RegressionLossWeights
from mlpf.model.losses import REGRESSION_FEATURES, event_loss, mlpf_loss, particle_loss
from mlpf.model.PFDataset import PFBatch


REGRESSION_WEIGHTS = RegressionLossWeights().model_dump()


def get_mock_data(batch_size=2, seq_len=10, num_classes=6):
    X = torch.randn(batch_size, seq_len, 20)
    # Mask some elements to 0 to simulate padding
    if batch_size > 0:
        X[0, seq_len // 2 :, 0] = 0
    if batch_size > 1:
        X[1, seq_len - 1 :, 0] = 0

    batch = PFBatch(X=X)

    y = {
        "cls_id": torch.randint(0, num_classes, (batch_size, seq_len)),
        "pt": torch.randn(batch_size, seq_len),
        "eta": torch.randn(batch_size, seq_len),
        "sin_phi": torch.randn(batch_size, seq_len),
        "cos_phi": torch.randn(batch_size, seq_len),
        "energy": torch.randn(batch_size, seq_len),
        "particle_number": torch.randint(0, 5, (batch_size, seq_len)),
        "momentum": torch.randn(batch_size, seq_len, 5),
    }

    ypred = {
        "cls_binary": torch.randn(batch_size, seq_len, 2),
        "cls_id_onehot": torch.randn(batch_size, seq_len, num_classes),
        "pt": torch.randn(batch_size, seq_len),
        "eta": torch.randn(batch_size, seq_len),
        "sin_phi": torch.randn(batch_size, seq_len),
        "cos_phi": torch.randn(batch_size, seq_len),
        "energy": torch.randn(batch_size, seq_len),
        "oc_beta": torch.rand(batch_size, seq_len, 1),
        "oc_coords": torch.randn(batch_size, seq_len, 3),
        "momentum": torch.randn(batch_size, seq_len, 5),
    }
    return batch, y, ypred


def test_mlpf_loss_standard():
    batch, y, ypred = get_mock_data()
    loss_opt, losses = mlpf_loss(y, ypred, batch, REGRESSION_WEIGHTS)

    assert "Total" in losses
    assert not torch.isnan(loss_opt)
    print("Standard loss basic test passed")


def test_mlpf_loss_standard_no_particles():
    batch, y, ypred = get_mock_data()
    y["cls_id"][:] = 0

    loss_opt, losses = mlpf_loss(y, ypred, batch, REGRESSION_WEIGHTS)

    assert "Total" in losses
    assert not torch.isnan(loss_opt)
    assert losses["Regression_pt"] == 0
    print("Standard loss no particles test passed")


def test_mlpf_loss_standard_single_particle():
    batch, y, ypred = get_mock_data(batch_size=1, seq_len=5)
    y["cls_id"][:] = 0
    y["cls_id"][0, 0] = 1

    loss_opt, losses = mlpf_loss(y, ypred, batch, REGRESSION_WEIGHTS)

    assert "Total" in losses
    assert not torch.isnan(loss_opt)
    assert losses["Regression_pt"] >= 0
    print("Standard loss single particle test passed")


def test_mlpf_loss_standard_large_batch():
    batch, y, ypred = get_mock_data(batch_size=8, seq_len=50)
    loss_opt, losses = mlpf_loss(y, ypred, batch, REGRESSION_WEIGHTS)

    assert "Total" in losses
    assert not torch.isnan(loss_opt)
    print("Standard loss large batch test passed")


def test_mlpf_loss_standard_perfect_prediction():
    batch, y, ypred = get_mock_data(batch_size=1, seq_len=5)
    # One particle
    y["cls_id"][:] = 0
    y["cls_id"][0, 0] = 1

    # Perfect ypred
    ypred["cls_binary"][:] = -10.0
    ypred["cls_binary"][0, 0, 1] = 10.0  # element 0 is particle
    ypred["cls_binary"][0, 1:, 0] = 10.0  # others are not

    ypred["cls_id_onehot"][:] = -10.0
    ypred["cls_id_onehot"][0, 0, 1] = 10.0  # class 1

    for k in ["pt", "eta", "sin_phi", "cos_phi", "energy"]:
        ypred[k][0, 0] = y[k][0, 0]

    loss_opt, losses = mlpf_loss(y, ypred, batch, REGRESSION_WEIGHTS)

    assert "Total" in losses
    assert not torch.isnan(loss_opt)
    # Regression losses should be 0
    assert torch.allclose(losses["Regression_pt"], torch.tensor(0.0), atol=1e-5)
    print("Standard loss perfect prediction test passed")


def test_mlpf_loss_standard_stability():
    batch, y, ypred = get_mock_data(batch_size=2, seq_len=10)

    ypred["pt"][0, 0] = 1e5
    ypred["energy"][1, 1] = torch.nan

    loss_opt, losses = mlpf_loss(y, ypred, batch, REGRESSION_WEIGHTS)

    assert not torch.isnan(loss_opt)
    assert not torch.isnan(losses["Regression_energy"])
    print("Standard loss stability test passed")


def test_mlpf_loss_standard_masks_no_target_without_mutating_inputs():
    batch, y, ypred = get_mock_data()
    y["cls_id"][:] = 0
    y["cls_id"][0, 0] = 1

    is_no_target = y["cls_id"] == 0
    y["pt"][is_no_target] = torch.nan
    ypred["eta"][is_no_target] = torch.nan
    y_pt_before = y["pt"].clone()
    ypred_eta_before = ypred["eta"].clone()

    loss_opt, losses = mlpf_loss(y, ypred, batch, REGRESSION_WEIGHTS)

    assert not torch.isnan(loss_opt)
    torch.testing.assert_close(y["pt"], y_pt_before, equal_nan=True)
    torch.testing.assert_close(ypred["eta"], ypred_eta_before, equal_nan=True)
    print("No-target regression masking test passed")


def test_event_loss_decomposes_into_particle_loss():
    batch, y, ypred = get_mock_data()
    valid = batch.mask.bool()
    is_no_target = y["cls_id"] == 0

    particle_targets = {"cls_id": y["cls_id"][valid]}
    particle_predictions = {
        "cls_binary": ypred["cls_binary"][valid],
        "cls_id_onehot": ypred["cls_id_onehot"][valid],
    }
    for feature in REGRESSION_FEATURES:
        target = torch.where(is_no_target, torch.zeros_like(y[feature]), y[feature])
        prediction = torch.where(is_no_target, torch.zeros_like(ypred[feature]), ypred[feature])
        particle_targets[feature] = target[valid]
        particle_predictions[feature] = prediction[valid]

    expected = particle_loss(particle_targets, particle_predictions, batch.X[..., 1][valid], REGRESSION_WEIGHTS)
    actual = event_loss(y, ypred, batch, REGRESSION_WEIGHTS)

    assert actual.keys() == expected.keys()
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key])


def test_regression_loss_weights_are_applied():
    batch, y, ypred = get_mock_data()
    baseline = event_loss(y, ypred, batch, REGRESSION_WEIGHTS)
    doubled_eta_weights = {**REGRESSION_WEIGHTS, "eta": 2 * REGRESSION_WEIGHTS["eta"]}
    reweighted = event_loss(y, ypred, batch, doubled_eta_weights)

    torch.testing.assert_close(reweighted["Regression_eta"], 2 * baseline["Regression_eta"])
    for feature in set(REGRESSION_FEATURES) - {"eta"}:
        torch.testing.assert_close(reweighted[f"Regression_{feature}"], baseline[f"Regression_{feature}"])


if __name__ == "__main__":
    test_mlpf_loss_standard()
    test_mlpf_loss_standard_no_particles()
    test_mlpf_loss_standard_single_particle()
    test_mlpf_loss_standard_large_batch()
    test_mlpf_loss_standard_perfect_prediction()
    test_mlpf_loss_standard_stability()
    test_mlpf_loss_standard_masks_no_target_without_mutating_inputs()
    test_event_loss_decomposes_into_particle_loss()
    test_regression_loss_weights_are_applied()
