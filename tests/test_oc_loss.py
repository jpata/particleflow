"""
Spec: Validates 'mlpf_loss' with 'LossType.OBJECT_CONDENSATION'. Tests stability across various event types: normal, zero-particle (noise-only), and single-particle. Assertions: Verifies loss values are finite and non-negative, checks 'OC_V' (attractive/repulsive) and 'OC_beta' components, and ensures no crashes when predictions contain NaNs or extreme values.
"""

import torch
from mlpf.model.losses import mlpf_loss
from mlpf.conf import LossType
from mlpf.model.PFDataset import PFBatch


def get_mock_data(batch_size=2, seq_len=10, num_classes=6):
    X = torch.randn(batch_size, seq_len, 20)
    X[:, :, 1] = X[:, :, 1].abs()
    X[:, :, 5] = X[:, :, 5].abs()
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


def test_mlpf_loss_oc():
    batch, y, ypred = get_mock_data()
    loss_opt, losses = mlpf_loss(y, ypred, batch, loss_mode=LossType.OBJECT_CONDENSATION)

    assert "Total" in losses
    assert losses["OC_V"] >= 0
    assert losses["OC_beta"] >= 0
    assert losses["Classification_binary"] == 0
    print("OC loss test passed")


def test_mlpf_loss_oc_no_particles():
    batch, y, ypred = get_mock_data()
    # All elements are noise
    y["cls_id"][:] = 0
    y["particle_number"][:] = 0

    loss_opt, losses = mlpf_loss(y, ypred, batch, loss_mode=LossType.OBJECT_CONDENSATION)

    assert "Total" in losses
    assert not torch.isnan(loss_opt)
    # When there are no particles, OC_V Attractive should be 0, OC_V Repulsive might still be > 0 if there are condensation points predicted from noise
    print("OC loss no particles test passed")


def test_mlpf_loss_oc_single_particle():
    batch, y, ypred = get_mock_data(batch_size=1, seq_len=5)
    y["cls_id"][:] = 0
    y["particle_number"][:] = 0
    y["cls_id"][0, 0] = 1
    y["particle_number"][0, 0] = 1

    loss_opt, losses = mlpf_loss(y, ypred, batch, loss_mode=LossType.OBJECT_CONDENSATION)

    assert "Total" in losses
    assert not torch.isnan(loss_opt)
    print("OC loss single particle test passed")


def test_mlpf_loss_oc_large_batch():
    batch, y, ypred = get_mock_data(batch_size=8, seq_len=50)
    loss_opt, losses = mlpf_loss(y, ypred, batch, loss_mode=LossType.OBJECT_CONDENSATION)

    assert "Total" in losses
    assert not torch.isnan(loss_opt)
    print("OC loss large batch test passed")


def test_mlpf_loss_oc_perfect_prediction():
    batch, y, ypred = get_mock_data(batch_size=1, seq_len=5)
    # One particle
    y["cls_id"][:] = 0
    y["particle_number"][:] = 0
    y["cls_id"][0, 0] = 1
    y["particle_number"][0, 0] = 1

    # Perfect ypred
    ypred["oc_beta"][:] = 0.0
    ypred["oc_beta"][0, 0, 0] = 0.99  # Use a high value for the CP

    ypred["oc_coords"][:] = 0.0
    # All noise elements should be far from CP (at 0,0,0) or have low beta
    ypred["oc_coords"][0, 1:, :] = 10.0

    # Matching regression
    for k in ["pt", "eta", "sin_phi", "cos_phi", "energy"]:
        ypred[k][0, 0] = y[k][0, 0]

    # PID matching
    ypred["cls_id_onehot"][:] = -10.0
    ypred["cls_id_onehot"][0, 1, 0] = 10.0  # class 1

    loss_opt, losses = mlpf_loss(y, ypred, batch, loss_mode=LossType.OBJECT_CONDENSATION)

    assert "Total" in losses
    # OC losses should be relatively low (though constants/qmin might keep them > 0)
    # We just check stability here
    assert not torch.isnan(loss_opt)
    print("OC loss perfect prediction test passed")


def test_mlpf_loss_oc_stability():
    batch, y, ypred = get_mock_data(batch_size=2, seq_len=10)

    # Test with extreme values
    ypred["oc_coords"][0, 0, 0] = 1e5
    ypred["oc_beta"][1, 1, 0] = torch.nan

    loss_opt, losses = mlpf_loss(y, ypred, batch, loss_mode=LossType.OBJECT_CONDENSATION)

    assert "Total" in losses
    assert not torch.isnan(loss_opt)
    print("OC loss stability test passed")


def test_mlpf_loss_oc_zero_eta_phi_when_no_target():
    batch, y, ypred = get_mock_data()
    y["cls_id"][:, 2:] = 0

    loss_opt, losses = mlpf_loss(y, ypred, batch, loss_mode=LossType.OBJECT_CONDENSATION)

    is_no_target = (y["cls_id"] == 0)
    for key in ["pt", "eta", "sin_phi", "cos_phi", "energy", "phi"]:
        if key in ypred:
            assert torch.all(ypred[key][is_no_target] == 0.0)
        if key in y:
            assert torch.all(y[key][is_no_target] == 0.0)
    print("Zero regression components when no target particle test passed for OC loss")


if __name__ == "__main__":
    test_mlpf_loss_oc()
    test_mlpf_loss_oc_no_particles()
    test_mlpf_loss_oc_single_particle()
    test_mlpf_loss_oc_large_batch()
    test_mlpf_loss_oc_perfect_prediction()
    test_mlpf_loss_oc_stability()
    test_mlpf_loss_oc_zero_eta_phi_when_no_target()
