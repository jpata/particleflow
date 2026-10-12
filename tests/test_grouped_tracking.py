import pytest
import torch

from mlpf.model.PFDataset import PFBatch
from mlpf.model.set_losses import event_consistency_loss, hit_grouping_loss
from mlpf.model.tracker_group_readout import tracker_group_geometry


def test_detector_balancing_is_invariant_to_calo_hit_replication():
    def loss(repeats, balanced):
        # Tracker assigns correctly; calorimeter is deliberately wrong.
        x = torch.ones(1, 1 + repeats, 15)
        x[0, 1:, 0] = 2
        truth = torch.zeros(1, 1 + repeats, 14)
        truth[..., 13] = 1
        targets = torch.zeros(1, 1, 14)
        targets[..., 0] = 1
        targets[..., 13] = 1
        batch = PFBatch(X=x, ytarget=truth, ytarget_set=targets)
        keys = torch.tensor([[[1.0, 0.0]] + [[0.0, 1.0]] * repeats], requires_grad=True)
        slots = torch.eye(2)[None].requires_grad_()
        bg = torch.full((1, 1 + repeats, 1), -10.0, requires_grad=True)
        value = hit_grouping_loss((keys, slots, bg), batch, [(torch.tensor([0]), torch.tensor([0]))], detector_balanced=balanced, chunk_size=2)
        value.backward()
        assert torch.isfinite(keys.grad).all() and torch.isfinite(slots.grad).all()
        return value.item()

    assert loss(1, True) == pytest.approx(loss(9, True))
    assert loss(9, False) > loss(1, False)


def test_event_consistency_uses_true_visible_sum_and_presence_gradients():
    truth = {
        "pt": torch.tensor([[10.0]]).log(),
        "energy": torch.tensor([[12.0]]).log(),
        "eta": torch.tensor([[1.0]]),
        "sin_phi": torch.tensor([[0.0]]),
        "cos_phi": torch.tensor([[1.0]]),
    }
    pred = {key: val.clone().requires_grad_() for key, val in truth.items()}
    pred["cls_binary"] = torch.tensor([[[-100.0, 100.0]]], requires_grad=True)
    mask = torch.ones(1, 1, dtype=torch.bool)
    assert event_consistency_loss(truth, pred, mask).item() == pytest.approx(0, abs=1e-12)
    pred["cls_binary"] = torch.zeros(1, 1, 2, requires_grad=True)
    loss = event_consistency_loss(truth, pred, mask)
    assert loss > 0
    loss.backward()
    assert pred["cls_binary"].grad[0, 0, 1] < 0
    assert pred["pt"].grad[0, 0] < 0
    assert event_consistency_loss(truth, pred, ~mask).isfinite()


def arc_features(radius=10.0):
    angle = torch.linspace(0.01, 0.18, 16, dtype=torch.float64)
    x = torch.zeros(16, 15, dtype=torch.float64)
    x[:, 0] = 1
    x[:, 6] = radius * torch.sin(angle) * 1000
    x[:, 7] = radius * (1 - torch.cos(angle)) * 1000
    x[:, 8] = angle * 1000
    x[:, 14] = torch.arange(16) // 2
    return x


def test_tracker_circle_readout_recovers_momentum_and_rejects_bad_groups():
    x = arc_features()
    summary, log_pt, valid = tracker_group_geometry(x, torch.zeros(16, dtype=torch.long), 2, 2.0)
    assert valid.tolist() == [True, False]
    assert log_pt[0].exp().item() == pytest.approx(0.299792458 * 2 * 10, rel=1e-4)
    assert summary[1].abs().sum() == 0
    # The same transverse shape translated off the origin still fits a circle.
    x[:, 6] += 50
    assert tracker_group_geometry(x, torch.zeros(16, dtype=torch.long), 1, 2.0)[2][0]
    x[:, 7] = 0  # collinear hits cannot determine a finite curvature
    assert not tracker_group_geometry(x, torch.zeros(16, dtype=torch.long), 1, 2.0)[2][0]
    assert not tracker_group_geometry(x[:2], torch.zeros(2, dtype=torch.long), 1, 2.0)[2][0]
    assert not tracker_group_geometry(x[:0], torch.empty(0, dtype=torch.long), 1, 2.0)[2][0]


@pytest.mark.parametrize("radius", [1., 10., 100.])
def test_tracker_fit_rotation_translation_order_and_measurement_noise(radius):
    # Fixed detector lever arm; increasing radius reduces the bending signal.
    angle = torch.linspace(.1 / radius, 1.6 / radius, 24, dtype=torch.float64)
    x = torch.zeros(24, 15, dtype=torch.float64)
    x[:, 0] = 1
    xy = torch.stack((radius * angle.sin(), radius * (1-angle.cos())), dim=-1)
    rotation = torch.tensor([[.6, -.8], [.8, .6]], dtype=torch.float64)
    xy = xy @ rotation.T + torch.tensor([.04, -.03])
    noise = torch.randn(24, 2, generator=torch.Generator().manual_seed(13), dtype=torch.float64) * .005
    x[:, 6:8] = xy * 1000 + noise  # 5 micron coordinate noise
    x[:, 14] = torch.arange(24) // 2
    permutation = torch.randperm(24, generator=torch.Generator().manual_seed(42))
    _, logpt, valid = tracker_group_geometry(x[permutation], torch.zeros(24, dtype=torch.long), 2, 2.)
    assert valid.tolist() == [True, False]
    assert logpt[0].exp().item() == pytest.approx(.299792458 * 2 * radius, rel=.01)


def test_tracker_fit_nonfinite_hits_and_insufficient_surface_coverage():
    x = arc_features()
    x[0, 6] = float("nan")
    x[1, 7] = float("inf")
    summary, logpt, valid = tracker_group_geometry(x, torch.zeros(16, dtype=torch.long), 2, 2.)
    assert torch.isfinite(summary).all() and torch.isfinite(logpt).all()
    assert valid.tolist() == [True, False]
    x[:, 14] = 0
    assert not tracker_group_geometry(x, torch.zeros(16, dtype=torch.long), 1, 2.)[2][0]
