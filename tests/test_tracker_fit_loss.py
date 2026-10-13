from types import SimpleNamespace

import pytest
import torch

from mlpf.conf import SetDecoderConfig
from mlpf.model.tracker_fit_loss import tracker_fit_losses, weighted_circle


def arc(radius=2., n=12):
    angle = torch.linspace(.03, .6, n, dtype=torch.float64)
    return torch.stack((radius * angle.sin(), radius * (1 - angle.cos())), -1)


def test_circle_accuracy_and_weight_gradient():
    xy = arc()
    xy[-1, 1] += .015
    weights = torch.ones(len(xy), dtype=torch.float64, requires_grad=True)
    centre, radius, valid = weighted_circle(xy, weights)
    assert valid
    radius.backward()
    assert torch.isfinite(weights.grad).all()
    assert weights.grad.abs().max() > .01
    assert torch.autograd.gradcheck(lambda w: weighted_circle(xy, w)[1], (weights,))
    _, exact_radius, valid = weighted_circle(arc(), torch.ones(12))
    assert valid
    assert exact_radius.item() == pytest.approx(2., rel=1e-5)


def fixture():
    xy = arc().float()
    xy[-1, 1] += .015
    x = torch.zeros(1, 12, 15)
    x[..., 0] = 1
    x[0, :, 6:8] = xy * 1000
    x[0, :, 14] = torch.arange(12)
    target = torch.zeros(1, 1, 14)
    target[0, 0, 0] = 1
    target[0, 0, 2] = torch.tensor(.299792458 * 2 * 2).log()
    target[..., 13] = 1
    hit_target = torch.zeros(1, 12, 14)
    hit_target[..., 13] = 1
    hit_target[0, -1, 13] = 2
    batch = SimpleNamespace(X=x, mask=torch.ones(1, 12, dtype=torch.bool), ytarget=hit_target,
                            ytarget_set=target, target_mask=torch.ones(1, 1, dtype=torch.bool))
    keys = torch.full((1, 12, 2), .1, requires_grad=True)
    slots = torch.tensor([[[1., 0.], [0., -1.]]], requires_grad=True)
    bg = torch.zeros(1, 12, 1, requires_grad=True)
    return (keys, slots, bg), batch, [(torch.tensor([0]), torch.tensor([0]))]


@pytest.mark.parametrize('name', ['TrackerGeometry', 'TrackerCurvature'])
def test_losses_backpropagate_to_ownership(name):
    grouping, batch, matches = fixture()
    losses = tracker_fit_losses(grouping, batch, matches, chunk_size=3)
    assert losses[name + 'Count'] == 1
    assert losses[name] > 0
    losses[name].backward()
    for tensor in grouping:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().max() > 0
    # Gradient descent should reduce the contaminated hit's ownership score.
    assert grouping[0].grad[0, -1, 0] > 0


@pytest.mark.parametrize('failure', ['empty', 'collinear', 'electron', 'nan', 'short'])
def test_unsupported_tracks_are_finite(failure):
    grouping, batch, matches = fixture()
    if failure == 'empty':
        batch.mask[:] = False
    elif failure == 'collinear':
        batch.X[..., 7] = 0
    elif failure == 'electron':
        batch.ytarget_set[..., 0] = 4
    elif failure == 'nan':
        batch.X[..., 6] = float('nan')
    elif failure == 'short':
        batch.X[..., 6:9] *= .001
    losses = tracker_fit_losses(grouping, batch, matches)
    assert losses['TrackerCurvatureCount'] == 0
    assert losses['TrackerGeometryCount'] == 0
    (losses['TrackerGeometry'] + losses['TrackerCurvature']).backward()
    assert torch.isfinite(grouping[0].grad).all()


def test_fit_loss_requires_ownership():
    with pytest.raises(ValueError, match='require hit_grouping'):
        SetDecoderConfig(tracker_curvature_loss_weight=.1)


def test_truth_ownership_never_selects_predicted_fit_candidates():
    grouping, batch, matches = fixture()
    before = tracker_fit_losses(grouping, batch, matches)
    batch.ytarget[..., 13] = 999
    after = tracker_fit_losses(grouping, batch, matches)
    assert after['TrackerGeometryCount'] == 0
    assert after['TrackerCurvatureCount'] == before['TrackerCurvatureCount']
    assert torch.equal(after['TrackerCurvature'], before['TrackerCurvature'])


def test_multiple_slot_checkpoint_backward():
    grouping, batch, matches = fixture()
    # Duplicate the track into a second slot/event to exercise delayed backward
    # through checkpointed ownership without capturing changing loop variables.
    grouping = tuple(torch.cat((t.detach(), t.detach()), 0).requires_grad_() for t in grouping)
    for name in ('X', 'mask', 'ytarget', 'ytarget_set', 'target_mask'):
        setattr(batch, name, torch.cat((getattr(batch, name), getattr(batch, name)), 0))
    losses = tracker_fit_losses(grouping, batch, matches * 2)
    assert losses['TrackerCurvatureCount'] == 2
    (losses['TrackerCurvature'] + losses['TrackerGeometry']).backward()
    for value in grouping:
        assert torch.allclose(value.grad[0], value.grad[1])
