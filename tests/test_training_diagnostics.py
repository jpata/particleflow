import torch

from mlpf.model.PFDataset import PFBatch
from mlpf.model.training import _accumulate_domain_losses_and_stats, _event_domain_labels, _finalize_diagnostics


def make_batch():
    X = torch.zeros(4, 5, 17)
    X[:, :3, 0] = torch.tensor([1, 2, 1])
    X[:, :3, 1] = 1.0
    X[:, :3, 5] = 2.0

    ytarget = torch.zeros(4, 5, 8)
    ytarget[:, :2, 0] = 1
    ytarget[:, :2, 2] = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    ytarget[:, :2, 3] = 0.0
    ytarget[:, :2, 4] = 0.0
    ytarget[:, :2, 5] = 1.0
    ytarget[:, :2, 6] = torch.tensor([[0.5], [1.0], [1.5], [2.0]])

    return PFBatch(
        X=X,
        ytarget=ytarget,
        genmet=torch.zeros(4, 2),
        source_id=torch.tensor([3, 2, 3, 2]),
        input_type_id=torch.tensor([1, 1, 2, 2]),
    )


def make_predictions(batch):
    shape = batch.ytarget.shape[:2]
    return {
        "cls_binary": torch.zeros(*shape, 2),
        "cls_id_onehot": torch.zeros(*shape, 6),
        "pt": batch.ytarget[..., 2] + 0.25,
        "eta": batch.ytarget[..., 3],
        "sin_phi": batch.ytarget[..., 4],
        "cos_phi": batch.ytarget[..., 5],
        "energy": batch.ytarget[..., 6] - 0.5,
    }


def make_targets(batch):
    return {
        "cls_id": batch.ytarget[..., 0].long(),
        "pt": batch.ytarget[..., 2],
        "eta": batch.ytarget[..., 3],
        "sin_phi": batch.ytarget[..., 4],
        "cos_phi": batch.ytarget[..., 5],
        "energy": batch.ytarget[..., 6],
    }


def test_event_domain_labels():
    batch = make_batch()
    assert _event_domain_labels(batch) == ["cld_hits", "clic_hits", "cld_pf", "clic_pf"]


def test_domain_loss_and_regression_diagnostics_are_grouped():
    batch = make_batch()
    accum = {}

    _accumulate_domain_losses_and_stats(
        batch,
        make_targets(batch),
        make_predictions(batch),
        {"pt": 1.0, "eta": 0.01, "sin_phi": 0.01, "cos_phi": 0.01, "energy": 1.0},
        accum,
    )
    metrics = _finalize_diagnostics(accum, world_size=1)

    for label in ["cld_hits", "clic_hits", "cld_pf", "clic_pf"]:
        assert f"diagnostic/loss/{label}/Regression_pt" in metrics
        assert f"diagnostic/loss/{label}/Regression_energy" in metrics
        assert metrics[f"diagnostic/composition/{label}/events"] == 1.0
        assert metrics[f"diagnostic/composition/{label}/target_particles"] == 2.0
        assert metrics[f"diagnostic/regression/{label}/pt_residual_mean"] == 0.25
        assert metrics[f"diagnostic/regression/{label}/energy_residual_mean"] == -0.5
        assert metrics[f"diagnostic/regression/{label}/pt_residual_rms"] == 0.25
        assert metrics[f"diagnostic/regression/{label}/energy_residual_rms"] == 0.5
