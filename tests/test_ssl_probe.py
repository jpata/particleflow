import copy

import pytest
import torch

from mlpf.conf import Y_FEATURES
from mlpf.model.losses import make_task_loss_weighter
from mlpf.model.mlpf import MLPF
from mlpf.model.ssl_probe import ProbeConfig, fit_probe
from tests.test_backbone_task_queries import make_config, make_inputs


@pytest.mark.parametrize("task_queries", [False, True])
def test_cached_readout_matches_full_forward(task_queries):
    config = make_config(dataset="cld", task_queries=task_queries)
    model = MLPF(config).eval()
    X, mask = make_inputs(config)
    mask[1, -2:] = False
    X[~mask] = 0
    with torch.no_grad():
        full = model(X, mask)
        cached = model.decode_backbone(model.encode_backbone(X, mask), X, mask)
    for a, b in zip(full, cached):
        torch.testing.assert_close(a, b)


def test_probe_trains_only_readout_preserves_ssl_and_rng(monkeypatch):
    torch.manual_seed(42)
    config = make_config(dataset="cld", task_queries=True)
    student = MLPF(config)
    student.task_loss_weighter = make_task_loss_weighter(config.task_loss_weights.model_dump())
    X, _ = make_inputs(config)
    X[1, -2:] = 0  # exercise stored padding and embedding/target alignment
    records = []
    for x in X:
        y = torch.zeros(x.shape[0], len(Y_FEATURES))
        y[:, 0] = (x[:, 0] != 0).float()
        y[:, 3:6] = x[:, 2:5]
        records.append({"X": x.numpy(), "ytarget": y.numpy(), "ytarget_pt_orig": x[:, 1].numpy(), "ytarget_e_orig": x[:, 5].numpy(), "source_id": 1})
    snapshot = copy.deepcopy(student.state_dict())
    rng = torch.get_rng_state().clone()
    calls = []
    original = MLPF.encode_backbone

    def counted(self, *args, **kwargs):
        calls.append(torch.is_grad_enabled())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MLPF, "encode_backbone", counted)
    settings = ProbeConfig(steps=20, batch_size=2)
    metrics, trained = fit_probe(student, records, records, settings, torch.device("cpu"), torch.float32)
    # One batch each for training/validation extraction, none during 20 head updates.
    assert calls == [False, False]
    assert student.training
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    for name, value in student.state_dict().items():
        torch.testing.assert_close(value, snapshot[name], rtol=0, atol=0)
    for name, value in trained.backbone.state_dict().items():
        torch.testing.assert_close(value, student.backbone.state_dict()[name], rtol=0, atol=0)
    assert all(p.grad is None for p in trained.backbone.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(trained.nn_binary_particle.parameters(), student.nn_binary_particle.parameters()))
    assert metrics["valid"]["losses"]["Classification_binary"] < metrics["initial"]["losses"]["Classification_binary"]
    # Repeating the same frozen probe gives the same fixed-budget result.
    repeated, _ = fit_probe(student, records, records, settings, torch.device("cpu"), torch.float32)
    assert repeated == metrics


def test_set_probe_uses_existing_matching_loss(tmp_path):
    from tests.test_set_prediction import make_config as set_config, make_target_tensor

    config = set_config()
    model = MLPF(config)
    model.task_loss_weighter = make_task_loss_weighter(config.task_loss_weights.model_dump())
    X = torch.zeros(8, config.input_dim)
    X[:, 0] = 2
    X[:, 1] = X[:, 5] = 1
    X[:, 4] = 1
    X[:, 10] = 2
    y = torch.zeros(8, len(Y_FEATURES))
    record = {
        "X": X.numpy(),
        "ytarget": y.numpy(),
        "ytarget_set": make_target_tensor()[0].numpy(),
        "ytarget_pt_orig": X[:, 1].numpy(),
        "ytarget_e_orig": X[:, 5].numpy(),
        "source_id": 1,
    }
    metrics, trained = fit_probe(model, [record], [record], ProbeConfig(steps=2, batch_size=1), torch.device("cpu"), torch.float32)
    assert torch.isfinite(torch.tensor(metrics["valid"]["losses"]["Total"]))
    assert any(p.grad is not None for p in trained.set_decoder.parameters())
    from mlpf.model.ssl_probe import save_probe_predictions
    import numpy as np

    path = tmp_path / "predictions.npz"
    save_probe_predictions(trained, [record], 1, path, torch.device("cpu"), torch.float32)
    with np.load(path, allow_pickle=False) as predictions:
        assert predictions["input_offsets"].tolist() == [0, 8]
        assert predictions["prediction_offsets"].tolist() == [0, 4]
        assert predictions["target_offsets"].tolist() == [0, 2]


def test_periodic_probe_schedule(tmp_path, monkeypatch):
    import mlpf.pretrain as pretrain

    config = make_config(dataset="cld", task_queries=False)
    X, _ = make_inputs(config)
    data = [{"X": x.numpy(), "source_id": 1} for x in X]
    monkeypatch.setattr(pretrain, "build_model_config", lambda _: config)
    monkeypatch.setattr(pretrain, "load_inputs", lambda *args: data)
    monkeypatch.setattr(pretrain, "check_splits", lambda _: None)
    calls = []

    def probe(*args):
        calls.append(args[4])
        return {"ssl_step": args[4]}

    monkeypatch.setattr(pretrain, "run_probe", probe)
    snapshots = []

    def snapshot(*args):
        snapshots.append(args[4])
        return {"step": args[4]}

    monkeypatch.setattr(pretrain, "save_snapshot", snapshot)
    settings = pretrain.PretrainConfig(
        data_dir="/tmp",
        datasets=["fixture"],
        steps=3,
        dtype="float32",
        probe=ProbeConfig(enabled=True, frequency=2),
        snapshots={"enabled": True, "frequency": 2},
    )
    pretrain.run(settings, tmp_path / "run", "cpu")
    assert calls == [0, 2, 3]
    assert snapshots == [0, 2, 3]
