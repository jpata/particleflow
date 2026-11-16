"""Learning-curve evaluation must leave a matched training trajectory intact."""

import copy
from types import SimpleNamespace

import torch
import pytest

from mlpf.conf import Y_FEATURES
from mlpf.model.losses import make_task_loss_weighter
from mlpf.model.mlpf import MLPF
from mlpf.model.ssl_probe import ProbeConfig, cache_embeddings
from mlpf.probe_learning_curves import fit_cached, tensor_digest
from tests.test_backbone_task_queries import make_config, make_inputs


def test_intermediate_evaluations_preserve_training_and_frozen_encoder(tmp_path):
    torch.manual_seed(31)
    config = make_config(dataset="cld", task_queries=True)
    model = MLPF(config).eval()
    model.task_loss_weighter = make_task_loss_weighter(
        config.task_loss_weights.model_dump()
    )
    X, _ = make_inputs(config)
    records = []
    for x in X:
        y = torch.zeros(x.shape[0], len(Y_FEATURES))
        y[:, 0] = (x[:, 0] != 0).float()
        y[:, 3:6] = x[:, 2:5]
        records.append(
            {
                "X": x.numpy(),
                "ytarget": y.numpy(),
                "ytarget_pt_orig": x[:, 1].numpy(),
                "ytarget_e_orig": x[:, 5].numpy(),
                "source_id": 1,
            }
        )
    cache = cache_embeddings(model, records, 2, torch.device("cpu"), torch.float32)
    settings = SimpleNamespace(dtype="float32", probe=ProbeConfig(batch_size=2))
    a, b = copy.deepcopy(model), copy.deepcopy(model)
    for name in ["sparse", "dense"]:
        (tmp_path / name).mkdir()
    sparse = fit_cached(a, cache, cache, settings, 42, [0, 5], tmp_path / "sparse")
    dense = fit_cached(b, cache, cache, settings, 42, [0, 1, 2, 5], tmp_path / "dense")
    assert sparse["history"][-1]["valid"] == dense["history"][-1]["valid"]
    assert sparse["initial_readout_sha256"] == dense["initial_readout_sha256"]
    assert sparse["encoder_sha256"] == dense["encoder_sha256"]
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)


def test_tensor_digest_handles_scalar_buffers_and_bfloat16():
    state = {
        "counter": torch.tensor(1),
        "calibrated": torch.tensor(False),
        "embedding": torch.ones(2, dtype=torch.bfloat16),
    }
    assert tensor_digest(state) == tensor_digest(copy.deepcopy(state))
    state["counter"] += 1
    assert tensor_digest(state) != tensor_digest({**state, "counter": torch.tensor(1)})


def test_interrupted_probe_resumes_exact_training_trajectory(tmp_path, monkeypatch):
    from tests.test_set_prediction import make_config as make_set_config, make_target_tensor
    from tests.test_ssl_hits import hit_records

    torch.manual_seed(17)
    config = make_set_config()
    model = MLPF(config).eval()
    model.task_loss_weighter = make_task_loss_weighter(config.task_loss_weights.model_dump())
    record = hit_records()[0]
    import numpy as np

    record.update(ytarget_set=make_target_tensor()[0].numpy(), ytarget_pt_orig=np.ones(24), ytarget_e_orig=np.ones(24), source_id=1)
    cache = cache_embeddings(model, [record], 1, torch.device("cpu"), torch.float32)
    settings = SimpleNamespace(dtype="float32", probe=ProbeConfig(batch_size=1))
    budgets = [0, 2, 5]
    for name in ("continuous", "interrupted"):
        (tmp_path / name).mkdir()
    continuous = copy.deepcopy(model)
    expected = fit_cached(continuous, cache, cache, settings, 42, budgets, tmp_path / "continuous")
    original_step = torch.optim.AdamW.step
    calls = 0

    def interrupt(optimizer, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("simulated interruption")
        return original_step(optimizer, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.AdamW, "step", interrupt)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            fit_cached(copy.deepcopy(model), cache, cache, settings, 42, budgets, tmp_path / "interrupted")
    resumed = copy.deepcopy(model)
    actual = fit_cached(resumed, cache, cache, settings, 42, budgets, tmp_path / "interrupted")
    for key, value in continuous.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    assert expected["initial_readout_sha256"] == actual["initial_readout_sha256"]
    assert [r["valid"] for r in expected["history"]] == [r["valid"] for r in actual["history"]]


def test_set_mode_plots_include_physics_metrics(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    from mlpf.probe_learning_curves import plot_results

    for step in (0, 500):
        path = tmp_path / f"step-{step}-seed-7"
        path.mkdir()
        valid = {
            "unweighted_task_loss": 3.0,
            "particle_f1": 0.3,
            "matched_pid_accuracy": 0.6,
            "particle_metrics": {"event/energy_response_mean": 1.25, "matching/purity": 0.4},
        }
        (path / "result.json").write_text(json.dumps({"seed": 7, "history": [{"updates": u, "train": valid, "valid": valid} for u in (0, 50, 200)]}))
    plot_results(tmp_path)
    main = json.loads((tmp_path / "summary.json").read_text())
    energy = json.loads((tmp_path / "energy_summary.json").read_text())
    assert main["results"][0]["particle_f1"]["mean"] == 0.3
    assert energy["results"][0]["particle_metrics.event/energy_response_mean"]["mean"] == 1.25
    assert (tmp_path / "energy_learning_curves.png").exists()


def test_shared_cache_reused_and_encoder_change_rejected(tmp_path, monkeypatch):
    import mlpf.probe_learning_curves as curves

    state = {"encoder": torch.ones(2), "head": torch.ones(1)}
    model = SimpleNamespace(state_dict=lambda: state)
    config = SimpleNamespace(datasets=["fixture"], data_dir="/tmp", train_split="train[:50%]", valid_split="train[50%:]",
                             dtype="float32", probe=ProbeConfig(batch_size=1))
    calls = []

    def extract(model, data, *args):
        calls.append(data)
        return [{"embedding": torch.ones(2)} for _ in data]

    monkeypatch.setattr(curves, "cache_embeddings", extract)
    args = (tmp_path, 0, model, {"head"}, [0, 1], [2], config, torch.device("cpu"), torch.float32)
    curves.shared_embedding_cache(*args)
    cached = curves.shared_embedding_cache(*args)
    assert calls == [[0, 1], [2]]
    assert [len(part) for part in cached] == [2, 1]
    state["encoder"].add_(1)
    with pytest.raises(ValueError, match="cache metadata differs"):
        curves.shared_embedding_cache(*args)
