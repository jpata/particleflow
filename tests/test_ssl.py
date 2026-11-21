"""Contracts for input-only distillation, correspondence, EMA, and transfer."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mlpf.model.mlpf import MLPF
from mlpf.model.PFDataset import TFDSDataSource
from mlpf.model.ssl import DistillationConfig, StudentTeacher, balanced_mean, collate_inputs, subsample_view
from tests.test_backbone_task_queries import make_config, make_inputs


def setup_model(dataset="cld"):
    config = make_config(dataset=dataset, task_queries=False)
    student = MLPF(config)
    X, _ = make_inputs(config)
    return StudentTeacher(student, 8, DistillationConfig(prototypes=8, hidden_dim=16, projection_dim=4)), X, config


@pytest.mark.parametrize("dataset", ["cld", "cms", "cld_hits"])
def test_gradients_and_ema(dataset):
    model, X, _ = setup_model(dataset)
    model.train()
    assert not model.teacher.training
    before = {k: v.clone() for k, v in model.teacher.named_parameters()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    loss, _ = model(X)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.student.backbone.parameters())
    assert all(p.grad is None for p in model.teacher.parameters())
    assert all(p.grad is None for p in model.student.nn_pid.parameters())
    optimizer.step()
    for name, p in model.teacher.named_parameters():
        torch.testing.assert_close(p, before[name])
    model.update_teacher()
    for name, p in model.teacher.named_parameters():
        expected = before[name] * model.config.teacher_momentum + dict(model.student.named_parameters())[name] * (1 - model.config.teacher_momentum)
        torch.testing.assert_close(p, expected)


def test_subsampling_correspondence_and_padding():
    X = torch.zeros(2, 10, 3)
    X[0, :8, 0] = torch.tensor([1, 1, 1, 1, 1, 1, 2, 2])
    X[1, :2, 0] = torch.tensor([1, 2])
    X[..., 1] = torch.arange(10)
    view, mask, idx = subsample_view(X, X[..., 0] != 0, 0.5, torch.Generator().manual_seed(7))
    for event in range(2):
        torch.testing.assert_close(view[event, mask[event]], X[event, idx[event, mask[event]]])
        assert set(view[event, mask[event], 0].tolist()) == {1, 2}
    assert mask.sum(1).tolist() == [4, 2]
    assert torch.all(view[~mask] == 0)


def test_balancing_does_not_reward_high_occupancy():
    values = torch.tensor([[1.0, 1.0, 1.0, 5.0], [10.0, 0.0, 0.0, 0.0]])
    types = torch.tensor([[1, 1, 1, 2], [1, 0, 0, 0]])
    assert balanced_mean(values, types, types != 0).item() == 6.5


def test_eval_does_not_update_teacher_or_center_and_padding_is_ignored():
    model, X, _ = setup_model()
    model.eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    a, _ = model(X, generator=torch.Generator().manual_seed(1))
    padded = torch.cat([X, torch.zeros(X.shape[0], 3, X.shape[-1])], dim=1)
    b, _ = model(padded, generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(a, b)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_split_backbone_fails_explicitly():
    student = MLPF(make_config(backbone_mode="split"))
    with pytest.raises(ValueError, match="shared"):
        StudentTeacher(student, 8, DistillationConfig())


def test_bfloat16_projection_and_center():
    model, X, _ = setup_model()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, _ = model(X)
    loss.backward()
    assert loss.dtype == torch.float32
    assert model.center.dtype == torch.float32
    assert torch.isfinite(model.center).all()


def test_heptv2_distillation():
    from mlpf.conf import MLPFConfig

    config = make_config(dataset="cld", task_queries=False).model_dump(mode="json")
    config["model"].update(
        type="heptv2",
        heptv2={"embedding_dim": 16, "num_heads": 2, "width": 32, "block_size": 8, "n_hashes": 2, "num_regions": 4, "num_w_per_dist": 2},
    )
    config["conv_type"] = "heptv2"
    config = MLPFConfig.model_validate(config)
    X, _ = make_inputs(config)
    model = StudentTeacher(MLPF(config), 16, DistillationConfig(prototypes=8))
    loss, _ = model(X)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.student.backbone.parameters())


def test_exact_resume(tmp_path, monkeypatch):
    import mlpf.pretrain as pretrain

    config = make_config(dataset="cld", task_queries=False)
    X, _ = make_inputs(config)
    data = [{"X": x.numpy(), "source_id": 1} for x in X]
    monkeypatch.setattr(pretrain, "build_model_config", lambda _: config)
    monkeypatch.setattr(pretrain, "load_inputs", lambda *args: data)
    monkeypatch.setattr(pretrain, "check_splits", lambda _: None)
    settings = pretrain.PretrainConfig(
        data_dir="/tmp",
        datasets=["fixture"],
        steps=2,
        dtype="float32",
        batch_size=2,
        validation_frequency=1,
        checkpoint_frequency=1,
        ssl=DistillationConfig(prototypes=8, hidden_dim=16, projection_dim=4),
    )
    pretrain.run(settings, tmp_path / "resumed", "cpu")
    settings.steps = 3
    pretrain.run(settings, tmp_path / "resumed", "cpu", resume=True)
    pretrain.run(settings, tmp_path / "continuous", "cpu")
    a = torch.load(tmp_path / "resumed/latest.pth", weights_only=False)
    b = torch.load(tmp_path / "continuous/latest.pth", weights_only=False)
    for key in a["ssl_state_dict"]:
        torch.testing.assert_close(a["ssl_state_dict"][key], b["ssl_state_dict"][key], rtol=0, atol=0)


def test_overlapping_tfds_slices_rejected(monkeypatch):
    import tensorflow_datasets as tfds
    import mlpf.pretrain as pretrain

    splits = tfds.core.SplitDict([tfds.core.SplitInfo(name="train", shard_lengths=[100], num_bytes=0)])
    monkeypatch.setattr(pretrain.tfds, "builder", lambda *a, **kw: SimpleNamespace(info=SimpleNamespace(splits=splits)))
    config = pretrain.PretrainConfig(data_dir="/tmp", datasets=["fixture"])
    pretrain.check_splits(config)
    config.valid_split = "train[90%:]"
    with pytest.raises(ValueError, match="Overlapping"):
        pretrain.check_splits(config)


@pytest.mark.parametrize("include_targets", [False, True])
def test_loader_accepts_no_labels_and_discards_existing_labels(include_targets):
    X = np.zeros((3, 19), np.float32)
    X[:, 0] = [1, 2, 1]
    X[:, 1] = [1, 3, 2]
    record = {"X": X}
    if include_targets:
        # Deliberately unusable targets: input-only preprocessing must not access them.
        record.update(ytarget=object(), ycand=object())
    ds = SimpleNamespace(
        dataset_info=SimpleNamespace(
            name="cld_edm_ttbar_pf", config_name="1", features=SimpleNamespace(deserialize_example_np=lambda *a, **kw: record.copy())
        ),
        data_source=SimpleNamespace(__getitems__=lambda ids: [b"record"]),
        decoders=None,
    )
    loader = TFDSDataSource(ds, sort=True, pad_to_multiple=4, input_only=True)
    result = loader[0]
    assert set(result) == {"X", "source_id", "input_type_id"}
    assert result["X"][:, 1].tolist() == [3, 2, 1, 0]
    batch = collate_inputs([result])
    assert batch["X"].shape == (1, 3, 19)


def test_checkpoint_transfer_and_ssl_restore(tmp_path):
    from mlpf.model.losses import make_task_loss_weighter
    from mlpf.pretrain import PretrainConfig, save_checkpoint

    model, X, config = setup_model()
    model.student.task_loss_weighter = make_task_loss_weighter(config.task_loss_weights.model_dump())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    loss, _ = model(X)
    loss.backward()
    optimizer.step()
    model.update_teacher()
    settings = PretrainConfig(data_dir="/tmp", datasets=["cld_edm_ttbar_pf/1:3.3.0"])
    save_checkpoint(tmp_path / "latest.pth", model, optimizer, 1, settings, config)
    restored = MLPF(config)
    restored.task_loss_weighter = make_task_loss_weighter(config.task_loss_weights.model_dump())
    restored.load_state_dict(torch.load(tmp_path / "student.pth", weights_only=False)["model_state_dict"], strict=True)
    model.student.eval()
    restored.eval()
    torch.testing.assert_close(restored.encode_backbone(X, X[..., 0] != 0), model.student.encode_backbone(X, X[..., 0] != 0))
    state = torch.load(tmp_path / "latest.pth", weights_only=False)
    model.load_state_dict(state["ssl_state_dict"], strict=True)
    assert state["step"] == 1
