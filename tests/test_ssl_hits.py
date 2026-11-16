import copy

import numpy as np
import pytest
import torch

from mlpf.conf import Y_FEATURES
from mlpf.model.mlpf import MLPF
from mlpf.model.ssl import encode_ssl_embeddings, subsample_view
from mlpf.model.ssl_hit_probe import HitPairProbeConfig, fit_hit_pair_probes
from tests.test_set_prediction import make_config


def test_hept_padding_restores_rows_masks_and_gradients(monkeypatch):
    model = MLPF(make_config())
    seen = []

    def encode(x, mask, source_id):
        assert x.shape[1] % model.config.heptv2.block_size == 0
        assert torch.equal(x[~mask], torch.zeros_like(x[~mask]))
        seen.append((mask.sum().item(), source_id))
        return x * 2

    monkeypatch.setattr(model, "encode_backbone", encode)
    x = torch.ones(1, 13, 15, requires_grad=True)
    mask = torch.ones(1, 13, dtype=torch.bool)
    source = torch.tensor([1])
    full = encode_ssl_embeddings(model, x, mask, source)
    view, valid, idx = subsample_view(x, mask, 0.6)
    retained = encode_ssl_embeddings(model, view, valid, source)
    torch.testing.assert_close(retained, full.gather(1, idx[..., None].expand_as(view)))
    (full.sum() + retained.sum()).backward()
    expected = torch.full_like(x, 2)
    expected.scatter_add_(1, idx[..., None].expand_as(view), torch.full_like(view, 2))
    torch.testing.assert_close(x.grad, expected)
    assert [n for n, _ in seen] == [13, 7]
    assert all(s is source for _, s in seen)


def hit_records():
    records = []
    for seed in range(4):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(24, 15)).astype(np.float32)
        x[:, 0] = np.repeat([1, 2], 12)
        labels = np.tile(np.repeat([1, 2, 3], 4), 2)
        y = np.zeros((24, len(Y_FEATURES)), dtype=np.float32)
        y[:, Y_FEATURES.index("particle_number")] = labels
        # Instance IDs are event-local: only their equality may be used.
        embedding = torch.eye(3)[torch.as_tensor(labels - 1)]
        records.append({"X": x, "ytarget": y, "embedding": embedding})
    return records


def test_hit_pairs_disjoint_fitting_determinism_and_event_local_ids(tmp_path):
    records = hit_records()
    config = HitPairProbeConfig(enabled=True, neighbors=11, epochs=20)
    path = tmp_path / "pairs.npz"
    result = fit_hit_pair_probes(records[:2], records[2:], config, 42, path)
    assert result == fit_hit_pair_probes(records[:2], records[2:], config, 42)
    for detector in ("tracker", "calorimeter"):
        values = result["detectors"][detector]
        assert values["embedding"]["mean"]["roc_auc"] == pytest.approx(1)
        assert values["raw"]["mean"]["positive_fraction"] == values["embedding"]["mean"]["positive_fraction"]
        assert values["embedding"]["train_events"] == 2
    renamed = copy.deepcopy(records)
    for i, record in enumerate(renamed):
        record["ytarget"][:, Y_FEATURES.index("particle_number")] += 100 * (i + 1)
    assert result == fit_hit_pair_probes(renamed[:2], renamed[2:], config, 42)
    # Alter only validation truth. Fitted coefficients/scalers must be unchanged.
    renamed[2]["ytarget"][:, Y_FEATURES.index("particle_number")] = 1
    other = tmp_path / "other.npz"
    fit_hit_pair_probes(renamed[:2], renamed[2:], config, 42, other)
    with np.load(path) as a, np.load(other) as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])


def test_set_probe_particle_metrics_match_standard_validation():
    from mlpf.model.losses import make_task_loss_weighter
    from mlpf.model.ssl_probe import cache_embeddings, cached_batch, evaluate_probe
    from mlpf.model.validation_metrics import compute_validation_particle_metrics, validation_particle_collections
    from tests.test_set_prediction import make_target_tensor

    config = make_config(presence_threshold=0.3)
    model = MLPF(config).eval()
    model.task_loss_weighter = make_task_loss_weighter(config.task_loss_weights.model_dump())
    record = hit_records()[0]
    record.update(ytarget_set=make_target_tensor()[0].numpy(), ytarget_pt_orig=np.ones(24), ytarget_e_orig=np.ones(24), source_id=1)
    device = torch.device("cpu")
    cache = cache_embeddings(model, [record], 1, device, torch.float32)
    result = evaluate_probe(model, cache, device, torch.float32)
    batch, _ = cached_batch(model, cache, device)
    with torch.no_grad():
        predicted = model.predict_particles(batch.X, batch.mask, batch.source_id)
    expected = compute_validation_particle_metrics(
        *validation_particle_collections(batch, predicted, model.output_mode), num_classes=model.num_classes
    )
    for key, (total, count) in expected.items():
        assert result["particle_metrics"][key] == (pytest.approx(total / count) if count else None)


def test_hit_grouping_chunk_size_preserves_loss_and_gradients():
    from mlpf.model.PFDataset import PFBatch
    from mlpf.model.set_losses import hit_grouping_loss
    from tests.test_set_prediction import make_target_tensor

    torch.manual_seed(42)
    x = torch.zeros(1, 19, 15)
    x[:, :, 0] = 2
    y = torch.zeros(1, 19, len(Y_FEATURES))
    y[:, :, Y_FEATURES.index("particle_number")] = torch.arange(19) % 3
    batch = PFBatch(X=x, ytarget=y, ytarget_set=make_target_tensor())
    matches = [(torch.tensor([0, 2]), torch.tensor([0, 1]))]
    base = [torch.randn(1, 19, 4), torch.randn(1, 4, 4), torch.randn(1, 19, 1)]
    results = []
    for chunk in (3, 8192):
        grouping = [v.clone().requires_grad_() for v in base]
        loss = hit_grouping_loss(grouping, batch, matches, chunk_size=chunk)
        loss.backward()
        results.append([loss.detach(), *[v.grad for v in grouping]])
    for a, b in zip(*results):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
