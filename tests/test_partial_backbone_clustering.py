import torch

from mlpf.conf import MLPFConfig, ClusteringLossConfig
from mlpf.model.PFDataset import PFBatch
from mlpf.model.losses import hit_particle_clustering_loss, mlpf_loss
from mlpf.model.mlpf import MLPF
from mlpf.model.training import model_step


def make_config(**overrides):
    config = {
        "dataset": "cld",
        "data_dir": "/tmp",
        "conv_type": "attention",
        "input_dim": 17,
        "num_classes": 6,
        "elemtypes_nonzero": [1, 2],
        "model": {
            "type": "attention",
            "input_encoding": "split",
            "task_queries": True,
            "backbone": {"mode": "partial", "num_convs": 2, "private_num_convs": 1},
            "input_stem": {"mode": "modality", "modality_embedding": False, "source_embedding": False, "input_norm": True},
            "attention": {
                "num_convs": 2,
                "head_dim": 4,
                "num_heads": 2,
                "attention_type": "math",
                "dropout_ff": 0.0,
            },
        },
    }
    config.update(overrides)
    return MLPFConfig(**config)


def make_batch(batch_size=2, y_dim=14):
    X = torch.zeros(batch_size, 8, 17)
    for iev in range(batch_size):
        X[iev, :6, 0] = torch.tensor([1, 2, 1, 2, 1, 2])
        X[iev, :6, 1] = 1.0
        X[iev, :6, 4] = 1.0
        X[iev, :6, 5] = 2.0

    ytarget = torch.zeros(batch_size, 8, y_dim)
    ytarget[:, :4, 0] = 1
    ytarget[:, :4, 2] = 0.1
    ytarget[:, :4, 4] = 0.0
    ytarget[:, :4, 5] = 1.0
    ytarget[:, :4, 6] = 0.1
    if y_dim > 13:
        ytarget[:, 0:2, 13] = 1
        ytarget[:, 2:4, 13] = 2

    return PFBatch(
        X=X,
        ytarget=ytarget,
        genmet=torch.zeros(batch_size, 2),
        source_id=torch.full((batch_size,), 3),
        input_type_id=torch.ones(batch_size, dtype=torch.long),
    )


def test_partial_backbone_forward_shapes():
    model = MLPF(make_config()).eval()
    batch = make_batch()

    with torch.no_grad():
        outputs = model(batch.X, batch.mask, source_id=batch.source_id, input_type_id=batch.input_type_id)
        embeddings = model.encode_backbone(batch.X, batch.mask, source_id=batch.source_id, input_type_id=batch.input_type_id)

    assert [tuple(t.shape) for t in outputs] == [(2, 8, 2), (2, 8, 6), (2, 8, 5), (2, 8, 2)]
    assert tuple(embeddings.shape) == (2, 8, 8)
    assert model.private_num_convs == 1
    assert model.shared_num_convs == 1


def test_clustering_loss_prefers_grouped_hit_embeddings():
    batch = make_batch(batch_size=1)
    y = {
        "cls_id": batch.ytarget[..., 0].long(),
        "particle_number": batch.ytarget[..., 13],
    }
    config = ClusteringLossConfig(weight=1.0, margin=1.0)

    grouped = torch.zeros(1, 8, 4)
    grouped[0, 0:2, 0] = 1.0
    grouped[0, 2:4, 1] = 1.0
    grouped[0, 4:, 2] = 1.0

    mixed = torch.zeros(1, 8, 4)
    mixed[0, 0, 0] = 1.0
    mixed[0, 1, 1] = 1.0
    mixed[0, 2, 0] = 1.0
    mixed[0, 3, 1] = 1.0
    mixed[0, 4:, 2] = 1.0

    assert hit_particle_clustering_loss(grouped, y, batch, config) < hit_particle_clustering_loss(mixed, y, batch, config)


def test_model_step_adds_clustering_loss_when_enabled():
    cfg = make_config(clustering_loss={"weight": 0.01, "margin": 1.0})
    model = MLPF(cfg)
    batch = make_batch()

    loss_opt, losses, *_ = model_step(
        batch,
        model,
        mlpf_loss,
        cfg.regression_loss_weights.model_dump(),
        clustering_config=cfg.clustering_loss,
    )

    assert "Clustering" in losses
    assert "ClusteringRaw" in losses
    assert losses["Clustering"].item() >= 0.0
    assert losses["ClusteringRaw"].item() >= 0.0
    assert torch.isclose(loss_opt.detach(), losses["Total"])
