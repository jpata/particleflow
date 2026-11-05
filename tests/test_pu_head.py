"""
Spec: Validates the optional per-element pileup head ('model.use_pu_head') and the target |eta| cut
('target_abs_eta_max'). Checks that the pileup loss is added only with the head, ignores elements
without a target, balances hard scatter and pileup per element-pT bin with 'pu_loss_pt_balance',
and is calibrated as one more task; that the eta cut removes the targets but keeps their elements;
and that the ColliderML PU200 recipe builds a model with the head.
"""

import numpy as np
import pytest
import torch

from mlpf.conf import MLPFConfig, TaskLossWeights
from mlpf.model.losses import LOSS_TASKS, PU_LOSS_TASK, REGRESSION_FEATURES, _pt_balance_weights, make_task_loss_weighter, mlpf_loss, pu_loss
from mlpf.model.PFDataset import PFBatch, TFDSDataSource
from mlpf.pipeline import get_parser
from tests.mock_data import MockTFDS

REGRESSION_WEIGHTS = {feature: 1.0 for feature in REGRESSION_FEATURES}


def _event(n=8):
    X = torch.ones(1, n, 17)
    X[..., 1] = torch.linspace(1.0, 50.0, n)  # element pt
    y = {"cls_id": torch.tensor([[1, 1, 2, 0, 3, 1, 0, 2]]), "ispu": torch.tensor([[0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]])}
    y.update({feature: torch.zeros(1, n) for feature in REGRESSION_FEATURES})
    ypred = {"cls_binary": torch.randn(1, n, 2), "cls_id_onehot": torch.randn(1, n, 6), "ispu": torch.randn(1, n, 2)}
    ypred.update({feature: torch.zeros(1, n) for feature in REGRESSION_FEATURES})
    return PFBatch(X=X), y, ypred


def test_pu_loss_only_with_the_head():
    batch, y, ypred = _event()
    _, losses, _ = mlpf_loss(y, ypred, batch, REGRESSION_WEIGHTS)
    assert PU_LOSS_TASK not in losses

    _, losses, _ = mlpf_loss(y, ypred, batch, REGRESSION_WEIGHTS, pu_head=True)
    assert torch.isfinite(losses[PU_LOSS_TASK]) and losses[PU_LOSS_TASK] > 0
    # with the task weights, the pileup loss is one more calibrated task
    weighter = make_task_loss_weighter(TaskLossWeights().model_dump(), tasks=LOSS_TASKS + (PU_LOSS_TASK,))
    loss_opt, losses, diagnostics = mlpf_loss(y, ypred, batch, REGRESSION_WEIGHTS, weighter, pu_head=True)
    assert PU_LOSS_TASK in diagnostics["weight"]
    torch.testing.assert_close(loss_opt, sum(losses[task] for task in LOSS_TASKS + (PU_LOSS_TASK,)))


def test_pu_loss_ignores_elements_without_target():
    _, y, ypred = _event()
    y = {k: v[0] for k, v in y.items()}
    ypred = {k: v[0] for k, v in ypred.items()}
    pt = torch.linspace(1.0, 50.0, 8)
    ref = pu_loss(y, ypred, pt)[PU_LOSS_TASK]
    ref_balanced = pu_loss(y, ypred, pt, pt_balance=True)[PU_LOSS_TASK]
    no_target = y["cls_id"] == 0
    ypred["ispu"][no_target] = 100.0 * torch.tensor([-1.0, 1.0])
    torch.testing.assert_close(pu_loss(y, ypred, pt)[PU_LOSS_TASK], ref)
    torch.testing.assert_close(pu_loss(y, ypred, pt, pt_balance=True)[PU_LOSS_TASK], ref_balanced)


def test_pt_balance_gives_both_classes_equal_weight_in_each_bin():
    gen = torch.Generator().manual_seed(0)
    pt = torch.exp(torch.empty(2000).uniform_(0.0, 5.0, generator=gen))
    # pileup dominates at low pt, hard scatter at high pt
    target = (torch.rand(2000, generator=gen) < torch.sigmoid(2.5 - torch.log(pt))).long()
    mask = torch.ones(2000, dtype=torch.bool)
    w = _pt_balance_weights(pt, target, mask)

    lp = torch.log(pt)
    edges = torch.linspace(float(lp.min()), float(lp.max()), 11)
    binidx = torch.bucketize(lp, edges)
    for b in torch.unique(binidx):
        inb = binidx == b
        if (target[inb] == 0).any() and (target[inb] == 1).any():
            torch.testing.assert_close(w[inb & (target == 0)].sum(), w[inb & (target == 1)].sum())
            torch.testing.assert_close(w[inb].sum(), inb.sum().float())
    # the rare class of a bin is upweighted
    low = binidx == 1
    rare = 0 if (target[low] == 0).sum() < (target[low] == 1).sum() else 1
    assert w[low & (target == rare)].min() > 1.0


def test_target_eta_cut_keeps_the_elements():
    X = np.array([[2, 10.0, 2.3, 0.0, 1.0, 20.0], [2, 5.0, 1.0, 0.0, 1.0, 6.0], [1, 3.0, 3.4, 0.0, 1.0, 45.0]], dtype=np.float32)
    ytarget = np.zeros((3, 14), dtype=np.float32)
    ytarget[:, 0] = [2, 3, 1]
    ytarget[:, 2] = [1.0, 4.0, 3.0]  # pt
    ytarget[:, 3] = [5.6, -1.2, -3.5]  # eta: beyond, within, on the edge
    ytarget[:, 6] = [500.0, 7.0, 50.0]  # energy
    data = {"X": X, "ytarget": ytarget, "ycand": np.zeros_like(ytarget)}

    ret = TFDSDataSource(MockTFDS([{k: v.copy() for k, v in data.items()}], name="colliderml_ttbar_pu200_pf"), sort=False, target_abs_eta_max=3.5)[0]
    np.testing.assert_array_equal(ret["X"], X)
    np.testing.assert_array_equal(ret["ytarget"][:, 0], [0, 3, 0])
    assert ret["ytarget_e_orig"][0] == 0 and ret["ytarget_e_orig"][2] == 0 and ret["ytarget_e_orig"][1] == 7.0

    ret = TFDSDataSource(MockTFDS([{k: v.copy() for k, v in data.items()}], name="colliderml_ttbar_pu200_pf"), sort=False)[0]
    np.testing.assert_array_equal(ret["ytarget"][:, 0], [2, 3, 1])


def _recipe_config(model_name):
    cmd = ["--spec-file", "particleflow_spec.yaml", "--model-name", model_name, "--production-name", "colliderml", "train"]
    args, extra_args = get_parser().parse_known_args(cmd)
    args.train, args.test, args.hpo, args.ray_train = True, False, None, False
    return MLPFConfig.from_spec(args.spec_file, args.model_name, args.production_name, args, extra_args)


def test_colliderml_pu_head_recipe():
    from mlpf.model.mlpf import MLPF

    config = _recipe_config("pyg-colliderml-pu200-puhead-v1")
    assert config.model.use_pu_head and config.model.pu_loss_pt_balance
    assert config.target_abs_eta_max == 3.5
    assert MLPF(config).nn_ispu is not None

    baseline = _recipe_config("pyg-colliderml-pu200-eta3p5-v1")
    assert not baseline.model.use_pu_head and baseline.target_abs_eta_max == 3.5
    assert MLPF(baseline).nn_ispu is None


def test_pt_balance_requires_the_head():
    config = _recipe_config("pyg-colliderml-pu200-puhead-v1")
    config.model.use_pu_head = False
    with pytest.raises(ValueError, match="pu_loss_pt_balance requires"):
        config.populate_defaults()
