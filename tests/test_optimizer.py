import torch
from mlpf.conf import MLPFConfig
from mlpf.optimizers import get_optimizer


class _Weighter(torch.nn.Module):
    pass


class _Model(torch.nn.Module):
    def __init__(self, with_weighter):
        super().__init__()
        self.body = torch.nn.Linear(4, 4)
        if with_weighter:
            self.task_loss_weighter = _Weighter()


def _config():
    return MLPFConfig.model_validate(
        {
            "dataset": "cms",
            "data_dir": "/tmp",
            "conv_type": "attention",
            "model": {
                "type": "attention",
                "input_encoding": "split",
                "task_queries": True,
                "backbone": {"mode": "shared", "num_convs": 2},
                "attention": {
                    "num_convs": 2,
                    "num_heads": 2,
                    "head_dim": 4,
                    "attention_type": "math",
                    "dropout_ff": 0.0,
                },
            },
            "lr": 0.001,
            "weight_decay": 0.01,
            "optimizer": "adamw",
        }
    )


def test_optimizer_single_group_without_task_weighter():
    cfg = _config()
    opt = get_optimizer(_Model(with_weighter=False), cfg)
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["lr"] == cfg.lr
    assert opt.param_groups[0]["weight_decay"] == cfg.weight_decay


def test_calibrated_task_weighter_adds_no_optimizer_group():
    cfg = _config()
    opt = get_optimizer(_Model(with_weighter=True), cfg)
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["lr"] == cfg.lr
    assert opt.param_groups[0]["weight_decay"] == cfg.weight_decay
