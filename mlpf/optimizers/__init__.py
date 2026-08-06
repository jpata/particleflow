import torch
from mlpf.optimizers.lamb import Lamb
from mlpf.logger import _logger
from mlpf.conf import MLPFConfig, OptimizerType


def get_optimizer(model: torch.nn.Module, config: MLPFConfig):
    """
    Returns the optimizer for the given model based on the configuration provided.
    Parameters:
    model (torch.nn.Module): The model for which the optimizer is to be created.
    config (MLPFConfig): Configuration object.
    Returns:
    torch.optim.Optimizer: The optimizer specified in the configuration.
    Raises:
    ValueError: If the specified optimizer type is not supported.
    """

    wd = config.weight_decay
    task_loss_weighter = getattr(model, "task_loss_weighter", None)
    if task_loss_weighter is None:
        parameters = model.parameters()
    else:
        task_loss_weight_param_ids = {
            id(param) for param in task_loss_weighter.parameters()
        }
        parameters = [
            {
                "params": [
                    param
                    for param in model.parameters()
                    if id(param) not in task_loss_weight_param_ids
                ]
            },
            {
                "params": list(task_loss_weighter.parameters()),
                "lr": config.lr,
                "weight_decay": 0.0,
                # Keep LAMB's layer-wise trust scaling for the weighter (the
                # pre-split optimizer applied it via weight_decay != 0); without
                # it the log variances adapt roughly 3x slower.
                "always_adapt": True,
            },
        ]

    if config.optimizer == OptimizerType.ADAMW:
        ret = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=wd)
    elif config.optimizer == OptimizerType.LAMB:
        ret = Lamb(parameters, lr=config.lr, weight_decay=wd)
    elif config.optimizer == OptimizerType.SGD:
        ret = torch.optim.SGD(parameters, lr=config.lr, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer type: {config.optimizer}")

    _logger.info(f"Created optimizer: {ret}")
    return ret
