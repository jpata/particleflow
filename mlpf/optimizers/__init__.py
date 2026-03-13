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
    if config.optimizer == OptimizerType.ADAMW:
        ret = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=wd)
    elif config.optimizer == OptimizerType.LAMB:
        ret = Lamb(model.parameters(), lr=config.lr, weight_decay=wd)
    elif config.optimizer == OptimizerType.SGD:
        ret = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer type: {config.optimizer}")

    _logger.info(f"Created optimizer: {ret}")
    return ret
