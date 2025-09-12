import torch
from mlpf.optimizers.lamb import Lamb
from mlpf.logger import _logger


def get_optimizer(model, config):
    """
    Returns the optimizer for the given model based on the configuration provided.
    Parameters:
    model (torch.nn.Module): The model for which the optimizer is to be created.
    config (dict): Configuration dictionary containing optimizer settings.
                   Must include the key "lr" for learning rate.
                   Optionally includes the key "optimizer" to specify the type of optimizer.
                   Supported values for "optimizer" are "adamw", "lamb", and "sgd".
                   If "optimizer" is not provided, "adamw" is used by default.
    Returns:
    torch.optim.Optimizer: The optimizer specified in the configuration.
    Raises:
    ValueError: If the specified optimizer type is not supported.
    """

    wd = config["weight_decay"] if "weight_decay" in config else 0.01
    if "optimizer" not in config:
        ret = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=wd)
    if config["optimizer"] == "adamw":
        ret = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=wd)
    elif config["optimizer"] == "lamb":
        ret = Lamb(model.parameters(), lr=config["lr"], weight_decay=wd)
    elif config["optimizer"] == "sgd":
        ret = torch.optim.SGD(model.parameters(), lr=config["lr"], weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer type: {config['optimizer']}")

    _logger.info(f"Created optimizer: {ret}")
    return ret
