import datetime
import platform
from pathlib import Path


def create_experiment_dir(prefix=None, suffix=None, backend="tf", rank=0):
    if prefix is None:
        train_dir = Path("experiments") / datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    else:
        train_dir = Path("experiments") / (prefix + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))

    if suffix is not None:
        train_dir = train_dir.with_name(train_dir.name + "." + platform.node())

    if backend == "pyg":
        if (rank == 0) or (rank == "cpu"):
            train_dir.mkdir(parents=True)
    else:
        train_dir.mkdir(parents=True)

    return str(train_dir)
