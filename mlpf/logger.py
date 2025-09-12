import logging
import psutil
import shutil
import subprocess
import os
from functools import lru_cache


def _logging(rank, _logger, msg):
    """Will log the message only on rank 0 or cpu."""
    if (rank == 0) or (rank == "cpu"):
        _logger.info(msg)


def _configLogger(name, filename=None, loglevel=logging.INFO):
    # define a Handler which writes INFO messages or higher to the sys.stdout
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    if filename:
        logfile = logging.FileHandler(filename)
        logfile.setLevel(loglevel)
        logfile.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
        logger.addHandler(logfile)


class ColoredLogger:
    color_dict = {
        "black": "\033[0;30m",
        "red": "\033[0;31m",
        "green": "\033[0;32m",
        "orange": "\033[0;33m",
        "blue": "\033[0;34m",
        "purple": "\033[0;35m",
        "cyan": "\033[0;36m",
        "lightgray": "\033[0;37m",
        "darkgray": "\033[1;30m",
        "lightred": "\033[1;31m",
        "lightgreen": "\033[1;32m",
        "yellow": "\033[1;33m",
        "lightblue": "\033[1;34m",
        "lightpurple": "\033[1;35m",
        "lightcyan": "\033[1;36m",
        "white": "\033[1;37m",
        "bold": "\033[1m",
        "endcolor": "\033[0m",
    }

    def __init__(self, name):
        self.logger = logging.getLogger(name)

    def colorize(self, msg, color):
        return self.color_dict[color] + msg + self.color_dict["endcolor"]

    def debug(self, msg, *args, color=None, **kwargs):
        if color:
            msg = self.colorize(msg, color)
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, color=None, **kwargs):
        if color:
            msg = self.colorize(msg, color)
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, color=None, **kwargs):
        if color:
            msg = self.colorize(msg, color)
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, color=None, **kwargs):
        if color:
            msg = self.colorize(msg, color)
        self.logger.error(msg, *args, **kwargs)


_logger = ColoredLogger("mlpf")


@lru_cache(10)
def warn_once(msg, logger=_logger):
    # Keep track of 10 different messages and then warn again
    logger.warning(msg)


def log_memory(stage, rank, tensorboard_writer=None, step=None):
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    _logger.debug(f"RAM memory usage at {stage} on rank {rank}: rss={mem.rss / 1024**2:.2f} MB, vms={mem.vms / 1024**2:.2f} MB")
    if tensorboard_writer and step:
        tensorboard_writer.add_scalar(f"memory/rss_MB/{stage}", mem.rss / 1024**2, step)
        tensorboard_writer.add_scalar(f"memory/vms_MB/{stage}", mem.vms / 1024**2, step)


def log_smi():
    smi_command = shutil.which("nvidia-smi") or shutil.which("rocm-smi")
    if smi_command:
        try:
            result = subprocess.run([smi_command], capture_output=True, text=True, check=True)
            _logger.info(result.stdout)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            _logger.info("SMI error: {}".format(e))
