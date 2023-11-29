"""
Simple script that tests if CUDA is installed on the number of gpus specefied.

Author: Farouk Mokhtar
"""

import argparse
import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
from pyg.logger import _logger

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()


parser.add_argument("--gpus", type=str, default="0", help="to use CPU set to empty string; else e.g., `0,1`")


def main():
    args = parser.parse_args()
    world_size = args.gpus  # will be 1 for both cpu ("") and single-gpu ("0")

    if args.gpus:
        assert (
            world_size <= torch.cuda.device_count()
        ), f"--gpus is too high (specefied {world_size} gpus but only {torch.cuda.device_count()} gpus are available)"

        torch.cuda.empty_cache()
        if world_size > 1:
            _logger.info(f"Will use torch.nn.parallel.DistributedDataParallel() and {world_size} gpus", color="purple")
            for rank in range(world_size):
                _logger.info(torch.cuda.get_device_name(rank), color="purple")

        elif world_size == 1:
            rank = 0
            _logger.info(f"Will use single-gpu: {torch.cuda.get_device_name(rank)}", color="purple")

    else:
        rank = "cpu"
        _logger.info("Will use cpu", color="purple")


if __name__ == "__main__":
    main()
