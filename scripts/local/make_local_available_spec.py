#!/usr/bin/env python3
import sys
from pathlib import Path

import yaml


LOCAL_DATASETS = {
    "pyg-cld-hits-v1": ("cld_hits", "cld_edm_ttbar_hits", ["1"]),
    "pyg-clic-hits-v1": ("clic_hits", "clic_edm_ttbar_hits", ["1"]),
    "pyg-cld-v1": ("cld", "cld_edm_ttbar_pf", [str(i) for i in range(1, 11)]),
    "pyg-clic-v1": ("clic", "clic_edm_ttbar_pf", [str(i) for i in range(1, 11)]),
}


def set_ttbar_only(model_config, dataset_key, sample_name, splits):
    sample = {"name": sample_name, "version": "3.2.1", "splits": splits}
    model_config["train_datasets"] = {"physical": {"batch_size": 1, "samples": [sample]}}
    model_config["validation_datasets"] = {"physical": {"batch_size": 1, "samples": [sample]}}
    model_config["test_datasets"] = [{"name": sample_name, "version": "3.2.1", "splits": splits}]
    model_config["dataset"] = dataset_key


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: make_local_available_spec.py INPUT_SPEC OUTPUT_SPEC")

    input_spec = Path(sys.argv[1])
    output_spec = Path(sys.argv[2])

    with input_spec.open() as f:
        spec = yaml.safe_load(f)

    for model_name, (dataset_key, sample_name, splits) in LOCAL_DATASETS.items():
        if model_name in spec["models"]:
            set_ttbar_only(spec["models"][model_name], dataset_key, sample_name, splits)

    output_spec.parent.mkdir(parents=True, exist_ok=True)
    with output_spec.open("w") as f:
        yaml.safe_dump(spec, f, sort_keys=False)


if __name__ == "__main__":
    main()
