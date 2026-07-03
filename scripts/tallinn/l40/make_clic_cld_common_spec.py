#!/usr/bin/env python3
import copy
import sys
from pathlib import Path

import yaml


SPLITS = [str(i) for i in range(1, 11)]
VERSION = "3.2.0"


def sample(name):
    return {"name": name, "version": VERSION, "splits": SPLITS}


def set_samples(model_config, dataset_key, samples):
    model_config["dataset"] = dataset_key
    model_config["train_datasets"] = {"physical": {"batch_size": 1, "samples": [sample(name) for name in samples]}}
    model_config["validation_datasets"] = {"physical": {"batch_size": 1, "samples": [sample(name) for name in samples]}}
    model_config["test_datasets"] = [sample(name) for name in samples]


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: make_clic_cld_common_spec.py INPUT_SPEC OUTPUT_SPEC")

    input_spec = Path(sys.argv[1])
    output_spec = Path(sys.argv[2])

    with input_spec.open() as f:
        spec = yaml.safe_load(f)

    hit_model = copy.deepcopy(spec["models"]["pyg-cld-hits-v1"])
    set_samples(hit_model, "cld_hits", ["cld_edm_ttbar_hits", "clic_edm_ttbar_hits"])
    spec["models"]["pyg-clic-cld-hits-v1"] = hit_model

    pf_model = copy.deepcopy(spec["models"]["pyg-cld-v1"])
    set_samples(pf_model, "cld", ["cld_edm_ttbar_pf", "clic_edm_ttbar_pf"])
    spec["models"]["pyg-clic-cld-v1"] = pf_model

    output_spec.parent.mkdir(parents=True, exist_ok=True)
    with output_spec.open("w") as f:
        yaml.safe_dump(spec, f, sort_keys=False)


if __name__ == "__main__":
    main()
