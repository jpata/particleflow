#!/usr/bin/env python3
import copy
import sys
from pathlib import Path

import yaml


SPLITS = [str(i) for i in range(1, 11)]
VERSION = "3.2.0"
HIT_TEST_SAMPLES = ["cld_edm_ttbar_hits", "clic_edm_ttbar_hits"]
CLD_HIT_CLUSTERING_WEIGHT = 0.10
CLIC_HIT_CLUSTERING_WEIGHT = 0.30
MIXED_HIT_CLUSTERING_WEIGHT = 0.30


def sample(name):
    return {"name": name, "version": VERSION, "splits": SPLITS}


def set_datasets(model_config, dataset_key, train_samples, test_samples=HIT_TEST_SAMPLES):
    model_config["dataset"] = dataset_key
    model_config["train_datasets"] = {"physical": {"batch_size": 1, "samples": [sample(name) for name in train_samples]}}
    model_config["validation_datasets"] = {"physical": {"batch_size": 1, "samples": [sample(name) for name in train_samples]}}
    model_config["test_datasets"] = [sample(name) for name in test_samples]


def set_clustering_weight(model_config, weight):
    model_config["clustering_loss"] = {**model_config.get("clustering_loss", {}), "weight": weight}


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: make_hit_fom_clean_spec.py INPUT_SPEC OUTPUT_SPEC")

    input_spec = Path(sys.argv[1])
    output_spec = Path(sys.argv[2])

    with input_spec.open() as f:
        spec = yaml.safe_load(f)

    cld_hits = copy.deepcopy(spec["models"]["pyg-cld-hits-v1"])
    set_datasets(cld_hits, "cld_hits", ["cld_edm_ttbar_hits"])
    set_clustering_weight(cld_hits, CLD_HIT_CLUSTERING_WEIGHT)
    spec["models"]["pyg-clean-cld-hits-v1"] = cld_hits

    clic_hits = copy.deepcopy(spec["models"]["pyg-clic-hits-v1"])
    set_datasets(clic_hits, "clic_hits", ["clic_edm_ttbar_hits"])
    set_clustering_weight(clic_hits, CLIC_HIT_CLUSTERING_WEIGHT)
    spec["models"]["pyg-clean-clic-hits-v1"] = clic_hits

    mixed_hits = copy.deepcopy(spec["models"]["pyg-cld-hits-v1"])
    set_datasets(mixed_hits, "cld_hits", ["cld_edm_ttbar_hits", "clic_edm_ttbar_hits"])
    set_clustering_weight(mixed_hits, MIXED_HIT_CLUSTERING_WEIGHT)
    spec["models"]["pyg-clean-mixed-hits-v1"] = mixed_hits

    mixed_hits_pf = copy.deepcopy(spec["models"]["pyg-cld-v1"])
    set_datasets(
        mixed_hits_pf,
        "cld",
        ["cld_edm_ttbar_hits", "clic_edm_ttbar_hits", "cld_edm_ttbar_pf", "clic_edm_ttbar_pf"],
        test_samples=HIT_TEST_SAMPLES,
    )
    spec["models"]["pyg-clean-mixed-hits-pf-v1"] = mixed_hits_pf

    output_spec.parent.mkdir(parents=True, exist_ok=True)
    with output_spec.open("w") as f:
        yaml.safe_dump(spec, f, sort_keys=False)


if __name__ == "__main__":
    main()
