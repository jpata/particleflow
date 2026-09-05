#!/usr/bin/env python3
import argparse
from pathlib import Path

import yaml


def set_ttbar_only(model_config, dataset_key, sample_name, version, splits):
    sample = {"name": sample_name, "version": version, "splits": splits}
    model_config["train_datasets"] = {"physical": {"batch_size": 1, "samples": [sample]}}
    model_config["validation_datasets"] = {"physical": {"batch_size": 1, "samples": [sample]}}
    model_config["test_datasets"] = [{"name": sample_name, "version": version, "splits": splits}]
    model_config["dataset"] = dataset_key


def main():
    parser = argparse.ArgumentParser(description="Restrict CLD/CLIC models to locally available ttbar datasets.")
    parser.add_argument("input_spec", type=Path)
    parser.add_argument("output_spec", type=Path)
    parser.add_argument("--hit-version", default="3.2.1")
    parser.add_argument("--hit-splits", nargs="+", default=["1"])
    parser.add_argument("--pf-version", default="3.2.0")
    parser.add_argument("--pf-splits", nargs="+", default=[str(i) for i in range(1, 11)])
    args = parser.parse_args()

    input_spec = args.input_spec
    output_spec = args.output_spec

    with input_spec.open() as f:
        spec = yaml.safe_load(f)

    local_datasets = {
        "pyg-cld-hits-v1": ("cld_hits", "cld_edm_ttbar_hits", args.hit_version, args.hit_splits),
        "pyg-cld-hits-set-v1": ("cld_hits", "cld_edm_ttbar_hits", args.hit_version, args.hit_splits),
        "pyg-clic-hits-v1": ("clic_hits", "clic_edm_ttbar_hits", args.hit_version, args.hit_splits),
        "pyg-cld-v1": ("cld", "cld_edm_ttbar_pf", args.pf_version, args.pf_splits),
        "pyg-clic-v1": ("clic", "clic_edm_ttbar_pf", args.pf_version, args.pf_splits),
    }
    for model_name, (dataset_key, sample_name, version, splits) in local_datasets.items():
        if model_name in spec["models"]:
            set_ttbar_only(spec["models"][model_name], dataset_key, sample_name, version, splits)

    output_spec.parent.mkdir(parents=True, exist_ok=True)
    with output_spec.open("w") as f:
        yaml.safe_dump(spec, f, sort_keys=False)


if __name__ == "__main__":
    main()
