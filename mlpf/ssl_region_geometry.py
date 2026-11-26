"""Freeze empirical detector support from training inputs before region SSL."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from mlpf.model.ssl_regions import hit_addresses
from mlpf.pretrain import PretrainConfig, build_model_config, load_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=int, default=2048, help="Training events per dataset used for support, never validation")
    parser.add_argument("--eta-width", type=float, default=.5)
    parser.add_argument("--phi-bins", type=int, default=16)
    parser.add_argument("--fine-bins", type=int, default=2)
    args = parser.parse_args()
    if args.events < 1 or args.eta_width <= 0 or args.phi_bins < 1 or args.fine_bins < 1:
        parser.error("Counts and bin sizes must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace an existing geometry catalogue: {args.output}")
    torch.set_num_threads(1)
    config = PretrainConfig.model_validate(yaml.safe_load(args.config.read_text()))
    if not config.train_split.startswith("train"):
        raise ValueError("Build empirical geometry from a training slice")
    model_config = build_model_config(config)
    data = load_inputs(config, model_config, config.train_split, args.events)
    support = {}
    for i in range(len(data)):
        x = torch.as_tensor(np.asarray(data[i]["X"])).float()
        x = x[x[:, 0] != 0]
        if x.shape[1] != 15 or not torch.isin(x[:, 0], torch.tensor([1., 2.])).all():
            raise ValueError("Expected EDM4hep raw-hit schema")
        address, fine = hit_addresses(x, args.eta_width, args.phi_bins, args.fine_bins)
        combined = torch.unique(torch.cat([address, fine[:, None]], 1), dim=0).tolist()
        for row in combined:
            region = tuple(row[:-1])
            bins = support.setdefault(region, [False] * args.fine_bins**2)
            bins[row[-1]] = True
        if (i + 1) % 128 == 0:
            print(f"Scanned {i + 1}/{len(data)} events; {len(support)} supported regions", flush=True)
    regions = sorted(support)
    result = dict(schema="edm4hep-hit-regions-v1", support="Empirical union of training-hit bins; not a full active-channel geometry map",
                  datasets=config.datasets, split=config.train_split, scanned_events=len(data),
                  eta_width=args.eta_width, phi_bins=args.phi_bins, fine_bins=args.fine_bins,
                  regions=regions, valid_bins=[support[r] for r in regions])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(f"Saved {len(regions)} regions to {args.output}")


if __name__ == "__main__":
    main()
