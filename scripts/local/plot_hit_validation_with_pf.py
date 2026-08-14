#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from mlpf.conf import Dataset
from mlpf.model.inference import load_pf_baseline, make_plots


def main():
    parser = argparse.ArgumentParser(
        description="Replot hit-based MLPF validation with a tracks-and-clusters PF baseline."
    )
    parser.add_argument("--validation-dir", required=True)
    parser.add_argument("--hit-sample", required=True)
    parser.add_argument("--pf-data-dir", required=True)
    parser.add_argument("--pf-sample", required=True)
    parser.add_argument("--dataset", required=True, choices=["cld_hits", "clic_hits"])
    parser.add_argument("--num-events", type=int, default=5000)
    parser.add_argument("--version", default="3.2.0")
    parser.add_argument("--splits", nargs="+", default=["1"])
    args = parser.parse_args()

    dataset = Dataset(args.dataset)
    baseline = load_pf_baseline(
        args.pf_data_dir,
        args.pf_sample,
        dataset,
        version=args.version,
        splits=args.splits,
        num_events=args.num_events,
    )
    metrics = make_plots(
        args.validation_dir,
        args.hit_sample,
        dataset,
        dir_name="_test",
        num_test_events=args.num_events,
        baseline_yvals=baseline,
    )
    metrics_path = Path(args.validation_dir) / "pf_baseline_metrics.json"
    with metrics_path.open("w") as handle:
        json.dump(metrics, handle, indent=2, default=float)
    print(json.dumps(metrics, indent=2, default=float))


if __name__ == "__main__":
    main()
