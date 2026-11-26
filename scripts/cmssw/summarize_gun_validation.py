#!/usr/bin/env python3
"""Summarize direct fiducial GEN-versus-target validation results."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--title", default="Fiducial GEN status 1 versus new targets")
    args = parser.parse_args()
    summary = json.loads((args.directory / "summary.json").read_text())
    lines = [
        args.title,
        "",
        "Reference: visible GEN status-1 particles with pT > 0.5 GeV and |eta| <= 3.2.",
        "The same independent particle cuts are applied to new assigned targets. No compatibility baseline is included.",
        "Jets: anti-kT R=0.4, |eta| <= 3.2; GEN pT >= 20 GeV, target matching pT >= 5 GeV, unique dR < 0.2 matching; boundary cones truncated.",
        "MET: negative sum of fiducial visible transverse momenta; magnitude and projected responses use GEN MET >= 20 GeV.",
        "Resolution: sigma68=(Q84-Q16)/2; relative response resolution=sigma68/abs(median).",
        "",
    ]
    for name, s in summary.items():
        physics = s["response"]
        lines.extend(
            [
                name,
                f"Events {s['events']}; targets {s['targets']}; splits {s['splits']}; merges {s['merges']}",
                f"Fiducial GEN particles {physics['gen_fiducial_particles']}; fiducial targets {physics['target_fiducial_particles']}",
            ]
        )
        for field in ("jet_response", "met_magnitude_response", "met_projected_response"):
            values = physics[field]
            lines.append(field + ": " + json.dumps(values))
        lines.extend(["Response plots: " + s["response_directory"], ""])
    (args.directory / "report.txt").write_text("\n".join(lines) + "\n")
    (args.directory / "findings.txt").write_text("\n".join(lines) + "\n")
    names = list(summary)
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, field, label in [
        (axes[0], "jet_response", "Matched jet pT response"),
        (axes[1], "met_magnitude_response", "Fiducial MET magnitude response"),
    ]:
        y = [summary[n]["response"][field]["median"] for n in names]
        widths = [summary[n]["response"][field]["sigma68"] for n in names]
        valid = [i for i, v in enumerate(y) if v is not None]
        ax.errorbar(x[valid], [y[i] for i in valid], yerr=[widths[i] for i in valid], fmt="o", capsize=5)
        ax.axhline(1, color="0.6", linestyle=":")
        ax.set(ylabel="Median response ± sigma68 (spread)", title=label, xticks=x, xticklabels=names)
    fig.suptitle(args.title + "\nSpread bars are central 68% widths, not errors on the median")
    fig.savefig(args.directory / "sample_comparison.png", dpi=160)
    plt.close(fig)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
