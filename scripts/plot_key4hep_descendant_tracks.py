#!/usr/bin/env python3
"""Compare direct and descendant-promoted track targets in Key4hep datasets.

The postprocessing propagates detector-object links from MCParticle descendants
to visible generatorStatus==1 ancestors.  This script matches postprocessed
track rows back to the unmodified EDM4hep truth links and labels a representative
track as:

* ``direct``: directly linked to at least one generatorStatus==1 MCParticle;
* ``descendant``: not directly linked to status 1, but its linked MCParticle has
  a status-1 ancestor;
* ``other``: neither of the above (reported, but not drawn).

Only target representatives (ytarget_track[:, PDG] != 0) enter the kinematic
plots.  This makes the diagnostic insensitive to detector calibration: it tests
whether the reconstructed helix is being labelled with the kinematics of the
same physical particle.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot


@dataclass(frozen=True)
class Detector:
    name: str
    energy: str
    post_dir: Path
    root_dir: Path
    relation: str


DEFAULT_DETECTORS = (
    Detector(
        "IDEA",
        "365 GeV",
        Path("/local/joosep/mlpf/idea/IDEA_o1_v03_fccconfig_a05a3a9/post/p8_ee_qq_ecm365"),
        Path("/local/joosep/mlpf/idea/IDEA_o1_v03_fccconfig_a05a3a9/gen/p8_ee_qq_ecm365/root"),
        "TracksFromGenParticlesAssociation",
    ),
    Detector(
        "CLD",
        "365 GeV",
        Path("/local/joosep/mlpf/cld/v1.2.5_key4hep_2025-05-29/post/p8_ee_qq_ecm365"),
        Path("/local/joosep/mlpf/cld/v1.2.5_key4hep_2025-05-29/gen/p8_ee_qq_ecm365/root"),
        "SiTracksMCTruthLink",
    ),
    Detector(
        "CLIC",
        "380 GeV",
        Path("/local/joosep/mlpf/clic/v1.2.5_key4hep_2025-05-29/post/p8_ee_qq_ecm380"),
        Path("/local/joosep/mlpf/clic/v1.2.5_key4hep_2025-05-29/gen/p8_ee_qq_ecm380/root"),
        "SiTracksMCTruthLink",
    ),
)


def _branches(relation: str) -> dict[str, str]:
    return {
        "status": "MCParticles/MCParticles.generatorStatus",
        "parent_begin": "MCParticles/MCParticles.parents_begin",
        "parent_end": "MCParticles/MCParticles.parents_end",
        "parent_index": "_MCParticles_parents/_MCParticles_parents.index",
        "weight": f"{relation}/{relation}.weight",
        "track": f"_{relation}_from/_{relation}_from.index",
        "particle": f"_{relation}_to/_{relation}_to.index",
    }


def _has_status1_ancestor(
    particle: int,
    status: np.ndarray,
    parent_begin: np.ndarray,
    parent_end: np.ndarray,
    parent_index: np.ndarray,
) -> bool:
    """Return whether a non-status-1 particle descends from status 1."""
    stack = [int(p) for p in parent_index[parent_begin[particle] : parent_end[particle]]]
    visited = {particle}
    while stack:
        parent = stack.pop()
        if parent in visited or parent < 0 or parent >= len(status):
            continue
        visited.add(parent)
        if status[parent] == 1:
            return True
        stack.extend(int(p) for p in parent_index[parent_begin[parent] : parent_end[parent]])
    return False


def _delta_r(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    phi_x = np.arctan2(x[:, 3], x[:, 4])
    phi_y = np.arctan2(y[:, 4], y[:, 5])
    dphi = np.arctan2(np.sin(phi_x - phi_y), np.cos(phi_x - phi_y))
    return np.hypot(x[:, 2] - y[:, 3], dphi)


def _seed(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1]


def matched_files(detector: Detector, max_files: int) -> list[tuple[Path, Path]]:
    post_by_seed = {_seed(path): path for path in detector.post_dir.glob("*.parquet")}
    root_by_seed = {_seed(path): path for path in detector.root_dir.glob("*.root")}
    seeds = sorted(post_by_seed.keys() & root_by_seed.keys(), key=int)
    if not seeds:
        raise FileNotFoundError(f"No matched parquet/ROOT files for {detector.name}")
    return [(post_by_seed[s], root_by_seed[s]) for s in seeds[:max_files]]


def analyze_detector(detector: Detector, max_files: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    samples: dict[str, list[np.ndarray]] = {
        "direct_dr": [],
        "direct_response": [],
        "direct_charge_match": [],
        "descendant_dr": [],
        "descendant_response": [],
        "descendant_charge_match": [],
    }
    counts = {
        "events": 0,
        "input_tracks": 0,
        "representative_tracks": 0,
        "direct_representatives": 0,
        "descendant_representatives": 0,
        "other_representatives": 0,
        "invalid_truth_links": 0,
    }
    used_seeds: list[int] = []
    names = _branches(detector.relation)

    for parquet_path, root_path in matched_files(detector, max_files):
        print(f"{detector.name}: {_seed(parquet_path)}", flush=True)
        post = ak.from_parquet(parquet_path, columns=["X_track", "ytarget_track"])
        tree = uproot.open(root_path, handler=uproot.source.file.MemmapSource)["events"]
        raw = tree.arrays(list(names.values()), library="ak", how=dict)
        if len(post["X_track"]) != tree.num_entries:
            raise ValueError(
                f"Event-count mismatch for {parquet_path.name}: "
                f"post={len(post['X_track'])}, ROOT={tree.num_entries}"
            )
        used_seeds.append(int(_seed(parquet_path)))

        for event in range(tree.num_entries):
            x = np.asarray(ak.to_numpy(post["X_track"][event]), dtype=np.float64)
            y = np.asarray(ak.to_numpy(post["ytarget_track"][event]), dtype=np.float64)
            status = np.asarray(ak.to_numpy(raw[names["status"]][event]), dtype=np.int32)
            parent_begin = np.asarray(ak.to_numpy(raw[names["parent_begin"]][event]), dtype=np.int64)
            parent_end = np.asarray(ak.to_numpy(raw[names["parent_end"]][event]), dtype=np.int64)
            parent_index = np.asarray(ak.to_numpy(raw[names["parent_index"]][event]), dtype=np.int64)
            weights = np.asarray(ak.to_numpy(raw[names["weight"]][event]), dtype=np.float64)
            tracks = np.asarray(ak.to_numpy(raw[names["track"]][event]), dtype=np.int64)
            particles = np.asarray(ak.to_numpy(raw[names["particle"]][event]), dtype=np.int64)

            counts["events"] += 1
            if len(x) == 0:
                continue
            if len(x) != len(y) or (x.ndim != 2 or y.ndim != 2 or x.shape[1] < 16 or y.shape[1] < 14):
                raise ValueError(f"Unexpected track matrix shape in {parquet_path.name}, event {event}: {x.shape}, {y.shape}")

            direct = np.zeros(len(x), dtype=bool)
            descendant = np.zeros(len(x), dtype=bool)
            for track, particle, weight in zip(tracks, particles, weights):
                if weight <= 0:
                    continue
                if track < 0 or track >= len(x) or particle < 0 or particle >= len(status):
                    counts["invalid_truth_links"] += 1
                    continue
                if status[particle] == 1:
                    direct[track] = True
                elif _has_status1_ancestor(particle, status, parent_begin, parent_end, parent_index):
                    descendant[track] = True

            # A direct link takes precedence if an ambiguous track has both kinds.
            descendant &= ~direct
            representative = y[:, 0] != 0
            other = representative & ~direct & ~descendant
            direct_rep = representative & direct
            descendant_rep = representative & descendant

            counts["input_tracks"] += len(x)
            counts["representative_tracks"] += int(np.sum(representative))
            counts["direct_representatives"] += int(np.sum(direct_rep))
            counts["descendant_representatives"] += int(np.sum(descendant_rep))
            counts["other_representatives"] += int(np.sum(other))

            for label, mask in (("direct", direct_rep), ("descendant", descendant_rep)):
                if not np.any(mask):
                    continue
                xx, yy = x[mask], y[mask]
                samples[f"{label}_dr"].append(_delta_r(xx, yy))
                samples[f"{label}_response"].append(xx[:, 1] / yy[:, 2])
                charged = np.abs(yy[:, 1]) > 0.5
                # omega has the charge sign for the positive solenoidal fields used here.
                samples[f"{label}_charge_match"].append((np.sign(xx[charged, 13]) == np.sign(yy[charged, 1])).astype(float))

    arrays = {
        key: np.concatenate(value) if value else np.empty(0, dtype=np.float64)
        for key, value in samples.items()
    }
    summary: dict[str, Any] = {
        "detector": detector.name,
        "process": f"p8_ee_qq, sqrt(s)={detector.energy}",
        "seeds": used_seeds,
        **counts,
    }
    nrep = max(counts["representative_tracks"], 1)
    summary["descendant_fraction_of_representatives"] = counts["descendant_representatives"] / nrep
    for label in ("direct", "descendant"):
        dr = arrays[f"{label}_dr"]
        response = arrays[f"{label}_response"]
        charge = arrays[f"{label}_charge_match"]
        summary[label] = {
            "n": len(dr),
            "delta_r_median": float(np.median(dr)) if len(dr) else None,
            "delta_r_gt_0p1_fraction": float(np.mean(dr > 0.1)) if len(dr) else None,
            "pt_response_median": float(np.median(response)) if len(response) else None,
            "pt_response_outside_0p5_2_fraction": float(np.mean((response < 0.5) | (response > 2.0))) if len(response) else None,
            "charge_mismatch_fraction": float(1.0 - np.mean(charge)) if len(charge) else None,
        }
    return summary, arrays


def _fraction_hist(ax: Any, values: np.ndarray, bins: np.ndarray, **kwargs: Any) -> None:
    if len(values):
        ax.hist(values, bins=bins, weights=np.full(len(values), 1.0 / len(values)), histtype="step", linewidth=1.8, **kwargs)


def make_plot(results: list[tuple[dict[str, Any], dict[str, np.ndarray]]], output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), sharex="row", sharey="row", constrained_layout=True)
    dr_bins = np.geomspace(1e-6, 10.0, 57)
    response_bins = np.geomspace(1e-3, 1e3, 61)
    colors = {"direct": "#2878b5", "descendant": "#e07a1f"}

    for column, (summary, arrays) in enumerate(results):
        ax_dr, ax_response = axes[:, column]
        for label in ("direct", "descendant"):
            title = "direct status-1 link" if label == "direct" else "descendant-only link"
            n = len(arrays[f"{label}_dr"])
            _fraction_hist(ax_dr, np.clip(arrays[f"{label}_dr"], dr_bins[0], dr_bins[-1]), dr_bins, color=colors[label], label=f"{title} (n={n:,})")
            _fraction_hist(
                ax_response,
                np.clip(arrays[f"{label}_response"], response_bins[0], response_bins[-1]),
                response_bins,
                color=colors[label],
                label=title,
            )

        fraction = 100.0 * summary["descendant_fraction_of_representatives"]
        ax_dr.set_title(f"{summary['detector']}  |  {summary['process'].split('=')[-1]}\n{fraction:.1f}% descendant-only representatives")
        ax_dr.set_xscale("log")
        ax_dr.set_yscale("log")
        ax_dr.set_ylim(5e-5, 1.2)
        ax_dr.grid(True, which="both", alpha=0.18)
        ax_dr.legend(frameon=False, fontsize=8, loc="upper right")

        ax_response.set_xscale("log")
        ax_response.set_yscale("log")
        ax_response.set_ylim(5e-5, 1.2)
        ax_response.axvline(1.0, color="0.35", linestyle=":", linewidth=1)
        ax_response.grid(True, which="both", alpha=0.18)

    axes[0, 0].set_ylabel("fraction of tracks / bin")
    axes[1, 0].set_ylabel("fraction of tracks / bin")
    for ax in axes[0]:
        ax.set_xlabel(r"$\Delta R(\mathrm{track},\,\mathrm{target})$")
    for ax in axes[1]:
        ax.set_xlabel(r"$p_{\mathrm{T}}^{\mathrm{track}} / p_{\mathrm{T}}^{\mathrm{target}}$")
    fig.suptitle(
        "Status-1 descendant propagation can assign a secondary track the ancestor kinematics\n"
        "Representative target tracks; raw EDM4hep truth-link provenance; Pythia8 $e^+e^- \\to q\\bar{q}$",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-files", type=int, default=5, help="Matched seed files per detector (100 events/file)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/descendant_tracks/descendant_track_diagnostic.png"),
        help="Output PNG path; PDF, SVG, and JSON files are written alongside it",
    )
    args = parser.parse_args()
    if args.max_files <= 0:
        parser.error("--max-files must be positive")

    results = [analyze_detector(detector, args.max_files) for detector in DEFAULT_DETECTORS]
    make_plot(results, args.output)
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps([summary for summary, _ in results], indent=2) + "\n")
    print(
        f"Wrote {args.output}, {args.output.with_suffix('.pdf')}, "
        f"{args.output.with_suffix('.svg')}, and {summary_path}"
    )


if __name__ == "__main__":
    main()
