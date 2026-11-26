#!/usr/bin/env python3
"""Compare scalar sum pT and transverse-vector balance in saved PF target events."""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def pt(vectors):
    return np.hypot(vectors[:, 0], vectors[:, 1])


def fiducial(vectors):
    transverse = pt(vectors)
    eta = np.arcsinh(vectors[:, 2] / np.maximum(transverse, 1e-20))
    return (transverse > 0.5) & (np.abs(eta) < 3)


def activity(vectors):
    vector = vectors[:, :2].sum(axis=0)
    return dict(scalar_sum_pt=float(pt(vectors).sum()), sum_px=float(vector[0]), sum_py=float(vector[1]), vector_sum_pt=float(np.linalg.norm(vector)))


def describe(values):
    values = np.asarray(values, dtype=float)
    return dict(n=len(values), min=float(values.min()), median=float(np.median(values)), mean=float(values.mean()), max=float(values.max()))


def event_metrics(result):
    gen = [p for p in result["gen"] if abs(p.pdg_id) not in (12, 14, 16)]
    gen_p4 = np.array([p.p4 for p in gen]).reshape(-1, 4)
    gen_all = np.array([p.p4 for p in result["gen"]]).reshape(-1, 4)
    targets = result["target_p4"][result["target_id"] > 0]
    roots = np.array([p.p4 for p in result["truth"]]).reshape(-1, 4)
    represented = sorted(set(int(c["truth"]) for c in result["provenance"]))
    represented_p4 = roots[represented]
    by_gen = {p.truth_id: np.zeros(4) for p in gen}
    unattributed = set()
    unattributed_p4 = np.zeros(4)
    for contribution in result["provenance"]:
        truth = int(contribution["truth"])
        ancestors = result["gen_ancestors"][truth]
        if len(ancestors) != 1:
            unattributed.add(truth)
            unattributed_p4 += float(contribution["fraction"]) * result["truth"][truth].p4
            continue
        if ancestors[0] in by_gen:
            by_gen[ancestors[0]] += float(contribution["fraction"]) * result["truth"][truth].p4
    per_gen = []
    for p, keep in zip(gen, fiducial(gen_p4)):
        if keep:
            original_pt = float(np.hypot(*p.p4[:2]))
            assigned_pt = float(np.hypot(*by_gen[p.truth_id][:2]))
            per_gen.append(dict(gen_id=p.truth_id, pdg_id=p.pdg_id, gen_pt=original_pt, assigned_pt=assigned_pt, pt_ratio=assigned_pt / original_pt))
    row = dict(
        event_id=list(result["event_id"]),
        gen_visible=activity(gen_p4),
        gen_all=activity(gen_all),
        targets=activity(targets),
        represented_roots=activity(represented_p4),
        gen_fiducial=activity(gen_p4[fiducial(gen_p4)]),
        targets_fiducial=activity(targets[fiducial(targets)]),
        fiducial_gen_pt_closure=per_gen,
        unattributed_truth_rows=sorted(unattributed),
        unattributed_p4=unattributed_p4.tolist(),
    )
    for gen_key, target_key, label in [("gen_visible", "targets", "inclusive"), ("gen_fiducial", "targets_fiducial", "fiducial")]:
        g, t = row[gen_key], row[target_key]
        residual = np.hypot(t["sum_px"] - g["sum_px"], t["sum_py"] - g["sum_py"])
        row[label] = dict(
            scalar_pt_ratio=t["scalar_sum_pt"] / g["scalar_sum_pt"],
            vector_residual_GeV=float(residual),
            vector_residual_over_gen_sum_pt=float(residual / g["scalar_sum_pt"]),
        )
    row["ancestry_fiducial_pt_ratio"] = sum(p["assigned_pt"] for p in per_gen) / sum(p["gen_pt"] for p in per_gen)
    row["merge_scalar_pt_reduction_GeV"] = row["represented_roots"]["scalar_sum_pt"] - row["targets"]["scalar_sum_pt"]
    row["represented_vector_residual_GeV"] = float(
        np.hypot(row["represented_roots"]["sum_px"] - row["targets"]["sum_px"], row["represented_roots"]["sum_py"] - row["targets"]["sum_py"])
    )
    if row["merge_scalar_pt_reduction_GeV"] < -1e-8 or row["represented_vector_residual_GeV"] > 1e-8:
        raise AssertionError("Invalid scalar/vector transverse closure against represented roots")
    return row


def plots(rows, output):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    colors = ["#1764ab", "#dd6b20"]
    for label, gen_key, target_key, color in zip(["inclusive", "fiducial"], ["gen_visible", "gen_fiducial"], ["targets", "targets_fiducial"], colors):
        x = [r[gen_key]["scalar_sum_pt"] for r in rows]
        y = [r[target_key]["scalar_sum_pt"] for r in rows]
        axes[0, 0].scatter(x, y, label=label, color=color)
        axes[0, 1].plot([r["event_id"][-1] for r in rows], [r[label]["scalar_pt_ratio"] for r in rows], "o-", label=label, color=color)
        axes[1, 0].scatter([r[gen_key]["vector_sum_pt"] for r in rows], [r[target_key]["vector_sum_pt"] for r in rows], label=label, color=color)
        axes[1, 1].plot(
            [r["event_id"][-1] for r in rows], [100 * r[label]["vector_residual_over_gen_sum_pt"] for r in rows], "o-", label=label, color=color
        )
    for ax in (axes[0, 0], axes[1, 0]):
        low, high = ax.get_xlim()
        ax.plot([0, high], [0, high], color="0.6", linestyle=":")
        ax.legend()
    axes[0, 0].set(xlabel="Visible GEN scalar sum pT [GeV]", ylabel="Target scalar sum pT [GeV]")
    axes[0, 1].axhline(1, color="0.6", linestyle=":")
    axes[0, 1].set(xlabel="Event", ylabel="Target / GEN scalar sum pT")
    axes[0, 1].legend()
    axes[1, 0].set(xlabel="|Vector sum pT|, visible GEN [GeV]", ylabel="|Vector sum pT|, targets [GeV]")
    axes[1, 1].set(xlabel="Event", ylabel="Transverse-vector residual / GEN scalar sum pT [%]")
    axes[1, 1].legend()
    axes[0, 2].plot([r["event_id"][-1] for r in rows], [r["ancestry_fiducial_pt_ratio"] for r in rows], "o-")
    axes[0, 2].axhline(1, color="0.6", linestyle=":")
    axes[0, 2].set(xlabel="Event", ylabel="Ancestry-grouped assigned / fiducial GEN sum pT")
    ratios = [p["pt_ratio"] for r in rows for p in r["fiducial_gen_pt_closure"]]
    axes[1, 2].hist(ratios, bins=np.linspace(0, max(2, max(ratios) * 1.01), 60), histtype="step")
    axes[1, 2].set(xlabel="Assigned descendant pT / GEN ancestor pT", ylabel="Fiducial GEN particles")
    if len(rows) > 100:
        # Distributions are easier to read than joined event traces at high statistics.
        for ax in (axes[0, 1], axes[0, 2], axes[1, 1]):
            ax.clear()
        residual_max = max(100 * r[label]["vector_residual_over_gen_sum_pt"] for r in rows for label in ("inclusive", "fiducial"))
        for label, color in zip(("inclusive", "fiducial"), colors):
            axes[0, 1].hist(
                [r[label]["scalar_pt_ratio"] for r in rows], bins=np.linspace(0.5, 1.05, 56), histtype="step", linewidth=2, label=label, color=color
            )
            axes[1, 1].hist(
                [100 * r[label]["vector_residual_over_gen_sum_pt"] for r in rows],
                bins=np.linspace(0, residual_max * 1.01, 60),
                histtype="step",
                linewidth=2,
                label=label,
                color=color,
            )
        axes[0, 1].axvline(1, color="0.6", linestyle=":")
        axes[0, 1].set(xlabel="Target / GEN scalar sum pT", ylabel="Events")
        axes[0, 1].legend()
        axes[0, 2].hist([r["ancestry_fiducial_pt_ratio"] for r in rows], bins=np.linspace(0.6, 1.02, 50), histtype="step", linewidth=2)
        axes[0, 2].axvline(1, color="0.6", linestyle=":")
        axes[0, 2].set(xlabel="Ancestry-grouped assigned / fiducial GEN sum pT", ylabel="Events")
        axes[1, 1].set(xlabel="Transverse-vector residual / GEN scalar sum pT [%]", ylabel="Events", yscale="log")
        axes[1, 1].legend()
        axes[1, 2].set_yscale("log")
    fig.suptitle(
        "PU0 ttbar: scalar transverse activity and transverse-vector balance\nFiducial: visible particles/targets with pT > 0.5 GeV, |eta| < 3; all reconstructed inputs retained"
    )
    fig.savefig(output / "transverse_activity.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [event_metrics(result) for result in pickle.loads(args.events.read_bytes())]
    closure = [p for row in rows for p in row["fiducial_gen_pt_closure"]]
    summary = dict(
        events=len(rows),
        reference="Visible status-1 particles of the full pp collision; not a hard-process-only selection",
        inclusive_gen_sum_pt=describe([r["gen_visible"]["scalar_sum_pt"] for r in rows]),
        inclusive_target_sum_pt=describe([r["targets"]["scalar_sum_pt"] for r in rows]),
        inclusive_pt_ratio=describe([r["inclusive"]["scalar_pt_ratio"] for r in rows]),
        aggregate_inclusive_pt_ratio=sum(r["targets"]["scalar_sum_pt"] for r in rows) / sum(r["gen_visible"]["scalar_sum_pt"] for r in rows),
        fiducial_pt_ratio=describe([r["fiducial"]["scalar_pt_ratio"] for r in rows]),
        aggregate_fiducial_pt_ratio=sum(r["targets_fiducial"]["scalar_sum_pt"] for r in rows) / sum(r["gen_fiducial"]["scalar_sum_pt"] for r in rows),
        ancestry_fiducial_pt_ratio=describe([r["ancestry_fiducial_pt_ratio"] for r in rows]),
        fiducial_gen_particles=len(closure),
        fiducial_gen_pt_within_1pct=sum(abs(p["pt_ratio"] - 1) < 0.01 for p in closure),
        max_represented_vector_residual_GeV=max(r["represented_vector_residual_GeV"] for r in rows),
        merge_scalar_pt_reduction_GeV=describe([r["merge_scalar_pt_reduction_GeV"] for r in rows]),
        inclusive_vector_residual_GeV=describe([r["inclusive"]["vector_residual_GeV"] for r in rows]),
        fiducial_vector_residual_GeV=describe([r["fiducial"]["vector_residual_GeV"] for r in rows]),
        gen_including_neutrinos_vector_sum_pt=describe([r["gen_all"]["vector_sum_pt"] for r in rows]),
        represented_roots_without_unique_gen_ancestry=sum(len(r["unattributed_truth_rows"]) for r in rows),
        unattributed_energy_GeV=sum(r["unattributed_p4"][3] for r in rows),
    )
    (args.output / "transverse_activity.json").write_text(json.dumps(dict(summary=summary, events=rows), indent=2, allow_nan=False) + "\n")
    lines = [
        "PU0 ttbar transverse-activity validation",
        "",
        "Scalar sum pT = sum_i hypot(px_i,py_i): visible transverse activity, reduced sensitivity to forward beam remnants.",
        "Vector sum pT = hypot(sum_i px_i,sum_i py_i): transverse imbalance; visible-GEN imbalance is related to neutrino missing pT.",
        "Reference: full visible status-1 GEN (neutrinos excluded). This includes underlying event, not just the hard process.",
        "Fiducial: pT > 0.5 GeV, |eta| < 3, applied to GEN and targets independently.",
        "Ancestry closure groups target contributions back to each selected GEN ancestor before computing pT.",
        "",
        f"Events: {len(rows)}",
        f"Visible GEN scalar sum pT: {summary['inclusive_gen_sum_pt']['min']:.2f} to {summary['inclusive_gen_sum_pt']['max']:.2f} GeV",
        f"Median inclusive target/GEN scalar sum pT: {summary['inclusive_pt_ratio']['median']:.6f}",
        f"Median fiducial target/GEN scalar sum pT: {summary['fiducial_pt_ratio']['median']:.6f}",
        f"Aggregate fiducial target/GEN scalar sum pT: {summary['aggregate_fiducial_pt_ratio']:.6f}",
        f"Median fiducial transverse-vector residual: {summary['fiducial_vector_residual_GeV']['median']:.3f} GeV",
        f"Per-GEN fiducial pT closure within 1%: {summary['fiducial_gen_pt_within_1pct']}/{summary['fiducial_gen_particles']}",
        f"Maximum target/represented-root vector residual: {summary['max_represented_vector_residual_GeV']:.4g} GeV",
        f"Median scalar sum pT reduction from merging: {summary['merge_scalar_pt_reduction_GeV']['median']:.4f} GeV",
        f"Represented roots without unique GEN ancestry: {summary['represented_roots_without_unique_gen_ancestry']}",
        f"Their energy: {summary['unattributed_energy_GeV']:.6f} GeV; retained in global sums, excluded from per-GEN ancestry closure.",
        "",
        "Scalar sum pT need not be conserved by vector merging. The transverse vector is conserved against represented roots.",
        "The fiducial deficit also includes missing input support and threshold migration; it is not a hard-process-only efficiency.",
    ]
    (args.output / "report.txt").write_text("\n".join(lines) + "\n")
    plots(rows, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
