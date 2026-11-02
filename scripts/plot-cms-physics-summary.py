"""Compare physics spectra with event-level sampling fluctuations."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OBSERVABLES = [
    ("particle_pt", "Particle pT", "Particle pT [GeV]", "Particles / event / GeV"),
    ("jet_pt", "Jet pT", "Jet pT [GeV]", "Jets / event / GeV"),
    ("sumpt", "Event sum-pT", "Event scalar sum-pT [GeV]", "Probability density [1/GeV]"),
]


def spectra(hist, key, min_events=20):
    reference = hist[key + "_event_reference"].astype(np.float64)
    candidate = hist[key + "_event_candidate"].astype(np.float64)
    assert reference.shape == candidate.shape and reference.ndim == 2
    assert np.isfinite(reference).all() and np.isfinite(candidate).all()
    r, c = reference.sum(0), candidate.sum(0)
    np.testing.assert_array_equal(r, hist[key + "_reference"])
    np.testing.assert_array_equal(c, hist[key + "_candidate"])
    # Events, rather than individual particles/jets, are independent sampling units.
    se = np.sqrt(len(reference) * reference.var(0, ddof=1))
    contributing = (reference > 0).sum(0)
    populated = (contributing >= min_events) & (r > 0) & (se > 0)
    populated[[0, -1]] = False
    significance = np.divide(c - r, se, out=np.zeros_like(r), where=se > 0)
    return r, c, se, contributing, populated, significance


def main(root, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    count = len(json.loads((root / "input-manifest.json").read_text())["cases"])
    rows, data, provenance = [], {}, {}
    for model in ["attention", "heptv2"]:
        for scenario in ["cuda-on", "amd-on", "cuda-off", "amd-off", "amd-pytorch-vs-nvidia"]:
            name = f"{model}-{scenario}"
            path = root / "analysis" / name / "histograms.npz"
            provenance[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
            source = root / "analysis" / name / "summary.json"
            provenance[str(source.relative_to(root))] = hashlib.sha256(source.read_bytes()).hexdigest()
            summary = json.loads(source.read_text())
            assert summary["events"] == count
            metrics = {}
            with np.load(path) as hist:
                for key, _, _, _ in OBSERVABLES:
                    assert hist[key + "_event_reference"].shape[0] == count
                    r, c, se, contributing, mask, shift = spectra(hist, key)
                    edges = hist[key + "_edges"]
                    assert len(edges) == len(r) - 1
                    populated = np.flatnonzero(mask)
                    assert len(populated)
                    beyond = populated[abs(shift[populated]) > 2]
                    worst = int(populated[np.argmax(abs(shift[populated]))])

                    def bin_record(i):
                        return dict(
                            low=float(edges[i - 1]),
                            high=float(edges[i]),
                            reference=float(r[i]),
                            candidate=float(c[i]),
                            contributing_events=int(contributing[i]),
                            shift_percent=float(100 * (c[i] - r[i]) / r[i]),
                            reference_sampling_se=float(se[i]),
                            shift_in_reference_se=float(shift[i]),
                        )

                    metrics[key] = dict(
                        populated_bins=len(populated),
                        bins_outside_2se=len(beyond),
                        max_abs_shift_in_reference_se=float(abs(shift[worst])),
                        worst_bin=bin_record(worst),
                        departures=[bin_record(i) for i in beyond],
                        sparse_regular_bins=len(edges) - 1 - len(populated),
                        candidate_entries_in_empty_reference_bins=float(c[(r == 0)].sum()),
                        reference_underflow=float(r[0]),
                        reference_overflow=float(r[-1]),
                        candidate_underflow=float(c[0]),
                        candidate_overflow=float(c[-1]),
                    )
                    lo = hist[key + "_difference_ci_low"].copy()
                    hi = hist[key + "_difference_ci_high"].copy()
                    data[name, key] = edges.copy(), r, c, se, mask, lo, hi
            rows.append(
                dict(
                    name=name,
                    model=model,
                    scenario=scenario,
                    events=count,
                    observables=metrics,
                    mean_sumpt_shift=summary["paired_event_shifts"]["sumpt"],
                )
            )

    for model in ["attention", "heptv2"]:
        for mode in ["on", "off"]:
            fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), gridspec_kw={"height_ratios": [2, 1]})
            scenarios = [(f"cuda-{mode}", "NVIDIA ONNX", "#2171b5", "-"), (f"amd-{mode}", "AMD ONNX", "#d35400", "-")]
            if mode == "off":
                scenarios.append(("amd-pytorch-vs-nvidia", "AMD native PyTorch", "#777777", ":"))
            for col, (key, title, xlabel, ylabel) in enumerate(OBSERVABLES):
                top, ratio = axes[:, col]
                edges, reference, _, se, mask, _, _ = data[f"{model}-cuda-{mode}", key]
                centers = (edges[:-1] + edges[1:]) / 2
                widths = np.diff(edges)
                top.stairs(reference[1:-1] / (count * widths), edges, color="black", label="NVIDIA native reference", linewidth=1.5)
                sampling = 100 * 2 * se[1:-1][mask[1:-1]] / reference[1:-1][mask[1:-1]]
                ratio.fill_between(centers[mask[1:-1]], -sampling, sampling, color="#bbbbbb", alpha=0.45, label="Reference sampling ±2σ")
                for scenario, label, color, style in scenarios:
                    e, r, c, error, valid, low, high = data[f"{model}-{scenario}", key]
                    np.testing.assert_array_equal(e, edges)
                    np.testing.assert_array_equal(r, reference)
                    top.stairs(c[1:-1] / (count * widths), edges, label=label, color=color, linestyle=style, linewidth=1.3)
                    use = valid[1:-1]
                    x = centers[use]
                    delta = 100 * (c[1:-1][use] - r[1:-1][use]) / r[1:-1][use]
                    ratio.plot(x, delta, color=color, linestyle=style, marker=".", markersize=3, linewidth=0.8)
                    ratio.vlines(x, 100 * low[1:-1][use] / r[1:-1][use], 100 * high[1:-1][use] / r[1:-1][use], color=color, alpha=0.65, linewidth=0.7)
                top.set_yscale("log")
                top.set_title(title)
                top.set_ylabel(ylabel)
                ratio.axhline(0, color="black", linewidth=0.6)
                ratio.set_ylabel("(Candidate − ref.) / ref. [%]")
                ratio.set_xlabel(xlabel)
                ratio.set_xlim(edges[0], edges[-1])
                top.set_xlim(edges[0], edges[-1])
                if key != "sumpt":
                    top.set_xscale("log")
                    ratio.set_xscale("log")
                for ax in [top, ratio]:
                    ax.grid(alpha=0.15)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            band, band_label = axes[1, 0].get_legend_handles_labels()
            fig.legend(handles + band, labels + band_label, loc="lower center", ncol=3, fontsize=9)
            fig.suptitle(
                ("Attention" if model == "attention" else "HEPTv2")
                + (" · fused ONNX" if mode == "on" else " · unfused ONNX and native AMD control")
                + f" · {count:,} ttbar events"
            )
            fig.tight_layout(rect=[0, 0.09, 1, 0.95])
            name = f"{model}-{'fused' if mode == 'on' else 'controls'}"
            for suffix in ["png", "pdf"]:
                fig.savefig(outdir / f"{name}.{suffix}", dpi=170)
            plt.close(fig)
    result = dict(
        events=count,
        comparison="Physics spectra relative to native NVIDIA PyTorch BF16",
        sampling_unit="event",
        reference_sampling_se="sqrt(N * sample variance of per-event bin counts), ddof=1",
        reference_band="±2 sampling standard errors of the 5000-event reference spectrum; not a paired-difference interval",
        difference_intervals="Existing paired-event bootstrap pointwise 95% intervals, 400 resamples",
        populated_bin_rule="At least 20 reference events contribute; regular bins only",
        interpretation="Pointwise comparison to finite-reference-sample fluctuations, not a global significance test or equivalence margin. Sparse tails and flows remain recorded.",
        rows=rows,
    )
    (outdir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    provenance["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (outdir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(dict(events=count, comparisons=len(rows), observables=3, figures=4)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.outdir)
