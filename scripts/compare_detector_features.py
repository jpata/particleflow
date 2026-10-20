#!/usr/bin/env python3
"""Compare ttbar parquet and ArrayRecord TFDS features with explicit column semantics."""
import argparse
from collections import defaultdict
from collections import Counter
import hashlib
import json
from pathlib import Path

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow_datasets as tfds  # noqa: E402

from mlpf.conf import EDM4HEP, X_FEATURES, ParticleFeatures  # noqa: E402
from mlpf.jet_utils import _match_jets_event  # noqa: E402


DETECTORS = ["colliderml", "clic", "cld", "idea", "maia"]
COLORS = ["#a23b72", "#0072b2", "#d55e00", "#009e73", "#9467bd"]
LABELS = np.array([0, 211, 130, 22, 11, 13])
CML_TRACK = [
    "elemtype",
    "pt",
    "eta",
    "sin_phi",
    "cos_phi",
    "p",
    "D0",
    "Z0",
    "theta",
    "qop",
    "tanLambda",
    "omega",
    "radiusOfInnermostHit",
    "n_meas",
    "unused14",
    "unused15",
    "unused16",
]


def matrix(value, width):
    arr = np.asarray(value)
    return arr.reshape(-1, width) if arr.size else np.empty((0, width))


def parquet_events(paths, limit, width):
    count = 0
    for path in paths:
        data = ak.from_parquet(path)
        for index in range(len(data["X_track"])):
            wanted = ("X_track", "X_cluster", "ytarget_track", "ytarget_cluster", "ycand_track", "ycand_cluster", "genmet", "genjet", "targetjet")
            event = {key: ak.to_numpy(data[key][index]) for key in wanted if key in data.fields}
            xt = np.asarray(event["X_track"])
            xc = np.asarray(event["X_cluster"])
            xt = xt.reshape(-1, xt.shape[-1]) if xt.size else np.empty((0, width))
            xc = matrix(xc, width)
            if xt.shape[1] < width:
                xt = np.pad(xt, ((0, 0), (0, width - xt.shape[1])))
            event["X"] = np.concatenate([xt, xc])
            event["ytarget"] = np.concatenate([matrix(event["ytarget_track"], 14), matrix(event["ytarget_cluster"], 14)])
            if "ycand_track" in event:
                event["ycand"] = np.concatenate([matrix(event["ycand_track"], 14), matrix(event["ycand_cluster"], 14)])
            event["genjets"] = event["genjet"]
            event["targetjets"] = event["targetjet"]
            yield event
            count += 1
            if count >= limit:
                return


def tfds_events(path, limit, split, seed):
    builder = tfds.builder_from_directory(str(path))
    source = builder.as_data_source(split=split)
    indices = np.sort(np.random.default_rng(seed).choice(len(source), min(limit, len(source)), replace=False))
    for index in indices:
        event = source[int(index)]
        for key in ("ytarget", "ycand"):
            event[key] = np.array(event[key], copy=True)
            classes = event[key][:, 0]
            if not np.all((classes == classes.astype(int)) & (classes >= 0) & (classes < len(LABELS))):
                raise ValueError(f"Invalid class indices in {path}: {classes}")
            event[key][:, 0] = LABELS[classes.astype(int)]
        yield event


def jet_matching_values(genjets, targetjets, match_dr):
    gen = matrix(genjets, 4)
    target = matrix(targetjets, 4)
    gen = gen[np.isfinite(gen).all(axis=1) & (gen[:, 0] > 0)]
    target = target[np.isfinite(target).all(axis=1) & (target[:, 0] > 0)]
    gi, ti = _match_jets_event(gen[:, 1], gen[:, 2], target[:, 1], target[:, 2], match_dr)
    gen_matched = np.zeros(len(gen), dtype=bool)
    target_matched = np.zeros(len(target), dtype=bool)
    gen_matched[gi] = True
    target_matched[ti] = True
    return {
        "gen_pt": gen[:, 0],
        "gen_eta": gen[:, 1],
        "gen_matched": gen_matched,
        "target_pt": target[:, 0],
        "target_eta": target[:, 1],
        "target_matched": target_matched,
        "matched_gen_pt": gen[gi, 0],
        "matched_gen_eta": gen[gi, 1],
        "pt_ratio": target[ti, 0] / gen[gi, 0],
    }


def collect(events, detector, features=None, match_dr=0.1):
    values = defaultdict(list)
    checks = {
        "events": 0,
        "nonfinite": {},
        "alignment_failures": 0,
        "phi_norm_failures": 0,
        "unknown_target_pids": 0,
        "target_jet_index_failures": 0,
        "unknown_element_types": 0,
        "serialization_events": 0,
        "serialization_mismatches": 0,
        "track_p_kinematics_failures": 0,
        "cluster_et_kinematics_failures": 0,
        "target_energy_below_momentum": 0,
    }
    track_names = CML_TRACK if detector == "colliderml" else EDM4HEP.TrackFeatures.get_names()
    cluster_names = EDM4HEP.ClusterFeatures.get_names()
    y_names = ParticleFeatures.get_names()

    def add(key, value):
        values[key].append(np.asarray(value).reshape(-1))

    for event in events:
        checks["events"] += 1
        # ColliderML keeps its own 17-column layout; the EDM4hep detectors share a wider schema
        x = matrix(event["X"], len(X_FEATURES[detector]))
        y = matrix(event["ytarget"], 14)
        checks["unknown_element_types"] += int((~np.isin(x[:, 0], [1, 2])).sum())
        if features is not None:
            encoded = {
                "X": x.astype(np.float32),
                "ytarget": y.astype(np.float32),
                "ycand": matrix(event.get("ycand", y if detector == "idea" else np.zeros_like(y)), 14).astype(np.float32),
                "genmet": np.float32(np.asarray(event["genmet"]).reshape(-1)[0]),
                "genjets": matrix(event["genjets"], 4).astype(np.float32),
                "targetjets": matrix(event["targetjets"], 4).astype(np.float32),
            }
            for key in ("ytarget", "ycand"):
                encoded[key] = encoded[key].copy()
                encoded[key][:, 0] = [int(np.flatnonzero(LABELS == pid)[0]) for pid in encoded[key][:, 0]]
            decoded = features.deserialize_example_np(features.serialize_example(encoded))
            checks["serialization_events"] += 1
            checks["serialization_mismatches"] += int(any(not np.array_equal(value, decoded[key], equal_nan=True) for key, value in encoded.items()))
        checks["alignment_failures"] += int(len(x) != len(y))
        for key in ("X", "ytarget", "ycand", "genmet", "genjets", "targetjets"):
            if key in event:
                nbad = int((~np.isfinite(np.asarray(event[key]))).sum())
                checks["nonfinite"][key] = checks["nonfinite"].get(key, 0) + nbad
        checks["phi_norm_failures"] += int((np.abs(x[:, 3] ** 2 + x[:, 4] ** 2 - 1) > 1e-3).sum())
        checks["unknown_target_pids"] += int((~np.isin(y[:, 0], LABELS)).sum())
        tracks = x[x[:, 0] == 1]
        clusters = x[x[:, 0] == 2]
        checks["track_p_kinematics_failures"] += int((~np.isclose(tracks[:, 5], tracks[:, 1] * np.cosh(tracks[:, 2]), rtol=1e-3, atol=1e-4)).sum())
        checks["cluster_et_kinematics_failures"] += int(
            (~np.isclose(clusters[:, 1], clusters[:, 5] / np.cosh(clusters[:, 2]), rtol=1e-3, atol=1e-4)).sum()
        )
        for kind, code, names in (("track", 1, track_names), ("cluster", 2, cluster_names)):
            rows = x[x[:, 0] == code]
            add(f"event/n_{kind}", len(rows))
            # ColliderML's narrower layout has no columns for the trailing EDM4hep cluster features
            for col, name in enumerate(names[: x.shape[1]]):
                if not name.startswith("unused") and name != "elemtype":
                    add(f"{kind}/{name}", rows[:, col])
            add(f"{kind}/phi", np.arctan2(rows[:, 3], rows[:, 4]))
        active = y[y[:, 0] != 0]
        checks["target_energy_below_momentum"] += int((active[:, 6] + 1e-3 < active[:, 2] * np.cosh(active[:, 3])).sum())
        add("event/n_target", len(active))
        add("event/target_energy_sum", active[:, 6].sum())
        add("event/target_pt_sum", active[:, 2].sum())
        add("event/target_active_fraction", len(active) / max(len(y), 1))
        add("event/track_p_sum", x[x[:, 0] == 1, 5].sum())
        add("event/cluster_energy_sum", x[x[:, 0] == 2, 5].sum())
        for col, name in enumerate(y_names):
            add(f"target/{name}", active[:, col])
        add("target/phi", np.arctan2(active[:, 4], active[:, 5]))
        add("event/genmet", event["genmet"])
        for kind in ("genjets", "targetjets"):
            jets = matrix(event[kind], 4)
            add(f"event/n_{kind}", len(jets))
            for col, name in enumerate(("pt", "eta", "phi", "energy")):
                add(f"{kind}/{name}", jets[:, col])
        for name, value in jet_matching_values(event["genjets"], event["targetjets"], match_dr).items():
            add(f"jetmatch/{name}", value)
        ji = active[:, y_names.index("jet_idx")]
        checks["target_jet_index_failures"] += int(((ji < -1) | (ji >= len(matrix(event["targetjets"], 4))) | (ji != np.floor(ji))).sum())
        if "ycand" in event:
            candidate = matrix(event["ycand"], 14)
            add("event/n_candidate", (candidate[:, 0] != 0).sum())
        add("diagnostic/tanLambda_minus_sinh_eta", tracks[:, track_names.index("tanLambda")] - np.sinh(tracks[:, 2]))
    return {key: np.concatenate(parts) for key, parts in values.items()}, checks


def stats(arr):
    finite = arr[np.isfinite(arr)].astype(np.float64, copy=False)
    return {
        "count": len(arr),
        "nonfinite": int(len(arr) - len(finite)),
        "zero_fraction": float(np.mean(finite == 0)) if len(finite) else None,
        "min": float(finite.min()) if len(finite) else None,
        "max": float(finite.max()) if len(finite) else None,
        "quantiles_01_50_99": np.quantile(finite, [0.01, 0.5, 0.99]).tolist() if len(finite) else [],
    }


def verify_persisted_colliderml(parquet_path, tfds_path, detector="colliderml"):
    if detector == "colliderml":
        from mlpf.heptfds.colliderml_utils.utils import generate_examples
    else:
        from mlpf.heptfds.edm4hep_utils.utils_pf import generate_examples

    def fingerprint(event):
        digest = hashlib.sha256()
        for key in sorted(event):
            value = np.ascontiguousarray(np.asarray(event[key], dtype=np.float32))
            digest.update(key.encode())
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
        return digest.hexdigest()

    expected = Counter(fingerprint(event) for _, event in generate_examples(sorted(Path(parquet_path).glob("*.parquet"))))
    builder = tfds.builder_from_directory(tfds_path)
    actual = Counter()
    for split in builder.info.splits:
        source = builder.as_data_source(split=split)
        for index in range(len(source)):
            actual[fingerprint(source[index])] += 1
    return {
        "parquet_examples": sum(expected.values()),
        "tfds_examples": sum(actual.values()),
        "missing": sum((expected - actual).values()),
        "unexpected": sum((actual - expected).values()),
        "exact_match": expected == actual,
    }


def layout_comparison_figure(fig, axes, title, synthetic=False):
    """Reserve a fixed-height header, including on short one-row sheets."""
    height = fig.get_figheight()
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.85 / height))
    main_title = fig.suptitle(title, fontsize=12, y=1 - 0.12 / height, va="top")
    column_titles = []
    for detector, ax in zip(DETECTORS, axes[0]):
        label = detector.upper() + (" (SYNTHETIC)" if detector == "colliderml" and synthetic else "")
        position = ax.get_position()
        column_titles.append(fig.text((position.x0 + position.x1) / 2, 1 - 0.48 / height, label, ha="center", va="top", fontweight="bold"))
    return main_title, column_titles


def plot_group(datasets, keys, title, path, synthetic):
    fig, axes = plt.subplots(len(keys), len(DETECTORS), figsize=(4 * len(DETECTORS), max(3.6, 2.35 * len(keys))), squeeze=False)
    for row, key in enumerate(keys):
        arrays = [datasets.get(d, {}).get(key, np.array([])) for d in DETECTORS]
        finite = [a[np.isfinite(a)] for a in arrays]
        pooled = np.concatenate(finite)
        if not len(pooled):
            continue
        # Give every detector equal influence on the visible range even when
        # their object multiplicities differ by orders of magnitude.
        lo = min(np.quantile(arr, 0.005) for arr in finite if len(arr))
        hi = max(np.quantile(arr, 0.995) for arr in finite if len(arr))
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        categorical = key == "target/PDG"
        if categorical:
            finite = [np.array([np.flatnonzero(LABELS == p)[0] for p in a]) for a in finite]
            lo, hi = -0.5, 5.5
        positive = pooled[pooled > 0]
        scale = max(float(np.quantile(positive, 0.01)), hi / 1000) if len(positive) else 1
        log_x = not categorical and lo >= 0 and hi / max(scale, 1e-9) > 100
        edges = np.arange(-0.5, 6.5) if categorical else np.linspace(lo, hi, 51)
        if log_x:
            edges = scale * np.expm1(np.linspace(np.log1p(lo / scale), np.log1p(hi / scale), 51))
        for col, (detector, arr, ax) in enumerate(zip(DETECTORS, finite, axes[row])):
            if len(arr):
                counts, _ = np.histogram(arr, edges)
                fractions = counts / len(arr)
                ax.stairs(fractions, edges, color=COLORS[col], linewidth=1.4)
                outside = np.mean((arr < lo) | (arr > hi))
                annotation = f"n={len(arr):,}" if categorical else f"n={len(arr):,}\nmedian={np.median(arr):.3g}\noutside={outside:.1%}"
                ax.text(
                    0.98,
                    0.94,
                    annotation,
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1},
                )
            else:
                ax.text(0.5, 0.5, "Unavailable", transform=ax.transAxes, ha="center")
            ax.set_xlim(lo, hi)
            if log_x:
                ax.set_xscale("symlog", linthresh=scale)
            ax.set_xlabel(key)
            if categorical:
                ax.set_xticks(range(6), ["none", "chhad", "nhad", "gamma", "e", "mu"])
            ax.grid(alpha=0.2)
            if col == 0:
                ax.set_ylabel("Fraction / bin")
        ymax = max(ax.get_ylim()[1] for ax in axes[row])
        for ax in axes[row]:
            ax.set_ylim(0, ymax)
    layout_comparison_figure(fig, axes, title + " — shared bins/axes; union of detector 0.5–99.5% ranges", synthetic)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def binned_fraction(values, matched, edges):
    total = np.histogram(values, edges)[0]
    passed = np.histogram(values[np.asarray(matched, dtype=bool)], edges)[0]
    fraction = np.divide(passed, total, out=np.full(len(total), np.nan), where=total > 0)
    # Wilson 68% interval remains nonzero at efficiency zero or one.
    denominator = 1 + np.divide(1.0, total, out=np.zeros(len(total)), where=total > 0)
    center = (fraction + np.divide(0.5, total, out=np.zeros(len(total)), where=total > 0)) / denominator
    half = (
        np.sqrt(
            np.divide(fraction * (1 - fraction), total, out=np.zeros(len(total)), where=total > 0)
            + np.divide(0.25, total**2, out=np.zeros(len(total)), where=total > 0)
        )
        / denominator
    )
    return fraction, center - half, center + half, passed, total


def plot_jet_matching(datasets, stage, path, match_dr):
    fig, axes = plt.subplots(5, len(DETECTORS), figsize=(4 * len(DETECTORS), 13), squeeze=False)
    all_pt = np.concatenate([v["jetmatch/gen_pt"] for v in datasets.values()])
    all_eta = np.concatenate([v["jetmatch/gen_eta"] for v in datasets.values()])
    all_target_pt = np.concatenate([v["jetmatch/target_pt"] for v in datasets.values()])
    all_target_eta = np.concatenate([v["jetmatch/target_eta"] for v in datasets.values()])
    max_pt = max(200.0, float(np.max(np.concatenate([all_pt, all_target_pt]), initial=0)) * 1.01)
    min_pt = min(3.0, float(np.min(np.concatenate([all_pt, all_target_pt]), initial=3)))
    pt_edges = np.geomspace(min_pt, max_pt, 13)
    eta_max = max(3.0, np.ceil(np.max(np.abs(np.concatenate([all_eta, all_target_eta])), initial=0)))
    eta_edges = np.linspace(-eta_max, eta_max, 13)
    response_edges = np.linspace(0.5, 1.5, 81)
    summaries = {}
    for col, detector in enumerate(DETECTORS):
        if detector not in datasets:
            for ax in axes[:, col]:
                ax.text(0.5, 0.5, "Unavailable", transform=ax.transAxes, ha="center")
            continue
        data = datasets[detector]
        ratio = data["jetmatch/pt_ratio"]
        counts = np.histogram(ratio, response_edges)[0]
        axes[0, col].stairs(counts / max(len(ratio), 1), response_edges, color=COLORS[col])
        axes[0, col].axvline(1, color="black", linestyle=":", linewidth=1)
        axes[0, col].set_xlim(0.5, 1.5)
        axes[0, col].set_xlabel(r"Matched $p_T^{target}/p_T^{gen}$")
        outside = np.mean((ratio < 0.5) | (ratio > 1.5)) if len(ratio) else 0
        median = f"{np.median(ratio):.3f}" if len(ratio) else "n/a"
        axes[0, col].text(
            0.98,
            0.95,
            f"matches={len(ratio)}\nmedian={median}\noutside={outside:.1%}",
            transform=axes[0, col].transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        summaries[detector] = {
            "match_dr": match_dr,
            "gen_jets": len(data["jetmatch/gen_pt"]),
            "target_jets": len(data["jetmatch/target_pt"]),
            "matched_pairs": len(ratio),
            "pt_ratio": stats(ratio),
            "bins": {},
        }
        for dim, edges, response_row, fraction_row in (("pt", pt_edges, 1, 3), ("eta", eta_edges, 2, 4)):
            centers = np.sqrt(edges[:-1] * edges[1:]) if dim == "pt" else (edges[:-1] + edges[1:]) / 2
            matched_x = data[f"jetmatch/matched_gen_{dim}"]
            quantiles = np.full((len(centers), 3), np.nan)
            for index in range(len(centers)):
                below_upper = matched_x <= edges[index + 1] if index == len(centers) - 1 else matched_x < edges[index + 1]
                selected = ratio[(matched_x >= edges[index]) & below_upper]
                if len(selected):
                    quantiles[index] = np.quantile(selected, [0.16, 0.5, 0.84])
            ax = axes[response_row, col]
            ax.plot(centers, quantiles[:, 1], "o-", color=COLORS[col], markersize=3)
            ax.fill_between(centers, quantiles[:, 0], quantiles[:, 2], color=COLORS[col], alpha=0.2)
            ax.axhline(1, color="black", linestyle=":", linewidth=1)
            ax.set_xlabel(r"Genjet $p_T$ [GeV]" if dim == "pt" else r"Genjet $\eta$")
            for collection, style, label in (("gen", "-", "Genjet efficiency"), ("target", "--", "Targetjet purity")):
                fraction, lower, upper, passed, total = binned_fraction(
                    data[f"jetmatch/{collection}_{dim}"], data[f"jetmatch/{collection}_matched"], edges
                )
                ax = axes[fraction_row, col]
                ax.plot(centers, fraction, style, color=COLORS[col], label=label, marker="o", markersize=3)
                ax.fill_between(centers, lower, upper, color=COLORS[col], alpha=0.12)
                summaries[detector]["bins"][f"{collection}_{dim}"] = {"edges": edges.tolist(), "matched": passed.tolist(), "total": total.tolist()}
            axes[fraction_row, col].set_ylim(0, 1.05)
            axes[fraction_row, col].set_xlabel(r"Jet $p_T$ [GeV] (each collection)" if dim == "pt" else r"Jet $\eta$ (each collection)")
            for row in (response_row, fraction_row):
                axes[row, col].set_xlim(edges[0], edges[-1])
                if dim == "pt":
                    axes[row, col].set_xscale("log")
        axes[3, col].legend(fontsize=8, loc="lower right")
    for row, label in enumerate(
        ("Fraction / bin", "pT ratio: median, 16–84%", "pT ratio: median, 16–84%", "Matched fraction (68% Wilson)", "Matched fraction (68% Wilson)")
    ):
        axes[row, 0].set_ylabel(label)
        if row < 3:
            upper = max(ax.get_ylim()[1] for ax in axes[row])
            lower = min(ax.get_ylim()[0] for ax in axes[row])
            for ax in axes[row]:
                ax.set_ylim(0, upper) if row == 0 else ax.set_ylim(min(0.9, lower), max(1.1, upper))
        for ax in axes[row]:
            ax.grid(alpha=0.2)
    layout_comparison_figure(fig, axes, f"ttbar • {stage} • genjet ↔ targetjet: one-to-one ΔR < {match_dr:g}; no pT response cut")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", action="append", default=[], help="detector=directory")
    parser.add_argument("--tfds", action="append", default=[], help="detector=version-directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-events", type=int, default=100)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--jet-match-dr", type=float, default=0.1, help="One-to-one angular jet matching cut; no response cut")
    parser.add_argument("--colliderml-synthetic", action="store_true")
    parser.add_argument(
        "--verify-colliderml-roundtrip", action="store_true", help="Compare all native loader examples against all stored ColliderML TFDS records"
    )
    parser.add_argument(
        "--verify-maia-roundtrip", action="store_true", help="Compare all native loader examples against all stored MAIA TFDS records"
    )
    parser.add_argument("--maia-validation-report", type=Path, help="Include the independent MAIA physics-gate report")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"settings": vars(args) | {"output_dir": str(args.output_dir)}, "samples": {}}
    if args.maia_validation_report:
        report["settings"]["maia_validation_report"] = str(args.maia_validation_report)
        report["maia_validation"] = json.loads(args.maia_validation_report.read_text())
    report["feature_columns"] = {
        "track_key4hep": EDM4HEP.TrackFeatures.get_names(),
        "track_colliderml": CML_TRACK,
        "cluster": EDM4HEP.ClusterFeatures.get_names(),
        "target": ParticleFeatures.get_names(),
    }
    tfds_paths = dict(spec.split("=", 1) for spec in args.tfds)
    parquet_paths = dict(spec.split("=", 1) for spec in args.parquet)
    if args.verify_colliderml_roundtrip:
        report["colliderml_persisted_roundtrip"] = verify_persisted_colliderml(parquet_paths["colliderml"], tfds_paths["colliderml"])
        print("Persisted ColliderML roundtrip:", report["colliderml_persisted_roundtrip"], flush=True)
    if args.verify_maia_roundtrip:
        report["maia_persisted_roundtrip"] = verify_persisted_colliderml(parquet_paths["maia"], tfds_paths["maia"], "maia")
        print("Persisted MAIA roundtrip:", report["maia_persisted_roundtrip"], flush=True)
    images = []
    for stage, specs in (("parquet", args.parquet), ("tfds", args.tfds)):
        datasets = {}
        for spec in specs:
            detector, path = spec.split("=", 1)
            if detector not in DETECTORS:
                raise ValueError(detector)
            paths = sorted(Path(path).glob("*.parquet")) if stage == "parquet" else [Path(path)]
            if not paths:
                raise FileNotFoundError(path)
            events = (
                parquet_events(paths, args.num_events, len(X_FEATURES[detector]))
                if stage == "parquet"
                else tfds_events(path, args.num_events, args.split, args.seed)
            )
            features = tfds.builder_from_directory(tfds_paths[detector]).info.features if stage == "parquet" and detector in tfds_paths else None
            values, checks = collect(events, detector, features, args.jet_match_dr)
            datasets[detector] = values
            report["samples"][f"{stage}/{detector}"] = {
                "paths": [str(p) for p in paths],
                "checks": checks,
                "features": {k: stats(v) for k, v in values.items()},
            }
            print(stage, detector, checks, flush=True)
        if not datasets:
            continue
        filename = f"{stage}_jet_matching.png"
        report.setdefault("jet_matching", {})[stage] = plot_jet_matching(datasets, stage, args.output_dir / filename, args.jet_match_dr)
        images.append(filename)
        allkeys = sorted(set().union(*(v.keys() for v in datasets.values())))
        for group in ("event", "track", "cluster", "target", "genjets", "targetjets", "diagnostic"):
            keys = [k for k in allkeys if k.startswith(group + "/")]
            preferred = {
                "event": ["n_track", "n_cluster", "n_target", "target_active_fraction", "target_energy_sum", "genmet"],
                "track": ["pt", "p", "eta", "phi", "D0", "Z0"],
                "cluster": ["energy", "et", "eta", "phi", "num_hits", "energy_ecal"],
                "target": ["PDG", "charge", "pt", "eta", "phi", "energy"],
            }.get(group, [])
            first = [f"{group}/{key}" for key in preferred if f"{group}/{key}" in keys]
            keys = first + [key for key in keys if key not in first]
            for start in range(0, len(keys), 6):
                filename = f"{stage}_{group}_{start // 6 + 1}.png"
                plot_group(datasets, keys[start : start + 6], f"ttbar • {stage} • {group}", args.output_dir / filename, args.colliderml_synthetic)
                images.append(filename)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    note = (
        "ColliderML is SYNTHETIC: this validates schema and plumbing only."
        if args.colliderml_synthetic
        else "ColliderML: pp ttbar; CLIC/CLD/IDEA: ee ttbar at 380/365/365 GeV; MAIA: muon-collider ttbar. Shape agreement is not expected."
    )
    html = "<!doctype html><meta charset='utf-8'><title>ttbar detector comparison</title><style>body{font:16px sans-serif;max-width:1600px;margin:30px auto}img{width:100%}</style>"
    html += f"<h1>ttbar detector feature comparison</h1><p>{note}</p><p>CLIC uses ee ttbar at 380 GeV; CLD and IDEA use ee ttbar at 365 GeV. IDEA tracks are truth-seeded and candidates are target oracles. ColliderML has no PF baseline. For CLIC/CLD/IDEA, stored TFDS and newly converted ROOT samples are independent; distributions are not an event-matched serialization test. ColliderML and MAIA TFDS are built from the plotted parquet samples. The serialization checks separately verify each parquet event against the TFDS feature encoder/decoder. Targets exclude PDG=0 padding/unassigned rows. Non-finite values are counted in summary.json and excluded from histograms. Tail fractions are shown per panel. Input paths, sample counts, full ranges, quantiles, zero fractions and validation results are in <a href='summary.json'>summary.json</a>.</p>"
    if (args.output_dir.parent / "event_displays" / "index.html").exists():
        html += "<p><a href='../event_displays/index.html'>Five-detector event-display gallery</a>: tracks, clusters, hits and truth guides on a common transverse scale.</p>"
    if "maia" in parquet_paths:
        maia_count = report["samples"]["parquet/maia"]["checks"]["events"]
        html += f"<p>MAIA uses {maia_count} events from the supplied public v04 ttbar ROOT sample (10 events in the file). With fewer events than the other detectors, its statistical precision is lower. The file has no CellIDEncoding metadata; tracker surface system/side/layer are unknown and filled with zero. The MAIA model and comparisons use track/cluster inputs, not those hit surface fields.</p>"
    if "maia_validation" in report:
        validation = report["maia_validation"]
        failures = [gate for gate in validation["gates"] if gate["status"] == "FAIL"]
        html += f"<h2>MAIA physics validation: {validation['overall']}</h2><p>{validation['n_pass']} pass, {validation['n_fail']} fail, {validation['n_warn']} warn. The smoke script runs validation in report mode, so successful pipeline execution does not imply all physics gates passed.</p>"
        html += "<ul>" + "".join(f"<li>{gate['gate_id']} — {gate['title']}: {gate['observed']}</li>" for gate in failures) + "</ul>"
    html += "<p>Features are matched by meaning, not raw column number. After the first six track columns, ColliderML stores ACTS perigee parameters while the other detectors store EDM4hep track features. GeV is used for momenta and energies; phi is in radians; eta is dimensionless. Positive features spanning several orders of magnitude use a shared symlog x-axis. Every row uses shared bins and y limits, with equal event limits per detector; object histograms count objects, not events.</p>"
    html += f"<p>Jet matching uses a maximum-cardinality, minimum-ΔR one-to-one assignment within ΔR &lt; {args.jet_match_dr:g}, with wrapped phi and no response cut. Response is targetjet pT / genjet pT; profiles show the median and 16–84% spread for matched pairs versus genjet coordinates. Solid matching-fraction curves use all genjets as denominator (efficiency); dashed curves use all targetjets (purity), each binned in its own coordinates. Shading shows 68% Wilson intervals; empty bins are omitted. Jets retain their original detector-specific clustering and pT cuts. Binned numerator and denominator counts are saved in summary.json.</p>"
    html += "<p><strong>ColliderML track update:</strong> tanLambda is now derived as sinh(eta), omega uses the nominal 3 T field with the Key4hep 3e-4 conversion factor, and radiusOfInnermostHit is the minimum transverse radius of associated tracker hits. These occupy track columns 10/11/12; the equivalent EDM4hep columns are 11/13/10. The diagnostic compares tanLambda with sinh(eta) for every detector. ColliderML's minimum hit radius and EDM4hep's AtFirstHit reference-point radius are related but not identical definitions.</p>"
    html += "<p><strong>Existing TFDS edge case:</strong> One sampled CLIC cluster and one sampled CLD cluster have iTheta approximately pi, nonzero energy, and ET=eta=0. They fail the ET=E/cosh(eta) relation after the polar-boundary eta fallback. This was observed in stored 3.2.1 datasets; the newly converted ROOT samples have no such rows.</p>"
    html += "<h2>Validation checks</h2><table border='1' cellpadding='6'><tr><th>Sample</th><th>Events</th><th>Non-finite</th><th>Alignment</th><th>Phi normalization</th><th>Jet indices</th><th>p / pt / eta</th><th>ET / E / eta</th><th>Target E &lt; p</th><th>Encoding mismatches</th></tr>"
    for sample, entry in report["samples"].items():
        check = entry["checks"]
        numbers = [check["events"], sum(check["nonfinite"].values())] + [
            check[k]
            for k in (
                "alignment_failures",
                "phi_norm_failures",
                "target_jet_index_failures",
                "track_p_kinematics_failures",
                "cluster_et_kinematics_failures",
                "target_energy_below_momentum",
                "serialization_mismatches",
            )
        ]
        html += f"<tr><td>{sample}</td>" + "".join(f"<td>{n}</td>" for n in numbers) + "</tr>"
    html += "</table>"
    html += "<h2>Median event content</h2><table border='1' cellpadding='6'><tr><th>Sample</th><th>Tracks</th><th>Clusters</th><th>Targets</th><th>Active target fraction</th><th>Target energy sum [GeV]</th></tr>"
    for sample, entry in report["samples"].items():
        keys = ("event/n_track", "event/n_cluster", "event/n_target", "event/target_active_fraction", "event/target_energy_sum")
        medians = [entry["features"][key]["quantiles_01_50_99"][1] for key in keys]
        html += f"<tr><td>{sample}</td>" + "".join(f"<td>{value:.4g}</td>" for value in medians) + "</tr>"
    html += "</table><p>Compare cluster multiplicity, hits per cluster and active target fraction together: fine clustering creates many small input objects with no assigned target. These quantities describe the sampled reconstruction and target-building output; they are not detector performance rankings.</p>"
    if "colliderml_persisted_roundtrip" in report:
        html += (
            "<p>Stored ColliderML records compared with native parquet loader examples (order-independent SHA-256 fingerprints): "
            + json.dumps(report["colliderml_persisted_roundtrip"])
            + "</p>"
        )
    if "maia_persisted_roundtrip" in report:
        html += (
            "<p>Stored MAIA records compared with native parquet loader examples (order-independent SHA-256 fingerprints): "
            + json.dumps(report["maia_persisted_roundtrip"])
            + "</p>"
        )
    html += "".join(f"<h2>{name}</h2><img src='{name}' loading='lazy'>" for name in images)
    (args.output_dir / "index.html").write_text(html)


if __name__ == "__main__":
    main()
