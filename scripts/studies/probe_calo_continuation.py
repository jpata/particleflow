"""Linear cross-layer, distance-matched calorimeter continuation probe."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scripts.studies.probe_hit_instances import read
from mlpf.model.calo_probe import (
    build_candidates,
    differences,
    particle_weights,
    ranking_metric,
    score_edges,
)  # noqa: F401


def evaluate(paths, prepared, rep, fit):
    rows = []
    for path in paths:
        event = read(path)
        candidates, triplets, support = prepared[path]
        row = dict(event=path.stem, **support)
        if len(triplets):
            row["particle_macro_triplet_accuracy"] = ranking_metric(
                event, triplets, rep, fit
            )
        anchors = np.array(list(candidates), dtype=np.int64)
        if not len(anchors):
            rows.append(row)
            continue
        a = np.concatenate([np.repeat(key, len(candidates[key])) for key in anchors])
        b = np.concatenate([candidates[key] for key in anchors])
        scores = score_edges(event, rep, fit, a, b)
        top, compete, has_positive, start = [], [], [], 0
        for key in anchors:
            c = candidates[key]
            s = scores[start : start + len(c)]
            positive = event["labels"][c] == event["labels"][key]
            # Fractional credit for exact top-score ties.
            top.append(float(positive[s == s.max()].mean()))
            compete.append(positive.any() and not positive.all())
            has_positive.append(positive.any())
            start += len(c)
        top, compete, has_positive = (
            np.array(top),
            np.array(compete),
            np.array(has_positive),
        )
        row["top1_all_candidate_anchors"] = float(
            particle_weights(event["labels"][anchors]) @ top
        )
        for name, mask in [
            ("competing", compete),
            ("positive_available", has_positive),
        ]:
            if mask.any():
                row["top1_" + name] = float(
                    particle_weights(event["labels"][anchors[mask]]) @ top[mask]
                )
        rows.append(row)
    keys = set().union(*(r.keys() for r in rows)) - {"event"}
    return dict(
        mean={k: float(np.mean([r[k] for r in rows if k in r])) for k in sorted(keys)},
        events=rows,
    )


def train(args):
    paths = {
        s: sorted((args.source / "cache/calorimeter" / s).glob("*.npz"))
        for s in ("train", "valid", "test")
    }
    if any(not v for v in paths.values()):
        raise ValueError("All three cached event splits are required")
    ids = [set(p.stem for p in paths[s]) for s in paths]
    if any(ids[i] & ids[j] for i in range(3) for j in range(i)):
        raise ValueError("Event overlap across probe splits")
    prepared = {}
    for split, files in paths.items():
        for path in files:
            event = read(path)
            prepared[path] = build_candidates(
                event["raw"],
                event["labels"],
                args.neighbors,
                args.layer_gap,
                args.distance_ratio,
                args.seed + int(path.stem),
            )
            target = args.output / "candidates" / split
            target.mkdir(parents=True, exist_ok=True)
            candidates, triplets, support = prepared[path]
            np.savez_compressed(
                target / path.name,
                triplets=triplets,
                anchors=np.array(list(candidates)),
                lengths=np.array([len(v) for v in candidates.values()]),
                neighbors=np.concatenate(list(candidates.values()))
                if candidates
                else np.empty(0, dtype=np.int64),
            )
            print(split, path.stem, support, flush=True)
    if any(
        not any(len(prepared[path][1]) for path in files) for files in paths.values()
    ):
        raise ValueError(
            "Each split needs at least one admissible distance-matched triplet"
        )
    results = {}
    for rep in ("xyz_distance", "direction_distance"):
        results[rep] = evaluate(paths["test"], prepared, rep, None)
    for rep in ("raw", "embedding"):
        scaler = StandardScaler()
        for path in paths["train"]:
            event = read(path)
            triples = prepared[path][1]
            if not len(triples):
                continue
            pos, neg = differences(event[rep], triples)
            weights = particle_weights(event["labels"][triples[:, 0]])
            scaler.partial_fit(
                np.concatenate([pos, neg]), sample_weight=np.tile(weights / 2, 2)
            )
        designs, weights = [], []
        for path in paths["train"]:
            event = read(path)
            triples = prepared[path][1]
            if not len(triples):
                continue
            pos, neg = differences(event[rep], triples)
            designs.append(scaler.transform(pos) - scaler.transform(neg))
            weights.append(particle_weights(event["labels"][triples[:, 0]]))
        design = np.concatenate(designs)
        weight = np.concatenate(weights)
        weight *= len(weight) / weight.sum()
        # Symmetric examples implement log(1+exp(-(score_positive-score_negative))).
        # The common pair-score intercept cancels, so fit no intercept.
        x = np.concatenate([design, -design])
        y = np.repeat([1, 0], len(design))
        w = np.tile(weight, 2)
        best, sweep = None, []
        for c in args.c_values:
            model = LogisticRegression(
                C=c, fit_intercept=False, max_iter=1000, tol=1e-6, solver="lbfgs"
            )
            model.fit(x, y, sample_weight=w)
            if model.n_iter_[0] >= model.max_iter:
                raise RuntimeError("Probe optimizer did not converge")
            fit = dict(scaler=scaler, model=model)
            vals = []
            for path in paths["valid"]:
                triples = prepared[path][1]
                if len(triples):
                    vals.append(ranking_metric(read(path), triples, rep, fit))
            val = float(np.mean(vals))
            sweep.append(
                dict(C=c, validation_accuracy=val, iterations=int(model.n_iter_[0]))
            )
            if best is None or val > best[0]:
                best = val, fit, c
            print(rep, sweep[-1], flush=True)
        fit = best[1]
        results[rep] = evaluate(paths["test"], prepared, rep, fit)
        results[rep].update(
            validation_sweep=sweep,
            selected_C=best[2],
            features=x.shape[1],
            training_triplets=len(design),
        )
        joblib.dump(fit, args.output / f"{rep}.joblib")
        (args.output / "results.json").write_text(json.dumps(results, indent=2))
        print(rep, results[rep]["mean"], flush=True)
    return results


def plot_results(results, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ["xyz_distance", "direction_distance", "raw", "embedding"]
    labels = [
        "Spatial\ndistance",
        "Directional\ndistance",
        "Raw\nprobe",
        "Embedding\nprobe",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, key, title in zip(
        axes,
        ["particle_macro_triplet_accuracy", "top1_competing"],
        ["Distance-matched triplet accuracy", "Top-1 on full competing candidate sets"],
    ):
        values = [100 * results[m]["mean"][key] for m in methods]
        bars = ax.bar(
            labels, values, color=["#87959c", "#a9b3b9", "#277DA1", "#13A895"]
        )
        ax.bar_label(bars, fmt="%.1f%%", padding=3)
        ax.set(ylim=(0, 108), ylabel="Accuracy [%]", title=title)
        if key == "particle_macro_triplet_accuracy":
            ax.axhline(50, color="black", ls="--", lw=1, label="Chance")
            ax.legend(loc="upper left")
    fig.suptitle("Calorimeter continuation: frozen grouped HEPTv2 at 50k")
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("experiments/hit_instance_probe_50k_initial"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--layer-gap", type=int, default=1)
    parser.add_argument("--distance-ratio", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--c-values", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0]
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory")
    if (
        min(args.neighbors, args.layer_gap, *args.c_values) <= 0
        or args.distance_ratio < 1
    ):
        parser.error("Invalid graph or fitting parameters")
    args.output.mkdir(parents=True)
    manifest = dict(
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        source_manifest=json.loads((args.source / "manifest.json").read_text()),
        source_cache_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((args.source / "cache/calorimeter").glob("*/*.npz"))
        },
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    plot_results(train(args), args.output)


if __name__ == "__main__":
    main()
