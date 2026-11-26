"""Matched multi-seed frozen-embedding probe curves from saved SSL probe checkpoints."""

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import yaml

from mlpf.conf import MLPFConfig
from mlpf.model.losses import make_task_loss_weighter
from mlpf.model.mlpf import MLPF
from mlpf.model.ssl_probe import (
    cache_embeddings,
    cached_batch,
    configure_probe_readouts,
    evaluate_probe,
    load_probe_data,
    probe_loss,
)
from mlpf.pretrain import PretrainConfig


def readout_keys(model):
    modules = configure_probe_readouts(model)
    ids = {id(module) for module in modules}
    prefixes = [
        name + "." for name, module in model.named_modules() if id(module) in ids
    ]
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    return {
        name
        for name in model.state_dict()
        if name in trainable or any(name.startswith(prefix) for prefix in prefixes)
    }


def tensor_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(
            tensor.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
    return digest.hexdigest()


def fit_cached(model, train, valid, config, seed, budgets, path):
    """Paired seed controls sampling/dropout; evaluation never changes training RNG."""
    device = next(model.parameters()).device
    dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32
    readouts = configure_probe_readouts(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=config.probe.lr, weight_decay=config.probe.weight_decay
    )
    keys = readout_keys(model)
    before = tensor_digest(
        {k: v for k, v in model.state_dict().items() if k not in keys}
    )
    initial_heads = tensor_digest(
        {k: v for k, v in model.state_dict().items() if k in keys}
    )
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    rows = []
    resume_path = path / "latest.pth"
    first_step, elapsed_before = 0, 0.0
    if resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if (saved["seed"], saved["budgets"], saved["encoder_sha256"], saved["initial_readout_sha256"]) != (seed, budgets, before, initial_heads):
            raise ValueError("Probe resume state does not match this trial")
        model.load_state_dict(saved["readout_state_dict"], strict=False)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        generator.set_state(saved["generator_state"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
        rows, elapsed_before = saved["history"], saved["elapsed_seconds"]
        first_step = saved["updates"] + 1
        print(f"Resuming {path.name} after {saved['updates']} updates", flush=True)
    started = time.perf_counter()
    with (path / "metrics.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
        stream.flush()
        for step in range(first_step, max(budgets) + 1):
            if step:
                for module in readouts:
                    module.train()
                indices = torch.randint(
                    len(train), (config.probe.batch_size,), generator=generator
                )
                batch, embeddings = cached_batch(
                    model, [train[int(i)] for i in indices], device
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device.type, dtype=dtype, enabled=dtype != torch.float32
                ):
                    loss, _, _ = probe_loss(model, batch, embeddings)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite probe loss at {step}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
            if step in budgets:
                devices = (
                    [
                        device.index
                        if device.index is not None
                        else torch.cuda.current_device()
                    ]
                    if device.type == "cuda"
                    else []
                )
                with torch.random.fork_rng(devices=devices):
                    row = {
                        "updates": step,
                        "train": evaluate_probe(model, train, device, dtype),
                        "valid": evaluate_probe(model, valid, device, dtype),
                        "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                    }
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                temporary = path / "latest.tmp"
                torch.save(
                    {
                        "readout_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items() if k in keys},
                        "optimizer_state_dict": optimizer.state_dict(),
                        "seed": seed,
                        "budgets": budgets,
                        "updates": step,
                        "encoder_sha256": before,
                        "initial_readout_sha256": initial_heads,
                        "generator_state": generator.get_state(),
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                        "history": rows,
                        "elapsed_seconds": row["elapsed_seconds"],
                    },
                    temporary,
                )
                temporary.replace(resume_path)
                print(
                    json.dumps(
                        {
                            "trial": path.name,
                            "updates": step,
                            "valid": row["valid"]["unweighted_task_loss"],
                            "train": row["train"]["unweighted_task_loss"],
                            "seconds": row["elapsed_seconds"],
                        }
                    ),
                    flush=True,
                )
    assert before == tensor_digest(
        {k: v for k, v in model.state_dict().items() if k not in keys}
    ), "Encoder changed during probe"
    torch.save(
        {
            "readout_state_dict": {
                k: v.detach().cpu() for k, v in model.state_dict().items() if k in keys
            },
            "seed": seed,
            "updates": max(budgets),
        },
        path / "readout.pth",
    )
    result = {
        "seed": seed,
        "encoder_sha256": before,
        "initial_readout_sha256": initial_heads,
        "history": rows,
    }
    temporary = path / "result.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path / "result.json")
    return result


def metric_value(result, key):
    value = result
    for part in key.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return float(value) if value is not None else np.nan


def primary_metrics(result):
    if "particle_f1" in result:
        return [("unweighted_task_loss", "Unweighted PF task loss"), ("particle_f1", "Matched particle F1"), ("matched_pid_accuracy", "Matched PID accuracy")]
    return [("unweighted_task_loss", "Unweighted PF task loss"), ("presence_f1", "Presence F1"), ("pid_accuracy_on_targets", "PID accuracy on targets")]


SET_PHYSICS_METRICS = [
    ("particle_metrics.matching/purity", "Particle precision"),
    ("particle_metrics.matching/efficiency", "Particle recall"),
    ("particle_metrics.count/prediction_mean", "Predicted particles/event"),
    ("particle_metrics.event/energy_response_mean", "Event energy response"),
    ("particle_metrics.event/met_abs_error", "MET absolute error (GeV)"),
    ("particle_metrics.matched/energy_relative_abs_mean", "Matched energy relative error"),
]


def plot_results(output, metrics=None, prefix=""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trials = [
        json.loads(p.read_text()) | {"ssl_step": int(p.parent.name.split("-")[1])}
        for p in output.glob("step-*-seed-*/result.json")
    ]
    ssl_steps = sorted({t["ssl_step"] for t in trials})
    if not trials:
        return
    metrics = metrics or primary_metrics(trials[0]["history"][-1]["valid"])
    fig, axes = plt.subplots(2, len(metrics), figsize=(4.3 * len(metrics), 7.5), layout="constrained")
    summary = []
    for ssl_step in ssl_steps:
        group = sorted(
            [t for t in trials if t["ssl_step"] == ssl_step], key=lambda t: t["seed"]
        )
        budgets = [r["updates"] for r in group[0]["history"]]
        display = np.asarray(budgets) >= 50
        for row, split in enumerate(["train", "valid"]):
            for ax, (key, title) in zip(axes[row], metrics):
                values = np.array(
                    [[metric_value(r[split], key) for r in t["history"]] for t in group]
                )
                mean, std = (
                    values.mean(0),
                    values.std(0, ddof=1 if len(group) > 1 else 0),
                )
                (line,) = ax.plot(
                    np.asarray(budgets)[display],
                    mean[display],
                    "o-",
                    label=f"SSL {ssl_step} (n={len(group)})",
                    markersize=3,
                )
                ax.fill_between(
                    np.asarray(budgets)[display],
                    (mean - std)[display],
                    (mean + std)[display],
                    color=line.get_color(),
                    alpha=0.15,
                )
                ax.set(
                    xlabel="Probe updates",
                    title=f"{split.capitalize()}: {title}",
                    xscale="symlog",
                )
                ax.set_xlim(20, max(budgets))
                ax.set_xticks([b for b in budgets if b >= 20])
                ax.set_xticklabels([b for b in budgets if b >= 20], rotation=35)
                ax.grid(alpha=0.2)
        for i, budget in enumerate(budgets):
            item = {
                "ssl_step": ssl_step,
                "updates": budget,
                "seeds": [t["seed"] for t in group],
            }
            for key, _ in metrics:
                values = [metric_value(t["history"][i]["valid"], key) for t in group]
                item[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1 if len(values) > 1 else 0)),
                    "values": values,
                }
            summary.append(item)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Matched frozen-embedding probes | mean ± one seed standard deviation\nSame training/validation events, minibatch sequence and initial head within each seed"
    )
    for extension in ["png", "pdf"]:
        fig.savefig(
            output / f"{prefix}learning_curves.{extension}", dpi=160, bbox_inches="tight"
        )
    plt.close(fig)
    paired = []
    baseline = {t["seed"]: t for t in trials if t["ssl_step"] == 0}
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.3 * len(metrics), 4), layout="constrained")
    for ssl_step in [s for s in ssl_steps if s != 0]:
        group = [
            t for t in trials if t["ssl_step"] == ssl_step and t["seed"] in baseline
        ]
        if not group:
            continue
        budgets = [r["updates"] for r in group[0]["history"]]
        display = np.asarray(budgets) >= 50
        for ax, (key, title) in zip(axes, metrics):
            delta = np.array(
                [
                    [
                        metric_value(r["valid"], key) - metric_value(b["valid"], key)
                        for r, b in zip(t["history"], baseline[t["seed"]]["history"])
                    ]
                    for t in group
                ]
            )
            mean, std = delta.mean(0), delta.std(0, ddof=1 if len(group) > 1 else 0)
            (line,) = ax.plot(
                np.asarray(budgets)[display],
                mean[display],
                "o-",
                label=f"SSL {ssl_step} − SSL 0",
                markersize=3,
                color=f"C{ssl_steps.index(ssl_step)}",
            )
            ax.fill_between(
                np.asarray(budgets)[display],
                (mean - std)[display],
                (mean + std)[display],
                color=line.get_color(),
                alpha=0.15,
            )
            ax.axhline(0, color="gray", ls=":")
            ax.set(
                xlabel="Probe updates",
                title=title,
                xscale="symlog",
                xlim=(20, max(budgets)),
            )
            ax.grid(alpha=0.2)
            ax.set_xticks([b for b in budgets if b >= 50])
            ax.set_xticklabels([b for b in budgets if b >= 50], rotation=35)
            paired.append(
                {
                    "ssl_step": ssl_step,
                    "metric": key,
                    "updates": budgets,
                    "seeds": [t["seed"] for t in group],
                    "deltas": delta.tolist(),
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                }
            )
    if paired:
        axes[0].legend(fontsize=8)
    fig.suptitle(
        "Paired validation differences from the initialized backbone | mean ± one seed standard deviation"
    )
    for extension in ["png", "pdf"]:
        fig.savefig(
            output / f"{prefix}paired_differences.{extension}", dpi=160, bbox_inches="tight"
        )
    plt.close(fig)
    (output / f"{prefix}summary.json").write_text(
        json.dumps({"results": summary, "paired": paired}, indent=2) + "\n"
    )
    if not prefix and "particle_f1" in trials[0]["history"][-1]["valid"]:
        plot_results(output, SET_PHYSICS_METRICS[:3], "particle_counts_")
        plot_results(output, SET_PHYSICS_METRICS[3:], "energy_")


def verify_initializations(output, ssl_steps, seeds):
    for seed in seeds:
        hashes = [
            json.loads((output / f"step-{s}-seed-{seed}/result.json").read_text())[
                "initial_readout_sha256"
            ]
            for s in ssl_steps
        ]
        assert len(set(hashes)) == 1, f"Initial heads differ for seed {seed}"


def run_parallel(args, output):
    """Separate processes isolate RNG/model state while sharing the GPU."""
    # Encoder extraction has a different memory peak from cached readout fitting.
    # Build each cache once on the whole GPU before launching concurrent workers.
    command = [sys.executable, "-m", "mlpf.probe_learning_curves", "--run", str(args.run), "--output", str(output),
               "--updates", str(args.updates), "--device", args.device, "--ssl-steps", *map(str, args.ssl_steps),
               "--seeds", *map(str, args.seeds), "--prepare-cache"]
    print("Preparing shared frozen-embedding caches before starting workers", flush=True)
    with (output / "cache.log").open("a") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    pending = [
        (step, seed)
        for seed in args.seeds
        for step in args.ssl_steps
        if not (output / f"step-{step}-seed-{seed}/result.json").exists()
    ]

    def worker(step, seed):
        path = output / f"step-{step}-seed-{seed}"
        path.mkdir(exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "mlpf.probe_learning_curves",
            "--run",
            str(args.run),
            "--output",
            str(output),
            "--updates",
            str(args.updates),
            "--device",
            args.device,
            "--ssl-steps",
            *map(str, args.ssl_steps),
            "--seeds",
            *map(str, args.seeds),
            "--worker-step",
            str(step),
            "--worker-seed",
            str(seed),
        ]
        if args.gpu_memory_fraction is not None:
            command.extend(["--gpu-memory-fraction", str(args.gpu_memory_fraction)])
        with (path / "worker.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        return path.name

    print(
        f"Running {len(pending)} unfinished trials with {args.parallel} workers",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(worker, step, seed) for step, seed in pending]
        for future in as_completed(futures):
            print(f"Completed {future.result()}", flush=True)
            plot_results(output)
    verify_initializations(output, args.ssl_steps, args.seeds)
    plot_results(output)


def run(args):
    config = PretrainConfig.model_validate(
        yaml.safe_load((args.run / "pretrain-config.yaml").read_text())
    )
    output = args.output or args.run / "probe_learning_curves"
    output.mkdir(parents=True, exist_ok=True)
    budgets = sorted(set([0, 50, 100, 200, 500, 1000, 2000, args.updates]))
    budgets = [b for b in budgets if b <= args.updates]
    manifest = {
        "run": str(args.run.resolve()),
        "ssl_steps": args.ssl_steps,
        "seeds": args.seeds,
        "budgets": budgets,
        "probe_config": config.probe.model_dump(),
        "dtype": config.dtype,
        "seed_design": "Original SSL readout at probe.seed; fresh MLPF readout initialization for other seeds. Paired initialization and minibatches across SSL steps.",
    }
    if (output / "config.json").exists():
        if json.loads((output / "config.json").read_text()) != manifest:
            raise ValueError(
                "Output configuration differs; select a new output directory"
            )
    else:
        (output / "config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.parallel > 1 and args.worker_step is None:
        run_parallel(args, output)
        return
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if args.gpu_memory_fraction is not None and device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
    dtype = getattr(torch, config.dtype)
    original = torch.load(
        args.run / "student.pth", map_location="cpu", weights_only=False
    )
    model_config = MLPFConfig.model_validate(original["model_config"])
    train_data = load_probe_data(
        config, model_config, config.train_split, config.probe.train_events_per_dataset
    )
    valid_data = load_probe_data(
        config, model_config, config.valid_split, config.probe.valid_events_per_dataset
    )
    for ssl_step in args.ssl_steps:
        if args.worker_step is not None and ssl_step != args.worker_step:
            continue
        source = args.run / "probes" / f"step-{ssl_step}.pth"
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        if (
            checkpoint["model_config"] != original["model_config"]
            or checkpoint["ssl_step"] != ssl_step
        ):
            raise ValueError(f"Checkpoint metadata differs: {source}")
        model = MLPF(model_config)
        model.task_loss_weighter = make_task_loss_weighter(
            model_config.task_loss_weights.model_dump()
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        keys = readout_keys(model)
        # Saved probe checkpoints contain trained heads; remove them completely.
        state = model.state_dict()
        state.update({k: original["model_state_dict"][k] for k in keys})
        model.load_state_dict(state)
        model.to(device).eval().requires_grad_(False)
        train, valid = shared_embedding_cache(output, ssl_step, model, keys, train_data, valid_data, config, device, dtype)
        print(
            f"Cached SSL {ssl_step}: {len(train)} training, {len(valid)} validation events",
            flush=True,
        )
        if args.prepare_cache:
            del model, train, valid
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        for seed in args.seeds:
            if args.worker_seed is not None and seed != args.worker_seed:
                continue
            path = output / f"step-{ssl_step}-seed-{seed}"
            path.mkdir(exist_ok=True)
            if (path / "result.json").exists():
                continue
            probe = copy.deepcopy(model)
            if seed != config.probe.seed:
                torch.manual_seed(seed)
                fresh = MLPF(model_config)
                fresh.task_loss_weighter = make_task_loss_weighter(
                    model_config.task_loss_weights.model_dump()
                )
                probe.load_state_dict(
                    {k: v for k, v in fresh.state_dict().items() if k in keys},
                    strict=False,
                )
                del fresh
            result = fit_cached(probe, train, valid, config, seed, budgets, path)
            if seed == config.probe.seed and config.probe.steps in budgets:
                old = json.loads(
                    (args.run / "probes" / f"step-{ssl_step}.json").read_text()
                )["valid"]
                current = next(r for r in result["history"] if r["updates"] == config.probe.steps)[
                    "valid"
                ]
                print(
                    f"Original {config.probe.steps}-update reproduction:",
                    ssl_step,
                    {
                        k: metric_value(current, k) - metric_value(old, k)
                        for k, _ in primary_metrics(current)
                    },
                    flush=True,
                )
            del probe
            if args.worker_step is None:
                plot_results(output)
        del model, train, valid
    if args.worker_step is not None or args.prepare_cache:
        return
    # Hash checks make paired initializations and fixed encoders auditable.
    verify_initializations(output, args.ssl_steps, args.seeds)
    plot_results(output)


def shared_embedding_cache(output, ssl_step, model, readout, train_data, valid_data, config, device, dtype):
    signature = {
        "encoder_sha256": tensor_digest({k: v for k, v in model.state_dict().items() if k not in readout}),
        "datasets": config.datasets,
        "data_dir": config.data_dir,
        "train_split": config.train_split,
        "valid_split": config.valid_split,
        "train_events": len(train_data),
        "valid_events": len(valid_data),
        "extraction_batch_size": config.probe.batch_size,
        "dtype": config.dtype,
    }
    path = output / f"embeddings-step-{ssl_step}.pth"
    if path.exists():
        cache = torch.load(path, map_location="cpu", weights_only=False)
        if cache["signature"] != signature:
            raise ValueError(f"Frozen-embedding cache metadata differs: {path}")
        return cache["train"], cache["valid"]
    train = cache_embeddings(model, train_data, config.probe.batch_size, device, dtype)
    valid = cache_embeddings(model, valid_data, config.probe.batch_size, device, dtype)
    temporary = path.with_suffix(".tmp")
    torch.save({"signature": signature, "train": train, "valid": valid}, temporary)
    temporary.replace(path)
    return train, valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ssl-steps", nargs="+", type=int, default=[0, 500, 5000])
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[31415, 31416, 31417, 31418, 31419]
    )
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-memory-fraction", type=float, help="Per-worker CUDA allocator limit for concurrent trials")
    parser.add_argument(
        "--parallel", type=int, default=1, help="Number of concurrent GPU probe workers"
    )
    parser.add_argument("--worker-step", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--prepare-cache", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        args.updates < 200
        or args.parallel < 1
        or len(set(args.seeds)) != len(args.seeds)
        or len(set(args.ssl_steps)) != len(args.ssl_steps)
    ):
        parser.error("Use at least 200 updates and unique seeds/SSL steps")
    if (args.worker_step is None) != (args.worker_seed is None):
        parser.error("Both worker selectors are required together")
    if args.gpu_memory_fraction is not None and not 0 < args.gpu_memory_fraction <= 1:
        parser.error("GPU memory fraction must be in (0, 1]")
    run(args)


if __name__ == "__main__":
    main()
