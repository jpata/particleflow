#!/usr/bin/env python3
"""Local frozen-backbone momentum probe; truth membership is diagnostic only.

Cache event summaries from train/test TFDS splits separately. Validation and final
held-out events occupy disjoint test indices; no held-out metrics select heads.
All three heads see identical minibatches and start with identical parameters.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch import nn
import yaml

from mlpf.conf import MLPFConfig, Y_FEATURES
from mlpf.model.PFDataset import PFDataset, Collater
from mlpf.model.mlpf import MLPF
from mlpf.model.losses import make_task_loss_weighter
from mlpf.model.set_losses import hungarian_match, SetMatcherWeights
from mlpf.model.tracker_momentum_readout import TrackerMomentumReadout, association_weights
from mlpf.model.utils import unpack_predictions, unpack_target
from scripts.studies.frozen_tracker_pooling import pool_hit_associations
from scripts.studies.frozen_tracker_filtering import match, momentum, stats, jet_metrics

ARMS = ("hard", "soft", "oracle")
PROCS = ("ttbar", "ww_fullhad", "qq")
FIELDS = dict(
    summary=((3, 256, 300), "float32"),
    base=((256, 5), "float32"),
    truth=((256, 5), "float32"),
    active=((256,), "bool"),
    match_index=((256,), "int64"),
    ownership=((3, 256, 3), "float32"),
)


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def charged(x):
    return (x == 1) | (x == 4) | (x == 5)


def oracle_associations(batch, matches, slots):
    """Truth membership maps only to slots assigned by the frozen matcher."""
    pn = Y_FEATURES.index("particle_number")
    destination = torch.zeros((*batch.mask.shape, 1), device=batch.X.device, dtype=torch.long)
    weight = torch.zeros_like(destination, dtype=torch.float32)
    for e, (si, ti) in enumerate(matches):
        ids = batch.ytarget_set[e, batch.target_mask[e], pn].long()[ti]
        hit_ids = batch.ytarget[e, :, pn].long()
        for slot, particle in zip(si, ids):
            owned = (hit_ids == particle) & (particle > 0) & batch.mask[e]
            destination[e, owned, 0] = slot
            weight[e, owned, 0] = 1
    return destination, weight


def ownership_diagnostics(batch, grouping, oracle, sparse, use_oracle):
    """Per-slot effective hit count, truth-pure mass fraction, truth coverage.

    All denominators refer only to tracker hits and the fixed training matcher.
    -1 denotes undefined purity/completeness rather than a measured zero.
    """
    mass = torch.zeros(256, device=batch.X.device)
    correct = torch.zeros_like(mass)
    possible = torch.zeros_like(mass)
    hits = torch.nonzero(batch.mask[0] & (batch.X[0, :, 0] == 1), as_tuple=True)[0]
    keys, slots, background = grouping
    for idx in hits.split(1024):
        truth_dest, truth_weight = (v[0, idx] for v in oracle)
        if use_oracle:
            dest, weight = truth_dest, truth_weight
        else:
            dest, weight = association_weights(keys[0, idx], slots[0], background[0, idx], 4, 1.0, sparse)
        possible.index_add_(0, truth_dest[:, 0], truth_weight[:, 0])
        for k in range(weight.shape[1]):
            mass.index_add_(0, dest[:, k], weight[:, k])
            same = (dest[:, k] == truth_dest[:, 0]) * truth_weight[:, 0]
            correct.index_add_(0, dest[:, k], weight[:, k] * same)
    purity = torch.where(mass > 0, correct / mass.clamp_min(1e-8), -1.0)
    completeness = torch.where(possible > 0, correct / possible.clamp_min(1), -1.0)
    return torch.stack([mass, purity, completeness], -1).cpu().numpy()


def load_backbone(args):
    cfg = MLPFConfig.model_validate(yaml.safe_load((args.run / "train-config.yaml").read_text()))
    cfg.model.attention.use_flash_attn_varlen = False
    cfg.model.heptv2.activation_checkpointing = False
    assert cfg.model.set_decoder.num_slots == 256 and cfg.model.heptv2.embedding_dim == 256
    assert not cfg.model.set_decoder.tracker_momentum_readout
    model = MLPF(cfg)
    model.task_loss_weighter = make_task_loss_weighter(cfg.task_loss_weights.model_dump())
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=True)
    del state
    model.requires_grad_(False).eval().cuda()
    return model, cfg


def extract_event(model, cfg, record):
    batch = Collater(["X", "ytarget", "ytarget_set"], [])([record]).to("cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        memory = model.encode_backbone(batch.X, batch.mask)
        out = model.set_decoder(memory, batch.mask, batch.X)
        target = unpack_target(batch.ytarget_set, model)
        pred = unpack_predictions(out.predictions)
        matches = hungarian_match(target, pred, batch.target_mask, SetMatcherWeights(**cfg.model.set_decoder.matcher.model_dump()))
        summaries, ownership = [], []
        oracle = oracle_associations(batch, matches, 256)
        for arm in ARMS:
            override = oracle if arm == "oracle" else None
            _, summary = pool_hit_associations(
                memory, batch.mask, batch.X, out.hit_grouping, 256, 1024, 4, 1.0, arm == "soft", True, fixed_associations=override
            )
            summaries.append(summary[0].float().cpu().numpy())
            ownership.append(ownership_diagnostics(batch, out.hit_grouping, oracle, arm == "soft", arm == "oracle"))
    pid = out[1][0, :, 1:].float().argmax(-1) + 1
    mom = out[2][0].float()
    base = torch.stack([pid, mom[:, 0], mom[:, 1], torch.atan2(mom[:, 2], mom[:, 3]), mom[:, 4]], -1).cpu().numpy()
    # Stored pT and energy are logs, converted only for physics metrics.
    t = batch.ytarget_set[0, batch.target_mask[0]]
    truth = torch.stack([t[:, 0], t[:, 2], t[:, 3], torch.atan2(t[:, 4], t[:, 5]), t[:, 6]], -1).cpu().numpy()
    active = (out[0][0].float().softmax(-1)[:, 1] >= cfg.model.set_decoder.presence_threshold).cpu().numpy()
    mapping = np.full(256, -1, dtype=np.int64)
    mapping[matches[0][0].cpu().numpy()] = matches[0][1].cpu().numpy()
    padded = np.zeros((256, 5), dtype=np.float32)
    padded[: len(truth)] = truth
    result = dict(summary=np.stack(summaries), base=base, truth=padded, active=active, match_index=mapping, ownership=np.stack(ownership))
    assert all(np.isfinite(x).all() for x in result.values())
    return result


def prepare_cache(args, split, model=None, cfg=None):
    directory = args.output / "cache" / split
    directory.mkdir(parents=True, exist_ok=True)
    per_proc = getattr(args, split + "_events")
    count = 3 * per_proc
    manifest = directory / "progress.json"
    progress = json.loads(manifest.read_text()) if manifest.exists() else dict(completed=0, events=count)
    assert progress["events"] == count
    if progress["completed"] == count:
        return model, cfg
    if model is None:
        model, cfg = load_backbone(args)
    arrays = {}
    for name, (shape, dtype) in FIELDS.items():
        path = directory / (name + ".npy")
        arrays[name] = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+", dtype=dtype, shape=(count, *shape))
    tfds_split = "train" if split == "train" else "test"
    # Skip the first 2048 test events, used by earlier local diagnostics.
    offset = 0 if split != "test" else args.valid_events
    ids = []
    datasets = []
    for process in PROCS:
        ds = PFDataset(
            str(args.data_dir),
            f"cld_edm_{process}_hits/1:3.2.1",
            tfds_split,
            pad_to_multiple=cfg.pad_to_multiple_elements,
            feature_dim=cfg.input_dim,
            max_open_readers=2,
            build_target_set=True,
        ).ds
        datasets.append(ds)
        if split == "train":
            selected = np.random.default_rng(args.seed).permutation(len(ds))[:per_proc]
        else:
            selected = np.arange(2048 + offset, 2048 + offset + per_proc)
        assert len(selected) == per_proc and max(selected) < len(ds)
        ids.append(selected.tolist())
    write_json(
        directory / "events.json",
        dict(
            tfds_split=tfds_split,
            dataset_config="1:3.2.1",
            processes=PROCS,
            indices=ids,
            order="round-robin by process: event number = 3 * within-process index + process index",
        ),
    )
    start = time.monotonic()
    begin = progress["completed"]
    for i in range(begin, count):
        torch.manual_seed(args.seed + i)
        process_index, within = i % 3, i // 3
        event = extract_event(model, cfg, datasets[process_index][ids[process_index][within]])
        for name, value in event.items():
            arrays[name][i] = value
        if (i + 1) % 16 == 0 or i + 1 == count:
            for arr in arrays.values():
                arr.flush()
            write_json(manifest, dict(completed=i + 1, events=count))
            rate = (i + 1 - begin) / (time.monotonic() - start)
            print(f"CACHE {split} {i+1}/{count} {rate:.2f} events/s peak_GPU={torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
    return model, cfg


def read_cache(args, split):
    root = args.output / "cache" / split
    progress = json.loads((root / "progress.json").read_text())
    assert progress["completed"] == progress["events"]
    return {k: np.load(root / (k + ".npy"), mmap_mode="r") for k in FIELDS}


def correction_loss(correction, base, truth, mapping):
    matched = mapping >= 0
    tlogpt = truth[:, :, 1].gather(1, mapping.clamp_min(0))
    corrected = base[:, :, 1] + correction * charged(base[:, :, 0])
    # Exact baseline pT regression form, with matching fixed and other losses frozen.
    squared = (corrected - tlogpt).square() * torch.exp(tlogpt).clamp_min(1e-6).sqrt()
    return (squared * matched).sum() / matched.sum().clamp_min(1)


def physical(values):
    values = values.copy()
    values[:, [1, 4]] = np.exp(np.clip(values[:, [1, 4]], -20, 20))
    return values


def evaluate(args, heads, cache, split, step, jets=False):
    results = {}
    for arm in ("baseline", *ARMS):
        rows = {p: [] for p in PROCS}
        residuals = {g: [] for g in ("charged", "charged_hadron", "neutral")}
        fixed = []
        corrections = []
        supported = []
        event_pairs = {p: [] for p in PROCS}
        for start in range(0, len(cache["base"]), 128):
            stop = min(start + 128, len(cache["base"]))
            base = np.array(cache["base"][start:stop])
            if arm != "baseline":
                with torch.no_grad():
                    summary = torch.tensor(np.array(cache["summary"][start:stop, ARMS.index(arm)]), device="cuda")
                    corr = heads[arm](summary).cpu().numpy()
                gate = charged(base[:, :, 0])
                counts = np.expm1(np.array(cache["summary"][start:stop, ARMS.index(arm), :, 256:]).reshape(-1, 256, 4, 11)[:, :, :, -1]).sum(-1)
                corrections.extend(corr[gate & (counts >= 1)].tolist())
                supported.extend((counts[gate] >= 1).tolist())
                base[:, :, 1] += corr * gate
            for j in range(stop - start):
                idx = start + j
                proc = PROCS[idx % 3]
                raw_truth = np.array(cache["truth"][idx])
                truth = physical(raw_truth[raw_truth[:, 0] != 0])
                active = cache["active"][idx]
                pred = physical(base[j][active])
                pi, ti, angular, rel = match(pred, truth)
                accepted = angular & (abs(rel) < 0.5)
                entry = dict(
                    accepted=int(accepted.sum()),
                    target=len(truth),
                    pred=len(pred),
                    p3=float(np.linalg.norm(momentum(pred).sum(0) - momentum(truth).sum(0))),
                )
                for group, classes in [("charged", (1, 4, 5)), ("charged_hadron", (1,)), ("neutral", (2, 3))]:
                    selected = np.isin(truth[ti, 0], classes)
                    residuals[group].extend(rel[angular & selected].tolist())
                    entry[group + "_accepted"] = int((accepted & selected).sum())
                    entry[group + "_target"] = int(np.isin(truth[:, 0], classes).sum())
                rows[proc].append(entry)
                if jets:
                    event_pairs[proc].append((truth, pred))
                mapping = cache["match_index"][idx]
                slots = np.flatnonzero((mapping >= 0) & active)
                targets = mapping[slots]
                dr = np.hypot(
                    base[j, slots, 2] - raw_truth[targets, 2],
                    np.arctan2(np.sin(base[j, slots, 3] - raw_truth[targets, 3]), np.cos(base[j, slots, 3] - raw_truth[targets, 3])),
                )
                keep = (dr < 0.1) & (raw_truth[targets, 0] == 1)
                fixed.extend(np.expm1(base[j, slots[keep], 1] - raw_truth[targets[keep], 1]).tolist())
        macro_f1 = np.mean([2 * sum(x["accepted"] for x in r) / max(1, sum(x["target"] + x["pred"] for x in r)) for r in rows.values()])
        flat = sum(rows.values(), [])
        result = dict(
            macro_f1=float(macro_f1),
            p3_error_mean=float(np.mean([x["p3"] for x in flat])),
            momentum={g: stats(v) for g, v in residuals.items()},
            fixed_charged_hadron=stats(fixed),
            efficiency={g: sum(x[g + "_accepted"] for x in flat) / max(1, sum(x[g + "_target"] for x in flat)) for g in residuals},
            supported_charged_slot_fraction=float(np.mean(supported)) if supported else None,
            correction_saturation_fraction=float(np.mean(np.abs(corrections) > 0.95 * math.log(1.5))) if corrections else None,
            oracle_diagnostic_only=arm == "oracle",
        )
        arm_index = ARMS.index(arm) if arm != "baseline" else 0
        ownership = np.array(cache["ownership"][:, arm_index])
        selected = np.asarray(cache["active"]) & charged(cache["base"][:, :, 0])
        result["tracker_ownership_active_charged_slots"] = {
            label: float(ownership[:, :, k][selected & (ownership[:, :, k] >= 0)].mean()) if np.any(selected & (ownership[:, :, k] >= 0)) else None
            for k, label in enumerate(("effective_hits_mean", "fractional_purity_mean", "fractional_completeness_mean"))
        }
        if jets:
            result["jets_by_process"] = {p: jet_metrics(es) for p, es in event_pairs.items()}
        results[arm] = result
        print(f'EVAL {split} step={step} {arm}: F1={macro_f1:.4f} charged_hadron_width={result["momentum"]["charged_hadron"]["half68"]}', flush=True)
    write_json(args.output / "history" / f"{split}_step_{step}.json", results)
    return results


def train_heads(args):
    torch.manual_seed(args.seed)
    head = TrackerMomentumReadout(256, math.log(1.5)).cuda()
    heads = nn.ModuleDict({n: copy.deepcopy(head) for n in ARMS})
    optimizer = torch.optim.AdamW(heads.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr * 0.1)
    train = read_cache(args, "train")
    valid = read_cache(args, "valid")
    rng = np.random.default_rng(args.seed)
    start_step = 0
    best = {a: float("inf") for a in ARMS}
    last = args.output / "checkpoints/latest.pth"
    if last.exists():
        state = torch.load(last, map_location="cuda", weights_only=False)
        heads.load_state_dict(state["heads"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = state["step"]
        rng.bit_generator.state = state["rng"]
        best = state["best"]
    elif start_step == 0:
        # The three interventions must reproduce the frozen model before updates.
        with torch.no_grad():
            for i, a in enumerate(ARMS):
                z = torch.tensor(np.array(valid["summary"][:1, i]), device="cuda")
                assert torch.count_nonzero(heads[a](z)) == 0
        initial = evaluate(args, heads, valid, "valid", 0)
        for arm in ARMS:
            score = initial[arm]["fixed_charged_hadron"]["half68"]
            if score is None:
                raise ValueError("Validation has no matched charged hadrons")
            best[arm] = score
            torch.save(dict(head=heads[arm].state_dict(), step=0, score=score), args.output / "checkpoints" / f"best_{arm}.pth")
    started = time.monotonic()
    for step in range(start_step + 1, args.steps + 1):
        indices = rng.choice(len(train["base"]), args.batch_size, replace=False)
        base, truth, mapping = [torch.tensor(np.array(train[k][indices]), device="cuda") for k in ["base", "truth", "match_index"]]
        optimizer.zero_grad(set_to_none=True)
        losses = {}
        for i, arm in enumerate(ARMS):
            summary = torch.tensor(np.array(train["summary"][indices, i]), device="cuda")
            loss = correction_loss(heads[arm](summary), base, truth, mapping)
            if not torch.isfinite(loss):
                raise FloatingPointError((step, arm, loss.item()))
            loss.backward()
            losses[arm] = loss.item()
        for arm in ARMS:
            nn.utils.clip_grad_norm_(heads[arm].parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step % 100 == 0 or step == start_step + 1:
            print(f"TRAIN step={step}/{args.steps} loss={losses} elapsed={time.monotonic()-started:.1f}s", flush=True)
        if step % args.val_freq == 0 or step == args.steps:
            metrics = evaluate(args, heads, valid, "valid", step)
            for arm in ARMS:
                # Predeclared score, frozen-cohort charged-hadron width, validation only.
                score = metrics[arm]["fixed_charged_hadron"]["half68"]
                if score is not None and score < best[arm]:
                    best[arm] = score
                    torch.save(dict(head=heads[arm].state_dict(), step=step, score=score), args.output / "checkpoints" / f"best_{arm}.pth")
            state = dict(
                heads=heads.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                step=step,
                rng=rng.bit_generator.state,
                best=best,
            )
            checkpoint = args.output / "checkpoints" / f"checkpoint-{step}.pth"
            torch.save(state, checkpoint)
            shutil.copy2(checkpoint, last)
    return heads


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=Path("/mnt/work/mlpf/tensorflow_datasets/cld"))
    p.add_argument("--train-events", type=int, default=4096, help="Per process; TFDS train subset")
    p.add_argument("--valid-events", type=int, default=256, help="Per process, disjoint TFDS test indices")
    p.add_argument("--test-events", type=int, default=512, help="Per process; evaluated once after head selection")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--val-freq", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--cache-only", action="store_true")
    args = p.parse_args()
    if min(args.train_events, args.valid_events, args.test_events, args.steps, args.val_freq) <= 0:
        p.error("Counts must be positive")
    if args.batch_size > 3 * args.train_events:
        p.error("Batch exceeds cached train events")
    args.checkpoint = args.run / "checkpoints/checkpoint-50000.pth"
    args.output.mkdir(parents=True, exist_ok=True)
    # A nonblocking lock prevents two extractors/trainers corrupting a run.
    import fcntl

    lock = (args.output / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()))
    lock.flush()
    for name in ["history", "checkpoints", "source"]:
        (args.output / name).mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    sources = [
        Path(__file__),
        Path("scripts/studies/frozen_tracker_pooling.py"),
        Path("mlpf/model/tracker_momentum_readout.py"),
        Path("scripts/studies/frozen_tracker_filtering.py"),
        args.run / "train-config.yaml",
        Path("mlpf/model/mlpf.py"),
        Path("mlpf/model/set_decoder.py"),
        Path("mlpf/model/set_losses.py"),
        Path("mlpf/model/PFDataset.py"),
    ]
    source_hashes = {str(p): hashlib.file_digest(p.open("rb"), "sha256").hexdigest() for p in sources}
    manifest = dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        cache_format=2,
        source_sha256=source_hashes,
        checkpoint_sha256=hashlib.file_digest(args.checkpoint.open("rb"), "sha256").hexdigest(),
        device=torch.cuda.get_device_name(),
        torch_version=torch.__version__,
        frozen_model=True,
        shared_pooling="unchanged hard baseline",
        final_output_only=True,
        head_parameters=sum(x.numel() for x in TrackerMomentumReadout(256, math.log(1.5)).parameters()),
        training_loss="baseline sqrt(target pT)-weighted log-pT MSE, fixed baseline training Hungarian matches",
        selection="minimum validation fixed-cohort charged-hadron width; all F1/tail tradeoffs also reported",
        limitations=[
            "One seed; finite train subset from local dataset config 1 only.",
            "Validation/final test disjoint TFDS test events, skipped first 2048; backbone may have seen them for validation.",
            "Oracle membership is diagnostic only; not deployable.",
            "Only pT changes; energy is frozen. Jet results describe this pT-only intervention.",
        ],
    )
    dest = args.output / "manifest.json"
    if dest.exists():
        old = json.loads(dest.read_text())
        old_args = {k: v for k, v in old["arguments"].items() if k != "cache_only"}
        new_args = {k: v for k, v in manifest["arguments"].items() if k != "cache_only"}
        assert old_args == new_args and old["checkpoint_sha256"] == manifest["checkpoint_sha256"]
        assert old["source_sha256"] == source_hashes, "Source changed; use a new output directory"
    else:
        write_json(dest, manifest)
        for source in sources:
            shutil.copy2(source, args.output / "source" / source.name)
    bytes_per_event = sum(np.prod(shape) * np.dtype(dtype).itemsize for shape, dtype in FIELDS.values())
    estimated = int(3 * (args.train_events + args.valid_events + args.test_events) * bytes_per_event)
    existing = sum(p.stat().st_size for p in (args.output / "cache").rglob("*.npy")) if (args.output / "cache").exists() else 0
    if shutil.disk_usage(args.output).free < max(0, estimated - existing) + 2 * 2**30:
        raise OSError("Insufficient disk space for summary cache plus 2 GiB headroom")
    print("START", json.dumps(manifest), flush=True)
    model, cfg = prepare_cache(args, "train")
    model, cfg = prepare_cache(args, "valid", model, cfg)
    del model
    torch.cuda.empty_cache()
    if args.cache_only:
        return
    heads = train_heads(args)
    selected = {}
    for arm in ARMS:
        state = torch.load(args.output / "checkpoints" / f"best_{arm}.pth", map_location="cuda", weights_only=False)
        heads[arm].load_state_dict(state["head"])
        selected[arm] = dict(step=state["step"], score=state["score"])
    write_json(args.output / "selected_heads.json", selected)
    model, cfg = prepare_cache(args, "test")
    del model
    torch.cuda.empty_cache()
    evaluate(args, heads, read_cache(args, "test"), "test", args.steps, jets=True)
    write_json(args.output / "completed.json", dict(steps=args.steps, selected=selected))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
