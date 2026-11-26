"""Input-only audits of SSL context dependence and individual-hit shortcuts."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from mlpf.conf import MLPFConfig
from mlpf.model.losses import make_task_loss_weighter
from mlpf.model.mlpf import MLPF
from mlpf.model.ssl import StudentTeacher, collate_inputs, encode_ssl_embeddings, subsample_view
from mlpf.pretrain import PretrainConfig, load_inputs


def stratified_indices(x, count, generator):
    indices = []
    for kind in x[:, 0].unique(sorted=True):
        rows = torch.where(x[:, 0] == kind)[0]
        indices.append(rows[torch.randperm(len(rows), generator=generator)[:count]])
    return torch.cat(indices).sort().values


def replace_context(x, query, donor, generator):
    """Keep query rows and event length fixed; replace other rows by detector type."""
    result = x.clone()
    replace = torch.ones(len(x), dtype=torch.bool)
    replace[query] = False
    reused = 0
    for kind in x[:, 0].unique():
        positions = torch.where(replace & (x[:, 0] == kind))[0]
        candidates = donor[donor[:, 0] == kind]
        if len(positions) and not len(candidates):
            raise ValueError("Donor has no hits of a required detector type")
        if len(positions) > len(candidates):
            reused += len(positions) - len(candidates)
            order = torch.cat([torch.randperm(len(candidates), generator=generator)
                               for _ in range((len(positions) + len(candidates) - 1) // len(candidates))])
        else:
            order = torch.randperm(len(candidates), generator=generator)
        result[positions] = candidates[order[:len(positions)]]
    assert torch.equal(result[query], x[query])
    return result, reused


def distribution_metrics(target, predicted):
    p, q = target.float().clamp_min(1e-12), predicted.float().clamp_min(1e-12)
    middle = (p + q) / 2
    return {
        "agreement": (p.argmax(-1) == q.argmax(-1)).float(),
        "cross_entropy": -(p * q.log()).sum(-1),
        "target_entropy": -(p * p.log()).sum(-1),
        "kl": (p * (p.log() - q.log())).sum(-1),
        "js": .5 * (p * (p.log() - middle.log()) + q * (q.log() - middle.log())).sum(-1),
    }


def group_metrics(target, predicted, types):
    metrics = distribution_metrics(target, predicted)
    return {str(int(k)): {key: float(v[types == k].mean()) for key, v in metrics.items()} for k in types.unique()}


def encode(model, x, source, device, dtype, isolated=False):
    x = x.to(device)
    x = x[:, None] if isolated else x[None]
    mask = x[..., 0] != 0
    sources = torch.full((len(x),), source, device=device, dtype=torch.long)
    with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
        result = encode_ssl_embeddings(model, x, mask, sources).float()
    return result[:, 0] if isolated else result[0]


def extract(args):
    device = torch.device(args.device)
    saved = torch.load(args.run / "latest.pth", map_location="cpu", weights_only=False)
    config = PretrainConfig.model_validate(saved["pretrain_config"])
    if config.ssl.mode != "thinning":
        raise ValueError("These diagnostics evaluate per-hit prototypes; region SSL needs diagnostics against its masked-region targets")
    architecture = MLPFConfig.model_validate(saved["model_config"])
    dtype = getattr(torch, config.dtype)
    student = MLPF(architecture)
    student.task_loss_weighter = make_task_loss_weighter(architecture.task_loss_weights.model_dump())
    dimension = saved["ssl_state_dict"]["student_head.mlp.1.weight"].shape[1]
    ssl = StudentTeacher(student, dimension, config.ssl)
    ssl.load_state_dict(saved["ssl_state_dict"])
    ssl.to(device).eval().requires_grad_(False)
    initial = MLPF(architecture)
    initial.task_loss_weighter = make_task_loss_weighter(architecture.task_loss_weights.model_dump())
    initial_state = torch.load(args.run / "probes/step-0.pth", map_location="cpu", weights_only=False)
    initial.load_state_dict(initial_state["model_state_dict"])
    initial.to(device).eval().requires_grad_(False)
    del saved, initial_state, student

    def probabilities(embedding, teacher=True):
        with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            head = ssl.teacher_head if teacher else ssl.student_head
            logits = head(embedding.to(dtype))
            return ((logits - ssl.center.reshape(1, -1)) / config.ssl.teacher_temperature).softmax(-1) if teacher else (logits / config.ssl.student_temperature).softmax(-1)

    generator = torch.Generator().manual_seed(args.seed)
    context = []
    for split, count in [("train", args.train_events), ("valid", args.valid_events)]:
        dataset = load_inputs(config, architecture, config.train_split if split == "train" else config.valid_split, count)
        cache = {key: [] for key in ["x", "target", "event"]}
        for event in range(len(dataset)):
            record = dataset[event]
            x = collate_inputs([record])["X"][0]
            source = int(record["source_id"])
            full_teacher = encode(ssl.teacher, x, source, device, dtype)
            selected = stratified_indices(x, args.hits_per_type, generator)
            cache["x"].append(x[selected])
            cache["target"].append(probabilities(full_teacher[selected.to(device)]).cpu())
            cache["event"].append(torch.full((len(selected),), event, dtype=torch.long))
            if split == "valid" and event < args.context_events:
                # Draw the actual training augmentation first, then choose queries among survivors.
                thin, mask, original = subsample_view(x[None], (x[:, 0] != 0)[None], config.ssl.keep_fraction, generator)
                thin = thin[0, mask[0]]
                thin_query = stratified_indices(thin, args.queries_per_type, generator)
                query = original[0, mask[0]][thin_query]
                donors = [collate_inputs([dataset[(event + offset) % len(dataset)]])["X"][0] for offset in [1, 2, 3, 4]]
                replacement, reused = replace_context(x, query, torch.cat(donors), generator)
                assert torch.equal(thin[thin_query], x[query])
                types = x[query, 0].to(device)
                target = probabilities(full_teacher[query.to(device)])
                for name, model in [("initial_encoder", initial), ("teacher", ssl.teacher), ("student", ssl.student)]:
                    full = full_teacher if name == "teacher" else encode(model, x, source, device, dtype)
                    reference = full[query.to(device)]
                    for condition in ["full_repeat", "keep_60pct", "foreign_context", "queries_only", "isolated_hits"]:
                        if condition == "full_repeat":
                            changed = encode(model, x, source, device, dtype)[query.to(device)]
                        elif condition == "keep_60pct":
                            changed = encode(model, thin, source, device, dtype)[thin_query.to(device)]
                        elif condition == "foreign_context":
                            changed = encode(model, replacement, source, device, dtype)[query.to(device)]
                        else:
                            changed = encode(model, x[query], source, device, dtype, isolated=condition == "isolated_hits")
                        row = dict(event=event, network=name, condition=condition, reused_donor_rows=reused, groups={})
                        for kind in types.unique():
                            keep = types == kind
                            a, b = reference[keep], changed[keep]
                            delta2 = (a - b).square().sum()
                            row["groups"][str(int(kind))] = dict(
                                hits=int(keep.sum()), cosine=float(F.cosine_similarity(a, b).mean()),
                                relative_l2=float((delta2 / a.square().sum().clamp_min(1e-20)).sqrt()),
                                relative_centered_rms=float((delta2 / (a-a.mean(0)).square().sum().clamp_min(1e-20)).sqrt()))
                        if name != "initial_encoder":
                            predicted = probabilities(changed, teacher=name == "teacher")
                            own_full = probabilities(reference, teacher=name == "teacher")
                            row["against_full_teacher"] = group_metrics(target, predicted, types)
                            row["against_own_full"] = group_metrics(own_full, predicted, types)
                        context.append(row)
                print(f"Context event {event + 1}/{args.context_events}", flush=True)
            if (event + 1) % 32 == 0:
                print(f"Extracted {split} {event + 1}/{len(dataset)}", flush=True)
        torch.save({key: torch.cat(value) for key, value in cache.items()}, args.output / f"{split}_targets.pth")
    (args.output / "context_events.json").write_text(json.dumps(context, indent=2) + "\n")


def sample_weights(cache):
    pairs = torch.stack([cache["event"], cache["x"][:, 0].long()], 1)
    unique, inverse, counts = torch.unique(pairs, dim=0, return_inverse=True, return_counts=True)
    events, type_counts = torch.unique(unique[:, 0], return_counts=True)
    per_group = type_counts[torch.searchsorted(events, unique[:, 0].contiguous())]
    return 1.0 / (counts[inverse] * per_group[inverse] * len(events))


def evaluate_predictions(cache, prediction):
    rows = distribution_metrics(cache["target"], prediction)
    result = {}
    for name, mask in [("all", torch.ones(len(prediction), dtype=torch.bool, device=prediction.device))] + [(str(int(k)), cache["x"][:, 0] == k) for k in cache["x"][:, 0].unique()]:
        weights = cache["weight"][mask]
        weights = weights / weights.sum()
        result[name] = {key: float((value[mask] * weights).sum()) for key, value in rows.items()}
    return result


def fit(args):
    device = torch.device(args.device)
    caches = {}
    for split in ["train", "valid"]:
        c = torch.load(args.output / f"{split}_targets.pth", weights_only=False)
        c["weight"] = sample_weights(c)
        caches[split] = {k: v.to(device) for k, v in c.items()}
    train, valid = caches["train"], caches["valid"]
    baseline = torch.zeros_like(valid["target"])
    for kind in train["x"][:, 0].unique():
        mask = train["x"][:, 0] == kind
        w = train["weight"][mask]
        marginal = (train["target"][mask] * (w/w.sum())[:, None]).sum(0)
        baseline[valid["x"][:, 0] == kind] = marginal
    summary = {"type_marginal_baseline": evaluate_predictions(valid, baseline), "trials": []}
    # Geometric metadata includes detector type, layer, positions and direction;
    # omit all three measurement columns: transverse energy, energy and time.
    for features, columns in [("geometry", [0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14]), ("all_raw", list(range(15)))]:
        x = train["x"][:, columns]
        mean = (x * train["weight"][:, None]).sum(0)
        std = (((x - mean).square() * train["weight"][:, None]).sum(0)).sqrt().clamp_min(1e-6)
        inputs = {split: ((c["x"][:, columns] - mean)/std).clamp(-20, 20) for split, c in caches.items()}
        for seed in args.fit_seeds:
            torch.manual_seed(seed)
            model = nn.Sequential(nn.Linear(len(columns), 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, train["target"].shape[1])).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
            sampler = torch.Generator(device=device).manual_seed(seed)
            history = []
            for step in range(args.fit_steps + 1):
                if step:
                    idx = torch.multinomial(train["weight"], 2048, replacement=True, generator=sampler)
                    loss = -(train["target"][idx] * model(inputs["train"][idx]).log_softmax(-1)).sum(-1).mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                if step in sorted(set([0, 200, 500, 1000, 2000, args.fit_steps])):
                    row = {"updates": step}
                    with torch.no_grad():
                        for split, c in caches.items():
                            predictions = torch.cat([model(z).softmax(-1) for z in inputs[split].split(4096)])
                            row[split] = evaluate_predictions(c, predictions)
                    history.append(row)
                    print(features, seed, step, row["valid"]["all"], flush=True)
            summary["trials"].append(dict(features=features, columns=columns, seed=seed, history=history))
            torch.save(dict(model=model.state_dict(), mean=mean.cpu(), std=std.cpu(), columns=columns), args.output / f"predictor-{features}-{seed}.pth")
            (args.output / "individual_hit_results.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", choices=["extract", "fit", "all"], default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-events", type=int, default=256)
    parser.add_argument("--valid-events", type=int, default=128)
    parser.add_argument("--context-events", type=int, default=64)
    parser.add_argument("--hits-per-type", type=int, default=256)
    parser.add_argument("--queries-per-type", type=int, default=16)
    parser.add_argument("--seed", type=int, default=78123)
    parser.add_argument("--fit-seeds", type=int, nargs="+", default=[8123, 8124, 8125])
    parser.add_argument("--fit-steps", type=int, default=4000)
    args = parser.parse_args()
    args.output = args.output or args.run / "diagnostics"
    args.output.mkdir(parents=True, exist_ok=True)
    if args.context_events > args.valid_events or args.valid_events < 5:
        parser.error("Need at least five validation events and context-events <= valid-events")
    (args.output / "config.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2) + "\n")
    torch.set_num_threads(4)
    if args.stage in ["extract", "all"]:
        extract(args)
    if args.stage in ["fit", "all"]:
        fit(args)


if __name__ == "__main__":
    main()
