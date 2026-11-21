"""Periodic supervised PF readout training on detached SSL embeddings."""

import copy
import json

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.data import ConcatDataset

from mlpf.conf import OutputMode
from mlpf.model.losses import LOSS_TASKS, REGRESSION_FEATURES, mlpf_loss
from mlpf.model.PFDataset import Collater, PFDataset
from mlpf.model.utils import unpack_predictions, unpack_target
from mlpf.model.ssl_snapshots import cohort_indices


class ProbeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    frequency: int = Field(default=500, ge=1)
    at_start: bool = True
    steps: int = Field(default=200, ge=1)
    batch_size: int = Field(default=8, ge=1)
    train_events_per_dataset: int = Field(default=64, ge=1)
    valid_events_per_dataset: int = Field(default=32, ge=1)
    lr: float = Field(default=0.001, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    seed: int = Field(default=31415, ge=0)
    save_events_per_dataset: int = Field(default=4, ge=0)


def probe_collater(model):
    fields = ["X", "ytarget", "ytarget_pt_orig", "ytarget_e_orig"]
    if model.output_mode == OutputMode.SET:
        fields.append("ytarget_set")
    return Collater(fields, ["source_id", "input_type_id"])


def load_probe_data(settings, model_config, split, count):
    return ConcatDataset(
        [
            PFDataset(
                settings.data_dir,
                name,
                split,
                num_samples=count,
                pad_to_multiple=None,
                feature_dim=model_config.input_dim,
                max_open_readers=4,
                build_target_set=model_config.model.output_mode == OutputMode.SET,
            ).ds
            for name in settings.datasets
        ]
    )


@torch.no_grad()
def cache_embeddings(model, dataset, batch_size, device, dtype):
    """Only this extraction calls the encoder; readout optimization uses CPU caches."""
    cache = []
    collate = probe_collater(model)
    for start in range(0, len(dataset), batch_size):
        records = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        batch = collate(records).to(device)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            embeddings = model.encode_backbone(batch.X, batch.mask, batch.source_id)
        for i, record in enumerate(records):
            valid = batch.mask[i]
            item = {key: record[key] for key in collate.per_particle_keys_to_get + collate.per_event_keys_to_get if key in record}
            # Also handle padding already stored in the source record.
            keep = valid[: len(record["X"])].cpu().numpy()
            for key in collate.per_particle_keys_to_get:
                if key != "ytarget_set":
                    item[key] = item[key][keep]
            item["embedding"] = embeddings[i, valid].detach().cpu()
            cache.append(item)
    return cache


def cached_batch(model, records, device):
    batch = probe_collater(model)(records).to(device)
    embeddings = torch.nn.utils.rnn.pad_sequence([r["embedding"] for r in records], batch_first=True).to(device)
    return batch, embeddings


def probe_loss(model, batch, embeddings):
    raw = model.decode_backbone(embeddings, batch.X, batch.mask)
    predictions = unpack_predictions(raw)
    weights = {name: 1.0 for name in REGRESSION_FEATURES}
    if model.output_mode == OutputMode.SET:
        from mlpf.model.set_losses import set_mlpf_loss
        from mlpf.model.training import _set_loss_kwargs

        targets = unpack_target(batch.ytarget_set, model)
        loss, terms, _ = set_mlpf_loss(targets, predictions, batch, weights, model.task_loss_weighter, **_set_loss_kwargs(model, raw))
    else:
        targets = unpack_target(batch.ytarget, model)
        loss, terms, _ = mlpf_loss(
            targets, predictions, batch, weights, model.task_loss_weighter, binary_focal_gamma=model.config.binary_classification_focal_gamma
        )
    return loss, terms, predictions


@torch.no_grad()
def evaluate_probe(model, cache, device, dtype):
    model.eval()
    totals = {}
    tp = fp = fn = correct = particles = 0
    # Equal-event validation, independent of the probe training batch size.
    for record in cache:
        batch, embeddings = cached_batch(model, [record], device)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            _, terms, predictions = probe_loss(model, batch, embeddings)
        for key, value in terms.items():
            totals[key] = totals.get(key, 0) + float(value) / len(cache)
        if model.output_mode == OutputMode.ELEMENTWISE:
            target = batch.ytarget[..., 0].long()
            positive = (target != 0) & batch.mask
            predicted = predictions["cls_binary"].argmax(-1).bool() & batch.mask
            tp += int((positive & predicted).sum())
            fp += int((~positive & predicted).sum())
            fn += int((positive & ~predicted).sum())
            correct += int(((predictions["cls_id_onehot"].argmax(-1) == target) & positive).sum())
            particles += int(positive.sum())
    # Calibrated task weights can differ between probes. Report the same fixed
    # sum of base task losses as well; Total remains the actual training objective.
    result = {"losses": totals, "events": len(cache), "unweighted_task_loss": sum(totals.get(task, 0) for task in LOSS_TASKS)}
    if model.output_mode == OutputMode.ELEMENTWISE:
        result.update(
            presence_precision=tp / max(tp + fp, 1),
            presence_recall=tp / max(tp + fn, 1),
            presence_f1=2 * tp / max(2 * tp + fp + fn, 1),
            pid_accuracy_on_targets=correct / max(particles, 1),
        )
    return result


def fit_probe(student, train_data, valid_data, config, device, dtype):
    """Fresh identical readout initialization each time; never mutate the SSL model.

    The student PF heads are frozen throughout SSL, so a copy gives the same
    initialization at every probe while updating only the copied readout weights.
    fork_rng also preserves the subsequent SSL augmentations and dropout stream.
    """
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed)
        model = copy.deepcopy(student).eval().requires_grad_(False)
        train_cache = cache_embeddings(model, train_data, config.batch_size, device, dtype)
        valid_cache = cache_embeddings(model, valid_data, config.batch_size, device, dtype)
        readouts = []
        for name in (
            "nn_binary_particle",
            "nn_pid",
            "nn_pt",
            "nn_eta",
            "nn_sin_phi",
            "nn_cos_phi",
            "nn_energy",
            "classification_norm",
            "regression_norm",
            "classification_readout",
            "regression_readout",
            "set_decoder",
            "task_loss_weighter",
        ):
            module = getattr(model, name, None)
            if module is not None:
                module.requires_grad_(True)
                readouts.append(module)
        for name in ("classification_query", "regression_query"):
            parameter = getattr(model, name, None)
            if parameter is not None:
                parameter.requires_grad_(True)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay)
        initial = evaluate_probe(model, valid_cache, device, dtype)
        for module in readouts:
            module.train()
        generator = torch.Generator().manual_seed(config.seed)
        for _ in range(config.steps):
            indices = torch.randint(len(train_cache), (config.batch_size,), generator=generator)
            batch, embeddings = cached_batch(model, [train_cache[int(i)] for i in indices], device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                loss, _, _ = probe_loss(model, batch, embeddings)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite supervised probe loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
        final = evaluate_probe(model, valid_cache, device, dtype)
        metrics = {
            "initial": initial,
            "valid": final,
            "train_events": len(train_cache),
            "steps": config.steps,
            "trainable_parameters": sum(p.numel() for p in parameters),
        }
        return metrics, model


def run_probe(student, settings, model_config, output, step, device, dtype):
    config = settings.probe
    train = load_probe_data(settings, model_config, settings.train_split, config.train_events_per_dataset)
    valid = load_probe_data(settings, model_config, settings.valid_split, config.valid_events_per_dataset)
    print(f"Training frozen-embedding PF probe at SSL step {step} ({config.steps} updates)", flush=True)
    metrics, model = fit_probe(student, train, valid, config, device, dtype)
    directory = output / "probes"
    directory.mkdir(exist_ok=True)
    metrics.update(ssl_step=step, config=config.model_dump(mode="json"))
    (directory / f"step-{step}.json").write_text(json.dumps(metrics, indent=2) + "\n")
    temporary = directory / f"step-{step}.tmp"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "extra_state": {"step": 0},
            "ssl_step": step,
            "probe": metrics,
            "model_config": model_config.model_dump(mode="json"),
        },
        temporary,
    )
    temporary.replace(directory / f"step-{step}.pth")
    if config.save_events_per_dataset:
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(config.seed)
            save_probe_predictions(model, valid, config.save_events_per_dataset, directory / f"step-{step}-predictions.npz", device, dtype)
    print(json.dumps({"step": step, "probe": metrics}), flush=True)
    return metrics


@torch.no_grad()
def save_probe_predictions(model, dataset, count, path, device, dtype):
    """Small fixed cohort: raw readout outputs plus supervised targets and inputs.

    Independent offsets preserve input/prediction/target axes in set mode.
    Momentum outputs and targets use training coordinates, not decoded four-vectors.
    """
    arrays = {key: [] for key in ("X", "targets", "presence_logits", "pid_logits", "momentum", "dataset_index", "event_index")}
    offsets = {key: [0] for key in ("input_offsets", "prediction_offsets", "target_offsets")}
    model.eval()
    for source, event, index in cohort_indices(dataset, count):
        batch = probe_collater(model)([dataset[index]]).to(device)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            raw = model(batch.X, batch.mask, batch.source_id)
        valid = batch.mask[0]
        pred_mask = torch.ones(raw[0].shape[1], dtype=torch.bool, device=device) if model.output_mode == OutputMode.SET else valid
        targets = batch.ytarget_set[0, batch.target_mask[0]] if model.output_mode == OutputMode.SET else batch.ytarget[0, valid]
        for key, value in zip(("presence_logits", "pid_logits", "momentum"), raw[:3]):
            arrays[key].append(value[0, pred_mask].float().cpu().numpy())
        arrays["X"].append(batch.X[0, valid].float().cpu().numpy())
        arrays["targets"].append(targets.float().cpu().numpy())
        arrays["dataset_index"].append(source)
        arrays["event_index"].append(event)
        offsets["input_offsets"].append(offsets["input_offsets"][-1] + int(valid.sum()))
        offsets["prediction_offsets"].append(offsets["prediction_offsets"][-1] + int(pred_mask.sum()))
        offsets["target_offsets"].append(offsets["target_offsets"][-1] + len(targets))
    data = {k: np.asarray(v) if k in ("dataset_index", "event_index") else np.concatenate(v) for k, v in arrays.items()}
    data.update({k: np.asarray(v) for k, v in offsets.items()})
    data["output_mode"] = np.asarray(model.output_mode.value)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **data)
    temporary.replace(path)
