"""Fixed, input-only validation cohorts for tracking encoder evolution."""

import hashlib
import json

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from mlpf.model.ssl import collate_inputs


class SnapshotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    frequency: int = Field(default=100, ge=1)
    events_per_dataset: int = Field(default=4, ge=1)
    elements_per_event: int = Field(default=256, ge=1)
    seed: int = Field(default=27182, ge=0)
    teacher: bool = True


def cohort_indices(dataset, count):
    """Identify events within each ConcatDataset constituent, including short ones."""
    parts = getattr(dataset, "datasets", [dataset])
    offset = 0
    indices = []
    for source, part in enumerate(parts):
        indices.extend((source, event, offset + event) for event in range(min(count, len(part))))
        offset += len(part)
    return indices


@torch.no_grad()
def save_snapshot(model, dataset, settings, output, step, device, dtype):
    config = settings.snapshots
    directory = output / "embeddings"
    directory.mkdir(exist_ok=True)
    metadata = {"datasets": settings.datasets, "split": settings.valid_split, "settings": config.model_dump(mode="json")}
    columns = {key: [] for key in ("X", "dataset_index", "event_index", "element_index", "student")}
    if config.teacher:
        columns["teacher"] = []
    was_training = model.training
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(config.seed)
            for source, event, index in cohort_indices(dataset, config.events_per_dataset):
                record = dataset[index]
                original = np.flatnonzero(record["X"][:, 0] != 0)
                rng = np.random.default_rng(np.random.SeedSequence([config.seed, source, event]))
                # Sample rows only after encoding the complete event context.
                selected = np.sort(rng.choice(len(original), min(len(original), config.elements_per_event), replace=False))
                batch = {k: v.to(device) for k, v in collate_inputs([record]).items()}
                mask = batch["X"][..., 0] != 0
                for name in ("student", "teacher") if config.teacher else ("student",):
                    encoder = getattr(model, name)
                    with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                        embeddings = encoder.encode_backbone(batch["X"], mask, batch["source_id"])
                    columns[name].append(embeddings[0, selected].float().cpu().numpy())
                columns["X"].append(np.asarray(record["X"])[original[selected]])
                columns["dataset_index"].append(np.full(len(selected), source, dtype=np.int64))
                columns["event_index"].append(np.full(len(selected), event, dtype=np.int64))
                columns["element_index"].append(original[selected])
    finally:
        model.train(was_training)
    arrays = {key: np.concatenate(value) for key, value in columns.items()}
    digest = hashlib.sha256()
    for key in ("X", "dataset_index", "event_index", "element_index"):
        digest.update(arrays[key].tobytes())
    metadata["cohort_sha256"] = digest.hexdigest()
    # Never silently compare a different event/row selection after resume.
    manifest_path = directory / "cohort.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in ("datasets", "split", "cohort_sha256"):
            if previous[key] != metadata[key]:
                raise ValueError("Embedding snapshot cohort changed; use a new output directory")
    else:
        manifest_path.write_text(json.dumps(metadata, indent=2) + "\n")
    arrays.update(step=np.asarray(step), cohort_sha256=np.asarray(metadata["cohort_sha256"]))
    temporary = directory / f"step-{step}.tmp"
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(directory / f"step-{step}.npz")
    return {"path": f"embeddings/step-{step}.npz", "elements": len(arrays["X"]), "cohort_sha256": metadata["cohort_sha256"]}
