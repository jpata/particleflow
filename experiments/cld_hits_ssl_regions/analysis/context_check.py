"""Post-training input-only region predictor ablation; no parameter updates."""

import json
from pathlib import Path

import numpy as np
import torch

from mlpf.conf import MLPFConfig
from mlpf.model.losses import make_task_loss_weighter
from mlpf.model.mlpf import MLPF
from mlpf.model.ssl import StudentTeacher, encode_ssl_embeddings
from mlpf.model.ssl_regions import measurement_loss
from mlpf.pretrain import PretrainConfig, batch_at, load_inputs


@torch.inference_mode()
def main():
    run = Path("experiments/cld_hits_ssl_regions")
    state = torch.load(run / "latest.pth", map_location="cpu", weights_only=False)
    config = PretrainConfig.model_validate(state["pretrain_config"])
    architecture = MLPFConfig.model_validate(state["model_config"])
    student = MLPF(architecture)
    student.task_loss_weighter = make_task_loss_weighter(
        architecture.task_loss_weights.model_dump()
    )
    dimension = state["ssl_state_dict"]["student_head.mlp.1.weight"].shape[1]
    model = StudentTeacher(student, dimension, config.ssl)
    model.load_state_dict(state["ssl_state_dict"])
    model.cuda().eval().requires_grad_(False)
    checkpoint_step = state["step"]
    del state
    dataset = load_inputs(config, architecture, config.valid_split, 64)
    generator = torch.Generator(device="cuda").manual_seed(config.seed + 1)
    cache = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for event in range(len(dataset)):
            batch = batch_at(dataset, [event], torch.device("cuda"))
            views = []
            for _ in range(config.ssl.student_views):
                sample = model.region_sampler(
                    batch["X"], batch["X"][..., 0] != 0, generator
                )
                embedding = encode_ssl_embeddings(
                    model.student, sample["view"], sample["mask"], batch["source_id"]
                )
                views.append(({k: v.cpu() for k, v in sample.items()}, embedding.cpu()))
            cache.append(views)
        print("Encoded 64 held-out events, two masks each", flush=True)
        rows = []
        for event, views in enumerate(cache):
            for view_index, (cpu_sample, cpu_embedding) in enumerate(views):
                sample = {k: v.cuda() for k, v in cpu_sample.items()}
                embedding = cpu_embedding.cuda()
                donor_sample, donor = cache[(event + 1) % len(cache)][view_index]
                replacement = torch.empty_like(cpu_embedding)
                # Preserve recipient token count and detector composition.
                for kind in [1, 2]:
                    selected = cpu_sample["view"][0, :, 0] == kind
                    candidates = donor[0, donor_sample["view"][0, :, 0] == kind]
                    if not len(candidates):
                        raise ValueError("Donor missing a detector type")
                    index = torch.linspace(
                        0, len(candidates) - 1, int(selected.sum())
                    ).long()
                    replacement[0, selected] = candidates[index]
                for condition, context in [
                    ("correct", embedding),
                    ("foreign_event", replacement.cuda()),
                    ("zero_context", torch.zeros_like(embedding)),
                ]:
                    prediction = model.region_predictor(
                        context, sample["mask"], sample["features"]
                    )
                    for name, kind in [
                        ("all", None),
                        ("tracker", 1),
                        ("calorimeter", 2),
                    ]:
                        target = dict(sample)
                        if kind is not None:
                            target["valid"] = sample["valid"] & (
                                sample["types"] == kind
                            ).unsqueeze(-1)
                        _, metrics = measurement_loss(
                            prediction, target, config.ssl.regions
                        )
                        keep = [
                            "occupancy_loss",
                            "count_loss",
                            "energy_loss",
                            "occupied_fraction",
                            "occupancy_precision",
                            "occupancy_recall",
                            "occupancy_f1",
                        ]
                        rows.append(
                            dict(
                                event=event,
                                view=view_index,
                                condition=condition,
                                detector=name,
                                **{k: float(metrics[k]) for k in keep},
                            )
                        )
    summary = {}
    rng = np.random.default_rng(27182)
    resamples = rng.integers(len(cache), size=(5000, len(cache)))
    for detector in ["all", "tracker", "calorimeter"]:
        summary[detector] = {}
        reference = None
        for condition in ["correct", "foreign_event", "zero_context"]:
            selected = [
                r
                for r in rows
                if r["condition"] == condition and r["detector"] == detector
            ]
            values = {
                k: np.array([r[k] for r in selected])
                .reshape(len(cache), config.ssl.student_views)
                .mean(1)
                for k in keep
            }
            summary[detector][condition] = {
                k: float(v.mean()) for k, v in values.items()
            }
            if condition == "correct":
                reference = values
            else:
                summary[detector][condition]["paired_difference_from_correct"] = {
                    k: dict(
                        mean=float((v - reference[k]).mean()),
                        event_bootstrap_95ci=np.quantile(
                            (v - reference[k])[resamples].mean(1), [0.025, 0.975]
                        ).tolist(),
                    )
                    for k, v in values.items()
                }
    result = dict(
        checkpoint_step=checkpoint_step,
        events=len(cache),
        views=config.ssl.student_views,
        note="Fixed recipient queries and targets. Foreign event memory preserves recipient token counts by type. Zero memory is an OOD control, not a fitted geometry-only model. Bootstrap measures event variation only; cyclic donor reuse is not independent.",
        summary=summary,
        rows=rows,
    )
    (run / "analysis/context_check.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
