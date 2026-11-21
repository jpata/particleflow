"""Observable-element self-distillation; no particle supervision or backbone changes.

Panda-inspired rather than a reproduction: a clean EMA teacher, modality-stratified
point dropout, and DINO centering/sharpening with event/type-balanced reductions.
"""

import copy
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from pydantic import BaseModel, ConfigDict, Field


class DistillationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prototypes: int = Field(default=256, ge=2)
    hidden_dim: int = Field(default=256, ge=1)
    projection_dim: int = Field(default=64, ge=1)
    keep_fraction: float = Field(default=0.6, gt=0, lt=1)
    student_views: int = Field(default=2, ge=1)
    student_temperature: float = Field(default=0.1, gt=0)
    teacher_temperature: float = Field(default=0.04, gt=0)
    teacher_momentum: float = Field(default=0.996, ge=0, lt=1)
    center_momentum: float = Field(default=0.9, ge=0, lt=1)


def collate_inputs(records):
    """Accept only measured inputs and metadata; preserve variable event lengths."""
    xs = [torch.as_tensor(r["X"], dtype=torch.float32) for r in records]
    xs = [x[x[:, 0] != 0] for x in xs]
    if any(len(x) == 0 for x in xs):
        raise ValueError("SSL requires at least one valid element per event")
    return {
        "X": nn.utils.rnn.pad_sequence(xs, batch_first=True),
        "source_id": torch.tensor([int(r["source_id"]) for r in records]),
    }


def type_counts(types, mask, num_types=None):
    num_types = int(types.max()) + 1 if num_types is None else num_types
    counts = torch.zeros(types.shape[0], num_types, device=types.device, dtype=torch.long)
    return counts.scatter_add_(1, types.long(), mask.long())


def element_weights(types, mask, num_types=None):
    counts = type_counts(types, mask, num_types)
    denominator = counts.gather(1, types.long()).clamp_min(1) * (counts > 0).sum(1, keepdim=True).clamp_min(1) * types.shape[0]
    return mask.float() / denominator


def weighted_mean(values, weights):
    # GEMV avoids materializing a [batch, elements, prototypes] weighted copy.
    with torch.autocast(values.device.type, enabled=False):
        result = weights.reshape(-1).to(values.dtype) @ values.reshape(weights.numel(), -1)
    return result.reshape(values.shape[2:])


def subsample_view(X, mask, fraction, generator=None, num_types=None):
    """Physically compact the view, retaining >=1 element of every observed type.

    Returned indices address the original padded event, never particle IDs.
    Feature engineering and HEPT hashing subsequently see only retained elements.
    """
    types = X[..., 0].long()
    counts = type_counts(types, mask, num_types)
    quotas = (counts.double() * fraction).long().clamp_min(1) * (counts > 0)
    lengths = quotas.sum(1)
    if int(lengths.min()) == 0:
        raise ValueError("SSL cannot subsample an empty event")
    # Draw only for valid rows so padding does not change the random views.
    scores = torch.full(mask.shape, float("inf"), device=X.device)
    scores.masked_scatter_(mask, torch.rand(int(mask.sum()), device=X.device, generator=generator))
    random_order = scores.argsort(dim=1, stable=True)
    random_types = types.gather(1, random_order)
    order = random_order.gather(1, random_types.argsort(dim=1, stable=True))
    ordered_types = types.gather(1, order)
    # Include padded rows in offsets (type zero), but never in the quotas.
    all_counts = type_counts(types, torch.ones_like(mask), counts.shape[1])
    starts = all_counts.cumsum(1) - all_counts
    ranks = torch.arange(X.shape[1], device=X.device)[None] - starts.gather(1, ordered_types)
    take = (ranks < quotas.gather(1, ordered_types)) & mask.gather(1, order)
    keep = torch.zeros_like(mask).scatter_(1, order, take)
    original = torch.arange(X.shape[1], device=X.device).expand_as(mask)
    idx = original.masked_fill(~keep, X.shape[1]).sort(dim=1).values[:, : int(lengths.max())]
    valid = idx < X.shape[1]
    idx = idx.clamp_max(X.shape[1] - 1)
    view = X.gather(1, idx[..., None].expand(-1, -1, X.shape[-1]))
    return view.masked_fill(~valid[..., None], 0), valid, idx


def balanced_mean(values, types, mask):
    """Average elements within type, types within event, and events within batch."""
    return weighted_mean(values, element_weights(types, mask))


class ProjectionHead(nn.Module):
    def __init__(self, dimension, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(dimension), nn.Linear(dimension, config.hidden_dim), nn.GELU(), nn.Linear(config.hidden_dim, config.projection_dim)
        )
        self.prototypes = nn.Parameter(torch.randn(config.prototypes, config.projection_dim))

    def forward(self, features):
        features = self.mlp(features)
        # Keep temperatures, softmax, and moving-center statistics in FP32.
        with torch.autocast(features.device.type, enabled=False):
            z = F.normalize(features.float(), dim=-1)
            return F.linear(z, F.normalize(self.prototypes.float(), dim=-1))


class StudentTeacher(nn.Module):
    def __init__(self, student, dimension, config):
        super().__init__()
        if student.use_split_backbone:
            raise ValueError("SSL requires backbone.mode=shared; encode_backbone exposes only the ID branch in split mode")
        self.config = config
        self.num_element_types = max(student.elemtypes_nonzero) + 1
        self.student = student
        # PF heads are neither trained nor used during pretraining.
        student.requires_grad_(False)
        for name in ("nn0", "backbone", "tracker_backbone", "calo_backbone", "feature_engineering", "feature_bag", "input_normalization"):
            module = getattr(student, name, None)
            if module is not None:
                module.requires_grad_(True)
        self.student_head = ProjectionHead(dimension, config)
        self.teacher = copy.deepcopy(student).requires_grad_(False).eval()
        self.teacher_head = copy.deepcopy(self.student_head).requires_grad_(False).eval()
        self.register_buffer("center", torch.zeros(1, 1, config.prototypes))

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        self.teacher_head.eval()
        return self

    @torch.no_grad()
    def update_teacher(self):
        """Call once after each successful optimizer step."""
        for teacher, student in ((self.teacher, self.student), (self.teacher_head, self.student_head)):
            torch._foreach_lerp_(list(teacher.parameters()), list(student.parameters()), 1 - self.config.teacher_momentum)
            for t, s in zip(teacher.buffers(), student.buffers()):
                t.copy_(s)

    def forward(self, X, source_id=None, generator=None, timings=None):
        cfg = self.config
        mask = X[..., 0] != 0
        full_weights = element_weights(X[..., 0], mask, self.num_element_types)
        with torch.no_grad(), timings.phase("teacher") if timings else nullcontext():
            teacher_logits = self.teacher_head(self.teacher.encode_backbone(X, mask, source_id))
            targets = ((teacher_logits - self.center) / cfg.teacher_temperature).softmax(-1)
        losses = []
        for _ in range(cfg.student_views):
            with timings.phase("view_sampling") if timings else nullcontext():
                view, valid, idx = subsample_view(X, mask, cfg.keep_fraction, generator, self.num_element_types)
            with timings.phase("student") if timings else nullcontext():
                logits = self.student_head(self.student.encode_backbone(view, valid, source_id))
            with timings.phase("loss") if timings else nullcontext():
                target = targets.gather(1, idx[..., None].expand(-1, -1, cfg.prototypes))
                ce = -(target * (logits / cfg.student_temperature).log_softmax(-1)).sum(-1)
                losses.append(weighted_mean(ce, element_weights(view[..., 0], valid, self.num_element_types)))
        loss = torch.stack(losses).mean()
        with torch.no_grad(), timings.phase("statistics") if timings else nullcontext():
            marginal = weighted_mean(targets, full_weights)
            entropy = -(targets * targets.clamp_min(1e-12).log()).sum(-1)
            metrics = {
                "loss": loss.detach(),
                "teacher_entropy": weighted_mean(entropy, full_weights),
                "effective_prototypes": (-(marginal * marginal.clamp_min(1e-12).log()).sum()).exp(),
            }
            if self.training:
                mean = weighted_mean(teacher_logits, full_weights)
                self.center.lerp_(mean[None, None], 1 - cfg.center_momentum)
        return loss, metrics
