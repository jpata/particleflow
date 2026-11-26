"""Turn a general ("full") TFDS example into the per-element layout used by training.

The full datasets (``*_full``) store tracks, clusters, tracker hits and calorimeter
hits separately, a target particle table and per-element truth labels (see
mlpf/data/full_targets.py). Training selects an ordered list of element kinds; this
module concatenates those kinds into ``X`` (element type = position in the list,
starting at 1), writes ``particle_number`` into ``ytarget`` and returns the labels as
``ylabel``. ``ytarget_set`` contains the target particles that own at least one
selected element, so the target definition is the same for every selection.
"""

from typing import Sequence

import numpy as np

from mlpf.conf import ParticleFeatures
from mlpf.data.full_targets import ELEMENT_KINDS, LABEL_FEATURES, PRIMARY_ORIGINS

Y_FEATURES = ParticleFeatures.get_names()
PN = Y_FEATURES.index("particle_number")
CHARGE = Y_FEATURES.index("charge")
LABEL_PN = LABEL_FEATURES.index("particle_number")
LABEL_ORIGIN = LABEL_FEATURES.index("origin")
# Class indices of CLASS_LABELS = [0, 211, 130, 22, 11, 13]
CHARGED_TO_NEUTRAL = {1: 2, 4: 3, 5: 2}
HIT_KINDS = ("hit_tracker", "hit_calo")
DEFAULT_ELEMENTS = HIT_KINDS


def is_full_example(example) -> bool:
    return "ytarget_particles" in example and "X_hit_tracker" in example


def validate_elements(elements: Sequence[str]) -> tuple:
    elements = tuple(elements)
    unknown = set(elements) - set(ELEMENT_KINDS)
    if not elements or unknown or len(set(elements)) != len(elements):
        raise ValueError(f"full_elements must be a non-empty list of distinct kinds from {ELEMENT_KINDS}, got {list(elements)}")
    return elements


def materialize_full_example(example, elements: Sequence[str] = DEFAULT_ELEMENTS):
    """Return a dict with X, ytarget, ycand, ylabel and ytarget_set for the selected kinds."""
    elements = validate_elements(elements)
    width = max(example[f"X_{kind}"].shape[1] for kind in elements)
    xs, labels = [], []
    for code, kind in enumerate(elements, start=1):
        x = np.asarray(example[f"X_{kind}"], dtype=np.float32)
        padded = np.zeros((len(x), width), dtype=np.float32)
        padded[:, : x.shape[1]] = x
        padded[:, 0] = code
        xs.append(padded)
        labels.append(np.asarray(example[f"ylabel_{kind}"], dtype=np.float32))
    X = np.concatenate(xs) if xs else np.zeros((0, width), dtype=np.float32)
    ylabel = np.concatenate(labels) if labels else np.zeros((0, len(LABEL_FEATURES)), dtype=np.float32)

    ytarget = np.zeros((len(X), len(Y_FEATURES)), dtype=np.float32)
    ytarget[:, PN] = ylabel[:, LABEL_PN]

    particles = np.asarray(example["ytarget_particles"], dtype=np.float32).copy()
    owners = ylabel[:, LABEL_PN].astype(np.int64)
    present = np.zeros(len(particles) + 1, dtype=bool)
    present[owners] = True
    # Charged particles keep their charged class only if a selected tracking element
    # (a track, or a tracker hit the particle produced itself) is available.
    tracking = np.zeros(len(particles) + 1, dtype=bool)
    offset = 0
    for kind, x in zip(elements, xs):
        kind_labels = ylabel[offset : offset + len(x)]
        if kind == "track":
            tracking[kind_labels[:, LABEL_PN].astype(np.int64)] = True
        elif kind == "hit_tracker":
            primary = np.isin(kind_labels[:, LABEL_ORIGIN].astype(np.int64), PRIMARY_ORIGINS)
            tracking[kind_labels[primary, LABEL_PN].astype(np.int64)] = True
        offset += len(x)
    rows = particles[:, PN].astype(np.int64)
    neutralize = ~tracking[rows]
    for charged, neutral in CHARGED_TO_NEUTRAL.items():
        change = neutralize & (particles[:, 0] == charged)
        particles[change, 0] = neutral
        particles[change, CHARGE] = 0
    ytarget_set = particles[present[rows]].copy()
    if len(ytarget_set):
        # Set outputs train pt and energy in absolute log space.
        ytarget_set[:, 2] = np.log(np.clip(ytarget_set[:, 2], 1e-8, None))
        ytarget_set[:, 6] = np.log(np.clip(ytarget_set[:, 6], 1e-8, None))
    return {
        "X": X,
        "ytarget": ytarget,
        "ycand": np.zeros_like(ytarget),
        "ylabel": ylabel,
        "ytarget_set": ytarget_set,
    }
