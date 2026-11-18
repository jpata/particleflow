"""Strict conversion of the key4hep full view (tracks, clusters, tracker and calorimeter hits) to TFDS examples."""

from pathlib import Path

import awkward as ak
import numpy as np

from mlpf.conf import EDM4HEP, ParticleFeatures
from mlpf.data.full_targets import ACCOUNTING_FEATURES, ELEMENT_KINDS, INSTANCE_FEATURES, LABEL_FEATURES, ORIGIN_NAMES
from mlpf.heptfds.track_calo_utils.utils import split_files  # noqa: F401  (shared deterministic file split)

Y_FEATURES = ParticleFeatures.get_names()
CLASS_LABELS = [0, 211, 130, 22, 11, 13]
X_FEATURES = {
    "track": EDM4HEP.TrackFeatures.get_names(),
    "cluster": EDM4HEP.ClusterFeatures.get_names(),
    "hit_tracker": EDM4HEP.HitFeatures.get_names(),
    "hit_calo": EDM4HEP.HitFeatures.get_names(),
}
PN = Y_FEATURES.index("particle_number")
JET = Y_FEATURES.index("jet_idx")
ENERGY = Y_FEATURES.index("energy")
LABEL_PN = LABEL_FEATURES.index("particle_number")
LABEL_INSTANCE = LABEL_FEATURES.index("instance_number")
LABEL_ORIGIN = LABEL_FEATURES.index("origin")
REQUIRED = {
    *(f"X_{kind}" for kind in ELEMENT_KINDS),
    *(f"ylabel_{kind}" for kind in ELEMENT_KINDS),
    "target_particles",
    "target_instances",
    "target_energy_accounting",
    "targetjet_full",
    "genmet",
    "genjet",
}


def matrix(value, width, name):
    array = np.asarray(ak.to_numpy(value))
    if array.size == 0:
        return np.zeros((0, width), dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name}: expected [N,{width}], got {array.shape}; regenerate the source with the current postprocessor")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: non-finite values")
    return array.astype(np.float32, copy=True)


def _integer_column(values, low, high, name):
    if np.any((values < low) | (values > high) | (values != np.floor(values))):
        raise ValueError(f"{name}: values outside [{low}, {high}] or not integers")


def convert_event(event):
    """Validate one Parquet event and return the TFDS example (without key)."""
    example = {}
    for kind in ELEMENT_KINDS:
        x = matrix(event[f"X_{kind}"], len(X_FEATURES[kind]), f"X_{kind}")
        labels = matrix(event[f"ylabel_{kind}"], len(LABEL_FEATURES), f"ylabel_{kind}")
        if len(labels) != len(x):
            raise ValueError(f"X_{kind} and ylabel_{kind} are not aligned")
        example[f"X_{kind}"] = x
        example[f"ylabel_{kind}"] = labels

    targets = matrix(event["target_particles"], len(Y_FEATURES), "target_particles")
    instances = matrix(event["target_instances"], len(INSTANCE_FEATURES), "target_instances")
    jets = matrix(event["targetjet_full"], 4, "targetjet_full")
    if not np.array_equal(targets[:, PN], np.arange(1, len(targets) + 1)):
        raise ValueError("target_particles must be numbered 1..P")
    if not np.array_equal(instances[:, 0], np.arange(1, len(instances) + 1)):
        raise ValueError("target_instances must be numbered 1..S")
    _integer_column(instances[:, 1], 1, len(targets), "instance particle_number")
    _integer_column(targets[:, JET], -1, len(jets) - 1, "target jet_idx")
    labelled = np.zeros(len(targets), dtype=bool)
    for kind in ELEMENT_KINDS:
        labels = example[f"ylabel_{kind}"]
        _integer_column(labels[:, LABEL_PN], 0, len(targets), f"ylabel_{kind} particle_number")
        _integer_column(labels[:, LABEL_INSTANCE], 0, len(instances), f"ylabel_{kind} instance_number")
        _integer_column(labels[:, LABEL_ORIGIN], 0, max(ORIGIN_NAMES), f"ylabel_{kind} origin")
        owned = labels[:, LABEL_PN] > 0
        if np.any((labels[:, LABEL_INSTANCE] > 0) != owned) or np.any((labels[:, LABEL_ORIGIN] > 0) != owned):
            raise ValueError(f"ylabel_{kind}: instance/origin must be set exactly for labelled elements")
        instance_owner = instances[labels[owned, LABEL_INSTANCE].astype(int) - 1, 1]
        if not np.array_equal(instance_owner, labels[owned, LABEL_PN]):
            raise ValueError(f"ylabel_{kind}: element instance belongs to a different target particle")
        labelled[labels[owned, LABEL_PN].astype(int) - 1] = True
    if not labelled.all():
        raise ValueError("every target particle must label at least one element")

    accounting = np.asarray(ak.to_numpy(event["target_energy_accounting"]), dtype=np.float64)
    if accounting.shape != (len(ACCOUNTING_FEATURES),) or not np.isfinite(accounting).all():
        raise ValueError("invalid target_energy_accounting")
    if not np.isclose(accounting[0], accounting[1:].sum(), rtol=1e-6, atol=1e-4) or not np.isclose(
        accounting[3], targets[:, ENERGY].astype(np.float64).sum(), rtol=1e-5, atol=1e-3
    ):
        raise ValueError("target energy accounting does not close")

    lookup = {pdg: index for index, pdg in enumerate(CLASS_LABELS)}
    unknown = set(targets[:, 0].astype(int)) - set(lookup)
    if unknown:
        raise ValueError(f"Unsupported target classes: {sorted(unknown)}")
    targets[:, 0] = [lookup[int(pdg)] for pdg in targets[:, 0]]

    met = np.asarray(ak.to_numpy(event["genmet"])).reshape(-1)
    if met.size != 1 or not np.isfinite(met).all():
        raise ValueError("genmet must be one finite scalar per event")
    example.update(
        {
            "ytarget_particles": targets,
            "target_instances": instances,
            "energy_accounting": accounting.astype(np.float32),
            "genmet": np.float32(met[0]),
            "genjets": matrix(event["genjet"], 4, "genjet"),
            "targetjets": jets,
        }
    )
    return example


def generate_examples(files):
    for filename in files:
        missing = REQUIRED - set(ak.metadata_from_parquet(filename)["form"].fields)
        if missing:
            raise ValueError(f"{filename}: missing {sorted(missing)}; regenerate Parquet with the current postprocessor")
        data = ak.from_parquet(filename, columns=sorted(REQUIRED))
        n_events = len(data["X_track"])
        for index in range(n_events):
            event = {name: data[name][index] for name in REQUIRED}
            try:
                example = convert_event(event)
            except ValueError as error:
                raise ValueError(f"{filename}, event {index}: {error}") from error
            yield f"{Path(filename).name}:{index}", example
