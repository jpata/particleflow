"""Strict conversion of the independent track/calo target view to TFDS examples."""

from pathlib import Path

import awkward as ak
import numpy as np

from mlpf.conf import EDM4HEP, ParticleFeatures

Y_FEATURES = ParticleFeatures.get_names()
CLASS_LABELS = [0, 211, 130, 22, 11, 13]
TRACK_EXTRAS = ["chi2", "ndf", "dEdx", "dEdxError", "radiusOfInnermostHit", "tanLambda", "D0", "omega", "Z0", "n_meas"]
X_FEATURES = EDM4HEP.HitFeatures.get_names() + TRACK_EXTRAS
X_FEATURES[1] = "pt_or_et"
X_FEATURES[5] = "p_or_energy"
PN = Y_FEATURES.index("particle_number")
JET = Y_FEATURES.index("jet_idx")
# Source columns refer to the documented ColliderML 17-column union schema.
COLLIDERML_TRACK_MAP = {"D0": 6, "Z0": 7, "tanLambda": 10, "omega": 11, "radiusOfInnermostHit": 12, "n_meas": 13}
KEY4HEP_TRACK_MAP = {name: i for i, name in enumerate(EDM4HEP.TrackFeatures.get_names()) if name in TRACK_EXTRAS or name == "time"}
REQUIRED = {
    "X_track",
    "X_hit_calo",
    "ytarget_track_calo_track",
    "ytarget_track_calo_hit",
    "track_calo_target_particles",
    "track_calo_original_to_target",
    "track_calo_energy_accounting",
    "targetjet_track_calo",
    "genmet",
    "genjet",
}


def required_fields(detector):
    # IDEA's legacy hit columns remain empty for backward compatibility.
    return (REQUIRED - {"X_hit_calo"}) | {"X_track_calo_hit"} if detector == "idea" else REQUIRED


def track_mapping(detector):
    if detector not in ("cld", "clic", "maia", "colliderml", "idea"):
        raise ValueError(f"Unsupported track/calo detector: {detector}")
    return COLLIDERML_TRACK_MAP if detector == "colliderml" else KEY4HEP_TRACK_MAP


def feature_availability(detector):
    """Schema availability, not a per-measurement quality or uncertainty mask."""
    tracks = [False] * len(X_FEATURES)
    tracks[:6] = [True] * 6
    for name in track_mapping(detector):
        tracks[X_FEATURES.index(name)] = True
    calo = [True] * 15 + [False] * len(TRACK_EXTRAS)
    if detector in ("maia", "colliderml", "idea"):
        for name in ("system", "side", "layer"):
            calo[X_FEATURES.index(name)] = False
    if detector == "idea":
        tracks[X_FEATURES.index("dEdx")] = False
        tracks[X_FEATURES.index("dEdxError")] = False
    if detector == "colliderml":
        calo[X_FEATURES.index("time")] = False
        calo[X_FEATURES.index("type")] = False
    return {"track": tracks, "calo_hit": calo}


def matrix(value, width, name):
    array = np.asarray(ak.to_numpy(value))
    if array.size == 0:
        return np.zeros((0, width), dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name}: expected [N,{width}], got {array.shape}; regenerate the source with the documented schema")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: non-finite values")
    return array.astype(np.float32, copy=True)


def input_features(event, detector):
    mapping = track_mapping(detector)
    track = matrix(event["X_track"], 17 if detector == "colliderml" else 16, "X_track")
    calo_name = "X_track_calo_hit" if detector == "idea" else "X_hit_calo"
    calo = matrix(event[calo_name], 15, calo_name)
    if np.any(track[:, 0] != 1) or np.any(calo[:, 0] != 2):
        raise ValueError("Expected elemtype=1 for reconstructed tracks and elemtype=2 for calorimeter hits")
    xtrack = np.zeros((len(track), len(X_FEATURES)), dtype=np.float32)
    xcalo = np.zeros((len(calo), len(X_FEATURES)), dtype=np.float32)
    xtrack[:, :6] = track[:, :6]
    for name, source in mapping.items():
        xtrack[:, X_FEATURES.index(name)] = track[:, source]
    xcalo[:, :15] = calo
    return xtrack, xcalo


def target_features(event, n_track, n_calo):
    track = matrix(event["ytarget_track_calo_track"], len(Y_FEATURES), "ytarget_track_calo_track")
    calo = matrix(event["ytarget_track_calo_hit"], len(Y_FEATURES), "ytarget_track_calo_hit")
    target = matrix(event["track_calo_target_particles"], len(Y_FEATURES), "track_calo_target_particles")
    jets = matrix(event["targetjet_track_calo"], 4, "targetjet_track_calo")
    if len(track) != n_track or len(calo) != n_calo:
        raise ValueError("Track/calo input and target rows are not aligned")
    rows = np.concatenate((track, calo))
    ids = rows[:, PN]
    if np.any((ids < 0) | (ids > len(target)) | (ids != np.floor(ids))):
        raise ValueError("Invalid particle_number membership labels")
    representatives = rows[rows[:, 0] != 0]
    representatives = representatives[np.argsort(representatives[:, PN])]
    if not np.array_equal(target[:, PN], np.arange(1, len(target) + 1)) or not np.array_equal(representatives, target):
        raise ValueError("Full representative rows must reproduce the unique track/calo particle table exactly")
    indices = target[:, JET]
    if np.any((indices < -1) | (indices >= len(jets)) | (indices != np.floor(indices))):
        raise ValueError("Invalid track/calo target jet indices")
    original = np.asarray(ak.to_numpy(event["track_calo_original_to_target"]))
    if not np.isfinite(original).all() or np.any((original < -1) | (original >= len(target)) | (original != np.floor(original))):
        raise ValueError("Invalid original-to-target mapping")
    accounting = np.asarray(ak.to_numpy(event["track_calo_energy_accounting"]), dtype=np.float64)
    if accounting.shape != (4,) or not np.isfinite(accounting).all() or np.any(accounting < 0):
        raise ValueError("Invalid track/calo energy accounting")
    if not np.isclose(accounting[0], accounting[1:].sum(), rtol=1e-8, atol=1e-7) or not np.isclose(
        accounting[3], target[:, 6].astype(float).sum(), rtol=1e-6, atol=1e-5
    ):
        raise ValueError("Track/calo target energy accounting does not close")
    # Fail on unknown classes instead of silently making them background.
    lookup = {pdg: index for index, pdg in enumerate(CLASS_LABELS)}
    unknown = set(rows[:, 0]) - set(lookup)
    if unknown:
        raise ValueError(f"Unsupported target PDG codes: {sorted(unknown)}")
    rows[:, 0] = [lookup[pdg] for pdg in rows[:, 0]]
    return rows, jets


def generate_examples(files, detector):
    required = required_fields(detector)
    for filename in files:
        missing = required - set(ak.metadata_from_parquet(filename)["form"].fields)
        if missing:
            raise ValueError(f"{filename}: missing {sorted(missing)}; regenerate Parquet with --track-calo-targets")
        # Avoid loading legacy targets, clusters and raw tracker hits.
        data = ak.from_parquet(filename, columns=sorted(required))
        n_events = len(data["X_track"])
        if any(len(data[name]) != n_events for name in required):
            raise ValueError(f"{filename}: event counts differ between columns")
        for index in range(n_events):
            event = {name: data[name][index] for name in required}
            try:
                track, calo = input_features(event, detector)
                ytarget, jets = target_features(event, len(track), len(calo))
                met = np.asarray(ak.to_numpy(event["genmet"])).reshape(-1)
                if met.size != 1 or not np.isfinite(met).all():
                    raise ValueError("genmet must be one finite scalar per event")
                genjets = matrix(event["genjet"], 4, "genjet")
            except ValueError as error:
                raise ValueError(f"{filename}, event {index}: {error}") from error
            # Preserve the original event index even for empty/background-only events.
            yield f"{Path(filename).name}:{index}", {
                "X": np.concatenate((track, calo)),
                "ytarget": ytarget,
                "ycand": np.zeros_like(ytarget),
                "genmet": np.float32(met[0]),
                "genjets": genjets,
                "targetjets": jets,
            }


def split_files(manual_dir, config_name):
    """Keep the existing deterministic file-based 90/10 split and 1..10 configs.

    Explicit train/ and test/ directories override the automatic split. Files
    must contain event-disjoint samples; this function never randomly splits rows.
    """
    root = Path(manual_dir)
    if (root / "train").is_dir() or (root / "test").is_dir():
        splits = {name: sorted((root / name).glob("*.parquet")) for name in ("train", "test")}
    else:
        files = sorted(root.glob("*.parquet"))
        boundary = int(0.9 * len(files))
        splits = {"train": files[:boundary], "test": files[boundary:]}
    if any(not files for files in splits.values()):
        raise ValueError("Provide event-disjoint Parquet in train/ and test/, or at least two files directly in manual_dir")
    if set(path.resolve() for path in splits["train"]) & set(path.resolve() for path in splits["test"]):
        raise ValueError("Train and test reference the same source file")
    selected = {}
    group = int(config_name) - 1
    for name, files in splits.items():
        # Reject ambiguous example keys (also avoids silently combining duplicate files).
        if len({file.name for file in files}) != len(files):
            raise ValueError(f"Duplicate file names in {name} split")
        size = len(files) // 10
        selected[name] = files[group * size : (group + 1) * size] if group < 9 else files[9 * size :]
        if not selected[name]:
            raise ValueError(f"No {name} files for config {config_name}; use config 10 for small samples")
    return selected
