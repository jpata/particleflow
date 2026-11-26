"""General (full) dataset: truth labels, TFDS conversion, loader selection and origin-aware grouping."""

import copy
from types import SimpleNamespace

import awkward as ak
import fastjet
import numpy as np
import pytest
import tensorflow_datasets as tfds
import torch

from mlpf.conf import INPUT_TYPE_IDS, ParticleFeatures, dataset_for_sample, dataset_input_type_id, Dataset
from mlpf.data.full_targets import (
    BIT_DECAYED_IN_TRACKER,
    BIT_VERTEX_IS_NOT_ENDPOINT_OF_PARENT,
    INSTANCE_FEATURES,
    LABEL_FEATURES,
    ORIGIN_DECAY,
    ORIGIN_PRIMARY,
    ORIGIN_PRIMARY_LOOPING,
    ORIGIN_PRIMARY_UNSAVED_SECONDARY,
    ORIGIN_SECONDARY,
    TruthLinks,
    build_full_targets,
    parents_from_daughters,
    target_ancestors,
)
from mlpf.heptfds.full_utils.utils import X_FEATURES, convert_event
from mlpf.model.full_view import materialize_full_example
from mlpf.model.set_losses import hit_grouping_loss

JETDEF = fastjet.JetDefinition(fastjet.ee_genkt_algorithm, 0.4, -1.0)
Y = ParticleFeatures.get_names()
PN, INSTANCE, ORIGIN, PURITY = (LABEL_FEATURES.index(name) for name in ("particle_number", "instance_number", "origin", "purity"))


def toy_event():
    """MC 0: pi+ (target) decays in the tracker to mu+ (MC 1), which emits a delta electron (MC 2).
    MC 3: photon (target). MC 4: neutrino daughter of the pion (never a hit producer).

    Hits: 0-4 tracker, 5-7 calorimeter. Track 0 belongs to the pion, track 1 to the muon.
    Cluster 0 holds calorimeter hits 5 and 6.
    """
    pdg = np.array([211, -13, 11, 22, 14])
    charge = np.array([1, 1, -1, 0, 0])
    status = np.zeros(5, dtype=np.int64)
    status[0] = 1 << BIT_DECAYED_IN_TRACKER
    status[2] = 1 << BIT_VERTEX_IS_NOT_ENDPOINT_OF_PARENT
    daughters_begin = np.array([0, 2, 3, 3, 3])
    daughters_end = np.array([2, 3, 3, 3, 3])
    daughter_index = np.array([1, 4, 2])
    parent = parents_from_daughters(5, daughters_begin, daughters_end, daughter_index)
    momentum = np.array([[3.0, 0, 1], [2.0, 0.5, 0.5], [0.01, 0.01, 0], [0, 5.0, 0], [1.0, 0, 0]])
    mc = {
        "PDG": pdg.astype(float),
        "charge": charge.astype(float),
        "simulatorStatus": status,
        "px": momentum[:, 0],
        "py": momentum[:, 1],
        "pz": momentum[:, 2],
        "energy": np.linalg.norm(momentum, axis=1) + 0.1,
        "vertex_x": np.array([0, 300.0, 600.0, 0, 300.0]),
        "vertex_y": np.zeros(5),
        "vertex_z": np.zeros(5),
    }
    hit_positions = np.array([[100.0, 0, 0], [200, 0, 0], [150, 0, 0], [500, 0, 0], [700, 0, 0], [2000, 0, 0], [2000, 50, 0], [0, 2000, 0]])
    tracker_momentum = np.zeros((8, 3), dtype=np.float32)
    tracker_momentum[:5] = [[3, 0, 1], [3, 0, 1], [-3, 0, 1], [2, 0.5, 0.5], [0.01, 0.01, 0]]
    unsaved = np.zeros(8, dtype=bool)
    unsaved[1] = True
    truth = TruthLinks(
        mc=mc,
        parent=parent,
        targets=np.array([0, 3]),
        mc_to_hit=(np.array([0, 0, 0, 1, 2, 1, 0, 3]), np.array([0, 1, 2, 3, 4, 5, 6, 7]), np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1.5, 0.5, 5.0])),
        mc_to_track=(np.array([0, 1]), np.array([0, 1]), np.array([1.0, 1.0])),
        hit_to_cluster=(np.array([5, 6]), np.array([0, 0]), np.array([1.0, 1.0])),
        tracker_momentum=tracker_momentum,
        tracker_unsaved_secondary=unsaved,
        is_tracker_hit=np.arange(8) < 5,
        hit_positions=hit_positions,
    )
    gen = {
        "PDG": np.array([211, 22]),
        "charge": np.array([1.0, 0.0]),
        "pt": np.array([3.0, 5.0]),
        "eta": np.array([0.33, 0.0]),
        "sin_phi": np.array([0.0, 1.0]),
        "cos_phi": np.array([1.0, 0.0]),
        "energy": np.array([3.3, 5.0]),
    }
    return truth, gen


def test_target_ancestors_and_parents():
    truth, _ = toy_event()
    assert truth.parent.tolist() == [-1, 0, 1, -1, 0]
    assert target_ancestors(truth.parent, np.array([0, 3])).tolist() == [0, 0, 0, 1, 0]


def test_labels_origins_instances_and_accounting():
    truth, gen = toy_event()
    out = build_full_targets(truth, gen, 2, 1, source_energy=9.0, jetdef=JETDEF, jet_ptcut=0.0)
    tracker = out["ylabel_hit_tracker"]
    assert tracker[:, PN].tolist() == [1, 1, 1, 1, 1]
    assert tracker[:, ORIGIN].tolist() == [ORIGIN_PRIMARY, ORIGIN_PRIMARY_UNSAVED_SECONDARY, ORIGIN_PRIMARY_LOOPING, ORIGIN_DECAY, ORIGIN_SECONDARY]
    # the pion's own hits share one instance; the muon and the delta electron are separate instances
    assert tracker[0, INSTANCE] == tracker[1, INSTANCE] == tracker[2, INSTANCE]
    assert len({tracker[0, INSTANCE], tracker[3, INSTANCE], tracker[4, INSTANCE]}) == 3
    np.testing.assert_allclose(tracker[0, 4:7], [3, 0, 1])
    calo = out["ylabel_hit_calo"]
    assert calo[:, PN].tolist() == [1, 1, 2]
    assert calo[:, ORIGIN].tolist() == [ORIGIN_DECAY, ORIGIN_PRIMARY, ORIGIN_PRIMARY]
    assert np.all(calo[:, 4:7] == 0)
    assert out["ylabel_track"][:, ORIGIN].tolist() == [ORIGIN_PRIMARY, ORIGIN_DECAY]
    # the cluster mixes the muon (1.5) and the pion (0.5): both belong to the pion target
    cluster = out["ylabel_cluster"][0]
    assert cluster[PN] == 1 and cluster[ORIGIN] == ORIGIN_DECAY and cluster[PURITY] == pytest.approx(1.0)
    targets = out["target_particles"]
    assert targets[:, Y.index("particle_number")].tolist() == [1, 2]
    assert targets[:, 0].tolist() == [211, 22]
    instances = out["target_instances"]
    assert instances.shape == (4, len(INSTANCE_FEATURES))
    assert instances[:, 1].tolist() == [1, 1, 1, 2]
    np.testing.assert_allclose(out["target_energy_accounting"], [9.0, 0.7, 0.0, 8.3], atol=1e-6)


def test_unlabelled_target_is_dropped_and_background_kept():
    truth, gen = toy_event()
    truth.mc_to_hit = tuple(x[:-1] for x in truth.mc_to_hit)  # the photon loses its only hit
    out = build_full_targets(truth, gen, 2, 1, source_energy=9.0, jetdef=JETDEF, jet_ptcut=0.0)
    assert len(out["target_particles"]) == 1
    assert out["ylabel_hit_calo"][2].tolist() == [0] * len(LABEL_FEATURES)
    np.testing.assert_allclose(out["target_energy_accounting"], [9.0, 0.7, 5.0, 3.3], atol=1e-6)


def parquet_event():
    truth, gen = toy_event()
    out = build_full_targets(truth, gen, 2, 1, source_energy=9.0, jetdef=JETDEF, jet_ptcut=0.0)
    rng = np.random.default_rng(1)
    event = {
        f"X_{kind}": rng.normal(size=(n, len(X_FEATURES[kind]))).astype(np.float32)
        for kind, n in (("track", 2), ("cluster", 1), ("hit_tracker", 5), ("hit_calo", 3))
    }
    event["X_hit_tracker"][:, 10] = 3
    event.update(out)
    event["genmet"] = 1.0
    event["genjet"] = np.zeros((0, 4), dtype=np.float32)
    return event


def test_convert_event_validates_and_maps_classes():
    example = convert_event(parquet_event())
    assert example["ytarget_particles"][:, 0].tolist() == [1, 3]
    assert example["X_hit_tracker"].shape == (5, 15)

    broken = parquet_event()
    broken["ylabel_hit_calo"][0, INSTANCE] = 4  # the photon's instance on a pion-owned hit
    with pytest.raises(ValueError, match="different target"):
        convert_event(broken)
    broken = parquet_event()
    broken["target_energy_accounting"][3] += 1
    with pytest.raises(ValueError, match="accounting"):
        convert_event(broken)
    broken = parquet_event()
    broken["ylabel_track"] = broken["ylabel_track"][:1]
    with pytest.raises(ValueError, match="aligned"):
        convert_event(broken)


@pytest.mark.parametrize(
    "elements,widths,pion_class",
    [
        (["hit_tracker", "hit_calo"], 15, 1),
        (["track", "hit_calo"], 16, 1),
        (["cluster", "hit_calo"], 19, 2),  # no tracking element: the pion becomes a neutral hadron target
        (["hit_calo"], 15, 2),
    ],
)
def test_materialize_selection(elements, widths, pion_class):
    example = convert_event(parquet_event())
    out = materialize_full_example(example, elements)
    counts = {"track": 2, "cluster": 1, "hit_tracker": 5, "hit_calo": 3}
    assert out["X"].shape == (sum(counts[kind] for kind in elements), widths)
    assert out["X"][:, 0].tolist() == sum(([code] * counts[kind] for code, kind in enumerate(elements, start=1)), [])
    assert out["ytarget"][:, Y.index("particle_number")].tolist() == out["ylabel"][:, PN].tolist()
    assert out["ytarget_set"][:, 0].tolist() == [pion_class, 3]
    if pion_class == 2:
        assert out["ytarget_set"][0, Y.index("charge")] == 0
    np.testing.assert_allclose(out["ytarget_set"][:, 2], np.log([3.0, 5.0]), rtol=1e-6)


def test_materialize_rejects_unknown_kind():
    with pytest.raises(ValueError, match="full_elements"):
        materialize_full_example(convert_event(parquet_event()), ["hits"])


def test_dataset_names():
    assert dataset_input_type_id("cld_edm_ttbar_full") == INPUT_TYPE_IDS["full"]
    assert dataset_for_sample("cld_edm_ttbar_full", Dataset.CLD) == Dataset.CLD_HITS


def test_tfds_roundtrip_and_loader(tmp_path):
    from mlpf.heptfds.cld_full.ttbar import CldEdmTtbarFull
    from mlpf.model.PFDataset import TFDSDataSource

    manual = tmp_path / "manual"
    for split in ("train", "test"):
        (manual / split).mkdir(parents=True)
        event = parquet_event()
        ak.to_parquet(
            ak.Array([{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in event.items()}]), manual / split / f"{split}.parquet"
        )
    builder = CldEdmTtbarFull(config="10", data_dir=tmp_path / "tfds")
    builder.download_and_prepare(download_config=tfds.download.DownloadConfig(manual_dir=manual, try_download_gcs=False))
    assert builder.name == "cld_edm_ttbar_full"
    loaded = tfds.builder_from_directory(builder.data_dir)
    assert loaded.info.metadata["input_type"] == "full"
    source = loaded.as_data_source(split="test")
    adapted = TFDSDataSource(source, sort=True, pad_to_multiple=16, build_target_set=True)[0]
    assert adapted["X"].shape == (16, 15) and adapted["ylabel"].shape == (16, len(LABEL_FEATURES))
    assert adapted["ytarget_set"].shape == (2, len(Y))
    assert adapted["input_type_id"] == INPUT_TYPE_IDS["hits"]
    assert not any(key.startswith("X_") for key in adapted)
    # analysis readers without a target set still get the materialized inputs and labels
    plain = TFDSDataSource(source, sort=False, build_target_set=False)[0]
    assert plain["X"].shape == (8, 15) and plain["ylabel"].shape == (8, len(LABEL_FEATURES))
    mixed = TFDSDataSource(source, sort=False, build_target_set=True, full_elements=["track", "hit_calo"])[0]
    assert mixed["input_type_id"] == INPUT_TYPE_IDS["full"]


def grouping_batch():
    example = convert_event(parquet_event())
    out = materialize_full_example(example, ["hit_tracker", "hit_calo"])
    batch = SimpleNamespace(
        X=torch.as_tensor(out["X"])[None],
        ytarget=torch.as_tensor(out["ytarget"])[None],
        ytarget_set=torch.as_tensor(out["ytarget_set"])[None],
        ylabel=torch.as_tensor(out["ylabel"])[None],
    )
    batch.mask = batch.X[..., 0] != 0
    batch.target_mask = batch.ytarget_set[..., 0] != 0
    torch.manual_seed(0)
    grouping = (torch.randn(1, 8, 4), torch.randn(1, 3, 4), torch.randn(1, 8, 1))
    matches = [(torch.tensor([0, 2]), torch.tensor([0, 1]))]
    return grouping, batch, matches


def test_grouping_origins():
    grouping, batch, matches = grouping_batch()
    inclusive = hit_grouping_loss(grouping, batch, matches)
    everything = hit_grouping_loss(grouping, batch, matches, origins=[1, 2, 3, 4, 5])
    assert torch.allclose(inclusive, everything)
    ignored = hit_grouping_loss(grouping, batch, matches, origins=[ORIGIN_PRIMARY])
    background = hit_grouping_loss(grouping, batch, matches, origins=[ORIGIN_PRIMARY], excluded_as="background")
    assert torch.isfinite(ignored) and torch.isfinite(background)
    assert not torch.allclose(ignored, inclusive) and not torch.allclose(ignored, background)
    # ignoring excluded hits equals the inclusive loss over the remaining hits only
    keep = torch.isin(batch.ylabel[0, :, ORIGIN].long(), torch.tensor([0, ORIGIN_PRIMARY]))
    reduced = copy.copy(batch)
    reduced.X, reduced.ytarget, reduced.ylabel = batch.X[:, keep], batch.ytarget[:, keep], batch.ylabel[:, keep]
    reduced.mask = batch.mask[:, keep]
    reduced_grouping = (grouping[0][:, keep], grouping[1], grouping[2][:, keep])
    assert torch.allclose(ignored, hit_grouping_loss(reduced_grouping, reduced, matches))
    batch.ylabel = None
    with pytest.raises(ValueError, match="ylabel"):
        hit_grouping_loss(grouping, batch, matches, origins=[ORIGIN_PRIMARY])


def full_model_config(elements, hit_features=True):
    from mlpf.conf import MLPFConfig

    samples = {"physical": {"batch_size": 1, "samples": {"cld_edm_ttbar_full": {"version": "1.0.0", "splits": ["10"]}}}}
    return MLPFConfig.model_validate(
        {
            "dataset": "cld_hits",
            "data_dir": "/tmp",
            "full_elements": elements,
            "train_dataset": {"cld_hits": samples},
            "model": {
                "type": "heptv2",
                "output_mode": "set",
                "input_encoding": "split",
                "heptv2": {"num_convs": 1, "num_heads": 2, "embedding_dim": 16, "width": 16, "block_size": 8},
                "set_decoder": {"num_slots": 6, "num_layers": 2, "num_heads": 2, "query_init": "input-conditioned", "hit_grouping": True},
                "hit_feature_engineering": {"enabled": hit_features},
            },
            "conv_type": "heptv2",
        }
    )


@pytest.mark.parametrize(
    "elements,width,types",
    [
        (["hit_tracker", "hit_calo"], 15, [1, 2]),
        (["track", "hit_calo"], 16, [1, 2]),
        (["track", "cluster"], 19, [1, 2]),
        (["track", "cluster", "hit_tracker", "hit_calo"], 19, [1, 2, 3, 4]),
    ],
)
def test_model_config_and_forward_for_selections(elements, width, types):
    from mlpf.model.mlpf import MLPF

    config = full_model_config(elements)
    assert config.uses_full_datasets()
    assert config.input_dim == width and config.elemtypes_nonzero == types
    model = MLPF(config)
    example = convert_event(parquet_event())
    out = materialize_full_example(example, elements)
    X = torch.as_tensor(out["X"])[None]
    X[..., 1] = X[..., 1].abs() + 0.1
    X[..., 5] = X[..., 5].abs() + 0.1
    X = torch.nn.functional.pad(X, (0, 0, 0, 16 - X.shape[1]))  # HEPTv2 block size; the loader pads to a multiple
    mask = X[..., 0] != 0
    engineered = model._engineer_input_features(X, mask)
    # raw features (including element codes) are untouched; engineered columns are zero off hit rows
    assert torch.equal(engineered[..., :width][mask], X[mask])
    hit_codes = [code for code, kind in enumerate(elements, start=1) if kind.startswith("hit_")]
    off_hit = ~torch.isin(X[..., 0], torch.tensor(hit_codes, dtype=X.dtype)) & mask
    assert torch.all(engineered[..., width:][off_hit] == 0)
    predictions = model(X, mask)
    assert all(torch.isfinite(output).all() for output in predictions)


def test_mixed_selection_matches_hit_selection_on_hit_rows():
    from mlpf.model.mlpf import MLPF

    hits = MLPF(full_model_config(["hit_tracker", "hit_calo"]))
    mixed = MLPF(full_model_config(["track", "hit_tracker", "hit_calo"]))
    example = convert_event(parquet_event())
    X_hits = torch.as_tensor(materialize_full_example(example, ["hit_tracker", "hit_calo"])["X"])[None]
    X_mixed = torch.as_tensor(materialize_full_example(example, ["track", "hit_tracker", "hit_calo"])["X"])[None]
    a = hits._engineer_input_features(X_hits, torch.ones(X_hits.shape[:2], dtype=torch.bool))
    b = mixed._engineer_input_features(X_mixed, torch.ones(X_mixed.shape[:2], dtype=torch.bool))
    # the same hits get the same engineered features whether or not tracks are present
    torch.testing.assert_close(b[0, 2:, 16:], a[0, :, 15:])


def test_decay_rule_uses_species_and_vertex_groups():
    from mlpf.data.full_targets import decay_products

    # 0 pi+ -> 1 mu+ (+ 2 nu at the same vertex); 0 pi+ also knocks out 3 p + 4 n elsewhere;
    # 5 K0S -> 6 pi+ 7 pi-; 8 pi- emits a lone 9 e- (delta-ray-like); 10 mu -> lone 11 e
    parent = np.array([-1, 0, 0, 0, 0, -1, 5, 5, -1, 8, -1, 10])
    pdg = np.array([211, -13, 14, 2212, 2112, 310, 211, -211, -211, 11, 13, 11])
    vertex = np.zeros((12, 3))
    vertex[[1, 2]] = [100, 0, 0]
    vertex[[3, 4]] = [200, 0, 0]
    vertex[[6, 7]] = [50, 0, 0]
    vertex[9] = [30, 0, 0]
    vertex[11] = [70, 0, 0]
    decay = decay_products(parent, pdg, vertex)
    assert decay.tolist() == [False, True, True, False, False, False, True, True, False, False, False, False]
    # along-path products (edm4hep bit 28) are never decays
    continues = np.zeros(12, dtype=bool)
    continues[[1, 2]] = True
    assert not decay_products(parent, pdg, vertex, continues)[[1, 2]].any()


def test_inward_hits_by_time_and_calorimeter_entry():
    from mlpf.data.colliderml.full_view import calorimeter_entry_rows
    from mlpf.data.full_targets import inward_hits_by_time

    producer = np.array([0, 0, 0, 0, 1])
    positions = np.array([[100.0, 0, 0], [300, 0, 0], [500, 0, 0], [350, 0, 0], [50, 0, 0]])
    times = np.array([0.1, 0.2, 0.3, 0.4, 0.0])
    assert inward_hits_by_time(producer, positions, times, np.ones(5, dtype=bool)).tolist() == [False, False, False, True, False]

    # 0 target photon (tracker region) -> 1 e (calorimeter) -> 2 e (calorimeter); 3 is a target
    # produced in the calorimeter region and stays itself.
    parent_row = np.array([-1, 0, 1, -1])
    vertex = np.array([[0.0, 0, 0], [1300, 0, 0], [1400, 0, 0], [1300, 0, 0]])
    target_index = np.array([0, -1, -1, 1])
    assert calorimeter_entry_rows(np.array([2, 1, 3, -1]), parent_row, vertex, target_index).tolist() == [0, 0, 3, -1]


def test_full_datasets_require_set_output_mode():
    from mlpf.conf import MLPFConfig

    config = full_model_config(["hit_tracker", "hit_calo"]).model_dump(mode="json")
    config["model"]["output_mode"] = "elementwise"
    with pytest.raises(ValueError, match="support only model.output_mode='set'"):
        MLPFConfig.model_validate(config)
