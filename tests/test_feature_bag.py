"""Tests of the feature bag: shared slots across detectors, aliases, the IDEA override,
constant and "Null" columns, scaling, and a normalized feature-bag MLPF on mixed detectors."""

import math

import pytest
import torch

from mlpf.conf import SOURCE_IDS, TRACK_CLUSTER_SOURCES, X_FEATURES, MLPFConfig, column_names
from mlpf.model.feature_bag import (
    CLUSTER,
    FEATURE_ALIASES,
    FEATURE_OVERRIDES,
    FEATURE_VOCABULARY,
    SOURCE_FEATURE_COLUMNS,
    TRACK,
    FeatureBag,
)
from mlpf.model.input_normalization import InputNormalization
from mlpf.model.mlpf import MLPF

SLOT = {name: index for index, name in enumerate(FEATURE_VOCABULARY)}
# Committed normalization statistics of the datasets used below.
VERSIONS = {"cld": "3.3.0", "clic": "3.3.0", "colliderml": "1.5.0", "idea": "0.1.0"}
SAMPLES = {"cld": "cld_edm_ttbar_pf", "clic": "clic_edm_qq_pf", "colliderml": "colliderml_ttbar_nopu_pf", "idea": "idea_edm_qq_pf"}


def one_row(elemtype, columns, input_dim=17):
    """A [1, 1, input_dim] event with the given {column: value}."""
    X = torch.zeros(1, 1, input_dim)
    X[..., 0] = elemtype
    for column, value in columns.items():
        X[..., column] = value
    return X


def scale(source, elemtype, input_dim=17):
    present = [column for column in SOURCE_FEATURE_COLUMNS[source][elemtype] if column < input_dim]
    return math.sqrt(len(FEATURE_VOCABULARY) / len(present))


def source(name):
    return torch.tensor([SOURCE_IDS[name]])


def test_vocabulary_is_the_union_of_the_dataset_features():
    assert list(FEATURE_VOCABULARY) == sorted(set(FEATURE_VOCABULARY))
    assert set(TRACK_CLUSTER_SOURCES) == {"cld", "clic", "colliderml", "idea", "maia"}
    # Features that are constant in today's data keep their slot.
    assert {"dEdx", "dEdxError", "time", "iTheta", "energy_other"} <= set(FEATURE_VOCABULARY)
    # The element type and "Null" columns are not features.
    assert not {"type", "elemtype", "Null"} & set(FEATURE_VOCABULARY)


def test_same_quantity_in_different_columns_shares_a_slot():
    bag = FeatureBag(17)
    # tanLambda is column 11 for CLD tracks and column 10 for ColliderML tracks.
    cld = bag(one_row(TRACK, {11: 0.7}), source("cld"))
    colliderml = bag(one_row(TRACK, {10: 0.7}), source("colliderml"))

    assert cld[0, 0, SLOT["tanLambda"]] / scale("cld", TRACK) == pytest.approx(0.7)
    assert colliderml[0, 0, SLOT["tanLambda"]] / scale("colliderml", TRACK) == pytest.approx(0.7)
    # Column 11 of a ColliderML track is omega, not tanLambda.
    assert bag(one_row(TRACK, {11: 0.3}), source("colliderml"))[0, 0, SLOT["tanLambda"]] == 0


def test_aliases_share_a_slot():
    bag = FeatureBag(17)
    # CLD names the impact parameter D0 (column 12), ColliderML d0 (column 6).
    assert bag(one_row(TRACK, {12: 0.5}), source("cld"))[0, 0, SLOT["d0"]] > 0
    assert bag(one_row(TRACK, {6: 0.5}), source("colliderml"))[0, 0, SLOT["d0"]] > 0
    # Cluster et and track pt (column 1), cluster energy and track p (column 5).
    for elemtype in (TRACK, CLUSTER):
        out = bag(one_row(elemtype, {1: 1.0, 5: 2.0}), source("cld"))[0, 0]
        assert out[SLOT["pt"]] > 0 and out[SLOT["energy"]] > 0
        assert torch.count_nonzero(out) == 2


def test_idea_override():
    bag = FeatureBag(17)
    idea = bag(one_row(CLUSTER, {10: 1.0, 11: 2.0}), source("idea"))[0, 0]
    assert idea[SLOT["energy_cherenkov"]] > 0 and idea[SLOT["energy_scintillation"]] > 0
    assert idea[SLOT["energy_ecal"]] == 0 and idea[SLOT["energy_hcal"]] == 0
    cld = bag(one_row(CLUSTER, {10: 1.0, 11: 2.0}), source("cld"))[0, 0]
    assert cld[SLOT["energy_ecal"]] > 0 and cld[SLOT["energy_hcal"]] > 0


def test_aliases_and_overrides_name_existing_columns():
    names = {name for source_name in TRACK_CLUSTER_SOURCES for elemtype in (TRACK, CLUSTER) for name in column_names(source_name, elemtype)}
    assert set(FEATURE_ALIASES) <= names
    for source_name, by_type in FEATURE_OVERRIDES.items():
        for elemtype, overrides in by_type.items():
            assert set(overrides) <= set(column_names(source_name, elemtype))


def test_constant_columns_keep_their_slot_and_arrive_as_zero_after_normalization():
    bag = FeatureBag(19)
    # CLD track time (column 15) is -1 in every 3.3.0 event: the feature bag maps it like any column...
    X = one_row(TRACK, {1: 5.0, 15: -1.0}, input_dim=19)
    assert bag(X, source("cld"))[0, 0, SLOT["time"]] / scale("cld", TRACK, 19) == pytest.approx(-1.0)
    # ...and input normalization, whose statistics have std 0 for it, zeroes it first.
    normalized = InputNormalization(19, {"cld": VERSIONS["cld"]})(X, source("cld"))
    assert bag(normalized, source("cld"))[0, 0, SLOT["time"]] == 0
    assert bag(normalized, source("cld"))[0, 0, SLOT["pt"]] != 0


def test_null_columns_and_element_type_are_left_out():
    bag = FeatureBag(17)
    # ColliderML track columns 14-16 are "Null".
    X = one_row(TRACK, {14: 1.0, 15: 1.0, 16: 1.0})
    assert torch.count_nonzero(bag(X, source("colliderml"))) == 0


def test_padding_rows_stay_zero_and_scaling_follows_present_features():
    bag = FeatureBag(17)
    X = torch.rand(2, 3, 17) + 0.1
    X[:, 0, 0] = TRACK
    X[:, 1, 0] = CLUSTER
    X[:, 2] = 0  # padding row
    out = bag(X, torch.tensor([SOURCE_IDS["clic"], SOURCE_IDS["colliderml"]]))

    assert out.shape == (2, 3, len(FEATURE_VOCABULARY))
    assert torch.count_nonzero(out[:, 2]) == 0
    for event, source_name in enumerate(["clic", "colliderml"]):
        for row, elemtype in enumerate([TRACK, CLUSTER]):
            present = [column for column in SOURCE_FEATURE_COLUMNS[source_name][elemtype] if column < 17]
            assert torch.count_nonzero(out[event, row]) == len(present)
            assert out[event, row].sum().item() == pytest.approx(X[event, row, present].sum().item() * scale(source_name, elemtype), rel=1e-5)


def test_missing_source_id_is_an_error():
    with pytest.raises(ValueError, match="source_id"):
        FeatureBag(17)(one_row(TRACK, {1: 1.0}), None)


def make_config(samples, input_normalization=True, input_dim=19):
    return MLPFConfig.model_validate(
        {
            "dataset": "cld",
            "data_dir": "/tmp",
            "input_dim": input_dim,
            "model": {
                "type": "attention",
                "feature_bag": True,
                "input_normalization": input_normalization,
                "attention": {"num_convs": 1, "num_heads": 2, "head_dim": 8, "attention_type": "simple"},
            },
            "conv_type": "attention",
            "test_dataset": {SAMPLES.get(name, name): {"version": VERSIONS.get(name, "1.0.0"), "splits": ["1"]} for name in samples},
        }
    )


def test_feature_bag_requires_input_normalization():
    with pytest.raises(ValueError, match="requires model.input_normalization"):
        make_config(["cld"], input_normalization=False)


def test_feature_bag_rejects_unsupported_samples():
    with pytest.raises(ValueError, match="feature_bag supports|input_normalization supports"):
        make_config(["cld", "cms_pf_ttbar"])
    with pytest.raises(ValueError, match="feature_bag supports|input_normalization supports"):
        make_config(["clic_edm_ttbar_hits"])


def test_feature_bag_reads_the_normalized_columns_of_a_batch_mixing_detectors():
    config = make_config(["cld", "colliderml", "idea"])
    model = MLPF(config)
    assert model.nn0[0].in_features == len(FEATURE_VOCABULARY)

    X = torch.rand(3, 16, 19) + 0.1
    X[..., 0] = torch.randint(1, 3, (3, 16)).float()
    X[1, :, 17:] = 0  # ColliderML has 17 columns, zero-padded to 19
    mask = torch.ones(3, 16, dtype=torch.bool)
    source_id = torch.tensor([SOURCE_IDS["cld"], SOURCE_IDS["colliderml"], SOURCE_IDS["idea"]])

    expected = model.feature_bag(model.input_normalization(X, source_id), source_id)
    seen = []
    handle = model.nn0.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
    preds_binary, preds_pid, preds_momentum, _ = model(X, mask, source_id=source_id)
    handle.remove()

    torch.testing.assert_close(seen[0], expected)
    assert preds_binary.shape == (3, 16, 2)
    assert preds_momentum.shape == (3, 16, 5)
    assert torch.isfinite(preds_pid).all() and torch.isfinite(preds_momentum).all()


@pytest.mark.parametrize("source_name", TRACK_CLUSTER_SOURCES)
def test_every_named_column_has_a_feature(source_name):
    for elemtype in (TRACK, CLUSTER):
        names = column_names(source_name, elemtype)
        named = {column for column, name in enumerate(names) if column > 0 and name != "Null"}
        assert set(SOURCE_FEATURE_COLUMNS[source_name][elemtype]) == named
        assert len(names) == len(X_FEATURES[source_name])
