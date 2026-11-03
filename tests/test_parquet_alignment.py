"""Regression coverage for native reconstructed-element/truth direction gates."""

import awkward as ak
import numpy as np
import pytest

from tests.validate_parquet import ParquetValidator, _has_pid_conflict


def test_grouped_pid_conflicts_match_original_check():
    rng = np.random.default_rng(123)
    for n in [0, 1, 10, 1000]:
        pn = rng.integers(0, 50, n)
        pid = rng.choice([0, 11, 13, 22, 130, 211], n)
        expected = any(len(np.unique(pid[(pn == p) & (pid > 0)])) > 1 for p in np.unique(pn[pn > 0]))
        assert _has_pid_conflict(pn, pid) == expected
    assert not _has_pid_conflict(np.array([1, 1, 0, 0]), np.array([211, 211, 11, 22]))


def make_validator(tmp_path, kind="track", angles=(0.0, 1.0), target_angles=None):
    x = np.zeros((len(angles), 17))
    y = np.zeros((len(angles), 14))
    x[:, 0] = 1 if kind == "track" else 2
    x[:, 1] = 2
    x[:, 3:5] = np.array([np.sin(angles), np.cos(angles)]).T
    target_angles = angles if target_angles is None else target_angles
    y[:, 0] = 211 if kind == "track" else 22
    y[:, 2] = 2
    y[:, 4:6] = np.array([np.sin(target_angles), np.cos(target_angles)]).T
    validator = ParquetValidator("unused.parquet", "colliderml", None, tmp_path, save_plots=False)
    validator.data = ak.Array({f"X_{kind}": [x.tolist()], f"ytarget_{kind}": [y.tolist()]})
    validator.nev_used = 1
    return validator


@pytest.mark.parametrize("kind", ["track", "cluster"])
@pytest.mark.parametrize("permuted", [False, True])
def test_native_alignment_detects_permutation(tmp_path, kind, permuted):
    v = make_validator(tmp_path, kind, target_angles=(1.0, 0.0) if permuted else None)
    v.gate_element_target_alignment()
    gate = next(g for g in v.gates if g.gate_id == ("track_target_direction" if kind == "track" else "cluster_target_direction"))
    assert gate.status == ("FAIL" if permuted else "PASS")
    assert v.alignment_metrics[kind]["all"]["fraction_aligned"] == (0 if permuted else 1)


def test_phi_wraparound(tmp_path):
    v = make_validator(tmp_path, angles=(np.pi - 0.01,), target_angles=(-np.pi + 0.01,))
    v.gate_element_target_alignment()
    assert v.gates[0].status == "PASS"
    assert v.alignment_metrics["track"]["all"]["median_delta_r"] == pytest.approx(0.02)


@pytest.mark.parametrize("kind", ["track", "cluster"])
def test_eta_misalignment_fails(tmp_path, kind):
    v = make_validator(tmp_path, kind)
    data = ak.to_list(v.data)
    for row in data[0][f"ytarget_{kind}"]:
        row[3] = 2.0
    v.data = ak.Array(data)
    v.gate_element_target_alignment()
    assert next(g for g in v.gates if g.gate_id == ("track_target_direction" if kind == "track" else "cluster_target_direction")).status == "FAIL"


def test_alignment_parquet_roundtrip(tmp_path):
    original = make_validator(tmp_path, target_angles=(1.0, 0.0))
    # Include both PF categories, but deliberately omit hit and jet tables.
    cluster = make_validator(tmp_path, "cluster")
    fields = {**{f: original.data[f] for f in original.data.fields}, **{f: cluster.data[f] for f in cluster.data.fields}}
    path = tmp_path / "input.parquet"
    ak.to_parquet(ak.zip(fields, depth_limit=1), path)
    v = ParquetValidator(path, "colliderml", None, tmp_path / "plots")
    v.plots_dir.mkdir()
    assert v.load()
    v.gate_element_target_alignment()
    assert not v.write_report(v.plots_dir / "validation_report.json")
    assert [g.gate_id for g in v.gates] == ["track_target_direction", "cluster_target_direction"]
    assert (tmp_path / "plots" / "validation_report.json").exists()


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), 0.0])
def test_invalid_direction_fails(tmp_path, invalid):
    v = make_validator(tmp_path, angles=(0.0,))
    data = ak.to_list(v.data)
    data[0]["X_track"][0][3:5] = [invalid, invalid]
    v.data = ak.Array(data)
    v.gate_element_target_alignment()
    assert v.gates[0].status == "FAIL"
    assert v.alignment_metrics["track"]["all"]["invalid"] == 1


def test_background_rows_excluded(tmp_path):
    v = make_validator(tmp_path, target_angles=(0.0, -1.0))
    data = ak.to_list(v.data)
    data[0]["ytarget_track"][1][0] = 0  # inclusive PN-only marks are not target reps
    data[0]["ytarget_track"][1][13] = 42
    v.data = ak.Array(data)
    v.gate_element_target_alignment()
    assert v.gates[0].status == "PASS"
    assert v.alignment_metrics["track"]["all"]["count"] == 1


def test_row_count_mismatch_fails(tmp_path):
    v = make_validator(tmp_path)
    data = ak.to_list(v.data)
    data[0]["ytarget_track"].pop()
    v.data = ak.Array(data)
    v.gate_element_target_alignment()
    assert v.gates[0].status == "FAIL"


def test_no_target_representatives_skips(tmp_path):
    v = make_validator(tmp_path)
    data = ak.to_list(v.data)
    for row in data[0]["ytarget_track"]:
        row[0] = 0
    v.data = ak.Array(data)
    v.gate_element_target_alignment()
    assert v.gates[0].status == "SKIP"
