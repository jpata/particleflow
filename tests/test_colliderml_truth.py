# Unit tests for ColliderML truth selection and attribution.
import awkward as ak
import numpy as np

from mlpf.data.colliderml.truth import DEFAULT_CALIBRATION, NEUTRINO_PDGS, compute_gen_tables
from mlpf.data.target_building import (
    map_charged_to_neutral,
    map_neutral_to_charged,
    map_pdgid_to_candid,
)


def _particles_event0():
    # 6 particles: two leaf primaries (pi+ id 1, pi0 id 2), one neutrino leaf (id 3),
    # two secondary gammas belonging to the pi0 (ids 4, 5), one invisible charged electron (id 6).
    ev = {
        "particle_id": [1, 2, 3, 4, 5, 6],
        "pdg_id": [211, 111, 12, 22, 22, 11],
        "charge": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "px": [10.0, 0.0, 1.0, 1.5, 1.5, 0.5],
        "py": [0.0, 3.0, 1.0, 0.0, 0.0, 0.0],
        "pz": [0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
        "energy": [10.0, 5.0, 2.0, 3.0, 2.0, 0.55],
        "vertex_primary": [1, 1, 1, 1, 1, 1],
        "primary": [True, True, True, False, False, True],
        "parent_id": [-1, -1, -1, 2, 2, -1],
    }
    return ak.Record({k: ak.Array(v) for k, v in ev.items()})


def _calo_hits_event0():
    # two hits, one with deposits from daughters 4 and 5 of the pi0 (pid 2)
    ev = {
        "detector": [10, 10],
        "total_energy": [3.0, 2.0],
        "x": [1.0, 2.0],
        "y": [0.0, 0.0],
        "z": [0.0, 0.0],
        "contrib_particle_ids": [[4], [5]],
        "contrib_energies": [[3.0], [2.0]],
        "contrib_times": [[0.0], [0.0]],
    }
    return ak.Record({k: ak.Array(v) for k, v in ev.items()})


def _tracks_event0():
    # one track majority-linked to the pi+ (particle id 1); flat per-event shape, as the real
    # converter hands one event's record to compute_gen_tables
    ev = {
        "d0": [0.01],
        "z0": [0.02],
        "phi": [0.3],
        "theta": [1.4],
        "qop": [0.1],
        "majority_particle_id": [1],
        "hit_ids": [[0, 1, 2]],
    }
    return ak.Record({k: ak.from_iter(v) for k, v in ev.items()})


def _tracker_hits_event0():
    # three tracker hits, all owned by the pi+ (particle id 1), matching the hit_ids of the
    # single track above -> (pi+, track0) fraction 1.0
    ev = {
        "particle_id": [1, 1, 1],
    }
    return ak.Record({k: ak.from_iter(v) for k, v in ev.items()})


def test_leaf_primary_selection_excludes_neutrinos_and_invisible():
    particles = _particles_event0()
    calo = _calo_hits_event0()
    tracks = _tracks_event0()
    tracker = _tracker_hits_event0()
    gen, gp_to_hit, gp_to_track, genref = compute_gen_tables(particles, calo, tracks, DEFAULT_CALIBRATION, tracker_ev=tracker)
    # pi+ (track) + pi0 (calo) survive; nu (leaf but invisible) and invisible e+ are excluded
    assert len(gen["PDG"]) == 2
    assert set(gen["PDG"].tolist()) == {211, 111}
    assert not any(abs(int(p)) in NEUTRINO_PDGS for p in gen["PDG"])
    # gamma deposits attributed to the pi0 leaf
    assert np.allclose(gen["energy"][gen["PDG"] == 111], 5.0, atol=1e-6)
    # exactly one track link for the pi+, with full hit share
    assert len(gp_to_track[0]) == 1
    assert np.allclose(gp_to_track[2], 1.0)
    # two calo contributions both pointing to the pi0's local index
    assert len(gp_to_hit[0]) == 2
    gp_idx = int(gp_to_track[0][0])
    hit_gp = set(int(g) for g in gp_to_hit[0])
    assert hit_gp == {int(g) for g in [0, 1] if g != gp_idx}
    # gp_to_track column = max hit share of the leaf's tracks (0 for untracked neutrals)
    assert np.isclose(gen["gp_to_track"][gen["PDG"] == 211][0], 1.0)
    assert np.isclose(gen["gp_to_track"][gen["PDG"] == 111][0], 0.0)


def test_neutrino_excluded_even_if_depositing():
    # neutrino cannot be a target even when it has calo deposits (mock case)
    calo = ak.Record(
        {
            "detector": ak.Array([10, 10]),
            "total_energy": ak.Array([1.0, 1.0]),
            "x": ak.Array([1.0, 2.0]),
            "y": ak.Array([0.0, 0.0]),
            "z": ak.Array([0.0, 0.0]),
            "contrib_particle_ids": ak.Array([[3], [3]]),
            "contrib_energies": ak.Array([[1.0], [1.0]]),
            "contrib_times": ak.Array([[0.0], [0.0]]),
        }
    )
    gen, _, _, genref = compute_gen_tables(_particles_event0(), calo, _tracks_event0(), DEFAULT_CALIBRATION, tracker_ev=_tracker_hits_event0())
    assert not any(abs(int(p)) in NEUTRINO_PDGS for p in gen["PDG"])
    assert not any(abs(int(p)) in NEUTRINO_PDGS for p in genref["PDG"])


def test_track_visibility_requires_min_hit_share():
    # Second track: majority id says pi+ (pid 1) but the invisible electron (pid 6) owns only
    # 1 of its 10 hits -> 0.1 < TRACK_HIT_FRACTION_MIN, so the electron must not become a target.
    tracks = ak.Record(
        {
            "d0": ak.from_iter([0.01, 0.01]),
            "z0": ak.from_iter([0.02, 0.02]),
            "phi": ak.from_iter([0.3, 0.3]),
            "theta": ak.from_iter([1.4, 1.4]),
            "qop": ak.from_iter([0.1, 0.1]),
            "majority_particle_id": ak.from_iter([1, 1]),
            "hit_ids": ak.from_iter([[0, 1, 2], list(range(3, 13))]),
        }
    )
    tracker = ak.Record({"particle_id": ak.from_iter([1, 1, 1] + [1] * 9 + [6])})
    gen, gp_to_hit, gp_to_track, _genref = compute_gen_tables(
        _particles_event0(), _calo_hits_event0(), tracks, DEFAULT_CALIBRATION, tracker_ev=tracker
    )
    # electron (pid 6) stays invisible: no calo deposit and no track with >= 20% hit share
    assert set(gen["PDG"].tolist()) == {211, 111}
    # the pi+ keeps both track links, with fractions 1.0 and 0.9; the 0.1 electron link is dropped
    assert len(gp_to_track[0]) == 2
    np.testing.assert_allclose(sorted(gp_to_track[2]), [0.9, 1.0], atol=1e-6)
    # gp_to_track target column carries the max fraction
    assert np.isclose(gen["gp_to_track"][gen["PDG"] == 211][0], 1.0)


def _particles_event0_extended():
    # Base event plus: pid 7, a 100 GeV photon with only a 0.375 GeV calibrated tail deposit
    # (fraction 0.00375 < 0.10, neutral so the absolute term does not apply); pid 8, a 100 GeV
    # charged pion with a 0.6 GeV calibrated deposit (fraction 0.006 < 0.10 but > MIP threshold,
    # charged). Used by the visibility and genref tests below.
    ev = {
        "particle_id": [1, 2, 3, 4, 5, 6, 7, 8],
        "pdg_id": [211, 111, 12, 22, 22, 11, 22, 211],
        "charge": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        "px": [10.0, 0.0, 1.0, 1.5, 1.5, 0.5, 100.0, 100.0],
        "py": [0.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 10.0],
        "pz": [0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "energy": [10.0, 5.0, 2.0, 3.0, 2.0, 0.55, 100.0, 100.0],
        "vertex_primary": [1, 1, 1, 1, 1, 1, 1, 1],
        "primary": [True, True, True, False, False, True, True, True],
        "parent_id": [-1, -1, -1, 2, 2, -1, -1, -1],
    }
    return ak.Record({k: ak.Array(v) for k, v in ev.items()})


def _calo_hits_event0_extended():
    # region 10 (37.5x): pi0 daughters 3.0+2.0, photon tail 0.01, mip pion 0.016
    return ak.Record(
        {
            "detector": ak.Array([10, 10, 10, 10]),
            "total_energy": ak.Array([3.0, 2.0, 0.01, 0.016]),
            "x": ak.Array([1.0, 2.0, 3.0, 4.0]),
            "y": ak.Array([0.0, 0.0, 0.0, 0.0]),
            "z": ak.Array([0.0, 0.0, 0.0, 0.0]),
            "contrib_particle_ids": ak.Array([[4], [5], [7], [8]]),
            "contrib_energies": ak.Array([[3.0], [2.0], [0.01], [0.016]]),
            "contrib_times": ak.Array([[0.0], [0.0], [0.0], [0.0]]),
        }
    )


def test_calo_visibility_uses_shared_fraction_and_mip_terms():
    # pid 7 (photon, tail-only deposit) -> NOT a target; pid 8 (charged pion, sub-fraction
    # deposit above the MIP threshold) -> target via the MIP term.
    particles = _particles_event0_extended()
    calo = _calo_hits_event0_extended()
    gen, gp_to_hit, gp_to_track, _genref = compute_gen_tables(
        particles, calo, _tracks_event0(), DEFAULT_CALIBRATION, tracker_ev=_tracker_hits_event0()
    )
    # pi+ (track) + pi0 (fraction) + pion 8 (MIP absolute term) survive; photon 7 (tail-only,
    # neutral) and the invisible e+ are out
    assert sorted(gen["PDG"].tolist()) == [111, 211, 211]
    # the MIP-admitted pion carries its low-fraction deposit alongside it
    pion_row = gen["energy"] == 100.0
    assert np.isclose(gen["gp_to_cluster"][pion_row][0], 37.5 * 0.016, atol=1e-4)


def test_genref_measurable_superset_of_visible():
    # The gen reference ("measurable truth") admits any leaf primary with a registration
    # (calibrated deposit or track hit share) regardless of visibility: the tail-only photon
    # (pid 7) fails the visibility mask but enters genref; the invisible electron (no deposit,
    # no track) and the neutrino enter neither; every target is contained in genref.
    gen, _, _, genref = compute_gen_tables(
        _particles_event0_extended(), _calo_hits_event0_extended(), _tracks_event0(), DEFAULT_CALIBRATION, tracker_ev=_tracker_hits_event0()
    )
    tgt_pdg = sorted(int(abs(p)) for p in gen["PDG"])
    ref_pdg = sorted(int(abs(p)) for p in genref["PDG"])
    assert tgt_pdg == [111, 211, 211]  # pi0 (fraction), pi+ (track), 100 GeV pion (MIP term)
    assert ref_pdg == [22, 111, 211, 211]  # + the tail-only photon; no neutrino, no invisible e
    # the photon's genref gp_to_cluster carries its calibrated deposit; it owns no track share
    ph = np.asarray(genref["PDG"]) == 22
    assert np.isclose(genref["gp_to_cluster"][ph][0], 37.5 * 0.01, atol=1e-4)
    assert np.isclose(genref["gp_to_track"][ph][0], 0.0)
    # pi0 keeps its full attributed deposit; the tracked pi+ its full hit share
    assert np.isclose(genref["gp_to_cluster"][np.asarray(genref["PDG"]) == 111][0], 37.5 * 5.0, atol=1e-3)
    assert np.isclose(genref["gp_to_track"][np.asarray(genref["PDG"]) == 211][0], 1.0)
    # same status conventions as the target features
    assert np.all(genref["generatorStatus"] == 1.0)
    assert np.all(genref["simulatorStatus"] == 0x01000000)


def test_class_forcing_helpers():
    assert map_pdgid_to_candid(22, 0.0) == 22
    assert map_pdgid_to_candid(11, 1.0) == 11
    assert map_pdgid_to_candid(13, -1.0) == 13
    assert map_pdgid_to_candid(211, 1.0) == 211
    assert map_pdgid_to_candid(111, 0.0) == 130
    assert map_charged_to_neutral(211) == 130
    assert map_charged_to_neutral(11) == 22
    assert map_neutral_to_charged(22) == 211
    assert map_neutral_to_charged(130) == 211
