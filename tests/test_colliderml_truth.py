# Unit tests for ColliderML truth selection and attribution.
import awkward as ak
import numpy as np
import pytest

from mlpf.data.colliderml.truth import DEFAULT_CALIBRATION, NEUTRINO_PDGS, calibration_factors, compute_gen_tables
from mlpf.data.target_building import (
    map_charged_to_neutral,
    map_neutral_to_charged,
    map_pdgid_to_candid,
)


def _particles_event0():
    # 6 particles: two leaf primaries (pi+ id 1, pi0 id 2), one neutrino leaf (id 3),
    # two Geant4 decay photons of the pi0 (ids 4, 5; they replace it as leaves), one invisible
    # charged electron (id 6).
    ev = {
        "particle_id": [1, 2, 3, 4, 5, 6],
        "pdg_id": [211, 111, 12, 22, 22, 11],
        "charge": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "px": [10.0, 0.0, 1.0, 0.0, 0.0, 0.5],
        "py": [0.0, 3.0, 1.0, 1.8, 1.2, 0.0],
        "pz": [0.0, 4.0, 0.0, 2.4, 1.6, 0.0],
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
    # pi+ (track) + the pi0's two photons (calo) survive; nu (leaf but invisible) and
    # invisible e+ are excluded
    assert len(gen["PDG"]) == 3
    assert sorted(gen["PDG"].tolist()) == [22, 22, 211]
    assert not any(abs(int(p)) in NEUTRINO_PDGS for p in gen["PDG"])
    # each photon is a target with its own kinematics and its own deposit
    photons = gen["PDG"] == 22
    np.testing.assert_allclose(sorted(gen["energy"][photons]), [2.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(sorted(gen["gp_to_cluster"][photons]), [37.5 * 2.0, 37.5 * 3.0], rtol=1e-6)
    # exactly one track link for the pi+, with full hit share
    assert len(gp_to_track[0]) == 1
    assert np.allclose(gp_to_track[2], 1.0)
    # two calo contributions, one per photon row (the zero-weight tracker links gp_to_hit also
    # carries are excluded here by weight)
    assert int(np.sum(np.asarray(gp_to_hit[2]) > 0)) == 2
    hit_gp = sorted(int(g) for g in np.asarray(gp_to_hit[0])[np.asarray(gp_to_hit[2]) > 0])  # calo links only
    assert hit_gp == sorted(np.nonzero(photons)[0].tolist())
    # gp_to_track column = max hit share of the leaf's tracks (0 for untracked neutrals)
    assert np.isclose(gen["gp_to_track"][gen["PDG"] == 211][0], 1.0)
    assert np.allclose(gen["gp_to_track"][photons], 0.0)


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
    assert set(gen["PDG"].tolist()) == {211, 22}
    # the pi+ keeps both track links, with fractions 1.0 and 0.9; the 0.1 electron link is dropped
    assert len(gp_to_track[0]) == 2
    np.testing.assert_allclose(sorted(gp_to_track[2]), [0.9, 1.0], atol=1e-6)
    # gp_to_track target column carries the max fraction
    assert np.isclose(gen["gp_to_track"][gen["PDG"] == 211][0], 1.0)


def test_tracker_links_are_zero_weight_and_offset_past_calo():
    # tracker hits join the hit adjacency at zero weight, offset past the calo hits
    gen, gp_to_hit, _, _ = compute_gen_tables(
        _particles_event0(), _calo_hits_event0(), _tracks_event0(), DEFAULT_CALIBRATION, tracker_ev=_tracker_hits_event0()
    )
    n_calo = 2
    sel = np.asarray(gp_to_hit[1]) >= n_calo
    assert np.asarray(gp_to_hit[2])[sel].tolist() == [0.0, 0.0, 0.0]
    assert sorted((np.asarray(gp_to_hit[1])[sel] - n_calo).tolist()) == [0, 1, 2]
    # the pi+ owns these hits (kept gen row with PDG 211)
    pi_row = int(np.argmax(np.asarray(gen["PDG"]) == 211))
    assert set(np.asarray(gp_to_hit[0])[sel].tolist()) == {pi_row}


def test_tracker_links_skip_unknown_and_unkept_owners():
    # hit owner id absent from the table and the neutrino leaf both yield no link
    tracker = ak.Record({"particle_id": ak.Array([1, 999, 3])})
    gen, gp_to_hit, _, _ = compute_gen_tables(_particles_event0(), _calo_hits_event0(), _tracks_event0(), DEFAULT_CALIBRATION, tracker_ev=tracker)
    sel = np.asarray(gp_to_hit[1]) >= 2
    # only the pi+-owned hit (tracker row 0) survives, at extended index 2
    assert sel.sum() == 1
    assert int(np.asarray(gp_to_hit[1])[sel][0]) == 2


def _particles_event0_extended():
    # Base event plus: pid 7, a 100 GeV photon with only a 0.375 GeV calibrated tail deposit
    # (fraction 0.00375 < 0.10, neutral so the absolute term does not apply); pid 8, a 100 GeV
    # charged pion with a 0.6 GeV calibrated deposit (fraction 0.006 < 0.10 but > MIP threshold,
    # charged). Used by the visibility and genref tests below.
    ev = {
        "particle_id": [1, 2, 3, 4, 5, 6, 7, 8],
        "pdg_id": [211, 111, 12, 22, 22, 11, 22, 211],
        "charge": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        "px": [10.0, 0.0, 1.0, 0.0, 0.0, 0.5, 100.0, 100.0],
        "py": [0.0, 3.0, 1.0, 1.8, 1.2, 0.0, 0.0, 10.0],
        "pz": [0.0, 4.0, 0.0, 2.4, 1.6, 0.0, 0.0, 0.0],
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
    # pi+ (track) + the pi0's photons (fraction) + pion 8 (MIP absolute term) survive; photon 7
    # (tail-only, neutral) and the invisible e+ are out
    assert sorted(gen["PDG"].tolist()) == [22, 22, 211, 211]
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
    assert tgt_pdg == [22, 22, 211, 211]  # pi0 photons (fraction), pi+ (track), 100 GeV pion (MIP term)
    assert ref_pdg == [22, 22, 22, 211, 211]  # + the tail-only photon; no neutrino, no invisible e
    # the tail-only photon's genref gp_to_cluster carries its calibrated deposit; no track share
    ph = (np.asarray(genref["PDG"]) == 22) & (np.asarray(genref["energy"]) == 100.0)
    assert np.isclose(genref["gp_to_cluster"][ph][0], 37.5 * 0.01, atol=1e-4)
    assert np.isclose(genref["gp_to_track"][ph][0], 0.0)
    # the pi0's photons keep their own attributed deposits; the tracked pi+ its full hit share
    pi0_photons = (np.asarray(genref["PDG"]) == 22) & (np.asarray(genref["energy"]) < 10.0)
    np.testing.assert_allclose(sorted(genref["gp_to_cluster"][pi0_photons]), [37.5 * 2.0, 37.5 * 3.0], rtol=1e-6)
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


def test_calibration_factors_reject_unknown_regions():
    k = calibration_factors(np.array([9, 12, 14]), DEFAULT_CALIBRATION)
    assert k.dtype == np.float32 and k.tolist() == [np.float32(DEFAULT_CALIBRATION[d]) for d in (9, 12, 14)]
    for bad in ([3], [15], [-1]):
        with pytest.raises(ValueError, match="no calibration factor"):
            calibration_factors(np.array([9] + bad), DEFAULT_CALIBRATION)


def _calo_hits(contrib_ids, raw):
    # one region-10 (37.5x) hit per contribution list; raw deposits per contribution
    n = len(contrib_ids)
    return ak.Record(
        {
            "detector": ak.Array([10] * n),
            "total_energy": ak.Array([float(sum(r)) for r in raw]),
            "x": ak.Array([float(i) for i in range(n)]),
            "y": ak.Array([0.0] * n),
            "z": ak.Array([0.0] * n),
            "contrib_particle_ids": ak.Array(contrib_ids),
            "contrib_energies": ak.Array(raw),
            "contrib_times": ak.Array([[0.0] * len(c) for c in contrib_ids]),
        }
    )


def _particles(rows):
    # rows: (particle_id, pdg_id, charge, px, py, pz, energy, vertex_primary, primary, parent_id)
    keys = ["particle_id", "pdg_id", "charge", "px", "py", "pz", "energy", "vertex_primary", "primary", "parent_id"]
    return ak.Record({k: ak.Array([r[i] for r in rows]) for i, k in enumerate(keys)})


def test_split_pi0_photons_replace_the_pi0_in_target_and_genref():
    # pileup vertex (vertex_primary 3): the photons inherit ispu from their own rows
    particles = _particles(
        [
            (1, 211, 1.0, 10.0, 0.0, 0.0, 10.0, 1, True, -1),
            (2, 111, 0.0, 0.0, 3.0, 4.0, 5.0, 3, True, -1),
            (4, 22, 0.0, 0.0, 1.8, 2.4, 3.0, 3, False, 2),
            (5, 22, 0.0, 0.0, 1.2, 1.6, 2.0, 3, False, 2),
        ]
    )
    gen, _, _, genref = compute_gen_tables(
        particles, _calo_hits([[4], [5]], [[3.0], [2.0]]), _tracks_event0(), DEFAULT_CALIBRATION, tracker_ev=_tracker_hits_event0()
    )
    for table in (gen, genref):
        assert 111 not in np.abs(table["PDG"]).tolist()
        photons = table["PDG"] == 22
        assert photons.sum() == 2
        np.testing.assert_allclose(sorted(table["pt"][photons]), [np.hypot(0.0, 1.2), np.hypot(0.0, 1.8)], rtol=1e-6)
        assert np.all(table["ispu"][photons] == 1.0)
        assert np.all(table["generatorStatus"][photons] == 1.0)


def test_split_pi0_dalitz_electrons_become_leaves():
    particles = _particles(
        [
            (1, 211, 1.0, 10.0, 0.0, 0.0, 10.0, 1, True, -1),
            (2, 111, 0.0, 0.0, 3.0, 4.0, 5.0, 1, True, -1),
            (4, 22, 0.0, 0.0, 1.8, 2.4, 3.0, 1, False, 2),
            (5, 11, -1.0, 0.0, 0.6, 0.8, 1.0, 1, False, 2),
            (6, -11, 1.0, 0.0, 0.6, 0.8, 1.0, 1, False, 2),
        ]
    )
    gen, _, _, _ = compute_gen_tables(
        particles, _calo_hits([[4], [5], [6]], [[3.0], [1.0], [1.0]]), _tracks_event0(), DEFAULT_CALIBRATION, tracker_ev=_tracker_hits_event0()
    )
    assert sorted(gen["PDG"].tolist()) == [-11, 11, 22, 211]


def test_deposit_credited_to_split_pi0_goes_to_its_most_energetic_product():
    # only one decay photon recorded (id 4); the release credits the unrecorded soft photon's
    # deposit directly to the pi0 id (2): it must land on photon 4, not be lost
    particles = _particles(
        [
            (1, 211, 1.0, 10.0, 0.0, 0.0, 10.0, 1, True, -1),
            (2, 111, 0.0, 0.0, 3.0, 4.0, 5.0, 1, True, -1),
            (4, 22, 0.0, 0.0, 2.88, 3.84, 4.8, 1, False, 2),
        ]
    )
    gen, gp_to_hit, _, _ = compute_gen_tables(
        particles, _calo_hits([[4], [2]], [[3.0], [0.1]]), _tracks_event0(), DEFAULT_CALIBRATION, tracker_ev=_tracker_hits_event0()
    )
    photon = gen["PDG"] == 22
    assert photon.sum() == 1
    assert np.isclose(gen["gp_to_cluster"][photon][0], 37.5 * 3.1, rtol=1e-6)
    assert int(np.sum(np.asarray(gp_to_hit[2]) > 0)) == 2


def test_conversion_tracks_of_a_split_photon_link_to_the_photon():
    # photon 4 converts (e- 7, e+ 8, Geant4 children of the photon); their tracker hits form a
    # track whose hits all walk to photon 4
    particles = _particles(
        [
            (1, 211, 1.0, 10.0, 0.0, 0.0, 10.0, 1, True, -1),
            (2, 111, 0.0, 0.0, 3.0, 4.0, 5.0, 1, True, -1),
            (4, 22, 0.0, 0.0, 1.8, 2.4, 3.0, 1, False, 2),
            (5, 22, 0.0, 0.0, 1.2, 1.6, 2.0, 1, False, 2),
            (7, 11, -1.0, 0.0, 0.9, 1.2, 1.5, 1, False, 4),
            (8, -11, 1.0, 0.0, 0.9, 1.2, 1.5, 1, False, 4),
        ]
    )
    tracks = ak.Record({"majority_particle_id": ak.from_iter([1, 7]), "hit_ids": ak.from_iter([[0, 1, 2], [3, 4, 5, 6]])})
    tracker = ak.Record({"particle_id": ak.from_iter([1, 1, 1, 7, 7, 8, 8])})
    gen, _, gp_to_track, _ = compute_gen_tables(
        particles, _calo_hits([[7, 8], [5]], [[1.0, 1.0], [2.0]]), tracks, DEFAULT_CALIBRATION, tracker_ev=tracker
    )
    ph4 = np.nonzero((gen["PDG"] == 22) & (gen["energy"] == 3.0))[0]
    assert len(ph4) == 1
    assert np.isclose(gen["gp_to_track"][ph4[0]], 1.0)
    assert np.isclose(gen["gp_to_cluster"][ph4[0]], 37.5 * 2.0, rtol=1e-6)
    links = {(int(g), int(t)) for g, t in zip(gp_to_track[0], gp_to_track[1])}
    assert (int(ph4[0]), 1) in links


def test_pi0_without_recorded_products_stays_a_leaf():
    particles = _particles(
        [
            (1, 211, 1.0, 10.0, 0.0, 0.0, 10.0, 1, True, -1),
            (2, 111, 0.0, 0.0, 3.0, 4.0, 5.0, 1, True, -1),
        ]
    )
    gen, _, _, _ = compute_gen_tables(particles, _calo_hits([[2]], [[3.0]]), _tracks_event0(), DEFAULT_CALIBRATION, tracker_ev=_tracker_hits_event0())
    assert sorted(gen["PDG"].tolist()) == [111, 211]


def test_generator_decayed_pi0_is_unchanged():
    # the pu200 pileup case: the generator decayed the pi0, so the photons are primary leaves
    # and the pi0 is not a leaf; nothing is split or redirected
    particles = _particles(
        [
            (1, 211, 1.0, 10.0, 0.0, 0.0, 10.0, 1, True, -1),
            (2, 111, 0.0, 0.0, 3.0, 4.0, 5.0, 2, True, -1),
            (4, 22, 0.0, 0.0, 1.8, 2.4, 3.0, 2, True, 2),
            (5, 22, 0.0, 0.0, 1.2, 1.6, 2.0, 2, True, 2),
            (9, 11, -1.0, 0.0, 0.9, 1.2, 1.5, 2, False, 4),
        ]
    )
    gen, _, _, _ = compute_gen_tables(
        particles, _calo_hits([[9], [5]], [[3.0], [2.0]]), _tracks_event0(), DEFAULT_CALIBRATION, tracker_ev=_tracker_hits_event0()
    )
    assert sorted(gen["PDG"].tolist()) == [22, 22, 211]
    ph4 = (gen["PDG"] == 22) & (gen["energy"] == 3.0)
    assert np.isclose(gen["gp_to_cluster"][ph4][0], 37.5 * 3.0, rtol=1e-6)
