# Deterministic synthetic ColliderML-shape fixture writer (no network).
#
# Writes 2 source-style shards (~30 events each) in the exact Release 1 schema into
# <out>/ttbar_pu0_{particles,tracks,calo_hits,tracker_hits}/data/ttbar_pu0_*/train-0000{0,1}-of-00002.parquet,
# plus metadata.json files mimicking the download layout, so the real converter and TFDS
# builder can run against it unchanged.
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _event_arrays(rng: np.random.Generator, iev: int):
    """One synthetic ttbar-like event: ~50 leaf primaries, some invisible, some neutrinos.
    The truth structure guarantees: every selected target is visible (track or calo), the
    meson->gamma parent chains exercise the attribution walk, and there are merged-pair
    situations for the merge/drop accounting."""
    # leaf primaries
    n_ch = 30
    n_pi0 = 10
    n_pho = 8
    n_n = 5
    n_nu = 6
    n_e = 2
    n_mu = 2
    pid_next = 1
    leaves = []  # (id, pdg, pt, eta, phi)
    for i in range(n_ch):
        leaves.append((pid_next, 211, rng.uniform(1, 50), rng.uniform(-2.5, 2.5), rng.uniform(-np.pi, np.pi)))
        pid_next += 1
    for i in range(n_pi0):
        leaves.append((pid_next, 111, rng.uniform(1, 30), rng.uniform(-2.5, 2.5), rng.uniform(-np.pi, np.pi)))
        pid_next += 1
    for i in range(n_pho):
        leaves.append((pid_next, 22, rng.uniform(2, 40), rng.uniform(-2.5, 2.5), rng.uniform(-np.pi, np.pi)))
        pid_next += 1
    for i in range(n_n):
        leaves.append((pid_next, 2112, rng.uniform(1, 30), rng.uniform(-2.5, 2.5), rng.uniform(-np.pi, np.pi)))
        pid_next += 1
    for i in range(n_nu):
        leaves.append((pid_next, [12, 14, 16][i % 3], rng.uniform(1, 40), rng.uniform(-2.5, 2.5), rng.uniform(-np.pi, np.pi)))
        pid_next += 1
    for i in range(n_e):
        leaves.append((pid_next, 11, rng.uniform(2, 40), rng.uniform(-2.5, 2.5), rng.uniform(-np.pi, np.pi)))
        pid_next += 1
    for i in range(n_mu):
        leaves.append((pid_next, 13, rng.uniform(2, 40), rng.uniform(-2.5, 2.5), rng.uniform(-np.pi, np.pi)))
        pid_next += 1

    # secondaries: two gammas for each pi0 (parent chains)
    secondaries = []
    for lid, pdg, ptv, eta, phi in leaves:
        if pdg != 111:
            continue
        e = ptv * np.cosh(eta)
        for k in range(2):
            g_id = pid_next
            pid_next += 1
            secondaries.append((g_id, 22, 0.5 * e, eta + rng.normal(0, 0.05), phi + rng.normal(0, 0.05), lid))

    # build particles table
    ids = [leaf[0] for leaf in leaves] + [s[0] for s in secondaries]
    pdgs = [leaf[1] for leaf in leaves] + [s[1] for s in secondaries]
    primary = [True] * len(leaves) + [False] * len(secondaries)
    parent_id = [-1] * len(leaves) + [s[5] for s in secondaries]
    charge_map = {211: 1, 111: 0, 22: 0, 2112: 0, 11: 1, 13: 1, 12: 0, 14: 0, 16: 0}
    charges = [float(charge_map[p]) for p in pdgs]
    pts = [leaf[2] for leaf in leaves] + [s[2] for s in secondaries]
    etas = [leaf[3] for leaf in leaves] + [s[3] for s in secondaries]
    phis = [leaf[4] for leaf in leaves] + [s[4] for s in secondaries]
    px = [p * np.cos(ph) for p, ph in zip(pts, phis)]
    py = [p * np.sin(ph) for p, ph in zip(pts, phis)]
    pz = [p * np.sinh(e_) for p, e_ in zip(pts, etas)]
    mass_map = {211: 0.1396, 111: 0.135, 22: 0.0, 2112: 0.9396, 11: 0.0005, 13: 0.1057, 12: 0.0, 14: 0.0, 16: 0.0}
    masses = [mass_map[p] for p in pdgs]
    energies = [np.sqrt(px[i] ** 2 + py[i] ** 2 + pz[i] ** 2 + masses[i] ** 2) for i in range(len(ids))]

    particles_row = {
        "event_id": np.uint32(iev),
        "particle_id": ids,
        "pdg_id": pdgs,
        "mass": masses,
        "energy": energies,
        "charge": charges,
        "vx": [0.0] * len(ids),
        "vy": [0.0] * len(ids),
        "vz": [0.0] * len(ids),
        "time": [0.0] * len(ids),
        "px": px,
        "py": py,
        "pz": pz,
        "perigee_d0": [0.0] * len(ids),
        "perigee_z0": [0.0] * len(ids),
        "vertex_primary": [1] * len(ids),
        "parent_id": parent_id,
        "primary": primary,
    }

    # tracks: one for each charged leaf with pt > 0.5 (except muons get one too); each track
    # owns 4 tracker hits on widening layers, wired through hit_ids so the converter's
    # hit-fraction gp->track links see a realistic wiring. Every 5th track donates its last
    # hit to the previous track's owner, so fractions below 1.0 are exercised too.
    track_rows = {k: [] for k in ["d0", "z0", "phi", "theta", "qop", "majority_particle_id", "hit_ids", "track_id"]}
    tracker_rows = {
        k: [] for k in ["x", "y", "z", "true_x", "true_y", "true_z", "time", "particle_id", "detector", "volume_id", "layer_id", "surface_id"]
    }
    tid = 0
    prev_lid = None
    for leaf in leaves:
        lid, pdg, ptv, eta, phi = leaf
        if charge_map[pdg] != 0 and ptv > 0.5:
            theta = 2 * np.arctan(np.exp(-eta))
            p = ptv / np.sin(theta)
            qop = charge_map[pdg] / p
            track_rows["d0"].append(float(rng.uniform(0, 0.05)))
            track_rows["z0"].append(float(rng.uniform(0, 0.05)))
            track_rows["phi"].append(float(phi + rng.normal(0, 0.005)))
            track_rows["theta"].append(float(theta))
            track_rows["qop"].append(float(qop))
            track_rows["majority_particle_id"].append(lid)
            hit_ids = []
            for ih, r in enumerate([50.0, 150.0, 300.0, 600.0]):
                owner = prev_lid if (tid % 5 == 4 and ih == 3 and prev_lid is not None) else lid
                x = float(r * np.cos(phi))
                y = float(r * np.sin(phi))
                z = float(r * np.sinh(eta))
                for k, v in [("x", x), ("y", y), ("z", z), ("true_x", x), ("true_y", y), ("true_z", z), ("time", 0.0)]:
                    tracker_rows[k].append(v)
                tracker_rows["particle_id"].append(int(owner))
                tracker_rows["detector"].append(0)
                tracker_rows["volume_id"].append(1)
                tracker_rows["layer_id"].append(ih)
                tracker_rows["surface_id"].append(tid)
                hit_ids.append(len(tracker_rows["x"]) - 1)
            track_rows["hit_ids"].append(hit_ids)
            track_rows["track_id"].append(tid)
            prev_lid = lid
            tid += 1
    tracks_row = {"event_id": np.uint32(iev), **{k: track_rows[k] for k in track_rows}}
    tracker_row = {"event_id": np.uint32(iev), **tracker_rows}

    # calo hits: ECAL barrel grid cloud per neutral EM (pi0->photons, photons); HCAL for
    # neutrons; small deposits in raw (uncalibrated) units
    calo = {"detector": [], "total_energy": [], "x": [], "y": [], "z": [], "contrib_particle_ids": [], "contrib_energies": [], "contrib_times": []}

    def _add_hit_cloud(center_xyz, sign_id, e_total, region):
        for _ in range(int(rng.integers(3, 8))):
            dx = rng.normal(0, 5 if region == 10 else 30)
            dy = rng.normal(0, 5 if region == 10 else 30)
            dz = rng.normal(0, 5 if region == 10 else 30)
            e = e_total / 5.0 * float(rng.uniform(0.5, 1.5))
            calo["detector"].append(region)
            calo["total_energy"].append(e)
            calo["x"].append(center_xyz[0] + dx)
            calo["y"].append(center_xyz[1] + dy)
            calo["z"].append(center_xyz[2] + dz)
            calo["contrib_particle_ids"].append([sign_id])
            calo["contrib_energies"].append([e])
            calo["contrib_times"].append([0.0])

    def _xyz_from_eta_phi(eta, phi, r_cyl):
        z = r_cyl * np.sinh(eta)
        return (r_cyl * np.cos(phi) * np.cosh(eta) / max(np.cosh(eta), 1e-9), r_cyl * np.sin(phi), z)

    for s in secondaries:
        sid, _, e, eta, phi, _ = s
        xyz = _xyz_from_eta_phi(eta, phi, 1400.0)
        _add_hit_cloud(xyz, sid, e / 30.0, 10)  # raw ~ e/30
    for leaf in leaves:
        lid, pdg, ptv, eta, phi = leaf
        if pdg == 22:
            e = ptv * np.cosh(eta)
            _add_hit_cloud(_xyz_from_eta_phi(eta, phi, 1400.0), lid, e / 30.0, 10)
        elif pdg == 2112:
            e = ptv * np.cosh(eta)
            _add_hit_cloud(_xyz_from_eta_phi(eta, phi, 2500.0), lid, e / 100.0, 13)
        elif pdg in (11,):
            e = ptv * np.cosh(eta)
            _add_hit_cloud(_xyz_from_eta_phi(eta, phi, 1400.0), lid, e / 30.0, 10)

    calo_row = {"event_id": np.uint32(iev), **{k: calo[k] for k in calo}}
    return particles_row, tracks_row, calo_row, tracker_row


def _list(schema_elem, values):
    return pa.array([values], type=pa.list_(schema_elem))


def write_shard(out_root: Path, shard_idx: int, n_shards: int, rng: np.random.Generator, events: list):
    """Write one shard (list of per-event row dicts) into the three tables."""
    evs = events
    p_tab = {
        "event_id": pa.array([r["event_id"] for r in [e[0] for e in evs]], type=pa.uint32()),
        "particle_id": pa.array([r["particle_id"] for r in [e[0] for e in evs]], type=pa.list_(pa.uint64())),
        "pdg_id": pa.array([r["pdg_id"] for r in [e[0] for e in evs]], type=pa.list_(pa.int64())),
        "mass": pa.array([r["mass"] for r in [e[0] for e in evs]], type=pa.list_(pa.float32())),
        "energy": pa.array([r["energy"] for r in [e[0] for e in evs]], type=pa.list_(pa.float32())),
        "charge": pa.array([r["charge"] for r in [e[0] for e in evs]], type=pa.list_(pa.float32())),
        "vx": pa.array([r["vx"] for r in [e[0] for e in evs]], type=pa.list_(pa.float32())),
        "vy": pa.array([r["vy"] for r in [e[0] for e in evs]], type=pa.list_(pa.float32())),
        "vz": pa.array([r["vz"] for r in [e[0] for e in evs]], type=pa.list_(pa.float32())),
        "time": pa.array([r["time"] for r in [e[0] for e in evs]], type=pa.list_(pa.float32())),
        "px": pa.array([r["px"] for r in [e[0] for e in evs]], type=pa.list_(pa.float32())),
        "py": pa.array([r["py"] for r in [e[0] for e in evs]], type=pa.list_(pa.float32())),
        "pz": pa.array([r["pz"] for r in [e[0] for e in evs]], type=pa.list_(pa.float32())),
        "perigee_d0": pa.array([r["perigee_d0"] for r in [e[0] for e in evs]], type=pa.list_(pa.float32())),
        "perigee_z0": pa.array([r["perigee_z0"] for r in [e[0] for e in evs]], type=pa.list_(pa.float32())),
        "vertex_primary": pa.array([r["vertex_primary"] for r in [e[0] for e in evs]], type=pa.list_(pa.uint16())),
        "parent_id": pa.array([r["parent_id"] for r in [e[0] for e in evs]], type=pa.list_(pa.int64())),
        "primary": pa.array([r["primary"] for r in [e[0] for e in evs]], type=pa.list_(pa.bool_())),
    }
    t_tab = {
        "event_id": pa.array([e[1]["event_id"] for e in evs], type=pa.uint32()),
        "d0": pa.array([e[1]["d0"] for e in evs], type=pa.list_(pa.float32())),
        "z0": pa.array([e[1]["z0"] for e in evs], type=pa.list_(pa.float32())),
        "phi": pa.array([e[1]["phi"] for e in evs], type=pa.list_(pa.float32())),
        "theta": pa.array([e[1]["theta"] for e in evs], type=pa.list_(pa.float32())),
        "qop": pa.array([e[1]["qop"] for e in evs], type=pa.list_(pa.float32())),
        "majority_particle_id": pa.array([e[1]["majority_particle_id"] for e in evs], type=pa.list_(pa.uint64())),
        "hit_ids": pa.array([e[1]["hit_ids"] for e in evs], type=pa.list_(pa.list_(pa.uint32()))),
        "track_id": pa.array([e[1]["track_id"] for e in evs], type=pa.list_(pa.uint16())),
    }
    c_tab = {
        "event_id": pa.array([e[2]["event_id"] for e in evs], type=pa.uint32()),
        "detector": pa.array([e[2]["detector"] for e in evs], type=pa.list_(pa.uint8())),
        "total_energy": pa.array([e[2]["total_energy"] for e in evs], type=pa.list_(pa.float32())),
        "x": pa.array([e[2]["x"] for e in evs], type=pa.list_(pa.float32())),
        "y": pa.array([e[2]["y"] for e in evs], type=pa.list_(pa.float32())),
        "z": pa.array([e[2]["z"] for e in evs], type=pa.list_(pa.float32())),
        "contrib_particle_ids": pa.array([e[2]["contrib_particle_ids"] for e in evs], type=pa.list_(pa.list_(pa.uint64()))),
        "contrib_energies": pa.array([e[2]["contrib_energies"] for e in evs], type=pa.list_(pa.list_(pa.float32()))),
        "contrib_times": pa.array([e[2]["contrib_times"] for e in evs], type=pa.list_(pa.list_(pa.float32()))),
    }
    # tracker_hits mirrors the release layout too (x/y/z/true_x/true_y/true_z/time/particle_id/
    # detector/volume_id/layer_id/surface_id); the converter joins them through tracks.hit_ids
    # to build the gp->track hit-share fractions.
    th_tab = {
        "event_id": pa.array([e[3]["event_id"] for e in evs], type=pa.uint32()),
        "x": pa.array([e[3]["x"] for e in evs], type=pa.list_(pa.float32())),
        "y": pa.array([e[3]["y"] for e in evs], type=pa.list_(pa.float32())),
        "z": pa.array([e[3]["z"] for e in evs], type=pa.list_(pa.float32())),
        "true_x": pa.array([e[3]["true_x"] for e in evs], type=pa.list_(pa.float32())),
        "true_y": pa.array([e[3]["true_y"] for e in evs], type=pa.list_(pa.float32())),
        "true_z": pa.array([e[3]["true_z"] for e in evs], type=pa.list_(pa.float32())),
        "time": pa.array([e[3]["time"] for e in evs], type=pa.list_(pa.float32())),
        "particle_id": pa.array([e[3]["particle_id"] for e in evs], type=pa.list_(pa.uint64())),
        "detector": pa.array([e[3]["detector"] for e in evs], type=pa.list_(pa.uint8())),
        "volume_id": pa.array([e[3]["volume_id"] for e in evs], type=pa.list_(pa.uint8())),
        "layer_id": pa.array([e[3]["layer_id"] for e in evs], type=pa.list_(pa.uint16())),
        "surface_id": pa.array([e[3]["surface_id"] for e in evs], type=pa.list_(pa.uint32())),
    }
    tables = {"particles": pa.table(p_tab), "tracks": pa.table(t_tab), "calo_hits": pa.table(c_tab), "tracker_hits": pa.table(th_tab)}
    for obj, tbl in tables.items():
        sub = out_root / f"ttbar_pu0_{obj}" / "data" / f"ttbar_pu0_{obj}"
        sub.mkdir(parents=True, exist_ok=True)
        pq.write_table(tbl, sub / f"train-{shard_idx:05d}-of-{n_shards:05d}.parquet")


def make_fixture(out_root: Path, n_events: int = 30, seed: int = 42):
    n_shards = 2
    rng = np.random.default_rng(seed)
    events0 = [_event_arrays(rng, iev) for iev in range(n_events // 2)]
    events1 = [_event_arrays(rng, iev) for iev in range(n_events // 2, n_events)]
    write_shard(out_root, 0, n_shards, rng, events0)
    write_shard(out_root, 1, n_shards, rng, events1)
    # metadata.json per table, mimicking the real download layout
    for obj in ["particles", "tracks", "calo_hits", "tracker_hits"]:
        sub = out_root / f"ttbar_pu0_{obj}"
        shards = sorted(str(p.relative_to(sub)) for p in (sub / "data" / f"ttbar_pu0_{obj}").glob("*.parquet"))
        with open(sub / "metadata.json", "w") as f:
            json.dump(
                {"config": f"ttbar_pu0_{obj}", "dataset_id": "CERN/ColliderML-Release-1", "revision": "synthetic-fixture", "shards": shards},
                f,
                indent=2,
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--num-events", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    make_fixture(Path(args.output), n_events=args.num_events, seed=args.seed)
    print(f"Wrote synthetic fixture to {args.output}")


if __name__ == "__main__":
    main()
