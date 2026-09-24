#!/usr/bin/env python3
"""Diagnose ColliderML clusterer quality against particle-level ground truth.

For each kept truth particle in each event:
  - total attributed deposit (inclusive; calibrated GeV)
  - number of clusters those hits landed in
  - fraction of the deposit inside the cluster the allocator chose as this particle's representative

The point of the diagnostic: choose the clusterer improvement by data. A median fraction-deposit-in-
owner near 1 with few clusters/particle => raise radius is enough. A median of ~0.1-0.3 with many
small clusters/particle => we need merging or a different seed rule.

Usage:
    python scripts/diagnose_clusterer.py --num-events 5
"""
import argparse
from pathlib import Path

import awkward as ak
import numpy as np

from mlpf.data.colliderml.clustering import cluster_event
from mlpf.data.colliderml.reader import iter_event_records
from mlpf.data.colliderml.truth import DEFAULT_CALIBRATION, compute_gen_tables
from mlpf.data.target_building import EventData, assign_genparticles_to_obj_and_merge

BASE = Path("/mnt/ceph/users/ewulff/data/colliderml/CERN__ColliderML-Release-1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-events", type=int, default=5)
    ap.add_argument("--shard", type=int, default=1)
    ap.add_argument(
        "--merge-frac", type=float, default=0.10, help="merge a smaller cluster into a larger neighbour if E_small < merge_frac * E_large"
    )
    ap.add_argument("--ecal-mm", type=float, default=25.0, help="ECAL link radius (mm)")
    ap.add_argument("--hcal-mm", type=float, default=90.0, help="HCAL link radius (mm)")
    args = ap.parse_args()
    radii_mm = {9: args.ecal_mm, 10: args.ecal_mm, 11: args.ecal_mm, 12: args.hcal_mm, 13: args.hcal_mm, 14: args.hcal_mm}

    it = iter_event_records(
        BASE / "ttbar_pu0_particles" / "data" / "ttbar_pu0_particles" / f"train-{args.shard:05d}-of-01000.parquet",
        BASE / "ttbar_pu0_tracks" / "data" / "ttbar_pu0_tracks" / f"train-{args.shard:05d}-of-01000.parquet",
        BASE / "ttbar_pu0_calo_hits" / "data" / "ttbar_pu0_calo_hits" / f"train-{args.shard:05d}-of-01000.parquet",
        BASE / "ttbar_pu0_tracker_hits" / "data" / "ttbar_pu0_tracker_hits" / f"train-{args.shard:05d}-of-01000.parquet",
    )

    n_spread = []
    frac_in_owner = []
    e_deposits = []
    e_truth = []
    for ev_idx, ev in enumerate(it):
        if ev_idx >= args.num_events:
            break
        x = np.asarray(ak.to_numpy(ev["calo_hits"]["x"]), np.float64)
        y = np.asarray(ak.to_numpy(ev["calo_hits"]["y"]), np.float64)
        z = np.asarray(ak.to_numpy(ev["calo_hits"]["z"]), np.float64)
        e_hit = np.asarray(ak.to_numpy(ev["calo_hits"]["total_energy"]), np.float64)
        det = np.asarray(ak.to_numpy(ev["calo_hits"]["detector"]))
        k = np.array([DEFAULT_CALIBRATION[int(d)] for d in det], np.float64)
        e_hit_calibrated = e_hit * k

        gen_features, gp_to_hit, gp_to_track, _genref_features = compute_gen_tables(
            ev["particles"], ev["calo_hits"], ev["tracks"], DEFAULT_CALIBRATION, tracker_ev=ev["tracker_hits"]
        )
        _, feats, hit_to_cluster, _ = cluster_event(x, y, z, e_hit_calibrated, det, radii_mm=radii_mm, merge_frac=args.merge_frac)
        n_cluster = feats.shape[0]
        n_hit = len(x)

        # run the allocator to get gp_to_obj (which cluster each gp owns, if any)
        # EventData needs the *lengths* of hit/cluster/track to size its matrices. Use typed
        # filler arrays — only the length is ever read, never the values.
        n_track = int(np.asarray(ak.to_numpy(ev["tracks"]["track_id"])).shape[0]) if "track_id" in ev["tracks"].fields else 0
        gd = EventData(
            gen_features,
            {"type": np.zeros(n_hit, np.float32)},
            {"type": np.zeros(n_cluster, np.float32)},
            {"type": np.zeros(n_track, np.float32)},
            gp_to_hit,
            gp_to_track,
            hit_to_cluster,
            (np.array([]), np.array([])),
        )
        gd2, gpo_cleaned, _, _, _, _ = assign_genparticles_to_obj_and_merge(gd)

        # gpo_cleaned rows index into gd2.gen_features_new (allocator post-apply-order, i.e.
        # cleaned). gen_features rows index into gd2.gen_features_new 1:1 by same ordering
        # (both are cleaned by the same mask). map "kept truth gp order" (cleaned slot) ->
        # gpo_cleaned row is the identity.
        n_cleaned = len(gd2.gen_features["energy"])
        owner_of = np.asarray(gpo_cleaned[:, 1], dtype=np.int64)
        assert len(owner_of) == n_cleaned, (len(owner_of), n_cleaned)

        # hit -> cluster lookup. hit_to_cluster COO rows are (hit_idx, cluster_idx, 1.0), one per hit.
        cluster_of_hh = np.zeros(n_hit, dtype=np.int64)
        cluster_of_hh[np.asarray(hit_to_cluster[0], np.int64)] = np.asarray(hit_to_cluster[1], np.int64)

        # The hits attributed to a gp live in a "filtered" gp_to_hit where rows are indexed
        # by the allocator's post-cleaning gp index (same as gd2.gen_features rows). That is
        # the same index space as gpo_cleaned.
        gpi = np.asarray(gd2.genparticle_to_hit[0], np.int64)  # cleaned gp index
        hi = np.asarray(gd2.genparticle_to_hit[1], np.int64)  # calo hit index
        w = np.asarray(gd2.genparticle_to_hit[2], np.float64)  # calibrated deposit weight

        # per cleaned gp
        for j in range(n_cleaned):
            sel = gpi == j
            if not np.any(sel):
                continue
            wj = w[sel]
            cl_of_deposit = cluster_of_hh[hi[sel]]
            n_cl = len(np.unique(cl_of_deposit))
            if owner_of[j] == -1:
                continue
            frac = float(wj[cl_of_deposit == owner_of[j]].sum() / max(wj.sum(), 1e-12))
            frac_in_owner.append(frac)
            n_spread.append(n_cl)
            e_deposits.append(float(wj.sum()))
            e_truth.append(float(gd2.gen_features["energy"][j]))

    frac_in_owner = np.asarray(frac_in_owner)
    n_spread = np.asarray(n_spread)
    e_deposits = np.asarray(e_deposits)
    e_truth = np.asarray(e_truth)
    print(f"particles diagnosed: {len(frac_in_owner)} over {args.num_events} events")
    print(f"  fraction of a particle's deposit in its owned cluster (median): {np.median(frac_in_owner):.3f}")
    print(f"    p10 p25 p50 p75 p90 p99: {np.percentile(frac_in_owner, [10, 25, 50, 75, 90, 99]).round(3).tolist()}")
    print(f"  clusters per particle (median): {np.median(n_spread):.0f}")
    print(f"    p10 p25 p50 p75 p90 p99: {np.percentile(n_spread, [10, 25, 50, 75, 90, 99]).round(1).tolist()}")
    print(f"  median match ratio for deposit > 1 GeV: {np.median(frac_in_owner[e_deposits > 1.0]):.3f}")
    print(f"    n={np.sum(e_deposits > 1.0)}")
    print(f"  median match ratio for E_truth > 1 GeV: {np.median(frac_in_owner[e_truth > 1.0]):.3f}")
    print(f"    n={np.sum(e_truth > 1.0)}")


if __name__ == "__main__":
    main()
