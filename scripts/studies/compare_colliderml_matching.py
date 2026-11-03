"""Compare greedy and joint ownership matching on fixed ColliderML clusters.

Use --workers for bounded CPU parallelism. Each worker runs all rules on one
original event, including the same merge and energy-accounting implementation.
"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import contextlib
import io
import json
import multiprocessing
from pathlib import Path
import time

import awkward as ak
import fastjet
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import vector
from scipy.sparse import coo_matrix

from mlpf.data.colliderml.reader import iter_event_records
from mlpf.data.colliderml.tracks import track_features_cml
from mlpf.data.colliderml.truth import DEFAULT_CALIBRATION, compute_gen_tables
from mlpf.data.target_building import (
    EventData,
    _segment_argmax_first,
    assign_genparticles_to_obj_and_merge,
    match_majority_tracks_then_energy,
    match_particles_to_inputs,
)
from mlpf.jet_utils import _match_jets_event

RULES = ("legacy", "majority_greedy", "ownership_greedy", "solver", "majority_energy")


def assign_tracks_by_hit_ownership(gp_to_track, track_hit_counts):
    """Select one majority-owned track per particle, without minority fallback.

    Matrix entries are fractions of *all* hits on each track, including hits
    without a selected truth owner. Duplicate tracks are ordered by purity,
    number of owned hits, then track index. A 50/50 track has no owner.
    """
    counts = np.asarray(track_hit_counts, dtype=float)
    if counts.shape != (gp_to_track.shape[1],) or np.any(~np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("track_hit_counts must contain one finite nonnegative count per track")
    matrix = gp_to_track.tocsc(copy=True)
    matrix.sum_duplicates()
    if np.any(~np.isfinite(matrix.data)) or np.any(matrix.data < 0) or np.any(np.asarray(matrix.sum(axis=0)) > 1 + 1e-6):
        raise ValueError("track links must be fractions of all track hits, with column sums <= 1")
    matrix.eliminate_zeros()
    owners = np.full(matrix.shape[1], -1, dtype=np.int64)
    _segment_argmax_first(matrix.indptr, matrix.indices, matrix.data, matrix.shape[0], owners)
    entries = matrix.tocoo()
    particles, tracks, weights = entries.row, entries.col, entries.data
    eligible = (particles == owners[tracks]) & (weights > 0.5) & (counts[tracks] > 0)
    particles, tracks, weights = particles[eligible], tracks[eligible], weights[eligible]
    order = np.lexsort((tracks, -weights * counts[tracks], -weights))
    selected = np.full(matrix.shape[0], -1, dtype=np.int64)
    for i in order:
        particle = particles[i]
        if selected[particle] == -1:
            selected[particle] = tracks[i]
    return selected


def array(a):
    return np.asarray(ak.to_numpy(a))


def delta_r(eta, phi, other_eta, other_phi):
    dp = phi - other_phi
    return np.hypot(eta - other_eta, np.arctan2(np.sin(dp), np.cos(dp)))


def greedy_assignment(track, cluster, energy, counts, majority=False):
    n, nt = track.shape
    result = np.full((n, 2), -1, dtype=np.int64)
    if majority:
        result[:, 0] = assign_tracks_by_hit_ownership(track, counts)
    used_t, used_c = set(), set()
    for particle in sorted(range(n), key=lambda p: energy[p], reverse=True):
        if not majority:
            lo, hi = track.indptr[particle : particle + 2]
            ids, values = track.indices[lo:hi], track.data[lo:hi]
            for t in ids[np.lexsort((ids, -values))]:
                if int(t) not in used_t:
                    result[particle, 0] = t
                    used_t.add(int(t))
                    break
        if result[particle, 0] >= 0:
            continue
        lo, hi = cluster.indptr[particle : particle + 2]
        ids, values = cluster.indices[lo:hi], cluster.data[lo:hi]
        for c in ids[np.lexsort((ids, -values))]:
            if int(c) not in used_c:
                result[particle, 1] = c
                used_c.add(int(c))
                break
    return result


def ownership_greedy(track, cluster, cluster_energy, particle_deposit):
    t, c = track.tocoo(), cluster.tocoo()
    keep = t.data > 0.5
    row = np.r_[t.row[keep], c.row]
    col = np.r_[t.col[keep], track.shape[1] + c.col]
    score = np.r_[1 + t.data[keep], np.minimum(2 * c.data / (cluster_energy[c.col] + particle_deposit[c.row]), 1)]
    order = np.lexsort((row, col, -score))
    result = np.full((track.shape[0], 2), -1, dtype=np.int64)
    used = np.zeros(track.shape[1] + cluster.shape[1], dtype=bool)
    for i in order:
        p, o = row[i], col[i]
        if np.any(result[p] >= 0) or used[o]:
            continue
        used[o] = True
        if o < track.shape[1]:
            result[p, 0] = o
        else:
            result[p, 1] = o - track.shape[1]
    return result


def extract_pairs(matrix, rows, columns):
    return np.asarray(matrix[rows, columns]).ravel() if len(rows) else np.empty(0)


def merge_accounting(energy, assignment, merged_particles, target_energy):
    """Count actual dropped particles separately from floating-point residuals."""
    dropped = ~np.any(assignment >= 0, axis=1)
    dropped[np.asarray(merged_particles, dtype=int)] = False
    dropped_energy = float(np.asarray(energy)[dropped].sum(dtype=np.float64))
    residual = target_energy + dropped_energy - float(np.sum(energy, dtype=np.float64))
    return dict(dropped_particles=int(dropped.sum()), dropped_energy=dropped_energy, energy_balance_residual=residual)


def jets_from_targets(g):
    pt, eta = array(g["pt"]), array(g["eta"])
    phi = np.arctan2(array(g["sin_phi"]), array(g["cos_phi"]))
    vectors = ak.zip(dict(px=pt * np.cos(phi), py=pt * np.sin(phi), pz=pt * np.sinh(eta), E=array(g["energy"])))
    sequence = fastjet.ClusterSequence(vectors, fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4))
    jets = vector.awk(sequence.inclusive_jets(min_pt=3.0))
    return np.column_stack([array(jets.pt), array(jets.eta), array(jets.phi)])


def process_event(task):
    sample, event, cached, output = task
    event_id = event["event_id"]
    p, t, c, h = [event[k] for k in ["particles", "tracks", "calo_hits", "tracker_hits"]]
    start = time.perf_counter()
    gen, gh, gt, _ = compute_gen_tables(p, c, t, DEFAULT_CALIBRATION, tracker_ev=h)
    features = track_features_cml(t, h)
    counts = array(ak.num(t["hit_ids"]))
    xc = array(cached["X_cluster"])
    assert event_id == int(cached["event_id"])
    co = array(cached["hit_to_cluster"]).astype(int)
    assert len(co) == len(c["x"])
    n, nt, nc, nh = len(gen["pt"]), len(counts), len(xc), len(c["x"]) + len(h["x"])
    good = co >= 0
    hit_to_cluster = (np.flatnonzero(good), co[good], np.ones(good.sum()))
    track = coo_matrix((gt[2], (gt[0], gt[1])), shape=(n, nt)).tocsr()
    hit = coo_matrix((gh[2], (gh[0], gh[1])), shape=(n, nh)).tocsr()
    h2c = coo_matrix((hit_to_cluster[2], (hit_to_cluster[0], hit_to_cluster[1])), shape=(nh, nc)).tocsr()
    cluster = (hit @ h2c).tocsr()
    track.eliminate_zeros()
    track.sort_indices()
    cluster.eliminate_zeros()
    cluster.sort_indices()
    gd = EventData(
        gen,
        {"type": np.zeros(nh)},
        {"type": np.zeros(nc)},
        {"type": np.ones(nt)},
        gh,
        gt,
        hit_to_cluster,
        (np.array([], dtype=int), np.array([], dtype=int)),
    )
    preparation = time.perf_counter() - start
    before_energy = float(np.sum(gen["energy"], dtype=np.float64))
    particle_energy, particle_pt, ispu = [np.asarray(gen[k]) for k in ["energy", "pt", "ispu"]]
    rows, objects = [], []
    assignments = {}
    for rule in RULES:
        start = time.perf_counter()
        if rule in ("legacy", "majority_greedy"):
            assignment = greedy_assignment(track, cluster, particle_energy, counts, rule == "majority_greedy")
        elif rule == "ownership_greedy":
            assignment = ownership_greedy(track, cluster, xc[:, 5], gen["gp_to_cluster"])
        elif rule == "solver":
            assignment = match_particles_to_inputs(track, cluster, xc[:, 5], gen["gp_to_cluster"])
        else:
            assignment = match_majority_tracks_then_energy(track, cluster, particle_energy)
        match_seconds = time.perf_counter() - start
        assignments[rule] = assignment
        tr_rows, cl_rows = np.flatnonzero(assignment[:, 0] >= 0), np.flatnonzero(assignment[:, 1] >= 0)
        tr, cl = assignment[tr_rows, 0], assignment[cl_rows, 1]
        fractions = extract_pairs(track, tr_rows, tr)
        deposit = extract_pairs(cluster, cl_rows, cl)
        purity = deposit / xc[cl, 5]
        completeness = deposit / np.asarray(gen["gp_to_cluster"])[cl_rows]
        overlap = 2 * deposit / (xc[cl, 5] + np.asarray(gen["gp_to_cluster"])[cl_rows])
        eligible = fractions > 0.5
        covered = np.any(assignment >= 0, axis=1)
        metrics = dict(
            sample=sample,
            event_id=event_id,
            rule=rule,
            preparation_seconds=preparation,
            matching_seconds=match_seconds,
            input_tracks=nt,
            input_clusters=nc,
            selected_particles=n,
            exclusive_particles=int(covered.sum()),
            exclusive_energy=float(particle_energy[covered].sum(dtype=np.float64)),
            selected_energy=before_energy,
            selected_hs=int((ispu == 0).sum()),
            exclusive_hs=int((covered & (ispu == 0)).sum()),
            selected_pt5=int((particle_pt >= 5).sum()),
            exclusive_pt5=int((covered & (particle_pt >= 5)).sum()),
            track_targets=len(tr),
            cluster_targets=len(cl),
            ineligible_track_targets=int((~eligible).sum()),
            track_hit_purity_sum=float(fractions.sum()),
            track_owned_hits=float((fractions * counts[tr]).sum()),
            track_hits=float(counts[tr].sum()),
            cluster_shared_deposit=float(deposit.sum()),
            cluster_input_energy=float(xc[cl, 5].sum(dtype=np.float64)),
            cluster_particle_deposit=float(np.asarray(gen["gp_to_cluster"])[cl_rows].sum(dtype=np.float64)),
            cluster_purity_sum=float(purity.sum()),
            cluster_completeness_sum=float(completeness.sum()),
            cluster_overlap_sum=float(overlap.sum()),
            ownership_objective=float((1 + fractions[eligible]).sum() + np.minimum(overlap, 1).sum()),
        )
        # Both rules use identical hit assignment, cluster merging and drop accounting.
        start = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            cleaned, assigned, *_ = assign_genparticles_to_obj_and_merge(gd, input_assignment=assignment)
        metrics["post_assignment_seconds"] = time.perf_counter() - start
        g = cleaned.gen_features
        g_energy, g_pt, g_eta, g_ispu = [array(g[k]) for k in ["energy", "pt", "eta", "ispu"]]
        g_phi = np.arctan2(array(g["sin_phi"]), array(g["cos_phi"]))
        tm, cm = assigned[:, 0] >= 0, assigned[:, 1] >= 0
        ti, ci = assigned[tm, 0], assigned[cm, 1]
        if rule == "legacy":
            for field, mask, indices in [("ytarget_track", tm, ti), ("ytarget_cluster", cm, ci)]:
                y = array(cached[field])
                assert np.array_equal(np.sort(indices), np.flatnonzero(y[:, 0] > 0))
                np.testing.assert_allclose(g_pt[mask], y[indices, 2], rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(g_eta[mask], y[indices, 3], rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(g_energy[mask], y[indices, 6], rtol=1e-5, atol=1e-4)
        target_energy = float(g_energy.sum(dtype=np.float64))
        metrics.update(
            merged_particles=len(cleaned.gp_merges[1]),
            target_energy=target_energy,
            **merge_accounting(particle_energy, assignment, cleaned.gp_merges[1], target_energy),
        )
        assert metrics["dropped_particles"] == n - len(assigned) - len(cleaned.gp_merges[1])
        merged = np.asarray(cleaned.gp_merges[1], dtype=int)
        metrics.update(
            merged_energy=float(particle_energy[merged].sum(dtype=np.float64)),
            merged_hs_energy=float(particle_energy[merged][ispu[merged] == 0].sum(dtype=np.float64)),
        )
        # final cluster-target energy vs its cluster's energy (after merges): the regression target scale
        log_ratio = np.abs(np.log(np.maximum(g_energy[cm], 1e-9) / np.maximum(xc[ci, 5], 1e-9)))
        metrics.update(
            cluster_energy_off3=int((log_ratio > np.log(3)).sum()),
            cluster_energy_off10=int((log_ratio > np.log(10)).sum()),
            cluster_energy_off3_energy=float(g_energy[cm][log_ratio > np.log(3)].sum(dtype=np.float64)),
            cluster_target_energy=float(g_energy[cm].sum(dtype=np.float64)),
        )
        target_met = float(np.hypot(np.sum(g_pt * np.cos(g_phi)), np.sum(g_pt * np.sin(g_phi))))
        metrics.update(target_met=target_met, genmet=float(cached["genmet"]), met_residual=target_met - float(cached["genmet"]))
        for kind, mask, ids, input_eta, input_phi, cut in [
            ("track", tm, ti, features["eta"][ti], array(t["phi"])[ti], 0.1),
            ("cluster", cm, ci, xc[ci, 2], np.arctan2(xc[ci, 3], xc[ci, 4]), 0.3),
        ]:
            dr = delta_r(input_eta, input_phi, g_eta[mask], g_phi[mask])
            for group, selected in [
                ("all", np.ones(len(ids), dtype=bool)),
                ("hard_scatter", g_ispu[mask] == 0),
                ("pileup", g_ispu[mask] == 1),
                ("pt5", g_pt[mask] >= 5),
            ]:
                metrics[f"{kind}_{group}_count"] = int(selected.sum())
                metrics[f"{kind}_{group}_bad"] = int((dr[selected] > cut).sum())
            objects.append(pd.DataFrame(dict(kind=kind, rule=rule, event_id=event_id, delta_r=dr, pt=g_pt[mask], ispu=g_ispu[mask])))
        candidate_jets = jets_from_targets(g)
        reference = array(cached["genjet"])
        reference = reference[reference[:, 0] > 3]
        r, q = _match_jets_event(reference[:, 1], reference[:, 2], candidate_jets[:, 1], candidate_jets[:, 2], 0.1)
        ratio = candidate_jets[q, 0] / reference[r, 0]
        objects.append(pd.DataFrame(dict(kind="jet", rule=rule, event_id=event_id, pt=reference[r, 0], response=ratio)))
        metrics.update(
            reference_jets=len(reference),
            target_jets=len(candidate_jets),
            matched_jets=len(r),
            response_qualified_jets=int((np.abs(ratio - 1) < 0.5).sum()),
        )
        rows.append(metrics)
    objective = {r["rule"]: r["ownership_objective"] for r in rows}
    assert objective["solver"] >= objective["ownership_greedy"] - 1e-4
    out = Path(output) / sample / "events"
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(objects, ignore_index=True).to_parquet(out / f"event_{event_id:06d}.parquet", index=False)
    (out / f"event_{event_id:06d}.json").write_text(json.dumps(rows, indent=2))
    return sample, event_id, rows


def tasks(args):
    for sample, source, processed in [("pu0", args.pu0_source, args.pu0_processed), ("pu200", args.pu200_source, args.pu200_processed)]:
        if (source / "particles.parquet").exists():
            paths = [source / f"{obj}.parquet" for obj in ["particles", "tracks", "calo_hits", "tracker_hits"]]
        else:
            paths = [
                next(source.glob(f"ttbar_{sample}_{obj}/data/*/train-00000-of-01000.parquet"))
                for obj in ["particles", "tracks", "calo_hits", "tracker_hits"]
            ]
        batches = iter(
            pq.ParquetFile(processed).iter_batches(
                batch_size=1, columns=["event_id", "X_cluster", "ytarget_track", "ytarget_cluster", "hit_to_cluster", "genjet", "genmet"]
            )
        )
        for i, event in enumerate(iter_event_records(*paths, batch_size=1)):
            if i >= args.events:
                break
            yield sample, event, ak.from_arrow(next(batches))[0], str(args.output)


def summarize(output):
    records = []
    for path in sorted(output.glob("*/events/event_*.json")):
        records.extend(json.loads(path.read_text()))
    frame = pd.DataFrame(records)
    frame.to_csv(output / "events.csv", index=False)
    objects = pd.concat([pd.read_parquet(p).assign(sample=p.parent.parent.name) for p in sorted(output.glob("*/events/event_*.parquet"))])
    jets = objects[(objects.kind == "jet") & (objects.pt > 20)]
    summaries = []
    for (sample, rule), z in frame.groupby(["sample", "rule"], sort=False):
        j = jets[(jets["sample"] == sample) & (jets.rule == rule)].response
        sums = z.select_dtypes(include="number").sum()
        r = dict(
            sample=sample,
            rule=rule,
            events=len(z),
            track_targets=int(sums.track_targets),
            cluster_targets=int(sums.cluster_targets),
            track_bad_fraction=float(sums.track_all_bad / sums.track_all_count),
            track_hs_bad_fraction=float(sums.track_hard_scatter_bad / sums.track_hard_scatter_count),
            cluster_bad_fraction=float(sums.cluster_all_bad / sums.cluster_all_count),
            cluster_pt5_bad_fraction=float(sums.cluster_pt5_bad / sums.cluster_pt5_count),
            mean_track_purity=float(sums.track_hit_purity_sum / sums.track_targets),
            hit_weighted_track_purity=float(sums.track_owned_hits / sums.track_hits),
            mean_cluster_purity=float(sums.cluster_purity_sum / sums.cluster_targets),
            mean_cluster_completeness=float(sums.cluster_completeness_sum / sums.cluster_targets),
            mean_cluster_overlap=float(sums.cluster_overlap_sum / sums.cluster_targets),
            matched_cluster_deposit=float(sums.cluster_shared_deposit),
            exclusive_particle_fraction=float(sums.exclusive_particles / sums.selected_particles),
            exclusive_hs_fraction=float(sums.exclusive_hs / sums.selected_hs),
            exclusive_pt5_fraction=float(sums.exclusive_pt5 / sums.selected_pt5),
            exclusive_energy_fraction=float(sums.exclusive_energy / sums.selected_energy),
            merged_fraction=float(sums.merged_particles / sums.selected_particles),
            dropped_particles=int(sums.dropped_particles),
            dropped_energy_fraction=float(sums.dropped_energy / sums.selected_energy),
            dropped_energy=float(sums.dropped_energy),
            jet_recall=float(sums.matched_jets / sums.reference_jets),
            jet_precision=float(sums.matched_jets / sums.target_jets),
            jet_response_recall=float(sums.response_qualified_jets / sums.reference_jets),
            median_matching_seconds=float(z.matching_seconds.median()),
            median_post_assignment_seconds=float(z.post_assignment_seconds.median()),
            maximum_energy_balance_residual=float(z.energy_balance_residual.abs().max()),
            ownership_objective=float(sums.ownership_objective),
            merged_energy=float(sums.merged_energy),
            merged_hs_energy=float(sums.merged_hs_energy),
            cluster_energy_off3_fraction=float(sums.cluster_energy_off3 / sums.cluster_targets),
            cluster_energy_off3_energy_fraction=float(sums.cluster_energy_off3_energy / sums.cluster_target_energy),
            cluster_energy_off10_fraction=float(sums.cluster_energy_off10 / sums.cluster_targets),
            met_residual_mean=float(z.met_residual.mean()),
            met_residual_rms=float(np.sqrt((z.met_residual**2).mean())),
            jet20_matched=int(len(j)),
            jet20_response_median=float(j.median()),
            jet20_response_iqr=float(j.quantile(0.75) - j.quantile(0.25)),
        )
        summaries.append(r)
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(pd.DataFrame(summaries).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pu0-source", type=Path, required=True)
    parser.add_argument("--pu200-source", type=Path, required=True)
    parser.add_argument("--pu0-processed", type=Path, required=True)
    parser.add_argument("--pu200-processed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.events < 1 or args.workers < 1:
        parser.error("events and workers must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "inputs.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    import scipy

    (args.output / "versions.json").write_text(json.dumps(dict(numpy=np.__version__, scipy=scipy.__version__), indent=2))
    start = time.perf_counter()
    source = iter(tasks(args))
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = set()
        while True:
            while len(pending) < args.workers:
                try:
                    task = next(source)
                except StopIteration:
                    break
                pending.add(pool.submit(process_event, task))
            if not pending:
                break
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                sample, event_id, _ = future.result()
                print(f"completed {sample} event {event_id}", flush=True)
    summarize(args.output)
    (args.output / "runtime.json").write_text(json.dumps(dict(wall_seconds=time.perf_counter() - start, workers=args.workers), indent=2))


if __name__ == "__main__":
    main()
