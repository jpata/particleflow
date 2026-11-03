"""Validate joint ownership matching against native CLD, CLIC, IDEA or CMS processing.

Detector-specific visibility, truth ancestry and fallback rules stay native.
Overcommitted track columns are excluded from solver eligibility and counted.
"""

import argparse
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import time
from unittest.mock import patch

import awkward as ak
import fastjet
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, diags
import uproot
import vector

from mlpf.conf import EDM4HEP
from mlpf.data.target_building import assign_genparticles_to_obj_and_merge, match_particles_to_inputs
from mlpf.jet_utils import _match_jets_event
from scripts.studies.compare_colliderml_matching import ownership_greedy, greedy_assignment, delta_r, merge_accounting

RULES = ("native", "ownership_greedy", "solver")


def arr(a):
    return np.asarray(ak.to_numpy(a))


def matrix(coo, n, m):
    out = coo_matrix((coo[2], (coo[0], coo[1])), shape=(n, m)).tocsr()
    out.sum_duplicates()
    out.eliminate_zeros()
    out.sort_indices()
    return out


def jets(pt, eta, phi, energy, detector):
    vectors = ak.zip(dict(px=pt * np.cos(phi), py=pt * np.sin(phi), pz=pt * np.sinh(eta), E=energy))
    definition = (
        fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4) if detector == "cms" else fastjet.JetDefinition(fastjet.ee_genkt_algorithm, 0.4, -1.0)
    )
    sequence = fastjet.ClusterSequence(vectors, definition)
    j = vector.awk(sequence.inclusive_jets(min_pt=3 if detector == "cms" else 5))
    return np.column_stack([arr(j.pt), arr(j.eta), arr(j.phi), arr(j.energy)])


def clean_scores(bundle):
    track = bundle["track"].copy()
    sums = np.asarray(track.sum(axis=0)).ravel()
    invalid = sums > 1 + 1e-6
    if np.any(invalid):
        track = track @ diags((~invalid).astype(float))
    cluster = bundle["cluster"].copy().tocoo()
    valid = (cluster.data >= 0) & (cluster.data <= bundle["signal"][cluster.col] * (1 + 1e-5))
    rejected = int((~valid).sum())
    cluster = coo_matrix((cluster.data[valid], (cluster.row[valid], cluster.col[valid])), shape=cluster.shape).tocsr()
    deposit = np.asarray(cluster.sum(axis=1)).ravel()
    # Tiny calibration roundoff is handled by the joint matcher, not renormalized.
    bundle["invalid_track_columns"] = int(invalid.sum())
    bundle["invalid_cluster_edges"] = rejected
    return track.tocsr(), cluster, deposit


def choose(bundle, rule):
    if rule == "native":
        return bundle["native"].copy()
    track, cluster, deposit = clean_scores(bundle)
    if rule == "solver":
        return match_particles_to_inputs(track, cluster, bundle["signal"], deposit)
    return ownership_greedy(track, cluster, bundle["signal"], deposit)


def post_shared(bundle, assignment):
    with contextlib.redirect_stdout(io.StringIO()):
        cleaned, obj, *_ = assign_genparticles_to_obj_and_merge(bundle["gd"], input_assignment=assignment)
    g = cleaned.gen_features
    masks = [obj[:, 0] >= 0, obj[:, 1] >= 0]
    rows = []
    for kind, mask in enumerate(masks):
        for i in np.flatnonzero(mask):
            rows.append(
                (
                    kind,
                    int(obj[i, kind]),
                    float(g["pt"][i]),
                    float(g["eta"][i]),
                    float(np.arctan2(g["sin_phi"][i], g["cos_phi"][i])),
                    float(g["energy"][i]),
                    float(g["ispu"][i]),
                )
            )
    accounting = merge_accounting(bundle["gen"]["energy"], assignment, cleaned.gp_merges[1], float(np.sum(arr(g["energy"]), dtype=np.float64)))
    return rows, dict(merged_particles=len(cleaned.gp_merges[1]), **accounting)


def post_idea(bundle, assignment):
    rows = []
    g = bundle["gen"]
    for p, objects in enumerate(assignment):
        for kind, obj in enumerate(objects):
            if obj >= 0:
                rows.append((kind, int(obj), float(g["pt"][p]), float(g["eta"][p]), float(g["phi"][p]), float(g["energy"][p]), 0.0))
    dropped = ~np.any(assignment >= 0, axis=1)
    return rows, dict(
        merged_particles=0, dropped_particles=int(dropped.sum()), dropped_energy=float(g["energy"][dropped].sum()), energy_balance_residual=0.0
    )


def measure(bundle, assignment, post, accounting, rule, seconds):
    g = bundle["gen"]
    p_t = np.flatnonzero(assignment[:, 0] >= 0)
    p_c = np.flatnonzero(assignment[:, 1] >= 0)
    tr, cl = assignment[p_t, 0], assignment[p_c, 1]
    tf = np.asarray(bundle["track"][p_t, tr]).ravel() if len(tr) else np.empty(0)
    dep = np.asarray(bundle["cluster"][p_c, cl]).ravel() if len(cl) else np.empty(0)
    signal = bundle["signal"][cl]
    total = np.asarray(bundle["cluster"].sum(axis=1)).ravel()[p_c]
    overlap = np.divide(2 * dep, signal + total, out=np.zeros(len(dep)), where=(signal + total) > 0)
    covered = np.any(assignment >= 0, axis=1)
    result = dict(
        detector=bundle["detector"],
        event_id=bundle["event_id"],
        rule=rule,
        matching_seconds=seconds,
        selected_particles=len(assignment),
        exclusive_particles=int(covered.sum()),
        track_matches=len(tr),
        cluster_matches=len(cl),
        input_tracks=bundle["track"].shape[1],
        invalid_track_columns=bundle.get("invalid_track_columns", 0),
        invalid_cluster_edges=bundle.get("invalid_cluster_edges", 0),
        track_purity_sum=float(tf.sum()),
        cluster_purity_sum=float(np.divide(dep, signal, out=np.zeros(len(dep)), where=signal > 0).sum()),
        overlap_objective=float(np.minimum(overlap, 1).sum() + (1 + tf[tf > 0.5]).sum()),
        selected_energy=float(np.sum(g["energy"], dtype=np.float64)),
        **accounting,
    )
    objects = pd.DataFrame(post, columns=["kind", "object", "pt", "eta", "phi", "energy", "ispu"])
    for kind, label, cut in [(0, "track", 0.1), (1, "cluster", 0.3)]:
        z = objects[objects.kind == kind].copy()
        ids = z.object.to_numpy(dtype=int)
        result[label + "_fallback_targets"] = int((~np.isin(ids, assignment[:, kind][assignment[:, kind] >= 0])).sum())
        dr = delta_r(bundle["eta"][kind][ids], bundle["phi"][kind][ids], z.eta.to_numpy(), z.phi.to_numpy())
        z["delta_r"] = dr
        objects.loc[z.index, "delta_r"] = dr
        result[label + "_targets"] = len(z)
        result[label + "_bad"] = int((dr > cut).sum())
        result[label + "_pt5_count"] = int((z.pt >= 5).sum())
        result[label + "_pt5_bad"] = int(((dr > cut) & (z.pt.to_numpy() >= 5)).sum())
    result["target_energy"] = float(objects.energy.sum())
    result["energy_balance_residual"] = result["target_energy"] + result["dropped_energy"] - result["selected_energy"]
    assert abs(result["energy_balance_residual"]) < 1e-6 * max(1, result["selected_energy"])
    candidates = jets(objects.pt.to_numpy(), objects.eta.to_numpy(), objects.phi.to_numpy(), objects.energy.to_numpy(), bundle["detector"])
    ref = bundle["reference"]
    a, b = _match_jets_event(ref[:, 1], ref[:, 2], candidates[:, 1], candidates[:, 2], 0.1)
    response = candidates[b, 0] / ref[a, 0]
    result.update(reference_jets=len(ref), target_jets=len(candidates), matched_jets=len(a), response_jets=int((np.abs(response - 1) < 0.5).sum()))
    return result, objects


def evaluate(bundle, out, native_post=None):
    rows = []
    objects = []
    clean_scores(bundle)
    for rule in RULES:
        start = time.perf_counter()
        assignment = choose(bundle, rule)
        seconds = time.perf_counter() - start
        if rule == "native":
            seconds = None  # native matching was already computed during detector preparation
        if native_post is not None and rule == "native":
            post, accounting = native_post
        elif bundle["detector"] == "idea":
            post, accounting = post_idea(bundle, assignment)
        else:
            post, accounting = post_shared(bundle, assignment)
        result, z = measure(bundle, assignment, post, accounting, rule, seconds)
        rows.append(result)
        z["rule"] = rule
        z["event_id"] = bundle["event_id"]
        objects.append(z)
    event = bundle["event_id"]
    (out / "events" / f"{event:05d}.json").write_text(json.dumps(rows, indent=2))
    pd.concat(objects, ignore_index=True).to_parquet(out / "events" / f"{event:05d}.parquet", index=False)
    print(f"completed {bundle['detector']} event {event}", flush=True)


def shared_bundle(gd, detector, iev, reference):
    n, nt, nc, nh = [
        len(x["type"] if name != "gen" else x["PDG"])
        for name, x in [("gen", gd.gen_features), ("track", gd.track_features), ("cluster", gd.cluster_features), ("hit", gd.hit_features)]
    ]
    track = matrix(gd.genparticle_to_track, n, nt)
    hit = matrix(gd.genparticle_to_hit, n, nh)
    h2c = matrix(gd.hit_to_cluster, nh, nc)
    cluster = (hit @ h2c).tocsr()
    cluster.eliminate_zeros()
    cluster.sort_indices()
    gen = {k: arr(gd.gen_features[k]) for k in ["pt", "eta", "energy", "ispu"]}
    gen["phi"] = (
        np.arctan2(arr(gd.gen_features["sin_phi"]), arr(gd.gen_features["cos_phi"]))
        if "sin_phi" in gd.gen_features.fields
        else arr(gd.gen_features["phi"])
    )
    # Use calibrated hit energy, not the potentially corrected Pandora cluster energy.
    signal = np.asarray(h2c.T @ arr(gd.hit_features["energy"])).ravel()
    native = greedy_assignment(track, cluster, gen["energy"], np.ones(nt), False)
    return dict(
        gd=gd,
        detector=detector,
        event_id=iev,
        gen=gen,
        track=track,
        cluster=cluster,
        signal=signal,
        native=native,
        eta=[arr(gd.track_features["eta"]), arr(gd.cluster_features["eta"])],
        phi=[arr(gd.track_features["phi"]), arr(gd.cluster_features["phi"])],
        reference=reference,
    )


def run_key4hep(args):
    from mlpf.data.key4hep import postprocessing as pp

    original_assign = pp.assign_genparticles_to_obj_and_merge
    original_jets = pp.compute_jets
    original_open = uproot.open
    captured = {}
    counter = [0]

    def compute(*a, **kw):
        value = original_jets(*a, **kw)
        if "reference" not in captured:
            captured["reference"] = value
        return value

    def assign(gd):
        native = original_assign(gd)
        i = counter[0]
        counter[0] += 1
        ref = captured["reference"][i]
        reference = np.column_stack([arr(ref.pt), arr(ref.eta), arr(ref.phi), arr(ref.energy)])
        bundle = shared_bundle(gd, args.detector, i, reference)
        # Verify that the reconstructed raw assignment gives native outputs.
        with contextlib.redirect_stdout(io.StringIO()):
            check = original_assign(gd, input_assignment=bundle["native"])
        np.testing.assert_array_equal(check[1], native[1])
        np.testing.assert_allclose(arr(check[0].gen_features["energy"]), arr(native[0].gen_features["energy"]), rtol=1e-6, atol=1e-6)
        evaluate(bundle, args.output)
        return native

    def opened(*a, **kw):
        kw["handler"] = uproot.source.file.MemmapSource
        return original_open(*a, **kw)

    with (
        patch.object(pp, "assign_genparticles_to_obj_and_merge", assign),
        patch.object(pp, "compute_jets", compute),
        patch.object(uproot, "open", opened),
    ):
        pp.process_one_file(str(args.input), str(args.output / "native.parquet"), args.detector, num_events=args.events)


def run_idea(args):
    from mlpf.data.key4hep import postprocessing as pp

    root = uproot.open(args.input, handler=uproot.source.file.MemmapSource)
    tree = root["events"]
    branches = [
        "MCParticles",
        "MCParticles.PDG",
        "MCParticles.momentum.x",
        "MCParticles.momentum.y",
        "MCParticles.momentum.z",
        "MCParticles.mass",
        "MCParticles.charge",
        "MCParticles.generatorStatus",
        "MCParticles.simulatorStatus",
        "MCParticles.daughters_begin",
        "MCParticles.daughters_end",
        "_MCParticles_daughters/_MCParticles_daughters.index",
        "TracksFromGenParticles",
        "_TracksFromGenParticles_trackStates",
        "TopoClusterAll",
        "TopoClusterAllCells",
        "_TopoClusterAll_hits/_TopoClusterAll_hits.index",
        "_TopoClusterAll_hits/_TopoClusterAll_hits.collectionID",
        "_TopoClusterAll_shapeParameters",
        "TracksFromGenParticlesAssociation.weight",
        "_TracksFromGenParticlesAssociation_from/_TracksFromGenParticlesAssociation_from.index",
        "_TracksFromGenParticlesAssociation_to/_TracksFromGenParticlesAssociation_to.index",
        "ClusterMCParticleLinks.weight",
        "_ClusterMCParticleLinks_from/_ClusterMCParticleLinks_from.index",
        "_ClusterMCParticleLinks_to/_ClusterMCParticleLinks_to.index",
    ]
    prop = tree.arrays(branches, entry_stop=args.events)
    meta = root["podio_metadata"]
    prefix = "events___CollectionTypeInfo/events___CollectionTypeInfo."
    cid = dict(zip(meta[prefix + "name"].array()[0], meta[prefix + "collectionID"].array()[0]))["TopoClusterAllCells"]
    for i in range(min(args.events, len(prop))):
        g = pp.gen_to_features(prop, i)
        t = pp.idea_track_to_features(prop, i, EDM4HEP.DETECTORS["idea"].b_field)
        c = pp.idea_cluster_to_features(prop, i, cid)
        ct, tt = pp.add_daughters_to_status1(
            g, pp._idea_relation(prop, "ClusterMCParticleLinks", i), pp._idea_relation(prop, "TracksFromGenParticlesAssociation", i)
        )
        track = matrix(tt, len(g["PDG"]), len(t["type"]))
        weights = matrix(ct, len(g["PDG"]), len(c["energy"]))

        def owners(m):
            a = m.toarray()
            owner = a.argmax(axis=0)
            owner[a.max(axis=0) <= 0] = -1
            return owner

        to, co = owners(track), owners(weights)
        represented = np.zeros(len(g["PDG"]), dtype=bool)
        represented[to[to >= 0]] = True
        represented[co[co >= 0]] = True
        keep = (arr(g["generatorStatus"]) == 1) & ~np.isin(np.abs(arr(g["PDG"])), [12, 14, 16]) & represented
        selected = np.flatnonzero(keep)
        lookup = {int(old): new for new, old in enumerate(selected)}
        native = np.full((len(selected), 2), -1, dtype=int)
        used = set()
        for kind, ownership in enumerate([to, co]):
            for obj, old in enumerate(ownership):
                if old in lookup and int(old) not in used:
                    native[lookup[int(old)], kind] = obj
                    used.add(int(old))
        gen = {k: arr(g[k])[keep] for k in ["pt", "eta", "phi", "energy", "ispu"]}
        track = track[keep]
        cluster = (weights[keep] @ diags(arr(c["energy"]))).tocsr()
        truth = (
            (arr(g["generatorStatus"]) == 1)
            & ~np.isin(np.abs(arr(g["PDG"])), [12, 14, 16])
            & ((arr(g["simulatorStatus"]).astype(np.int64) & 0x0F000000) != 0)
        )
        ref = jets(arr(g["pt"])[truth], arr(g["eta"])[truth], arr(g["phi"])[truth], arr(g["energy"])[truth], "idea")
        bundle = dict(
            detector="idea",
            event_id=i,
            gen=gen,
            track=track,
            cluster=cluster,
            signal=arr(c["energy"]),
            native=native,
            eta=[arr(t["eta"]), arr(c["eta"])],
            phi=[np.arctan2(arr(t["sin_phi"]), arr(t["cos_phi"])), np.arctan2(arr(c["sin_phi"]), arr(c["cos_phi"]))],
            reference=ref,
        )
        evaluate(bundle, args.output)


def cms_bundle(graph, particles, elements, iev):
    lookup = {p: i for i, p in enumerate(particles)}
    tracks = [e for e in elements if graph.nodes[e]["typ"] in [1, 6]]
    clusters = [e for e in elements if graph.nodes[e]["typ"] not in [1, 6]]
    gen = {k: np.array([graph.nodes[p][k] for p in particles]) for k in ["pt", "eta", "phi", "energy", "ispu"]}
    matrices = []
    for kind, inputs in enumerate([tracks, clusters]):
        r = []
        c = []
        w = []
        for j, e in enumerate(inputs):
            for p in graph.predecessors(e):
                if p not in lookup:
                    continue
                value = graph.edges[p, e]["weight"]
                if kind == 0:
                    value = value / max(graph.nodes[e]["energy"], 1e-12)
                if value > 0:
                    r.append(lookup[p])
                    c.append(j)
                    w.append(value)
        matrices.append(coo_matrix((w, (r, c)), shape=(len(particles), len(inputs))).tocsr())
    pythia = [n for n in graph if n[0] == "gen" and graph.nodes[n]["status"] == 1 and abs(graph.nodes[n]["pid"]) not in [12, 14, 16]]
    ref = jets(*[np.array([graph.nodes[p][k] for p in pythia]) for k in ["pt", "eta", "phi", "energy"]], "cms")
    bundle = dict(
        detector="cms",
        event_id=iev,
        gen=gen,
        track=matrices[0],
        cluster=matrices[1],
        signal=np.array([graph.nodes[e]["energy"] for e in clusters]),
        eta=[np.array([graph.nodes[e]["eta"] for e in inputs]) for inputs in [tracks, clusters]],
        phi=[np.array([graph.nodes[e]["phi"] for e in inputs]) for inputs in [tracks, clusters]],
        reference=ref,
        inputs=[tracks, clusters],
    )
    return bundle


def run_cms(args):
    from mlpf.data.cms import postprocessing2 as cms

    tree = uproot.open(args.input, handler=uproot.source.file.MemmapSource)["pfana/pftree"]
    ev = tree.arrays(entry_stop=args.events, library="np")
    for iev in range(min(args.events, tree.num_entries)):
        with contextlib.redirect_stdout(io.StringIO()):
            graph = cms.make_graph(ev, iev)
        results = []
        all_objects = []
        baseline = None
        for rule in RULES:
            saved = {}

            def assignment_fn(g, particles, elements):
                # Match the native sorted order; set order for particles is retained for this event.
                bundle = cms_bundle(g, particles, elements, iev)
                start = time.perf_counter()
                if rule == "native":
                    e2p = {}
                    p2e = {}
                    for typ in [6, 1, 4, 5, 8, 9, 10, 11]:
                        cms.find_representative_elements(g, e2p, p2e, typ)
                    result = np.full((len(particles), 2), -1, dtype=int)
                    lookup = {p: i for i, p in enumerate(particles)}
                    for kind, inputs in enumerate(bundle["inputs"]):
                        for j, e in enumerate(inputs):
                            if e in e2p:
                                result[lookup[e2p[e]], kind] = j
                    bundle["native"] = result
                else:
                    bundle["native"] = baseline
                    result = choose(bundle, rule)
                    e2p = {}
                    p2e = {}
                    for i, p in enumerate(particles):
                        for kind, j in enumerate(result[i]):
                            if j >= 0:
                                e = bundle["inputs"][kind][j]
                                e2p[e] = p
                                p2e[p] = e
                saved.update(bundle=bundle, assignment=result, seconds=time.perf_counter() - start)
                return e2p, p2e

            g = copy.deepcopy(graph)
            with contextlib.redirect_stdout(io.StringIO()):
                x, _, y = cms.prepare_normalized_table(g, iev, assignment_fn=assignment_fn)
            b = saved["bundle"]
            assignment = saved["assignment"]
            clean_scores(b)
            if rule == "native":
                baseline = assignment.copy()
                # Verify the callback reproduces the unmodified native path.
                if iev == 0:
                    with contextlib.redirect_stdout(io.StringIO()):
                        _, _, original = cms.prepare_normalized_table(copy.deepcopy(graph), iev)
                    for field in y.dtype.names:
                        np.testing.assert_allclose(y[field], original[field], rtol=1e-6, atol=1e-6)
            post = []
            maps = [{e: j for j, e in enumerate(inputs)} for inputs in b["inputs"]]
            elements = sorted([e for e in g if e[0] == "elem"])
            for i, e in enumerate(elements):
                if y["pid"][i] == 0:
                    continue
                kind = 0 if x["typ"][i] in [1, 6] else 1
                post.append(
                    (
                        kind,
                        maps[kind][e],
                        float(y["pt"][i]),
                        float(y["eta"][i]),
                        float(np.arctan2(y["sin_phi"][i], y["cos_phi"][i])),
                        float(y["energy"][i]),
                        float(y["ispu"][i]),
                    )
                )
            accounting = dict(merged_particles=len(assignment) - len(post), dropped_particles=0, dropped_energy=0.0)
            r, z = measure(b, assignment, post, accounting, rule, saved["seconds"])
            results.append(r)
            z["rule"] = rule
            z["event_id"] = iev
            all_objects.append(z)
        (args.output / "events" / f"{iev:05d}.json").write_text(json.dumps(results, indent=2))
        pd.concat(all_objects, ignore_index=True).to_parquet(args.output / "events" / f"{iev:05d}.parquet", index=False)
        print(f"completed cms event {iev}", flush=True)


def summarize(args):
    rows = []
    for p in sorted((args.output / "events").glob("*.json")):
        rows.extend(json.loads(p.read_text()))
    df = pd.DataFrame(rows)
    df.to_csv(args.output / "events.csv", index=False)
    results = []
    for rule, z in df.groupby("rule", sort=False):
        s = z.select_dtypes(include="number").sum()
        r = dict(
            detector=args.detector,
            rule=rule,
            events=len(z),
            track_targets=int(s.track_targets),
            cluster_targets=int(s.cluster_targets),
            track_bad_fraction=float(s.track_bad / s.track_targets),
            cluster_bad_fraction=float(s.cluster_bad / s.cluster_targets),
            cluster_pt5_bad_fraction=float(s.cluster_pt5_bad / s.cluster_pt5_count),
            mean_track_purity=float(s.track_purity_sum / s.track_matches),
            mean_cluster_purity=float(s.cluster_purity_sum / s.cluster_matches),
            exclusive_fraction=float(s.exclusive_particles / s.selected_particles),
            merged_fraction=float(s.merged_particles / s.selected_particles),
            dropped_particles=int(s.dropped_particles),
            dropped_energy=float(s.dropped_energy),
            jet_recall=float(s.matched_jets / s.reference_jets),
            jet_response_recall=float(s.response_jets / s.reference_jets),
            median_matching_seconds=float(z.matching_seconds.median()) if z.matching_seconds.notna().any() else None,
            invalid_track_columns=int(s.invalid_track_columns),
            invalid_cluster_edges=int(s.invalid_cluster_edges),
            max_energy_residual=float(z.energy_balance_residual.abs().max()),
            track_fallback_targets=int(s.track_fallback_targets),
            cluster_fallback_targets=int(s.cluster_fallback_targets),
            overlap_objective=float(s.overlap_objective),
        )
        results.append(r)
    (args.output / "summary.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(results).to_csv(args.output / "summary.csv", index=False)
    print(pd.DataFrame(results).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", choices=["cld", "clic", "idea", "cms"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=int, default=50)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "events").mkdir(exist_ok=True)
    start = time.perf_counter()
    h = hashlib.sha256()
    with args.input.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    (args.output / "inputs.json").write_text(
        json.dumps(dict(detector=args.detector, input=str(args.input), events=args.events, sha256=h.hexdigest()), indent=2)
    )
    if args.detector in ["cld", "clic"]:
        run_key4hep(args)
    elif args.detector == "idea":
        run_idea(args)
    else:
        run_cms(args)
    summarize(args)
    (args.output / "runtime.json").write_text(json.dumps(dict(wall_seconds=time.perf_counter() - start), indent=2))


if __name__ == "__main__":
    main()
