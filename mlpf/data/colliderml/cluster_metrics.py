# Truth-based quality metrics for calorimeter clusterings, detector- and clusterer-agnostic.
#
# A clustering is a per-hit label array (`cluster_of`, -1 = unclustered), scored against per-hit
# truth contributions (hit, particle, energy) that a detector-specific adapter builds, e.g.
# mlpf/data/colliderml/cluster_eval.py for ColliderML (Geant4 contributions walked to the leaf
# particles the converter targets).
#
# Notation: M[c, p] = energy particle p deposited in the hits of cluster c; T_p = p's total
# deposit over ALL hits (clustered or not); A_c = sum_p M[c, p] (the cluster's truth-attributed
# energy). Hit counts: each hit belongs to its main contributor (largest deposit); H[c, p] = hits
# of p in c, H_p = all hits of p, H_c = hits in c that belong to some particle. "Best cluster of
# p" = argmax_c M[c, p]; "dominant particle of c" = argmax_p M[c, p].
#
# Counts and sizes (no truth needed):
#   n_clusters, n_clusters_gt0p1, n_clusters_1hit   clusters per event (all / E > 0.1 GeV / one hit)
#   clu_nhit_max, clu_E_max                          largest cluster: hits / energy (GeV)
#   dropped_E_frac                                   hit energy in no cluster / all hit energy
#   n_removed_hits                                   hits in no cluster ("removed"), per event
#   n_tracks, n_visible, clusters_per_visible, elements_per_visible   (with n_tracks/n_visible)
#
# Removed hits vs the targets (needs truth["target"]: a per-row mask of the target particles,
# i.e. the converter's visible particles, whose energy ends up in a target, merged or not):
#   n_removed_hits_target     removed hits whose main contributor is a target particle, per event
#   removed_hits_target_frac  n_removed_hits_target / n_removed_hits
#   removed_target_E_frac     the targets' deposit in removed hits / the targets' total deposit
#   n_targets_lost_best       targets (with a deposit) that lost their best cluster to the removal:
#                             more of their deposit sits in removed hits than in any kept cluster
#                             (removed hits counted as one pseudo-cluster), per event
#   targets_lost_best_frac    n_targets_lost_best / targets with a deposit
#   n_targets_lost_all        targets with a deposit that keep none of it in any cluster, per event
#   targets_lost_all_frac     n_targets_lost_all / targets with a deposit
#
# Energy sharing without any matching:
#   purity_E        sum_c max_p M / sum_c A_c    (every cluster vs its dominant particle)
#   completeness_E  sum_p max_c M / sum_p T_p    (every particle vs its best cluster)
#
# Clustering quality under a matching rule (calorimeter only; particles with T_p > 1 GeV,
# clusters with E > 1 GeV for the fake rate):
#   rule "dm"    double majority: c and p match if c is p's best cluster, p is c's dominant
#                particle, M[c, p] >= 0.5 A_c and M[c, p] >= 0.5 T_p (one-to-one; standard in
#                ML tracking)
#   rule "alloc" the postprocessing allocator's cluster step (target_building
#                .assign_genparticles_to_obj_and_merge without tracks): particles in decreasing
#                true energy each take their best still-free cluster (by M, any share)
#   eff_<rule>          matched particles / particles
#   eff_dm_hs           eff_dm for hard-scatter particles only
#   eff_dm_E[_hs]       eff_dm weighted by deposit: sum T_p (matched) / sum T_p
#   fake_<rule>         clusters not matched to any particle / clusters
#   purity_<rule>_{E,hits}   over matched pairs: sum M / sum A_c   |  sum H / sum H_c
#   recall_<rule>_{E,hits}   over matched pairs: sum M / sum T_p   |  sum H / sum H_p
#
# Training view, the full postprocessing allocator (tracks first, then clusters; all visible
# particles, i.e. the converter's target candidates), from pp_metrics:
#   eff_pp              visible particles that get a track or cluster of their own / visible
#   fake_pp             elements without a target / elements (clusters E > 1 GeV, tracks p > 1 GeV)
#   purity_pp_clu_{E,hits}, recall_pp_clu_{E,hits}   as above, over cluster-assigned pairs
#   purity_pp_trk_hits  over track-assigned pairs: sum (p's hits on the track) / sum (track hits)
#   recall_pp_trk_hits  over track-assigned pairs: sum (p's hits on the track) / sum (p's tracker hits)
#   (tracks have no energy-based version: ColliderML tracker hits carry no energy)
# and from allocator_metrics: n_targets, merged_frac[_E,_hs], n_massive,
# clusters_per_target, elements_per_target.
#
# Distributions for plots (cluster_metrics(..., dists=True) -> out["_dist"]; `aggregate` skips it):
#   per cluster:   clu_nhit   number of hits
#                  clu_E      energy (GeV)
#                  clu_dm     matched under the double majority (not matched = fake)
#                  clu_alloc  matched by the allocator, clusters only
#   per particle with T_p >= DIST_PARTICLE_MIN_E:
#                  par_T      calorimeter deposit T_p (GeV)
#                  par_dm     matched under the double majority (efficiency)
#                  par_alloc  matched by the allocator, clusters only
import contextlib
import io
from typing import Any, Dict, List, Tuple

import numpy as np

from mlpf.data.target_building import EventData, assign_genparticles_to_obj_and_merge

# Bump whenever a metric's definition changes: caches of stored metric results should be keyed
# on it, so they are recomputed instead of silently reused.
METRICS_VERSION = "2026-10-13.5"

PARTICLE_MIN_E = 1.0  # GeV of deposit, particles counted in the rule-based efficiencies/purities
CLUSTER_MIN_E = 1.0  # GeV, clusters counted in the fake rates
TRACK_MIN_P = 1.0  # GeV, tracks counted in the training-view fake rate
DIST_PARTICLE_MIN_E = 0.1  # GeV of deposit, particles kept in the per-particle distributions


def _first_per_group(group: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Index of the max-value entry per group (ties: first in input order)."""
    order = np.lexsort((np.arange(len(group)), -value, group))
    g = group[order]
    first = np.ones(len(g), dtype=bool)
    first[1:] = g[1:] != g[:-1]
    return order[first]


def _sum_pairs(a: np.ndarray, b: np.ndarray, w: np.ndarray, n_b: int):
    """Sum duplicate (a, b) pairs: returns unique a, b and summed w (sorted by a, then b)."""
    key = np.asarray(a, np.int64) * np.int64(max(n_b, 1)) + np.asarray(b, np.int64)
    ukey, inv = np.unique(key, return_inverse=True)
    return ukey // max(n_b, 1), ukey % max(n_b, 1), np.bincount(inv, weights=np.asarray(w, np.float64), minlength=len(ukey))


def _candidates(p: np.ndarray, o: np.ndarray, w: np.ndarray, n_p: int):
    """CSR of each particle's objects with positive weight, sorted by weight desc, then index
    asc: the allocator's _row_sorted_by_weight order."""
    keep = w > 0
    p, o, w = p[keep], o[keep], w[keep]
    order = np.lexsort((o, -w, p))
    ptr = np.zeros(n_p + 1, np.int64)
    np.cumsum(np.bincount(p, minlength=n_p), out=ptr[1:])
    return ptr, o[order]


def greedy_assign(energy: np.ndarray, cand_lists, n_objs) -> List[np.ndarray]:
    """The allocator's assignment loop: particles in decreasing energy (ties: index asc) take
    the first still-free object of the first candidate list that has one (e.g. tracks, then
    clusters). Returns, per list, each particle's object index (-1 if none)."""
    order = np.lexsort((np.arange(len(energy)), -np.asarray(energy, np.float64)))
    owner = [np.full(len(energy), -1, np.int64) for _ in cand_lists]
    used = [np.zeros(n, bool) for n in n_objs]
    for i in order:
        for k, (ptr, objs) in enumerate(cand_lists):
            for o in objs[ptr[i] : ptr[i + 1]]:
                if not used[k][o]:
                    used[k][o] = True
                    owner[k][i] = o
                    break
            if owner[k][i] >= 0:
                break
    return owner


def _pair_sums(pairs_c, pairs_p, Mw_of, A, T, Hw_of, Hc, Hp):
    """purity / recall numerators and denominators over matched (cluster, particle) pairs."""
    m = np.array([Mw_of.get((c, p), 0.0) for c, p in zip(pairs_c, pairs_p)])
    h = np.array([Hw_of.get((c, p), 0) for c, p in zip(pairs_c, pairs_p)], dtype=np.float64)
    return {
        "purity_E": (m.sum(), A[pairs_c].sum()),
        "recall_E": (m.sum(), T[pairs_p].sum()),
        "purity_hits": (h.sum(), Hc[pairs_c].sum()),
        "recall_hits": (h.sum(), Hp[pairs_p].sum()),
    }


def cluster_metrics(
    cluster_of: np.ndarray,
    hit_e: np.ndarray,
    truth: Dict[str, np.ndarray],
    n_tracks: int | None = None,
    n_visible: int | None = None,
    dists: bool = False,
) -> Dict[str, Any]:
    """Metrics of one event's clustering; see the module header for definitions.

    truth: hit/leaf/w COO, n_rows, ispu, and `energy` (true particle energy per row, for the
    allocator ordering); optional `target` (bool per row) enables the removed-hit metrics.
    n_tracks / n_visible, if both given, enable the per-visible ratios. dists=True adds the
    per-cluster / per-particle arrays under "_dist".
    Returns per-event values plus `_num`/`_den` pairs that `aggregate` sums over events.
    """
    cluster_of = np.asarray(cluster_of, dtype=np.int64)
    hit_e = np.asarray(hit_e, dtype=np.float64)
    out: Dict[str, float] = {}
    n_cl = int(cluster_of.max()) + 1 if len(cluster_of) and cluster_of.max() >= 0 else 0
    inc = cluster_of >= 0
    cl_E = np.bincount(cluster_of[inc], weights=hit_e[inc], minlength=n_cl)
    cl_n = np.bincount(cluster_of[inc], minlength=n_cl)
    out["n_clusters"] = n_cl
    out["n_clusters_gt0p1"] = int(np.sum(cl_E > 0.1))
    out["n_clusters_1hit"] = int(np.sum(cl_n == 1))
    out["clu_nhit_max"] = int(cl_n.max()) if n_cl else 0
    out["clu_E_max"] = float(cl_E.max()) if n_cl else 0.0
    out["dropped_E_frac_num"] = float(hit_e[~inc].sum())
    out["dropped_E_frac_den"] = float(hit_e.sum())
    out["n_removed_hits"] = int(np.sum(~inc))
    if n_tracks is not None and n_visible is not None:
        out["n_tracks"] = int(n_tracks)
        out["n_visible"] = int(n_visible)
        out["clusters_per_visible_num"] = n_cl
        out["clusters_per_visible_den"] = int(n_visible)
        out["elements_per_visible_num"] = n_cl + int(n_tracks)
        out["elements_per_visible_den"] = int(n_visible)

    # ---- contingency: energy M[c, p] (clustered hits) and hit counts H[c, p]
    hit, leaf, w = truth["hit"], truth["leaf"], truth["w"]
    n_rows = truth["n_rows"]
    T = np.bincount(leaf, weights=w, minlength=n_rows)
    c = cluster_of[hit]
    m = c >= 0
    Mc, Mp, Mw = _sum_pairs(c[m], leaf[m], w[m], n_rows)
    A = np.bincount(Mc, weights=Mw, minlength=n_cl)
    owner_idx = _first_per_group(hit, w)  # each hit's main contributor
    h_hit, h_p = hit[owner_idx], leaf[owner_idx]
    Hp = np.bincount(h_p, minlength=n_rows).astype(np.float64)
    hc = cluster_of[h_hit]
    hm = hc >= 0
    Hc_, Hp_, Hw = _sum_pairs(hc[hm], h_p[hm], np.ones(int(hm.sum())), n_rows)
    Hc = np.bincount(hc[hm], minlength=n_cl).astype(np.float64)
    Mw_of = dict(zip(zip(Mc.tolist(), Mp.tolist()), Mw.tolist()))
    Hw_of = dict(zip(zip(Hc_.tolist(), Hp_.tolist()), Hw.tolist()))

    # ---- energy sharing without matching
    dom = _first_per_group(Mc, Mw)  # dominant particle per (non-empty) cluster
    best = _first_per_group(Mp, Mw)  # best cluster per particle
    out["purity_E_num"] = float(Mw[dom].sum())
    out["purity_E_den"] = float(A.sum())
    out["completeness_E_num"] = float(Mw[best].sum())
    out["completeness_E_den"] = float(T.sum())

    # ---- removed hits (in no cluster) vs the target particles
    tgt = truth.get("target")
    if tgt is not None:
        tgt = np.asarray(tgt, bool)
        n_from = int(np.sum(~inc[h_hit] & tgt[h_p]))  # removed hits whose main contributor is a target
        out["n_removed_hits_target"] = n_from
        out["removed_hits_target_frac_num"] = n_from
        out["removed_hits_target_frac_den"] = out["n_removed_hits"]
        tw = tgt[leaf]
        out["removed_target_E_frac_num"] = float(w[tw & ~m].sum())
        out["removed_target_E_frac_den"] = float(w[tw].sum())
        # targets losing their best cluster: their removed deposit beats every kept cluster
        R = np.bincount(leaf[~m], weights=w[~m], minlength=n_rows)
        best_kept = np.zeros(n_rows)
        best_kept[Mp[best]] = Mw[best]
        has = tgt & (T > 0)
        lost = has & (R > best_kept)
        n_has = int(has.sum())
        out["n_targets_lost_best"] = out["targets_lost_best_frac_num"] = int(lost.sum())
        out["n_targets_lost_all"] = out["targets_lost_all_frac_num"] = int(np.sum(has & (best_kept == 0)))
        out["targets_lost_best_frac_den"] = out["targets_lost_all_frac_den"] = n_has

    # ---- rule-based matching
    ref = T > PARTICLE_MIN_E
    big = cl_E > CLUSTER_MIN_E
    dom_c = np.full(n_cl, -1, np.int64)
    dom_c[Mc[dom]] = Mp[dom]
    bp, bc, bw = Mp[best], Mc[best], Mw[best]
    dm = (dom_c[bc] == bp) & (bw >= 0.5 * T[bp]) & (bw >= 0.5 * np.maximum(A[bc], 1e-12))
    pairs = {"dm": (bc[dm], bp[dm])}
    ptr, objs = _candidates(Mp, Mc, Mw, n_rows)
    (owner,) = greedy_assign(truth["energy"], [(ptr, objs)], [n_cl])
    ap = np.nonzero(owner >= 0)[0]
    pairs["alloc"] = (owner[ap], ap)
    ispu = np.asarray(truth["ispu"], bool)
    matched = {}  # rule -> (matched particles, matched clusters)
    for rule, (pc, pp) in pairs.items():
        matched_p = np.zeros(n_rows, bool)
        matched_p[pp] = True
        matched_c = np.zeros(n_cl, bool)
        matched_c[pc] = True
        matched[rule] = (matched_p, matched_c)
        out[f"eff_{rule}_num"] = int(np.sum(ref & matched_p))
        out[f"eff_{rule}_den"] = int(np.sum(ref))
        out[f"fake_{rule}_num"] = int(np.sum(big & ~matched_c))
        out[f"fake_{rule}_den"] = int(np.sum(big))
        sel = ref[pp]
        for k, (num, den) in _pair_sums(pc[sel], pp[sel], Mw_of, A, T, Hw_of, Hc, Hp).items():
            name = k.replace("_", f"_{rule}_")
            out[f"{name}_num"] = float(num)
            out[f"{name}_den"] = float(den)
        if rule == "dm":
            out["eff_dm_hs_num"] = int(np.sum(ref & matched_p & ~ispu))
            out["eff_dm_hs_den"] = int(np.sum(ref & ~ispu))
            # energy-weighted: each particle counts by its deposit T_p
            out["eff_dm_E_num"] = float(T[ref & matched_p].sum())
            out["eff_dm_E_den"] = float(T[ref].sum())
            out["eff_dm_E_hs_num"] = float(T[ref & matched_p & ~ispu].sum())
            out["eff_dm_E_hs_den"] = float(T[ref & ~ispu].sum())

    if dists:
        sel = T >= DIST_PARTICLE_MIN_E
        out["_dist"] = dict(
            clu_nhit=cl_n.astype(np.int32),
            clu_E=cl_E.astype(np.float32),
            clu_dm=matched["dm"][1],
            clu_alloc=matched["alloc"][1],
            par_T=T[sel].astype(np.float32),
            par_dm=matched["dm"][0][sel],
            par_alloc=matched["alloc"][0][sel],
        )
    return out


def pp_metrics(
    gen_energy: np.ndarray,
    gp_trk: Tuple[np.ndarray, np.ndarray, np.ndarray],
    gp_calo: Tuple[np.ndarray, np.ndarray, np.ndarray],
    gp_tracker_nhits: np.ndarray,
    cluster_of: np.ndarray,
    hit_e: np.ndarray,
    track_nhits: np.ndarray,
    track_p: np.ndarray,
) -> Dict[str, float]:
    """Training-view metrics: the full postprocessing allocator (tracks first, then clusters)
    on the visible particles. See the module header.

    gen_energy: true energy per visible particle. gp_trk: (particle, track, share of the track's
    hits). gp_calo: (particle, calo hit, deposit in GeV), calo hits indexed like cluster_of.
    gp_tracker_nhits: tracker hits per particle. track_nhits / track_p: per track.
    """
    n_gp = len(gen_energy)
    n_trk = len(track_nhits)
    cluster_of = np.asarray(cluster_of, np.int64)
    n_cl = int(cluster_of.max()) + 1 if len(cluster_of) and cluster_of.max() >= 0 else 0
    hit_e = np.asarray(hit_e, np.float64)
    inc = cluster_of >= 0
    cl_E = np.bincount(cluster_of[inc], weights=hit_e[inc], minlength=n_cl)

    tp, tt, tw = _sum_pairs(*gp_trk, n_trk) if len(gp_trk[0]) else (np.zeros(0, np.int64),) * 2 + (np.zeros(0),)
    gp, gh, gw = (np.asarray(a) for a in gp_calo)
    gp, gh, gw = gp.astype(np.int64), gh.astype(np.int64), gw.astype(np.float64)
    c = cluster_of[gh]
    m = c >= 0
    Mp, Mc, Mw = _sum_pairs(gp[m], c[m], gw[m], n_cl)
    T = np.bincount(gp, weights=gw, minlength=n_gp)
    A = np.bincount(Mc, weights=Mw, minlength=n_cl)
    own_trk, own_clu = greedy_assign(gen_energy, [_candidates(tp, tt, tw, n_gp), _candidates(Mp, Mc, Mw, n_gp)], [n_trk, n_cl])

    out: Dict[str, float] = {}
    has = (own_trk >= 0) | (own_clu >= 0)
    out["eff_pp_num"] = int(has.sum())
    out["eff_pp_den"] = n_gp
    clu_used = np.zeros(n_cl, bool)
    clu_used[own_clu[own_clu >= 0]] = True
    trk_used = np.zeros(n_trk, bool)
    trk_used[own_trk[own_trk >= 0]] = True
    big_c, big_t = cl_E > CLUSTER_MIN_E, np.asarray(track_p) > TRACK_MIN_P
    out["fake_pp_num"] = int(np.sum(big_c & ~clu_used) + np.sum(big_t & ~trk_used))
    out["fake_pp_den"] = int(big_c.sum() + big_t.sum())

    # cluster-assigned pairs: energy and hit counts (each calo hit belongs to its main contributor)
    owner_idx = _first_per_group(gh, gw)
    h_hit, h_p = gh[owner_idx], gp[owner_idx]
    Hp = np.bincount(h_p, minlength=n_gp).astype(np.float64)
    hc = cluster_of[h_hit]
    hm = hc >= 0
    Hpp, Hcc, Hw = _sum_pairs(h_p[hm], hc[hm], np.ones(int(hm.sum())), n_cl)
    Hc = np.bincount(hc[hm], minlength=n_cl).astype(np.float64)
    pcl = np.nonzero(own_clu >= 0)[0]
    sums = _pair_sums(
        own_clu[pcl],
        pcl,
        dict(zip(zip(Mc.tolist(), Mp.tolist()), Mw.tolist())),
        A,
        T,
        dict(zip(zip(Hcc.tolist(), Hpp.tolist()), Hw.tolist())),
        Hc,
        Hp,
    )
    for k, (num, den) in sums.items():
        name = k.replace("_", "_pp_clu_")
        out[f"{name}_num"] = float(num)
        out[f"{name}_den"] = float(den)

    # track-assigned pairs: hits
    ptk = np.nonzero(own_trk >= 0)[0]
    share = dict(zip(zip(tp.tolist(), tt.tolist()), tw.tolist()))
    nh = np.asarray(track_nhits, np.float64)
    on_track = np.array([share[(p, t)] * nh[t] for p, t in zip(ptk.tolist(), own_trk[ptk].tolist())])
    out["purity_pp_trk_hits_num"] = float(on_track.sum())
    out["purity_pp_trk_hits_den"] = float(nh[own_trk[ptk]].sum())
    out["recall_pp_trk_hits_num"] = float(on_track.sum())
    out["recall_pp_trk_hits_den"] = float(np.asarray(gp_tracker_nhits, np.float64)[ptk].sum())
    out["_n_with_element"] = int(has.sum())  # for the parity check against the allocator
    return out


def allocator_metrics(gd: EventData, n_cluster: int, n_track: int) -> Dict[str, float]:
    """Run the shared target allocator on one event and summarise what it does to the targets.

    gd: the event's EventData with the clustering under test as hit_to_cluster (gen_features =
    visible particles before allocation). Returns n_targets (the targets after merging: what the
    model predicts), merged_frac[_E,_hs] (share of targets merged into a host for lack of a free
    track/cluster: count / energy / hard scatter only), n_massive (targets with mass > 5 GeV
    after the 4-vector merges), clusters_per_target = clusters / targets and
    elements_per_target = (clusters + tracks) / targets.
    """
    gen = gd.gen_features
    with contextlib.redirect_stdout(io.StringIO()):
        cleaned = assign_genparticles_to_obj_and_merge(gd)[0]
    e_in = np.asarray(gen["energy"], np.float64)
    keys = gen.fields if hasattr(gen, "fields") else gen.keys()
    ispu = np.asarray(gen["ispu"]) > 0 if "ispu" in keys else np.zeros(len(e_in), bool)
    merged = np.zeros(len(e_in), bool)
    if len(cleaned.gp_merges[1]):
        merged[np.asarray(cleaned.gp_merges[1], np.int64)] = True
    f = cleaned.gen_features
    e, pt, eta = (np.asarray(f[k], np.float64) for k in ("energy", "pt", "eta"))
    mass = np.sqrt(np.maximum(e**2 - (pt * np.cosh(eta)) ** 2, 0.0))
    return {
        "n_targets": len(e),
        "merged_frac_num": int(merged.sum()),
        "merged_frac_den": len(e_in),
        "merged_frac_E_num": float(e_in[merged].sum()),
        "merged_frac_E_den": float(e_in.sum()),
        "merged_frac_hs_num": int((merged & ~ispu).sum()),
        "merged_frac_hs_den": int((~ispu).sum()),
        "n_massive": int(np.sum(mass > 5.0)),
        "clusters_per_target_num": n_cluster,
        "clusters_per_target_den": len(e),
        "elements_per_target_num": n_cluster + n_track,
        "elements_per_target_den": len(e),
    }


def aggregate(per_event: List[Dict[str, float]]) -> Dict[str, float]:
    """Sum `_num`/`_den` pairs over events; max for `_max` keys; mean for the rest. Keys
    starting with "_" are internal (not aggregated)."""
    agg: Dict[str, float] = {}
    for k in per_event[0].keys():
        if k.startswith("_") or k.endswith("_den"):
            continue
        if k.endswith("_num"):
            base = k[:-4]
            num = sum(e[k] for e in per_event)
            den = sum(e[base + "_den"] for e in per_event)
            agg[base] = num / den if den else np.nan
        elif k.endswith("_max"):
            agg[k] = float(np.max([e[k] for e in per_event]))
        else:
            vals = [e[k] for e in per_event]
            agg[k] = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else np.nan
    return agg
