# Detector-/source-independent machinery for turning per-event truth-particle assignments into
# MLPF per-element target arrays. Extracted from mlpf/data/key4hep/postprocessing.py; that module
# re-imports all of these, so CLIC/CLD behaviour and tests are untouched. Most functions moved
# over unchanged; three deliberate differences:
#   - assign_genparticles_to_obj_and_merge keeps the key4hep allocation logic verbatim but works
#     on sparse accessors instead of dense n_gp x n_obj matrices (bit-identical output, bounded
#     memory for high-pileup events)
#   - the visibility cut is a shared callable (visibility_mask), wrapping the formula that used
#     to be inline in the key4hep converter
#   - compute_jets takes jetdef/min_pt as parameters instead of reading key4hep module globals
#     (ColliderML runs anti-kt R=0.4; key4hep ee runs ee_genkt)
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import awkward
import fastjet
import numpy as np
import vector
from scipy.sparse import coo_matrix

# Type aliases for clarity
SparseMatrixCOO = Tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass
class EventData:
    gen_features: awkward.Record  # feature matrix of the genparticles
    hit_features: awkward.Record  # feature matrix of the calo hits
    cluster_features: awkward.Record  # feature matrix of the calo clusters
    track_features: awkward.Record  # feature matrix of the tracks
    genparticle_to_hit: SparseMatrixCOO  # sparse COO matrix of genparticles to hits (idx_gp, idx_hit, weight)
    genparticle_to_track: SparseMatrixCOO  # sparse COO matrix of genparticles to tracks (idx_gp, idx_track, weight)
    hit_to_cluster: SparseMatrixCOO  # sparse COO matrix of hits to clusters (idx_hit, idx_cluster, weight)
    gp_merges: Tuple[np.ndarray, np.ndarray]  # sparse COO matrix of any merged genparticles

    def __post_init__(self):
        self.genparticle_to_hit = (
            np.array(self.genparticle_to_hit[0]),
            np.array(self.genparticle_to_hit[1]),
            np.array(self.genparticle_to_hit[2]),
        )
        self.genparticle_to_track = (
            np.array(self.genparticle_to_track[0]),
            np.array(self.genparticle_to_track[1]),
            np.array(self.genparticle_to_track[2]),
        )
        self.hit_to_cluster = (
            np.array(self.hit_to_cluster[0]),
            np.array(self.hit_to_cluster[1]),
            np.array(self.hit_to_cluster[2]),
        )
        self.gp_merges = (np.array(self.gp_merges[0]), np.array(self.gp_merges[1]))


def map_pdgid_to_candid(pdgid: int, charge: float) -> int:
    if pdgid == 0:
        return 0

    # photon, electron, muon
    if pdgid in [22, 11, 13]:
        return pdgid

    # charged hadron
    if abs(charge) > 0:
        return 211

    # neutral hadron
    return 130


def map_charged_to_neutral(pdg: int) -> int:
    if pdg == 0:
        return 0
    if pdg == 11 or pdg == 22:
        return 22
    return 130


def map_neutral_to_charged(pdg: int) -> int:
    if pdg == 130 or pdg == 22:
        return 211
    return pdg


def sanitize(arr: np.ndarray) -> None:
    arr[np.isnan(arr)] = 0.0
    arr[np.isinf(arr)] = 0.0


# https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
def weighted_avg_and_std(values: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    """
    Return the weighted average and standard deviation.

    They weights are in effect first normalized so that they
    sum to 1 (and so they must not all be 0).

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights)
    return (average, math.sqrt(variance))


def filter_adj(adj: SparseMatrixCOO, all_to_filtered: Dict[int, int]) -> SparseMatrixCOO:
    i0s_new = []
    i1s_new = []
    ws_new = []
    for i0, i1, w in zip(*adj):
        if i0 in all_to_filtered:
            i0_new = all_to_filtered[i0]
            i0s_new.append(i0_new)
            i1s_new.append(i1)
            ws_new.append(w)
    return np.array(i0s_new), np.array(i1s_new), np.array(ws_new)


# Per-particle visibility thresholds shared by all target builders (#463); moved verbatim from
# mlpf/data/key4hep/postprocessing.py
visible_energy_fraction = 0.10
visible_energy_deposit = 0.5  # GeV


def visibility_mask(energy_in_hits: np.ndarray, energy: np.ndarray, charged: np.ndarray, in_tracker: np.ndarray | None = None) -> np.ndarray:
    """Hit-based visibility mask for target particles (#463).

    The hit weights are calibrated hit energies in GeV, so `energy_in_hits` is what the
    particle actually deposited and the first term is a true energy fraction.

    The fractional term alone is a shower-containment criterion and so misses MIPs: a muon
    deposits a roughly constant few GeV whatever its momentum, so its fraction falls with
    energy. The absolute term recovers these. The tracker term (`in_tracker`) does not help
    the muons (none have a track link above threshold), but it does recover charged hadrons
    whose calorimeter deposit falls below the absolute threshold (e.g. low-momentum pions and
    kaons decaying in the tracker): their momentum is measured by the tracker even though
    their shower is sub-threshold.

    The absolute term exists to catch MIPs, which are charged by definition, so it is
    restricted to charged particles. A neutral that deposits more than the threshold but less
    than the fractional cut is one that leaked: nothing measured its momentum and it has no
    track to supply it, so asking the model to predict its energy is asking for something
    unlearnable.

    The absolute term is a detector-dependent scale: it has to sit below the MIP peak and above
    the noise, and the MIP peak scales with calorimeter depth. Median muon deposit is ~4 GeV on
    MAIA but only ~2 GeV on CLD, so it is set to 0.5 GeV.

    Hit scope note: key4hep callers sum over tracker and calorimeter hits alike; ColliderML
    passes calo-only sums (its tracker_hits table has no energy field).
    """
    energy_in_hits = np.asarray(energy_in_hits, dtype=np.float64)
    energy = np.asarray(energy, dtype=np.float64)
    charged = np.asarray(charged, dtype=bool)
    mask = (energy_in_hits / np.maximum(energy, 1e-9) > visible_energy_fraction) | ((energy_in_hits > visible_energy_deposit) & charged)
    if in_tracker is not None:
        mask = mask | np.asarray(in_tracker, dtype=bool)
    return mask


# for each PF element (track, cluster), get the index of the best-matched particle (gen or reco)
# if the PF element has no best-matched particle, returns -1
def assign_to_recoobj(n_obj: int, obj_to_ptcl: Dict[int, int], used_particles: np.ndarray) -> np.ndarray:
    obj_to_ptcl_all = -1 * np.ones(n_obj, dtype=np.int64)
    for iobj in range(n_obj):
        if iobj in obj_to_ptcl:
            iptcl = obj_to_ptcl[iobj]
            obj_to_ptcl_all[iobj] = iptcl
            assert used_particles[iptcl] == 0
            used_particles[iptcl] = 1
    return obj_to_ptcl_all


def get_particle_feature_matrix(pfelem_to_particle: np.ndarray, feature_dict: Any, features: List[str]) -> np.ndarray:
    feats = []
    for feat in features:
        feat_arr = feature_dict[feat]
        if len(feat_arr) == 0:
            feat_arr_reordered = np.zeros(len(pfelem_to_particle))
        else:
            feat_arr_reordered = awkward.to_numpy(feat_arr[pfelem_to_particle])
            feat_arr_reordered[pfelem_to_particle == -1] = 0.0
        feats.append(feat_arr_reordered)
    feats = np.array(feats)
    return feats.T


def get_feature_matrix(feature_dict: Any, features: List[str]) -> np.ndarray:
    feats = []
    for feat in features:
        feat_arr = awkward.to_numpy(feature_dict[feat])
        feats.append(feat_arr)
    feats = np.array(feats)
    return feats.T


def compute_met(p4: Any) -> np.ndarray:
    sum_px = awkward.sum(p4.px, axis=1)
    sum_py = awkward.sum(p4.py, axis=1)
    met = np.sqrt(sum_px**2 + sum_py**2)
    return met


def compute_jets(particles_p4: Any, min_pt: float, jetdef: fastjet.JetDefinition, with_indices: bool = False) -> Any:
    cluster = fastjet.ClusterSequence(particles_p4, jetdef)
    jets = vector.awk(cluster.inclusive_jets(min_pt=min_pt))
    jets = vector.awk(awkward.zip({"energy": jets["t"], "px": jets["x"], "py": jets["y"], "pz": jets["z"]}))
    jets = awkward.Array({"pt": jets.pt, "eta": jets.eta, "phi": jets.phi, "energy": jets.energy})
    ret = jets
    if with_indices:
        indices = cluster.constituent_index(min_pt=min_pt)
        ret = jets, indices
    return ret


def assign_genparticles_to_obj_and_merge(  # noqa: C901
    gpdata: EventData,
) -> Tuple[EventData, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_gp = awkward.count(gpdata.gen_features["PDG"])
    n_track = awkward.count(gpdata.track_features["type"])
    n_hit = awkward.count(gpdata.hit_features["type"])
    n_cluster = awkward.count(gpdata.cluster_features["type"])

    # Sparse accessors instead of the old dense n_gp x n_obj weight matrices, which do not fit
    # in memory for high-multiplicity (pileup) events. The helpers below reproduce the dense
    # code's output exactly: column argmax with -1 for all-zero columns, and each row's
    # entries sorted by descending weight (ties by index). Duplicate COO coordinates occur in
    # real data (a particle re-depositing in the same calo cell), and float addition is not
    # associative: summing duplicates in encounter order before CSR construction keeps the
    # result bit-identical to the old dense accumulation.
    def _stable_sum_dup_coo(row, col, val, n_cols):
        row = np.asarray(row, dtype=np.int64)
        col = np.asarray(col, dtype=np.int64)
        val = np.asarray(val)
        if len(val) == 0:
            return row, col, val
        key = row * np.int64(n_cols) + col
        order = np.argsort(key, kind="stable")  # stable: within a pair, encounter order kept
        ks, rs, cs, vs = key[order], row[order], col[order], val[order]
        starts = np.concatenate([[0], np.nonzero(np.diff(ks))[0] + 1])
        # np.add.reduceat uses pairwise summation -> NOT bit-identical to todense's sequential
        # accumulate. Fold instead: acc_g += k-th entry, one rounded add at a time.
        lengths = np.diff(np.concatenate([starts, [len(vs)]]))
        acc = vs[starts].copy()
        mult = int(lengths.max())
        for k in range(1, mult):
            has = lengths > k
            acc[has] = acc[has] + vs[starts[has] + k]
        return rs[starts], cs[starts], acc

    r, c, v = _stable_sum_dup_coo(gpdata.genparticle_to_track[0], gpdata.genparticle_to_track[1], gpdata.genparticle_to_track[2], n_track)
    gp_to_track = coo_matrix((v, (r, c)), shape=(n_gp, n_track)).tocsr()
    gp_to_track.eliminate_zeros()

    r, c, v = _stable_sum_dup_coo(gpdata.genparticle_to_hit[0], gpdata.genparticle_to_hit[1], gpdata.genparticle_to_hit[2], n_hit)
    gp_to_hit = coo_matrix((v, (r, c)), shape=(n_gp, n_hit)).tocsr()
    gp_to_hit.eliminate_zeros()

    calohit_to_cluster = coo_matrix(
        (gpdata.hit_to_cluster[2], (gpdata.hit_to_cluster[0], gpdata.hit_to_cluster[1])), shape=(n_hit, n_cluster)
    ).tocsr()

    gp_to_cluster = (gp_to_hit @ calohit_to_cluster).tocsr()
    gp_to_cluster.eliminate_zeros()

    def _col_argmax_and_mask(m_sp):
        csc = m_sp.tocsc()
        out = -1 * np.ones(m_sp.shape[1], dtype=np.int32)
        for j in np.nonzero(np.diff(csc.indptr))[0]:
            s, e = csc.indptr[j], csc.indptr[j + 1]
            out[j] = csc.indices[s + np.argmax(csc.data[s:e])]
        return out

    def _row_sorted_by_weight(m_sp, i):
        s, e = m_sp.indptr[i], m_sp.indptr[i + 1]
        idx, val = m_sp.indices[s:e], m_sp.data[s:e]
        order = np.lexsort((idx, -val))
        return idx[order], val[order]

    # Inclusive mapping: for each track/cluster/hit, find the genparticle that contributes the most weight
    track_to_gp_inclusive = -1 * np.ones(n_track, dtype=np.int32)
    if n_gp > 0 and n_track > 0:
        track_to_gp_inclusive = _col_argmax_and_mask(gp_to_track)

    cluster_to_gp_inclusive = -1 * np.ones(n_cluster, dtype=np.int32)
    if n_gp > 0 and n_cluster > 0:
        cluster_to_gp_inclusive = _col_argmax_and_mask(gp_to_cluster)

    hit_to_gp_inclusive = -1 * np.ones(n_hit, dtype=np.int32)
    if n_gp > 0 and n_hit > 0:
        hit_to_gp_inclusive = _col_argmax_and_mask(gp_to_hit)

    # map each genparticle to a track or a cluster
    gp_to_obj = -1 * np.ones((n_gp, 2), dtype=np.int32)
    set_used_tracks = set([])
    set_used_clusters = set([])
    gps_sorted_energy = sorted(range(n_gp), key=lambda x: gpdata.gen_features["energy"][x], reverse=True)

    for igp in gps_sorted_energy:

        # first check if we can match the genparticle to a track
        trks, _ = _row_sorted_by_weight(gp_to_track, igp)
        for trk in trks:
            # if the track was not already used for something else
            if trk not in set_used_tracks:
                gp_to_obj[igp, 0] = trk
                set_used_tracks.add(trk)
                break

        # if there was no matched track, try a cluster
        if gp_to_obj[igp, 0] == -1:
            clusters, _ = _row_sorted_by_weight(gp_to_cluster, igp)
            for cl in clusters:
                if cl not in set_used_clusters:
                    gp_to_obj[igp, 1] = cl
                    set_used_clusters.add(cl)
                    break

    # assign genparticle to hit separately
    # we use a set to ensure each genparticle is assigned to a unique hit
    # this prevents errors where multiple genparticles are mapped to the same hit
    gp_to_hit_idx = -1 * np.ones(n_gp, dtype=np.int32)
    set_used_hits = set([])
    for igp in gps_sorted_energy:
        hits, _ = _row_sorted_by_weight(gp_to_hit, igp)
        for ihit in hits:
            if ihit not in set_used_hits:
                gp_to_hit_idx[igp] = ihit
                set_used_hits.add(ihit)
                break

    # the genparticles that could not be matched to a track or cluster are merged to the closest genparticle
    unmatched = np.where((gp_to_obj[:, 0] == -1) & (gp_to_obj[:, 1] == -1))[0]
    mask_gp_unmatched = np.ones(n_gp, dtype=bool)

    pt_arr = np.array(awkward.to_numpy(gpdata.gen_features["pt"]))
    eta_arr = np.array(awkward.to_numpy(gpdata.gen_features["eta"]))
    phi_arr = np.array(awkward.to_numpy(gpdata.gen_features["phi"]))
    energy_arr = np.array(awkward.to_numpy(gpdata.gen_features["energy"]))

    # now merge unmatched genparticles to their closest genparticle
    gp_merges_gp0 = []
    gp_merges_gp1 = []
    dropped_gps = []  # indices removed from the target because they have no track/cluster host
    dropped_energy = 0.0
    for igp_unmatched in unmatched:
        mask_gp_unmatched[igp_unmatched] = False

        # find closest cluster that this particle is matched to
        s, e = gp_to_cluster.indptr[igp_unmatched], gp_to_cluster.indptr[igp_unmatched + 1]
        if n_cluster > 0 and e > s:
            idx_best_cluster = gp_to_cluster.indices[s + np.argmax(gp_to_cluster.data[s:e])]
            # get the first genparticle matched to that cluster
            idx_gp_bestcluster = np.where(gp_to_obj[:, 1] == idx_best_cluster)[0]
        else:
            idx_gp_bestcluster = []

        # If the genparticle is not matched to any cluster, then it left a few hits to some other
        # track. This happens only for low-pT particles with no calorimeter deposit at all, so it
        # cannot be represented in the target: every kept genparticle must own a track or cluster
        # (asserted downstream). Removing it does lose its energy from the target, so record the
        # drop explicitly here; the two-sided accounting check below depends on it.
        if len(idx_gp_bestcluster) != 1:
            dropped_gps.append(int(igp_unmatched))
            dropped_energy += float(energy_arr[igp_unmatched])
            continue

        idx_gp_bestcluster = idx_gp_bestcluster[0]

        gp_merges_gp0.append(idx_gp_bestcluster)
        gp_merges_gp1.append(igp_unmatched)

        vec0 = vector.obj(
            pt=gpdata.gen_features["pt"][igp_unmatched],
            eta=gpdata.gen_features["eta"][igp_unmatched],
            phi=gpdata.gen_features["phi"][igp_unmatched],
            e=gpdata.gen_features["energy"][igp_unmatched],
        )
        # read the host from the running arrays, not from gen_features, so that several
        # unmatched particles merging into the same host accumulate instead of overwriting
        vec1 = vector.obj(
            pt=pt_arr[idx_gp_bestcluster],
            eta=eta_arr[idx_gp_bestcluster],
            phi=phi_arr[idx_gp_bestcluster],
            e=energy_arr[idx_gp_bestcluster],
        )
        vec = vec0 + vec1
        pt_arr[idx_gp_bestcluster] = vec.pt
        eta_arr[idx_gp_bestcluster] = vec.eta
        phi_arr[idx_gp_bestcluster] = vec.phi
        energy_arr[idx_gp_bestcluster] = vec.energy

    idx_all_masked = np.where(mask_gp_unmatched)[0]
    gen_features_new = {
        "PDG": np.abs(gpdata.gen_features["PDG"][mask_gp_unmatched]),
        "charge": gpdata.gen_features["charge"][mask_gp_unmatched],
        "pt": pt_arr[mask_gp_unmatched],
        "eta": eta_arr[mask_gp_unmatched],
        "sin_phi": np.sin(phi_arr[mask_gp_unmatched]),
        "cos_phi": np.cos(phi_arr[mask_gp_unmatched]),
        "energy": energy_arr[mask_gp_unmatched],
        "ispu": gpdata.gen_features["ispu"][mask_gp_unmatched],
        "generatorStatus": gpdata.gen_features["generatorStatus"][mask_gp_unmatched],
        "simulatorStatus": gpdata.gen_features["simulatorStatus"][mask_gp_unmatched],
        "gp_to_track": gpdata.gen_features["gp_to_track"][mask_gp_unmatched],
        "gp_to_cluster": gpdata.gen_features["gp_to_cluster"][mask_gp_unmatched],
        "jet_idx": gpdata.gen_features["jet_idx"][mask_gp_unmatched],
        "particle_number": np.arange(len(idx_all_masked), dtype=np.float32) + 1,
    }
    # The merges conserve energy exactly (hosts accumulate in the running arrays), and the only
    # particles removed from the target are the explicitly accounted drops above. Check the full
    # accounting two-sided so that any silent energy loss (a merge overwrite, an unaccounted drop,
    # a double-count) fails loudly instead of passing a one-sided inequality that can only ever
    # hold when energy is lost.
    assert len(gp_merges_gp0) + len(dropped_gps) == len(
        unmatched
    ), "every unmatched genparticle must be either merged into a host or explicitly dropped"
    sum_energy_original = float(np.sum(np.asarray(awkward.to_numpy(gpdata.gen_features["energy"]))))
    sum_energy_after = float(np.sum(gen_features_new["energy"]))
    assert abs(sum_energy_after + dropped_energy - sum_energy_original) < 1e-6 * max(1.0, sum_energy_original), (
        f"genparticle merge energy accounting mismatch: after + dropped = "
        f"{sum_energy_after + dropped_energy:.6g} GeV, original = {sum_energy_original:.6g} GeV"
    )
    if dropped_gps:
        print(
            f"Dropped {len(dropped_gps)} unmatched genparticles without a track/cluster host "
            f"(pT {float(np.sum(pt_arr[dropped_gps])):.1f} GeV, E {dropped_energy:.1f} GeV)"
        )

    genpart_idx_all_to_filtered = {idx_all: idx_filtered for idx_filtered, idx_all in enumerate(idx_all_masked)}
    genparticle_to_hit = filter_adj(gpdata.genparticle_to_hit, genpart_idx_all_to_filtered)
    genparticle_to_track = filter_adj(gpdata.genparticle_to_track, genpart_idx_all_to_filtered)
    gp_to_obj = gp_to_obj[mask_gp_unmatched]
    gp_to_hit_idx = gp_to_hit_idx[mask_gp_unmatched]

    # Map inclusive indices to filtered ones
    track_to_gp_inclusive_filtered = -1 * np.ones(n_track, dtype=np.int32)
    for itrk in range(n_track):
        igp_all = track_to_gp_inclusive[itrk]
        if igp_all in genpart_idx_all_to_filtered:
            track_to_gp_inclusive_filtered[itrk] = genpart_idx_all_to_filtered[igp_all]

    cluster_to_gp_inclusive_filtered = -1 * np.ones(n_cluster, dtype=np.int32)
    for icl in range(n_cluster):
        igp_all = cluster_to_gp_inclusive[icl]
        if igp_all in genpart_idx_all_to_filtered:
            cluster_to_gp_inclusive_filtered[icl] = genpart_idx_all_to_filtered[igp_all]

    hit_to_gp_inclusive_filtered = -1 * np.ones(n_hit, dtype=np.int32)
    for ihit in range(n_hit):
        igp_all = hit_to_gp_inclusive[ihit]
        if igp_all in genpart_idx_all_to_filtered:
            hit_to_gp_inclusive_filtered[ihit] = genpart_idx_all_to_filtered[igp_all]

    return (
        EventData(
            gen_features_new,
            gpdata.hit_features,
            gpdata.cluster_features,
            gpdata.track_features,
            genparticle_to_hit,
            genparticle_to_track,
            gpdata.hit_to_cluster,
            (np.array(gp_merges_gp0), np.array(gp_merges_gp1)),
        ),
        gp_to_obj,
        gp_to_hit_idx,
        track_to_gp_inclusive_filtered,
        cluster_to_gp_inclusive_filtered,
        hit_to_gp_inclusive_filtered,
    )
