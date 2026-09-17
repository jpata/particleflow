import numpy as np

import numba
import awkward
import vector
from scipy.optimize import linear_sum_assignment


@numba.njit
def deltaphi(phi1, phi2):
    diff = phi1 - phi2
    return np.arctan2(np.sin(diff), np.cos(diff))


@numba.njit
def deltar(eta1, phi1, eta2, phi2):
    deta = eta1 - eta2
    dphi = deltaphi(phi1, phi2)
    return np.sqrt(deta**2 + dphi**2)


def _match_jets_event(j1_eta, j1_phi, j2_eta, j2_phi, deltaR_cut):
    """Return a maximum-cardinality, minimum-deltaR one-to-one assignment."""

    if deltaR_cut <= 0:
        raise ValueError("deltaR_cut must be positive")

    num_jets_1 = len(j1_eta)
    num_jets_2 = len(j2_eta)
    if num_jets_1 == 0 or num_jets_2 == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    delta_eta = j1_eta[:, None] - j2_eta[None, :]
    delta_phi = j1_phi[:, None] - j2_phi[None, :]
    delta_phi = np.arctan2(np.sin(delta_phi), np.cos(delta_phi))
    delta_r = np.sqrt(delta_eta**2 + delta_phi**2)
    valid = np.isfinite(delta_r) & (delta_r < deltaR_cut)

    # Give every jet in the first collection its own dummy unmatched column.
    # The unmatched cost dominates the sum of all valid normalized distances,
    # so the assignment first maximizes cardinality and then minimizes deltaR.
    max_pairs = min(num_jets_1, num_jets_2)
    unmatched_cost = float(max_pairs + 1)
    invalid_cost = unmatched_cost * float(num_jets_1 + 1)
    cost = np.full((num_jets_1, num_jets_2 + num_jets_1), invalid_cost, dtype=np.float64)
    cost[:, :num_jets_2] = np.where(valid, delta_r / deltaR_cut, invalid_cost)
    cost[np.arange(num_jets_1), num_jets_2 + np.arange(num_jets_1)] = unmatched_cost

    jet_inds_1, columns = linear_sum_assignment(cost)
    real_match = columns < num_jets_2
    jet_inds_1 = jet_inds_1[real_match]
    jet_inds_2 = columns[real_match]
    accepted = valid[jet_inds_1, jet_inds_2]
    return jet_inds_1[accepted], jet_inds_2[accepted]


def jet_matching_metrics(response_ratios, num_reference_jets, num_candidate_jets, response_rel_pt_cut=0.5):
    """Summarize unique angular matches and response-qualified matches.

    ``response_ratios`` contains candidate/reference pT for the angularly
    matched pairs returned by :func:`match_jets`. A response-qualified match
    additionally satisfies ``abs(candidate/reference - 1) < response_rel_pt_cut``.
    """

    if response_rel_pt_cut <= 0:
        raise ValueError("response_rel_pt_cut must be positive")

    ratios = np.asarray(response_ratios, dtype=np.float64).reshape(-1)
    num_reference_jets = int(num_reference_jets)
    num_candidate_jets = int(num_candidate_jets)
    num_angular_matches = len(ratios)
    if num_angular_matches > min(num_reference_jets, num_candidate_jets):
        raise ValueError("one-to-one angular matches cannot exceed either jet collection")

    num_response_matches = int(np.sum(np.isfinite(ratios) & (np.abs(ratios - 1.0) < response_rel_pt_cut)))

    def fraction(numerator, denominator):
        return float(numerator / denominator) if denominator else float("nan")

    metrics = {
        "num_reference_jets": num_reference_jets,
        "num_candidate_jets": num_candidate_jets,
        "num_angular_matches": num_angular_matches,
        "num_response_qualified_matches": num_response_matches,
        "response_rel_pt_cut": float(response_rel_pt_cut),
        "angular_recall": fraction(num_angular_matches, num_reference_jets),
        "angular_precision": fraction(num_angular_matches, num_candidate_jets),
        "angular_f1": fraction(2 * num_angular_matches, num_reference_jets + num_candidate_jets),
        "angular_fake_rate": fraction(num_candidate_jets - num_angular_matches, num_candidate_jets),
        "response_qualified_recall": fraction(num_response_matches, num_reference_jets),
        "response_qualified_precision": fraction(num_response_matches, num_candidate_jets),
        "response_qualified_f1": fraction(2 * num_response_matches, num_reference_jets + num_candidate_jets),
    }
    # Keep the historical key readable by existing dashboards. Its semantics
    # are now the one-to-one angular recall rather than target-wise nearest-neighbor recall.
    metrics["match_frac"] = metrics["angular_recall"]
    return metrics


def match_jets(jets1, jets2, deltaR_cut):
    iev = len(jets1)
    jet_inds_1_ev = []
    jet_inds_2_ev = []
    for ev in range(iev):
        j1 = jets1[ev]  # first jet collection
        j2 = jets2[ev]  # second jet collection

        j1_eta = np.asarray(j1.eta, dtype=np.float64)
        j1_phi = np.asarray(j1.phi, dtype=np.float64)
        j2_eta = np.asarray(j2.eta, dtype=np.float64)
        j2_phi = np.asarray(j2.phi, dtype=np.float64)

        jet_inds_1, jet_inds_2 = _match_jets_event(j1_eta, j1_phi, j2_eta, j2_phi, deltaR_cut)
        jet_inds_1_ev.append(jet_inds_1.tolist())
        jet_inds_2_ev.append(jet_inds_2.tolist())
    return jet_inds_1_ev, jet_inds_2_ev


def squeeze_if_one(arr):
    if arr.shape[-1] == 1:
        return np.squeeze(arr, axis=-1)
    else:
        return arr


def build_dummy_array(num, dtype=np.int64):
    return awkward.Array(
        awkward.contents.ListOffsetArray(
            awkward.index.Index64(np.zeros(num + 1, dtype=np.int64)),
            awkward.from_numpy(np.array([], dtype=dtype), highlevel=False),
        )
    )


def to_p4(p4_obj):
    return vector.awk(
        awkward.zip(
            {
                "E": p4_obj.E,
                "px": p4_obj.px,
                "py": p4_obj.py,
                "pz": p4_obj.pz,
            }
        )
    )


def to_p4_sph(p4_obj):
    return awkward.zip({"pt": p4_obj.pt, "eta": p4_obj.eta, "phi": p4_obj.phi, "E": p4_obj.E})


def match_two_jet_collections(jets_coll, name1, name2, jet_match_dr):
    num_events = len(jets_coll[name1])

    vec1 = to_p4_sph(to_p4(jets_coll[name1]))
    vec2 = to_p4_sph(to_p4(jets_coll[name2]))
    ret = match_jets(vec1, vec2, jet_match_dr)
    j1_idx = awkward.from_iter(ret[0])
    j2_idx = awkward.from_iter(ret[1])

    num_jets = len(awkward.flatten(j1_idx))

    # In case there are no jets matched, create dummy array to ensure correct types
    if num_jets > 0:
        c1_to_c2 = awkward.Array({name1: j1_idx, name2: j2_idx})
    else:
        dummy = build_dummy_array(num_events)
        c1_to_c2 = awkward.Array({name1: dummy, name2: dummy})

    return c1_to_c2


def get_jet_config(dataset):
    import fastjet
    from mlpf.conf import JET_CONFIG

    ds_name = dataset.value
    if ds_name not in JET_CONFIG:
        raise Exception(f"jet configuration for dataset {ds_name} not implemented")

    config = JET_CONFIG[ds_name]
    algo = getattr(fastjet, config["algo"])

    if "p" in config:
        jetdef = fastjet.JetDefinition(algo, config["r"], config["p"])
    else:
        jetdef = fastjet.JetDefinition(algo, config["r"])

    return jetdef, config["ptcut"], config["match_dr"]
