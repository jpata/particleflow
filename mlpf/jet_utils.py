import numpy as np

# import numba
import awkward
import vector


# @numba.njit
def deltaphi(phi1, phi2):
    diff = phi1 - phi2
    return np.arctan2(np.sin(diff), np.cos(diff))


# @numba.njit
def deltar(eta1, phi1, eta2, phi2):
    deta = eta1 - eta2
    dphi = deltaphi(phi1, phi2)
    return np.sqrt(deta**2 + dphi**2)


# @numba.njit
def match_jets(jets1, jets2, deltaR_cut):
    iev = len(jets1)
    jet_inds_1_ev = []
    jet_inds_2_ev = []
    for ev in range(iev):
        j1 = jets1[ev]
        j2 = jets2[ev]

        jet_inds_1 = []
        jet_inds_2 = []
        for ij1 in range(len(j1)):
            drs = np.zeros(len(j2), dtype=np.float64)
            for ij2 in range(len(j2)):
                eta1 = j1.eta[ij1]
                eta2 = j2.eta[ij2]
                phi1 = j1.phi[ij1]
                phi2 = j2.phi[ij2]

                # Workaround for https://github.com/scikit-hep/vector/issues/303
                # dr = j1[ij1].deltaR(j2[ij2])
                dr = deltar(eta1, phi1, eta2, phi2)
                drs[ij2] = dr
            if len(drs) > 0:
                min_idx_dr = np.argmin(drs)
                if drs[min_idx_dr] < deltaR_cut:
                    jet_inds_1.append(ij1)
                    jet_inds_2.append(min_idx_dr)
        jet_inds_1_ev.append(jet_inds_1)
        jet_inds_2_ev.append(jet_inds_2)
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
