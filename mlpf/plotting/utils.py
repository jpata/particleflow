import awkward
import numpy as np
import numba
import boost_histogram as bh
from scipy.optimize import curve_fit


@numba.njit
def deltaphi_nb(phi1, phi2):
    diff = phi1 - phi2
    return np.arctan2(np.sin(diff), np.cos(diff))


@numba.njit
def deltar_nb(eta1, phi1, eta2, phi2):
    deta = eta1 - eta2
    dphi = deltaphi_nb(phi1, phi2)
    return np.sqrt(deta**2 + dphi**2)


# algo from http://cms.cern.ch/iCMS/jsp/openfile.jsp?tp=draft&files=AN2023_061_v2.pdf, section 3.2 (jet matching)
@numba.njit
def match_jets_nb(j1_eta, j1_phi, j2_eta, j2_phi, deltar_cut):
    assert len(j1_eta) == len(j2_eta)
    assert len(j1_phi) == len(j2_phi)
    assert len(j1_eta) == len(j1_phi)
    iev = len(j1_eta)
    jet_inds_1_ev = []
    jet_inds_2_ev = []
    drs_ev = []
    for ev in range(iev):
        jet_inds_1 = []
        jet_inds_2 = []
        drs = []

        while True:
            if len(j1_eta[ev]) == 0 or len(j2_eta[ev]) == 0:
                jet_inds_1_ev.append(np.array(jet_inds_1, dtype=np.int64))
                jet_inds_2_ev.append(np.array(jet_inds_2, dtype=np.int64))
                drs_ev.append(np.array(drs, dtype=np.float64))
                break

            drs_jets = 999 * np.ones((len(j1_eta[ev]), len(j2_eta[ev])), dtype=np.float64)

            for ij1 in range(len(j1_eta[ev])):
                if ij1 in jet_inds_1:
                    continue
                for ij2 in range(len(j2_eta[ev])):
                    if ij2 in jet_inds_2:
                        continue

                    eta1 = j1_eta[ev][ij1]
                    eta2 = j2_eta[ev][ij2]
                    phi1 = j1_phi[ev][ij1]
                    phi2 = j2_phi[ev][ij2]
                    dr = deltar_nb(eta1, phi1, eta2, phi2)
                    drs_jets[ij1, ij2] = dr

            if np.all(drs_jets == 999):
                jet_inds_1_ev.append(np.array(jet_inds_1, dtype=np.int64))
                jet_inds_2_ev.append(np.array(jet_inds_2, dtype=np.int64))
                drs_ev.append(np.array(drs, dtype=np.float64))
                break

            flat_index = np.argmin(drs_jets)
            num_rows, num_cols = drs_jets.shape
            ij1_min = flat_index // num_cols
            ij2_min = flat_index % num_cols

            jet_inds_1.append(ij1_min)
            jet_inds_2.append(ij2_min)
            drs.append(drs_jets[ij1_min, ij2_min])

            if len(jet_inds_1) == len(j1_eta[ev]) or len(jet_inds_2) == len(j2_eta[ev]):
                jet_inds_1_ev.append(np.array(jet_inds_1, dtype=np.int64))
                jet_inds_2_ev.append(np.array(jet_inds_2, dtype=np.int64))
                drs_ev.append(np.array(drs, dtype=np.float64))
                break
    return jet_inds_1_ev, jet_inds_2_ev, drs_ev


def compute_response(data, jet_coll="Jet", genjet_coll="GenJet", deltar_cut=0.2):
    rj_idx, gj_idx, drs = match_jets_nb(
        data[jet_coll + "_eta"], data[jet_coll + "_phi"], data[genjet_coll + "_eta"], data[genjet_coll + "_phi"], deltar_cut
    )

    rj_idx = awkward.Array(rj_idx)
    gj_idx = awkward.Array(gj_idx)
    drs = awkward.Array(drs)

    # sort by genjet pt, pick leading 3 gen jets
    pair_sort = awkward.argsort(data[genjet_coll + "_pt"][gj_idx], axis=1, ascending=False)[:, :3]

    gj_pt = data[genjet_coll + "_pt"][gj_idx][pair_sort]
    gj_eta = data[genjet_coll + "_eta"][gj_idx][pair_sort]

    if jet_coll + "_pt_corr" not in data.fields:
        data[jet_coll + "_pt_corr"] = data[jet_coll + "_pt_raw"]

    rj_pt_corr = data[jet_coll + "_pt_corr"][rj_idx][pair_sort]
    rj_pt_raw = data[jet_coll + "_pt_raw"][rj_idx][pair_sort]
    rj_eta = data[jet_coll + "_eta"][rj_idx][pair_sort]
    dr = drs[pair_sort]

    mask_top3 = dr < deltar_cut

    # get response based on top 3 jet pairs
    if jet_coll == "Jet":
        mask_top3 = mask_top3 & (((data["Jet_neMultiplicity"][rj_idx][pair_sort] > 1)) | (np.abs(data["Jet_eta"][rj_idx][pair_sort]) < 3))

    response_corr = rj_pt_corr / gj_pt
    response_raw = rj_pt_raw / gj_pt

    # For efficiency and purity, use all jets
    mask = drs < deltar_cut
    if jet_coll == "Jet":
        mask = mask & (((data["Jet_neMultiplicity"][rj_idx] > 1)) | (np.abs(data["Jet_eta"][rj_idx]) < 3))

    gj_pt_unfiltered = data[genjet_coll + "_pt"][gj_idx]
    gj_eta_unfiltered = data[genjet_coll + "_eta"][gj_idx]
    rj_pt_raw_unfiltered = data[jet_coll + "_pt_raw"][rj_idx]
    rj_pt_corr_unfiltered = data[jet_coll + "_pt_corr"][rj_idx]
    rj_eta_unfiltered = data[jet_coll + "_eta"][rj_idx]

    return {
        # Top 3 jet pairs, sorted by genjet pt
        "response": response_corr[mask_top3],
        "response_raw": response_raw[mask_top3],
        "dr": dr[mask_top3],
        jet_coll + "_pt_corr": rj_pt_corr[mask_top3],
        jet_coll + "_pt_raw": rj_pt_raw[mask_top3],
        jet_coll + "_eta": rj_eta[mask_top3],
        genjet_coll + "_pt": gj_pt[mask_top3],
        genjet_coll + "_eta": gj_eta[mask_top3],
        # All jet pairs, for efficiency/purity calculation
        f"{jet_coll}_pt_corr_unfiltered": rj_pt_corr_unfiltered[mask],
        f"{jet_coll}_pt_raw_unfiltered": rj_pt_raw_unfiltered[mask],
        f"{jet_coll}_eta_unfiltered": rj_eta_unfiltered[mask],
        f"{genjet_coll}_pt_unfiltered": gj_pt_unfiltered[mask],
        f"{genjet_coll}_eta_unfiltered": gj_eta_unfiltered[mask],
    }


def to_bh(data, bins):
    h1 = bh.Histogram(bh.axis.Variable(bins))
    h1.fill(data)
    return h1


def Gauss(x, a, x0, sigma):
    if sigma > 0:
        return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))
    else:
        return 0


def compute_scale_res(response):
    h0 = to_bh(response, np.linspace(0, 2, 100))
    if h0.values().sum() > 0:
        try:
            parameters1, _ = curve_fit(
                Gauss,
                h0.axes[0].centers,
                h0.values() / h0.values().sum(),
                p0=[1.0, 1.0, 1.0],
                maxfev=1000000,
                method="dogbox",
                bounds=[(-np.inf, 0.5, 0.0), (np.inf, 1.5, 2.0)],
            )
            norm = parameters1[0] * h0.values().sum()
            mean = parameters1[1]
            sigma = parameters1[2]
            return norm, mean, sigma
        except RuntimeError:
            return 0, 0, 0
    else:
        return 0, 0, 0
