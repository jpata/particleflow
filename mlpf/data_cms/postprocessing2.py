import math
import os

# to prevent https://stackoverflow.com/questions/52026652/openblas-blas-thread-init-pthread-create-resource-temporarily-unavailable
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pickle

import networkx as nx
import numpy as np
import tqdm
import uproot
import vector
import awkward
from sklearn.neighbors import KDTree
import fastjet

# enable this to save the full graph for debugging (very large outputs)
save_debugging_pickle = False

jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
jet_ptcut = 3

elem_branches = [
    "typ",
    "pt",
    "eta",
    "phi",
    "energy",
    "layer",
    "depth",
    "charge",
    "trajpoint",
    "eta_ecal",
    "phi_ecal",
    "eta_hcal",
    "phi_hcal",
    "muon_dt_hits",
    "muon_csc_hits",
    "muon_type",
    "px",
    "py",
    "pz",
    "sigma_x",
    "sigma_y",
    "sigma_z",
    "deltap",
    "sigmadeltap",
    "gsf_electronseed_trkorecal",
    "gsf_electronseed_dnn1",
    "gsf_electronseed_dnn2",
    "gsf_electronseed_dnn3",
    "gsf_electronseed_dnn4",
    "gsf_electronseed_dnn5",
    "num_hits",
    "cluster_flags",
    "corr_energy",
    "corr_energy_err",
    "vx",
    "vy",
    "vz",
    "pterror",
    "etaerror",
    "phierror",
    "lambd",
    "lambdaerror",
    "theta",
    "thetaerror",
    "time",
    "timeerror",
    "etaerror1",
    "etaerror2",
    "etaerror3",
    "etaerror4",
    "phierror1",
    "phierror2",
    "phierror3",
    "phierror4",
]

particle_feature_order = [
    "pid",
    "charge",
    "pt",
    "eta",
    "sin_phi",
    "cos_phi",
    "energy",
    "ispu",
    "generatorStatus",
    "simulatorStatus",
    "cp_to_track",
    "cp_to_cluster",
    "jet_idx",
]


def compute_jets(particles_p4, min_pt=jet_ptcut, with_indices=False):
    cluster = fastjet.ClusterSequence(particles_p4, jetdef)
    jets = vector.awk(cluster.inclusive_jets(min_pt=min_pt))
    jets = vector.awk(awkward.zip({"energy": jets["t"], "px": jets["x"], "py": jets["y"], "pz": jets["z"]}))
    jets = awkward.Array({"pt": jets.pt, "eta": jets.eta, "phi": jets.phi, "energy": jets.energy})
    ret = jets
    if with_indices:
        indices = cluster.constituent_index(min_pt=min_pt)
        ret = jets, indices
    return ret


def print_gen(g, min_pt=1):
    gen_nodes = [
        n for n in g.nodes if n[0] == "gen" and ((g.nodes[n]["status"] == 1) or (g.nodes[n]["status"] == 2 and g.nodes[n]["num_daughters"] == 0))
    ]
    for node in gen_nodes:
        print(node, g.nodes[node]["pt"], g.nodes[node]["eta"], g.nodes[node]["phi"], g.nodes[node]["typ"])

    elem_nodes = [(n, g.nodes[n]["pt"]) for n in g.nodes if n[0] == "elem" and g.nodes[n]["typ"] != 7]
    elem_nodes = sorted(elem_nodes, key=lambda x: x[1], reverse=True)
    elem_nodes = [n[0] for n in elem_nodes]
    for node in elem_nodes:
        if g.nodes[node]["pt"] > min_pt:
            print(node, g.nodes[node]["pt"], g.nodes[node]["eta"], g.nodes[node]["phi"], g.nodes[node]["typ"])

    cp_nodes = [n for n in g.nodes if n[0] == "cp"]
    for node in cp_nodes:
        children = list(g.successors(node))
        print(node, g.nodes[node]["pt"], g.nodes[node]["eta"], g.nodes[node]["phi"], g.nodes[node]["pid"], children)


def map_pdgid_to_candid(pdgid, charge):
    if pdgid in [22, 11, 13]:
        return pdgid

    # charged hadron
    if abs(charge) > 0:
        return 211

    # neutral hadron
    return 130


def deltar_pairs(eta_vec, phi_vec, dr_cut):

    deta = np.abs(np.subtract.outer(eta_vec, eta_vec))
    dphi = np.mod(np.subtract.outer(phi_vec, phi_vec) + np.pi, 2 * np.pi) - np.pi

    dr2 = deta**2 + dphi**2
    dr2 *= np.tri(*dr2.shape)
    dr2[dr2 == 0] = 999

    ind_pairs = np.where(dr2 < dr_cut)

    return ind_pairs


def get_charge(pid):
    abs_pid = abs(pid)
    if pid in [130, 22, 1, 2]:
        return 0.0
    # 13: mu-, 11: e-
    elif abs_pid in [11, 13]:
        return -math.copysign(1.0, pid)
    # 211: pi+
    elif abs_pid in [211]:
        return math.copysign(1.0, pid)
    else:
        raise Exception("Unknown pid: ", pid)


def compute_gen_met(g, calopart=None):
    if calopart is None:
        calopart = [elem for elem in g.nodes if elem[0] == "cp"]
    sum_px = np.sum([g.nodes[elem]["pt"] * np.cos(g.nodes[elem]["phi"]) for elem in calopart])
    sum_py = np.sum([g.nodes[elem]["pt"] * np.sin(g.nodes[elem]["phi"]) for elem in calopart])
    met = np.sqrt(sum_px**2 + sum_py**2)
    return met


def split_caloparticles(g, elem_type):

    # loop over caloparticles
    cps = [(g.nodes[n]["pt"], n) for n in g.nodes if n[0] == "cp"]
    for _, cp in cps:

        # get all associated elements with type==elem_type that received a contribution from this caloparticle
        sucs = [(suc, g.edges[cp, suc]["weight"], g.nodes[suc]["energy"]) for suc in g.successors(cp) if g.nodes[suc]["typ"] == elem_type]
        sum_sucs_w = sum([s[1] for s in sucs])
        sucs = [s[0] for s in sucs]
        if len(sucs) > 1:
            lv = vector.obj(
                pt=g.nodes[cp]["pt"],
                eta=g.nodes[cp]["eta"],
                phi=g.nodes[cp]["phi"],
                energy=g.nodes[cp]["energy"],
            )
            lv_fracs = []
            for suc in sucs:
                frac_e = g.edges[cp, suc]["weight"] / sum_sucs_w
                lv_fracs.append(lv * frac_e)

            max_cp = max([n[1] for n in g.nodes if n[0] == "cp"]) + 1
            new_cp_index = max_cp
            for lv_frac, suc in zip(lv_fracs, sucs):
                g.add_node(
                    ("cp", new_cp_index),
                    pt=lv_frac.pt,
                    eta=lv_frac.eta,
                    phi=lv_frac.phi,
                    energy=lv_frac.e,
                    charge=g.nodes[cp]["charge"],
                    pid=g.nodes[cp]["pid"],
                    ispu=g.nodes[cp]["ispu"],
                    generatorStatus=0,
                    simulatorStatus=1,
                    cp_to_track=g.nodes[cp]["cp_to_track"] * (lv_frac.e / lv.e),
                    cp_to_cluster=g.nodes[cp]["cp_to_cluster"] * (lv_frac.e / lv.e),
                    jet_idx=-1,
                )
                g.add_edge(("cp", new_cp_index), suc, weight=g.edges[cp, suc]["weight"])
                new_cp_index += 1
            g.remove_node(cp)


def find_representative_elements(g, elem_to_cp, cp_to_elem, elem_type):
    unused_elems = []
    elems = [(g.nodes[e]["pt"], e) for e in g.nodes if e[0] == "elem" and g.nodes[e]["typ"] == elem_type]
    elems_sorted = sorted(elems, key=lambda x: x[0], reverse=True)
    for _, elem in elems_sorted:
        cps = list(g.predecessors(elem))
        cps_weight = [(g.edges[(cp, elem)]["weight"], cp) for cp in cps if cp not in cp_to_elem if cp[0] == "cp"]
        cps_weight_sorted = sorted(cps_weight, key=lambda x: x[0], reverse=True)
        if len(cps_weight_sorted) > 0:
            cp = cps_weight_sorted[0][1]
            elem_to_cp[elem] = cp
            cp_to_elem[cp] = elem
        else:
            unused_elems.append(elem)


def prepare_normalized_table(g, iev):
    # if there's more than one track per caloparticle, the caloparticle should be distributed among the tracks
    split_caloparticles(g, 1)
    print("split, met={:.2f}".format(compute_gen_met(g)))

    if save_debugging_pickle:
        pickle.dump(g, open("split_g_{}.pkl".format(iev), "wb"), pickle.HIGHEST_PROTOCOL)

    # get the full list of caloparticles, elements and pfcandidates
    all_caloparticles = []
    all_elements = []
    all_pfcandidates = []
    for node in g.nodes:
        if node[0] == "elem":
            all_elements += [node]
            for parent in g.predecessors(node):
                if parent[0] == "cp":
                    all_caloparticles += [parent]
        elif node[0] == "pfcand":
            all_pfcandidates += [node]

    all_caloparticles = list(set(all_caloparticles))
    print("all caloparticles, met={:.2f}".format(compute_gen_met(g, all_caloparticles)))
    all_elements = sorted(list(set(all_elements)))

    elem_to_cp = {}  # map of element -> caloparticles
    cp_to_elem = {}  # map of caloparticle -> element

    # assign caloparticles in reverse pt order uniquely to best element
    find_representative_elements(g, elem_to_cp, cp_to_elem, 1)  # tracks
    find_representative_elements(g, elem_to_cp, cp_to_elem, 6)  # gsf
    find_representative_elements(g, elem_to_cp, cp_to_elem, 4)  # ecal
    find_representative_elements(g, elem_to_cp, cp_to_elem, 5)  # hcal
    find_representative_elements(g, elem_to_cp, cp_to_elem, 8)  # HF
    find_representative_elements(g, elem_to_cp, cp_to_elem, 9)  # HF
    find_representative_elements(g, elem_to_cp, cp_to_elem, 10)  #
    find_representative_elements(g, elem_to_cp, cp_to_elem, 11)

    s1 = set(list(cp_to_elem.keys()))
    s2 = set(all_caloparticles)
    unmatched_cp = list(s2 - s1)

    # assign unmatched caloparticles to best element, allowing for overlaps
    elem_to_cp = {k: [v] for k, v in elem_to_cp.items()}
    for cp in sorted(unmatched_cp, key=lambda x: g.nodes[x]["pt"], reverse=True):
        elems = [e for e in g.successors(cp)]
        elems_sorted = sorted(
            [(g.edges[cp, e]["weight"], e) for e in elems],
            key=lambda x: x[0],
            reverse=True,
        )
        _, elem = elems_sorted[0]
        elem_to_cp[elem] += [cp]

    # Find primary element for each PFCandidate
    unmatched_cand = []
    elem_to_cand = {}
    for cand in sorted(all_pfcandidates, key=lambda x: g.nodes[x]["pt"], reverse=True):
        tp = g.nodes[cand]["pid"]
        neighbors = list(g.predecessors(cand))

        chosen_elem = None

        # Pions, muons and electrons will be assigned to the best associated track
        if tp in [211, 13, 11]:
            for elem in neighbors:
                tp_neighbor = g.nodes[elem]["typ"]

                # track or gsf
                if tp_neighbor == 1 or tp_neighbor == 6:
                    if not (elem in elem_to_cand):
                        chosen_elem = elem
                        elem_to_cand[elem] = cand
                        break

        # other particles will be assigned to the highest-energy cluster (ECAL, HCAL, HFEM, HFHAD, SC)
        else:
            sorted_neighbors = sorted(
                neighbors,
                key=lambda x: g.edges[(x, cand)]["weight"],
                reverse=True,
            )
            for elem in sorted_neighbors:
                if not (elem in elem_to_cand):
                    chosen_elem = elem
                    elem_to_cand[elem] = cand
                    break

        if chosen_elem is None:
            # print("unmatched candidate {}, {}".format(cand, g.nodes[cand]))
            unmatched_cand += [cand]

    Xelem = np.recarray(
        (len(all_elements),),
        dtype=[(name, np.float32) for name in elem_branches],
    )
    Xelem.fill(0.0)
    ytarget = np.recarray(
        (len(all_elements),),
        dtype=[(name, np.float32) for name in particle_feature_order],
    )
    ytarget.fill(0.0)
    ycand = np.recarray(
        (len(all_elements),),
        dtype=[(name, np.float32) for name in particle_feature_order],
    )
    ycand.fill(0.0)

    # find primary element for each CaloParticle and PFCandidate
    for ielem, elem in enumerate(all_elements):
        caloparticles = sorted(
            elem_to_cp.get(elem, []),
            key=lambda x: g.edges[(x, elem)]["weight"],
            reverse=True,
        )

        # keep track which CaloParticles were already used
        for cp in caloparticles:
            all_caloparticles.remove(cp)

        candidate = elem_to_cand.get(elem, None)

        for j in range(len(elem_branches)):
            Xelem[elem_branches[j]][ielem] = g.nodes[elem][elem_branches[j]]

        if not (candidate is None):
            for j in range(len(particle_feature_order)):
                ycand[particle_feature_order[j]][ielem] = g.nodes[candidate][particle_feature_order[j]]

        lv = vector.obj(x=0, y=0, z=0, t=0)

        # if several CaloParticles are associated to ONLY this element, merge them, as they are not reconstructable separately
        if len(caloparticles) > 0:
            pids_e = sorted([(g.nodes[cp]["pid"], g.nodes[cp]["energy"]) for cp in caloparticles], key=lambda x: x[1], reverse=True)
            # get the pid of the highest-energy particle associated with this element
            pid = pids_e[0][0]

            charge = g.nodes[caloparticles[0]]["charge"]
            # pid = map_pdgid_to_candid(pid, charge)

            sum_pu = 0.0
            sum_tot = 0.0
            for cp in caloparticles:
                lv += vector.obj(
                    pt=g.nodes[cp]["pt"],
                    eta=g.nodes[cp]["eta"],
                    phi=g.nodes[cp]["phi"],
                    energy=g.nodes[cp]["energy"],
                )
                sum_pu += g.nodes[cp]["ispu"] * g.nodes[cp]["energy"]
                sum_tot += g.nodes[cp]["energy"]

            cp = {
                "pt": lv.rho,
                "eta": lv.eta,
                "sin_phi": np.sin(lv.phi),
                "cos_phi": np.cos(lv.phi),
                "energy": lv.t,
                "pid": pid,
                "px": lv.x,
                "py": lv.y,
                "pz": lv.z,
                "ispu": sum_pu / sum_tot,
                "charge": charge,
                "cp_to_track": np.sum([g.nodes[cp]["cp_to_track"] for cp in caloparticles]),
                "cp_to_cluster": np.sum([g.nodes[cp]["cp_to_cluster"] for cp in caloparticles]),
                "generatorStatus": 0,
                "simulatorStatus": 2,
                "jet_idx": -1,
            }

            for j in range(len(particle_feature_order)):
                ytarget[particle_feature_order[j]][ielem] = cp[particle_feature_order[j]]

    px = np.sum(ytarget["pt"] * ytarget["cos_phi"])
    py = np.sum(ytarget["pt"] * ytarget["sin_phi"])
    met = np.sqrt(px**2 + py**2)
    print("normalized, met={:.2f}".format(met))
    assert len(all_caloparticles) == 0

    return Xelem, ycand, ytarget


# end of prepare_normalized_table


# This function prepares a directed graph.
# The node types are:
#   - CaloParticles: cp
#   - SimClusters: sc
#   - TrackingParticles: tp
#   - PFCandidates: pfcand
#   - PFElements: elem
# The final form of the graph is: cp->elem, pfcand->elem.
# To achieve this, first, we prepare a graph with the edges:
#   cp->elem
#   cp->sc->elem *
#   cp->tp->elem *
#   pfcand->elem
# Then, the intermediate sc/tp from edges marked with * are collapsed , such that only cp->elem remains
def make_graph(ev, iev):
    element_type = ev["element_type"][iev]
    element_pt = ev["element_pt"][iev]
    element_e = ev["element_energy"][iev]
    element_eta = ev["element_eta"][iev]
    element_phi = ev["element_phi"][iev]
    element_eta_ecal = ev["element_eta_ecal"][iev]
    element_phi_ecal = ev["element_phi_ecal"][iev]
    element_eta_hcal = ev["element_eta_hcal"][iev]
    element_phi_hcal = ev["element_phi_hcal"][iev]
    element_trajpoint = ev["element_trajpoint"][iev]
    element_layer = ev["element_layer"][iev]
    element_charge = ev["element_charge"][iev]
    element_depth = ev["element_depth"][iev]
    element_deltap = ev["element_deltap"][iev]
    element_sigmadeltap = ev["element_sigmadeltap"][iev]
    element_px = ev["element_px"][iev]
    element_py = ev["element_py"][iev]
    element_pz = ev["element_pz"][iev]
    element_sigma_x = ev["element_sigma_x"][iev]
    element_sigma_y = ev["element_sigma_y"][iev]
    element_sigma_z = ev["element_sigma_z"][iev]
    element_muon_dt_hits = ev["element_muon_dt_hits"][iev]
    element_muon_csc_hits = ev["element_muon_csc_hits"][iev]
    element_muon_type = ev["element_muon_type"][iev]
    element_gsf_electronseed_trkorecal = ev["element_gsf_electronseed_trkorecal"][iev]
    element_gsf_electronseed_dnn1 = ev["element_gsf_electronseed_dnn1"][iev]
    element_gsf_electronseed_dnn2 = ev["element_gsf_electronseed_dnn2"][iev]
    element_gsf_electronseed_dnn3 = ev["element_gsf_electronseed_dnn3"][iev]
    element_gsf_electronseed_dnn4 = ev["element_gsf_electronseed_dnn4"][iev]
    element_gsf_electronseed_dnn5 = ev["element_gsf_electronseed_dnn5"][iev]
    element_num_hits = ev["element_num_hits"][iev]
    element_cluster_flags = ev["element_cluster_flags"][iev]
    element_corr_energy = ev["element_corr_energy"][iev]
    element_corr_energy_err = ev["element_corr_energy_err"][iev]
    element_pterror = ev["element_pterror"][iev]
    element_etaerror = ev["element_etaerror"][iev]
    element_phierror = ev["element_phierror"][iev]
    element_lambda = ev["element_lambda"][iev]
    element_theta = ev["element_theta"][iev]
    element_lambdaerror = ev["element_lambdaerror"][iev]
    element_thetaerror = ev["element_thetaerror"][iev]
    element_vx = ev["element_vx"][iev]
    element_vy = ev["element_vy"][iev]
    element_vz = ev["element_vz"][iev]
    element_time = ev["element_time"][iev]
    element_timeerror = ev["element_timeerror"][iev]
    element_etaerror1 = ev["element_etaerror1"][iev]
    element_etaerror2 = ev["element_etaerror2"][iev]
    element_etaerror3 = ev["element_etaerror3"][iev]
    element_etaerror4 = ev["element_etaerror4"][iev]
    element_phierror1 = ev["element_phierror1"][iev]
    element_phierror2 = ev["element_phierror2"][iev]
    element_phierror3 = ev["element_phierror3"][iev]
    element_phierror4 = ev["element_phierror4"][iev]

    trackingparticle_pid = ev["trackingparticle_pid"][iev]
    trackingparticle_charge = ev["trackingparticle_charge"][iev]
    trackingparticle_pt = ev["trackingparticle_pt"][iev]
    trackingparticle_e = ev["trackingparticle_energy"][iev]
    trackingparticle_eta = ev["trackingparticle_eta"][iev]
    trackingparticle_phi = ev["trackingparticle_phi"][iev]
    trackingparticle_ev = ev["trackingparticle_ev"][iev]

    caloparticle_pid = ev["caloparticle_pid"][iev]
    caloparticle_charge = ev["caloparticle_charge"][iev]
    caloparticle_pt = ev["caloparticle_pt"][iev]
    caloparticle_e = ev["caloparticle_energy"][iev]
    caloparticle_eta = ev["caloparticle_eta"][iev]
    caloparticle_phi = ev["caloparticle_phi"][iev]
    caloparticle_ev = ev["caloparticle_ev"][iev]
    caloparticle_idx_trackingparticle = ev["caloparticle_idx_trackingparticle"][iev]

    simcluster_pid = ev["simcluster_pid"][iev]
    simcluster_pt = ev["simcluster_pt"][iev]
    simcluster_e = ev["simcluster_energy"][iev]
    simcluster_eta = ev["simcluster_eta"][iev]
    simcluster_phi = ev["simcluster_phi"][iev]
    simcluster_idx_trackingparticle = ev["simcluster_idx_trackingparticle"][iev]
    simcluster_idx_caloparticle = ev["simcluster_idx_caloparticle"][iev]

    pfcandidate_pdgid = ev["pfcandidate_pdgid"][iev]
    pfcandidate_pt = ev["pfcandidate_pt"][iev]
    pfcandidate_e = ev["pfcandidate_energy"][iev]
    pfcandidate_eta = ev["pfcandidate_eta"][iev]
    pfcandidate_phi = ev["pfcandidate_phi"][iev]

    gen_pdgid = ev["gen_pdgid"][iev]
    gen_pt = ev["gen_pt"][iev]
    gen_e = ev["gen_energy"][iev]
    gen_eta = ev["gen_eta"][iev]
    gen_phi = ev["gen_phi"][iev]
    gen_status = ev["gen_status"][iev]
    gen_daughters = ev["gen_daughters"][iev]

    g = nx.DiGraph()
    for iobj in range(len(element_type)):

        # PF input features
        g.add_node(
            ("elem", iobj),
            typ=element_type[iobj],
            pt=element_pt[iobj],
            energy=element_e[iobj],
            eta=element_eta[iobj],
            phi=element_phi[iobj],
            eta_ecal=element_eta_ecal[iobj],
            phi_ecal=element_phi_ecal[iobj],
            eta_hcal=element_eta_hcal[iobj],
            phi_hcal=element_phi_hcal[iobj],
            trajpoint=element_trajpoint[iobj],
            layer=element_layer[iobj],
            charge=element_charge[iobj],
            depth=element_depth[iobj],
            deltap=element_deltap[iobj],
            sigmadeltap=element_sigmadeltap[iobj],
            px=element_px[iobj],
            py=element_py[iobj],
            pz=element_pz[iobj],
            sigma_x=element_sigma_x[iobj],
            sigma_y=element_sigma_y[iobj],
            sigma_z=element_sigma_z[iobj],
            muon_dt_hits=element_muon_dt_hits[iobj],
            muon_csc_hits=element_muon_csc_hits[iobj],
            muon_type=element_muon_type[iobj],
            gsf_electronseed_trkorecal=element_gsf_electronseed_trkorecal[iobj],
            gsf_electronseed_dnn1=element_gsf_electronseed_dnn1[iobj],
            gsf_electronseed_dnn2=element_gsf_electronseed_dnn2[iobj],
            gsf_electronseed_dnn3=element_gsf_electronseed_dnn3[iobj],
            gsf_electronseed_dnn4=element_gsf_electronseed_dnn4[iobj],
            gsf_electronseed_dnn5=element_gsf_electronseed_dnn5[iobj],
            num_hits=element_num_hits[iobj],
            cluster_flags=element_cluster_flags[iobj],
            corr_energy=element_corr_energy[iobj],
            corr_energy_err=element_corr_energy_err[iobj],
            pterror=element_pterror[iobj],
            etaerror=element_etaerror[iobj],
            phierror=element_phierror[iobj],
            lambd=element_lambda[iobj],
            theta=element_theta[iobj],
            lambdaerror=element_lambdaerror[iobj],
            thetaerror=element_thetaerror[iobj],
            vx=element_vx[iobj],
            vy=element_vy[iobj],
            vz=element_vz[iobj],
            time=element_time[iobj],
            timeerror=element_timeerror[iobj],
            etaerror1=element_etaerror1[iobj],
            etaerror2=element_etaerror2[iobj],
            etaerror3=element_etaerror3[iobj],
            etaerror4=element_etaerror4[iobj],
            phierror1=element_phierror1[iobj],
            phierror2=element_phierror2[iobj],
            phierror3=element_phierror3[iobj],
            phierror4=element_phierror4[iobj],
        )

    # Pythia generator particles
    for iobj in range(len(gen_pdgid)):
        g.add_node(
            ("gen", iobj),
            pid=abs(gen_pdgid[iobj]),
            pt=gen_pt[iobj],
            energy=gen_e[iobj],
            eta=gen_eta[iobj],
            phi=gen_phi[iobj],
            status=gen_status[iobj],
            num_daughters=len(gen_daughters[iobj]),
        )
    for iobj in range(len(gen_daughters)):
        for idau in range(len(gen_daughters[iobj])):
            g.add_edge(("gen", iobj), ("gen", idau))

    # TrackingParticles
    for iobj in range(len(trackingparticle_pid)):
        g.add_node(
            ("tp", iobj),
            pid=abs(trackingparticle_pid[iobj]),
            charge=trackingparticle_charge[iobj],
            pt=trackingparticle_pt[iobj],
            energy=trackingparticle_e[iobj],
            eta=trackingparticle_eta[iobj],
            phi=trackingparticle_phi[iobj],
            ispu=float(trackingparticle_ev[iobj] != 0),
            generatorStatus=0,
            simulatorStatus=0,
            cp_to_track=0,
            cp_to_cluster=0,
            jet_idx=-1,
        )

    # CaloParticles
    for iobj in range(len(caloparticle_pid)):
        g.add_node(
            ("cp", iobj),
            pid=abs(caloparticle_pid[iobj]),
            charge=caloparticle_charge[iobj],
            pt=caloparticle_pt[iobj],
            energy=caloparticle_e[iobj],
            eta=caloparticle_eta[iobj],
            phi=caloparticle_phi[iobj],
            ispu=float(caloparticle_ev[iobj] != 0),
            generatorStatus=0,
            simulatorStatus=0,
            cp_to_track=0,
            cp_to_cluster=0,
            jet_idx=-1,
        )
        itp = caloparticle_idx_trackingparticle[iobj]
        if itp != -1:
            g.add_edge(("cp", iobj), ("tp", itp))

    # SimClusters
    for iobj in range(len(simcluster_pid)):
        g.add_node(
            ("sc", iobj),
            pid=abs(simcluster_pid[iobj]),
            pt=simcluster_pt[iobj],
            eta=simcluster_eta[iobj],
            phi=simcluster_phi[iobj],
            energy=simcluster_e[iobj],
        )
        icp = simcluster_idx_caloparticle[iobj]
        g.add_edge(("cp", icp), ("sc", iobj))

        itp = simcluster_idx_trackingparticle[iobj]
        if itp != -1:
            g.add_edge(("sc", iobj), ("tp", itp))

    # baseline PF for cross-checks
    for iobj in range(len(pfcandidate_pdgid)):
        g.add_node(
            ("pfcand", iobj),
            pid=abs(pfcandidate_pdgid[iobj]),
            pt=pfcandidate_pt[iobj],
            energy=pfcandidate_e[iobj],
            eta=pfcandidate_eta[iobj],
            sin_phi=np.sin(pfcandidate_phi[iobj]),
            cos_phi=np.cos(pfcandidate_phi[iobj]),
            charge=get_charge(pfcandidate_pdgid[iobj]),
            ispu=0.0,  # for PF candidates, we don't know if it was PU or not
            generatorStatus=0,
            simulatorStatus=0,
            cp_to_track=0,
            cp_to_cluster=0,
            jet_idx=-1,
        )

    trackingparticle_to_element_first = ev["trackingparticle_to_element.first"][iev]
    trackingparticle_to_element_second = ev["trackingparticle_to_element.second"][iev]
    trackingparticle_to_element_cmp = ev["trackingparticle_to_element_cmp"][iev]
    for iobj, elem, c in zip(
        trackingparticle_to_element_first,
        trackingparticle_to_element_second,
        trackingparticle_to_element_cmp,
    ):
        if g.nodes[("elem", elem)]["typ"] in [2, 3, 7]:
            continue
        if ("tp", iobj) in g.nodes and ("elem", elem) in g.nodes:
            # print(("tp", iobj), ("elem", elem), c)
            g.add_edge(("tp", iobj), ("elem", elem), weight=c * g.nodes[("elem", elem)]["energy"])

    caloparticle_to_element_first = ev["caloparticle_to_element.first"][iev]
    caloparticle_to_element_second = ev["caloparticle_to_element.second"][iev]
    caloparticle_to_element_cmp = ev["caloparticle_to_element_cmp"][iev]
    for iobj, elem, c in zip(
        caloparticle_to_element_first,
        caloparticle_to_element_second,
        caloparticle_to_element_cmp,
    ):
        if not (g.nodes[("elem", elem)]["typ"] in [2, 3, 7]):
            if ("cp", iobj) in g.nodes and ("elem", elem) in g.nodes:
                g.add_edge(("cp", iobj), ("elem", elem), weight=c)

    simcluster_to_element_first = ev["simcluster_to_element.first"][iev]
    simcluster_to_element_second = ev["simcluster_to_element.second"][iev]
    simcluster_to_element_cmp = ev["simcluster_to_element_cmp"][iev]
    for iobj, elem, c in zip(
        simcluster_to_element_first,
        simcluster_to_element_second,
        simcluster_to_element_cmp,
    ):
        if not (g.nodes[("elem", elem)]["typ"] in [2, 3, 7]):
            if ("sc", iobj) in g.nodes and ("elem", elem) in g.nodes:
                g.add_edge(("sc", iobj), ("elem", elem), weight=c)

    print("make_graph init, met={:.2f}".format(compute_gen_met(g)))

    # add children of trackingparticle (tracks) to parents (simcluster/caloparticle)
    tps = [n for n in g.nodes if n[0] == "tp"]
    for tp in tps:
        preds = g.predecessors(tp)
        sucs = g.successors(tp)
        for pred in preds:
            for suc in sucs:
                if (pred, suc) not in g.edges:
                    # print(pred, tp, suc)
                    g.add_edge(pred, suc, weight=g.edges[(tp, suc)]["weight"])

    # remove tracking particles from graph
    g.remove_nodes_from(tps)

    if save_debugging_pickle:
        pickle.dump(g, open("init_g_{}.pkl".format(iev), "wb"), pickle.HIGHEST_PROTOCOL)

    # add any remaining links between SimClusters and Elements using delta-R proximity with dR<0.05
    # note: this may have issues with phi wraparound
    elems = [n for n in g.nodes if n[0] == "elem"]
    scs = [node for node in g.nodes if node[0] == "sc"]
    sc_coords = np.array([[g.nodes[n]["eta"] for n in scs], [g.nodes[n]["phi"] for n in scs]])
    if len(sc_coords.T) > 0:
        tree = KDTree(sc_coords.T, leaf_size=32)
        for elem in elems:
            eta = g.nodes[elem]["eta"]
            phi = g.nodes[elem]["phi"]
            nearby_scs = tree.query_radius([[eta, phi]], 0.05)[0]
            for isc in nearby_scs:
                if scs[isc] in g.nodes:
                    if (scs[isc], elem) not in g.edges:
                        g.add_edge(scs[isc], elem, weight=g.nodes[elem]["energy"])

    # add children of simcluster (elems) to parent (caloparticle)
    scs = [n for n in g.nodes if n[0] == "sc"]
    for sc in scs:
        preds = g.predecessors(sc)
        sucs = g.successors(sc)
        for pred in preds:
            for suc in sucs:
                if (pred, suc) not in g.edges:
                    # print(pred, sc, suc)
                    g.add_edge(pred, suc, weight=g.edges[(sc, suc)]["weight"])

    # remove simclusters from graph
    g.remove_nodes_from(scs)

    print("make_graph duplicates removed, met={:.2f}".format(compute_gen_met(g)))

    # now remove PS and BREM elements, as they are not that informative
    elems = [n for n in g.nodes if n[0] == "elem"]
    nodes_to_remove = []
    for elem in elems:
        if g.nodes[elem]["typ"] in [2, 3, 7]:
            nodes_to_remove.append(elem)
    g.remove_nodes_from(nodes_to_remove)

    print("cleanup done, met={:.2f}".format(compute_gen_met(g)))

    element_to_candidate_first = ev["element_to_candidate.first"][iev]
    element_to_candidate_second = ev["element_to_candidate.second"][iev]
    for elem, pfcand in zip(element_to_candidate_first, element_to_candidate_second):
        if ("elem", elem) in g.nodes:
            g.add_edge(("elem", elem), ("pfcand", pfcand), weight=1.0)

    num_gen = len([n for n in g.nodes if n[0] == "gen" and g.nodes[n]["status"] == 1])
    num_cp = len([n for n in g.nodes if n[0] == "cp"])
    num_sc = len([n for n in g.nodes if n[0] == "sc"])
    num_tp = len([n for n in g.nodes if n[0] == "tp"])
    num_pf = len([n for n in g.nodes if n[0] == "pfcand"])
    num_elem = len([n for n in g.nodes if n[0] == "elem"])
    print(f"GEN={num_gen} CP={num_cp} SC={num_sc} TP={num_tp} PF={num_pf} EL={num_elem}")

    for node in g.nodes:
        if node[0] == "cp":
            elems_children = list(g.successors(node))
            cp_to_track = 0
            cp_to_cluster = 0
            for elem in elems_children:
                w = g.edges[node, elem]["weight"]
                elem_type = g.nodes[elem]["typ"]
                if elem_type in [1, 6]:
                    cp_to_track += w
                else:
                    cp_to_cluster += w
            g.nodes[node]["cp_to_track"] = cp_to_track
            g.nodes[node]["cp_to_cluster"] = cp_to_cluster

    if save_debugging_pickle:
        pickle.dump(g, open("cleanup_g_{}.pkl".format(iev), "wb"), pickle.HIGHEST_PROTOCOL)
    return g


def cleanup_graph(g):
    all_removed_edges = []
    elems = [n for n in g.nodes if n[0] == "elem"]
    for elem in elems:
        edges_to_remove = []
        for pred in g.predecessors(elem):
            edge = (pred, elem)
            if g.edges[edge]["weight"] / g.nodes[elem]["energy"] < 0.1:
                edges_to_remove.append(edge)
        all_removed_edges += edges_to_remove
    # print("removed edges:", all_removed_edges)
    # for edge in all_removed_edges:
    #     print(g.nodes[edge[0]]["energy"], g.nodes[edge[1]]["energy"], g.edges[edge]["weight"])
    g.remove_edges_from(all_removed_edges)
    return g


def process(args):
    infile = args.input
    outpath = os.path.join(args.outpath, os.path.basename(infile).split(".")[0])
    tf = uproot.open(infile)

    tt = tf["pfana/pftree"]

    if args.num_events == -1:
        args.num_events = tt.num_entries
    events_to_process = [i for i in range(args.num_events)]

    all_data = []
    ev = tt.arrays(library="np")
    for iev in tqdm.tqdm(events_to_process):
        print("processing iev={}, genmet_cmssw={:.2f}".format(iev, ev["genmet_pt"][iev][0]))
        g = make_graph(ev, iev)
        # g = cleanup_graph(g)

        # associate target particles to input elements
        Xelem, ycand, ytarget = prepare_normalized_table(g, iev)
        data = {}

        # produce a list of stable pythia particles for downstream validation
        # stable: status=1 (typical) or status=2 and no daughters (B hadrons)
        ptcls_pythia = [
            n
            for n in g.nodes
            if n[0] == "gen" and ((g.nodes[n]["status"] == 1) or ((g.nodes[n]["status"] == 2) and g.nodes[n]["num_daughters"] == 0))
        ]
        feats = ["pid", "pt", "eta", "phi", "energy"]
        arr_ptcls_pythia = np.array([[g.nodes[n][f] for f in feats] for n in ptcls_pythia])

        # genjet from CMSSW currently contains neutrinos (ak4GenJet), so it's not good to use
        # genjet_pt = ev["genjet_pt"][iev]
        # genjet_eta = ev["genjet_eta"][iev]
        # genjet_phi = ev["genjet_phi"][iev]
        # genjet_energy = ev["genjet_energy"][iev]
        # genjet = np.stack(
        #     [awkward.to_numpy(genjet_pt), awkward.to_numpy(genjet_eta), awkward.to_numpy(genjet_phi), awkward.to_numpy(genjet_energy)], axis=-1
        # )

        # produce pythia-level genjets and genmet
        pythia_p4 = vector.awk(
            awkward.zip(
                {
                    "pt": arr_ptcls_pythia[:, 1],
                    "eta": arr_ptcls_pythia[:, 2],
                    "phi": arr_ptcls_pythia[:, 3],
                    "energy": arr_ptcls_pythia[:, 4],
                }
            )
        )
        pythia_jets = compute_jets(pythia_p4)
        genjet = np.stack(
            [
                awkward.to_numpy(pythia_jets.pt),
                awkward.to_numpy(pythia_jets.eta),
                awkward.to_numpy(pythia_jets.phi),
                awkward.to_numpy(pythia_jets.energy),
            ],
            axis=-1,
        )

        genmet_pt = ev["genmet_pt"][iev]
        genmet_phi = ev["genmet_phi"][iev]
        genmet = np.stack([genmet_pt, genmet_phi], axis=-1)

        # cluster target particles to jets, save per-particle jet index
        ytarget_constituents = -1 * np.ones(len(ytarget), dtype=np.int64)
        valid = ytarget["pid"] != 0
        # save mapping of index after masking -> index before masking as numpy array
        # inspired from:
        # https://stackoverflow.com/questions/432112/1044443#comment54747416_1044443
        cumsum = np.cumsum(valid) - 1
        _, index_mapping = np.unique(cumsum, return_index=True)
        ytarget_p4 = vector.awk(
            awkward.zip(
                {
                    "pt": ytarget[valid]["pt"],
                    "eta": ytarget[valid]["eta"],
                    "phi": np.arctan2(ytarget[valid]["sin_phi"], ytarget[valid]["cos_phi"]),
                    "energy": ytarget[valid]["energy"],
                }
            )
        )
        target_jets, target_jets_indices = compute_jets(ytarget_p4, with_indices=True)
        sorted_jet_idx = awkward.argsort(target_jets.pt, axis=-1, ascending=False).to_list()
        target_jets_indices = target_jets_indices.to_list()
        for jet_idx in sorted_jet_idx:
            jet_constituents = [index_mapping[idx] for idx in target_jets_indices[jet_idx]]  # map back to constituent index *before* masking
            ytarget_constituents[jet_constituents] = jet_idx
        ytarget["jet_idx"] = ytarget_constituents

        targetjets = np.stack(
            [
                awkward.to_numpy(target_jets.pt),
                awkward.to_numpy(target_jets.eta),
                awkward.to_numpy(target_jets.phi),
                awkward.to_numpy(target_jets.energy),
            ],
            axis=-1,
        )
        data = {
            "Xelem": Xelem,
            "ycand": ycand,
            "ytarget": ytarget,
            "pythia": arr_ptcls_pythia,
            "genjet": genjet,
            "genmet": genmet,
            "targetjet": targetjets,
        }

        if args.save_full_graph:
            data["full_graph"] = g

        all_data += [data]

    with open(outpath + ".pkl", "wb") as fi:
        pickle.dump(all_data, fi)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input file from PFAnalysis", required=True)
    parser.add_argument("--outpath", type=str, default="raw", help="output path")
    parser.add_argument(
        "--save-full-graph",
        action="store_true",
        help="save the full event graph",
    )
    parser.add_argument(
        "--num-events",
        type=int,
        help="number of events to process",
        default=-1,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    process(args)
