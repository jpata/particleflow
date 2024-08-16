import math
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tqdm
import uproot
import vector
import awkward

matplotlib.use("Agg")

elem_branches = [
    "typ",
    "pt",
    "eta",
    "phi",
    "e",
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

target_branches = ["typ", "charge", "pt", "eta", "sin_phi", "cos_phi", "e", "ispu", "orig_pid"]


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

    gen_nodes = [n for n in g.nodes if n[0] == "cp" and g.nodes[n]["pt"] > min_pt]
    for node in gen_nodes:
        children = [(g.nodes[suc]["typ"], g.edges[node, suc]["weight"]) for suc in g.successors(node)]
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


def draw_event(g):
    pos = {}
    for node in g.nodes:
        pos[node] = (g.nodes[node]["eta"], g.nodes[node]["phi"])

    fig = plt.figure(figsize=(10, 10))

    nodes_to_draw = [n for n in g.nodes if n[0] == "elem"]
    nx.draw_networkx(
        g,
        pos=pos,
        with_labels=False,
        node_size=5,
        nodelist=nodes_to_draw,
        edgelist=[],
        node_color="red",
        node_shape="s",
        alpha=0.5,
    )

    nodes_to_draw = [n for n in g.nodes if n[0] == "pfcand"]
    nx.draw_networkx(
        g,
        pos=pos,
        with_labels=False,
        node_size=10,
        nodelist=nodes_to_draw,
        edgelist=[],
        node_color="green",
        node_shape="x",
        alpha=0.5,
    )

    nodes_to_draw = [n for n in g.nodes if (n[0] == "cp")]
    nx.draw_networkx(
        g,
        pos=pos,
        with_labels=False,
        node_size=1,
        nodelist=nodes_to_draw,
        edgelist=[],
        node_color="blue",
        node_shape=".",
        alpha=0.5,
    )

    # draw edges between genparticles and elements
    edges_to_draw = [e for e in g.edges if e[0] in nodes_to_draw]
    nx.draw_networkx_edges(g, pos, edgelist=edges_to_draw, arrows=False, alpha=0.1)

    plt.xlim(-6, 6)
    plt.ylim(-4, 4)
    plt.tight_layout()
    plt.axis("on")
    return fig


def compute_gen_met(g):
    genpart = [elem for elem in g.nodes if elem[0] == "cp"]
    px = np.sum([g.nodes[elem]["pt"] * np.cos(g.nodes[elem]["phi"]) for elem in genpart])
    py = np.sum([g.nodes[elem]["pt"] * np.sin(g.nodes[elem]["phi"]) for elem in genpart])
    met = np.sqrt(px**2 + py**2)
    return met


def merge_closeby_particles(g, deltar_cut=0.01, max_iter=100):
    print("merging closeby met={:.2f}".format(compute_gen_met(g)))

    for it in range(max_iter):
        particles_to_merge = [elem for elem in g.nodes if elem[0] == "cp"]
        part_eta = [g.nodes[node]["eta"] for node in particles_to_merge]
        part_phi = [g.nodes[node]["phi"] for node in particles_to_merge]

        # find pairs that are close by in deltaR
        # note that if there are >2 particles close by to each other, only the closest 2 get merged
        merge_pairs = []
        pairs_0, pairs_1 = deltar_pairs(part_eta, part_phi, deltar_cut)

        # no closeby particles, break
        if len(pairs_0) == 0:
            break
        merge_pairs = [(particles_to_merge[p0], particles_to_merge[p1]) for p0, p1 in zip(pairs_0, pairs_1)]

        print("merging {} pairs".format(len(merge_pairs)))
        for pair in merge_pairs:
            if pair[0] in g.nodes and pair[1] in g.nodes:
                lv = vector.obj(pt=0, eta=0, phi=0, E=0)
                sum_pu = 0.0
                sum_tot = 0.0
                for gp in pair:
                    lv += vector.obj(
                        pt=g.nodes[gp]["pt"],
                        eta=g.nodes[gp]["eta"],
                        phi=g.nodes[gp]["phi"],
                        E=g.nodes[gp]["e"],
                    )
                    sum_pu += g.nodes[gp]["ispu"] * g.nodes[gp]["e"]
                    sum_tot += g.nodes[gp]["e"]

                # now update the remaining particle properties
                g.nodes[pair[0]]["pt"] = lv.pt
                g.nodes[pair[0]]["eta"] = lv.eta
                g.nodes[pair[0]]["phi"] = lv.phi
                g.nodes[pair[0]]["e"] = lv.energy
                g.nodes[pair[0]]["ispu"] = sum_pu / sum_tot
                orig_pid = g.nodes[pair[0]]["pid"]
                if g.nodes[pair[1]]["e"] > g.nodes[pair[0]]["e"]:
                    orig_pid = g.nodes[pair[1]]["pid"]
                g.nodes[pair[0]]["pid"] = orig_pid

                # add edge weights from the deleted particle to the remaining particle
                for suc in g.successors(pair[1]):
                    if (pair[0], suc) in g.edges:
                        g.edges[(pair[0], suc)]["weight"] += g.edges[(pair[1], suc)]["weight"]
                g.remove_nodes_from([pair[1]])
    print("done merging, met={:.2f}".format(compute_gen_met(g)))


def cleanup_graph(g, node_energy_threshold=0.1, edge_energy_threshold=0.05):
    g = g.copy()

    print("start cleanup, met={:.2f}".format(compute_gen_met(g)))

    # For each truth particle, compute the energy in tracks or calorimeter clusters
    for node in g.nodes:

        # CaloParticles or TrackingParticles
        if node[0] == "cp":
            E_track = 0.0
            E_calo = 0.0
            E_other = 0.0
            E_hf = 0.0
            E_hfem = 0.0
            E_hfhad = 0.0

            # remap PID to PF-like
            g.nodes[node]["remap_pid"] = map_pdgid_to_candid(abs(g.nodes[node]["pid"]), g.nodes[node]["charge"])

            for suc in g.successors(node):
                elem_type = g.nodes[suc]["typ"]
                if elem_type in [1, 6]:
                    E_track += g.edges[node, suc]["weight"]
                elif elem_type in [4, 5, 10, 11]:
                    E_calo += g.edges[node, suc]["weight"]
                elif elem_type in [8, 9]:
                    if elem_type == 8:
                        E_hfem += g.edges[node, suc]["weight"]
                    elif elem_type == 9:
                        E_hfhad += g.edges[node, suc]["weight"]
                    E_hf += g.edges[node, suc]["weight"]
                else:
                    E_other += g.edges[node, suc]["weight"]

            g.nodes[node]["E_track"] = E_track
            g.nodes[node]["E_calo"] = E_calo
            g.nodes[node]["E_other"] = E_other
            g.nodes[node]["E_hf"] = E_hf
            g.nodes[node]["E_hfem"] = E_hfem
            g.nodes[node]["E_hfhad"] = E_hfhad

    # If there are multiple tracks matched to a gen/sim particle, keep the association to the closest one by dR
    for node in g.nodes:
        if node[0] == "cp":
            # collect tracks or GSFs
            tracks = []
            for suc in g.successors(node):
                typ = g.nodes[suc]["typ"]
                if typ == 1 or typ == 6:
                    tracks.append(suc)
            if len(tracks) > 1:
                n0 = g.nodes[node]
                drs = []
                for tr in tracks:
                    n1 = g.nodes[tr]
                    deta = np.abs(n0["eta"] - n1["eta"])
                    dphi = np.mod(n0["phi"] - n1["phi"] + np.pi, 2 * np.pi) - np.pi
                    dr2 = deta**2 + dphi**2
                    drs.append(dr2)
                imin = np.argmin(drs)

                # set the weight of the edge to the other tracks to 0
                for itr in range(len(tracks)):
                    if itr != imin:
                        g.edges[(node, tracks[itr])]["weight"] = 0.0

    for node in g.nodes:
        if node[0] == "cp":
            remap_pid = g.nodes[node]["remap_pid"]

            # charged particles that leave no track should not be reconstructed as charged
            if remap_pid in [211, 13] and g.nodes[node]["E_track"] == 0:
                g.nodes[node]["remap_pid"] = 130
                g.nodes[node]["charge"] = 0
            if remap_pid in [11] and g.nodes[node]["E_track"] == 0:
                g.nodes[node]["remap_pid"] = 22
                g.nodes[node]["charge"] = 0

            # if a particle only leaves deposits in the HF, it should be reconstructed as an HF candidate
            if (g.nodes[node]["E_track"] == 0) and (g.nodes[node]["E_calo"] == 0) and (g.nodes[node]["E_other"] == 0) and g.nodes[node]["E_hf"] > 0:
                if g.nodes[node]["E_hfhad"] > g.nodes[node]["E_hfem"]:
                    g.nodes[node]["remap_pid"] = 1
                    g.nodes[node]["charge"] = 0
                else:
                    g.nodes[node]["remap_pid"] = 2
                    g.nodes[node]["charge"] = 0

    # CaloParticles contain a lot of electrons and muons with a soft pt spectrum
    # these should not be attempted to be reconstructed as ele/mu, but rather as charged or neutral hadrons
    for node in g.nodes:
        if node[0] == "cp":
            nd = g.nodes[node]
            if nd["pt"] < 1.0 and (abs(nd["remap_pid"]) == 11 or abs(nd["remap_pid"]) == 13):
                if g.nodes[node]["E_track"] > g.nodes[node]["E_calo"]:
                    g.nodes[node]["remap_pid"] = 211
                else:
                    if abs(nd["remap_pid"]) == 11:
                        g.nodes[node]["remap_pid"] = 22
                    else:
                        g.nodes[node]["remap_pid"] = 130
                    g.nodes[node]["charge"] = 0

    # remove calopart/trackingpart not linked to any elements
    # as these are not reconstructable in principle
    nodes_to_remove = []
    for node in g.nodes:
        if node[0] == "cp":
            deg = g.degree[node]
            if deg == 0:
                nodes_to_remove += [node]
    g.remove_nodes_from(nodes_to_remove)
    print("unlinked cleanup, met={:.2f}".format(compute_gen_met(g)))

    return g


def prepare_normalized_table(g, genparticle_energy_threshold=0.2):
    # rg = g.reverse()

    all_genparticles = []
    all_elements = []
    all_pfcandidates = []
    for node in g.nodes:
        if node[0] == "elem":
            all_elements += [node]
            for parent in g.predecessors(node):
                if parent[0] == "cp":
                    all_genparticles += [parent]
        elif node[0] == "pfcand":
            all_pfcandidates += [node]
    all_genparticles = list(set(all_genparticles))
    all_elements = sorted(all_elements)

    # assign genparticles in reverse energy order uniquely to best element
    elem_to_gp = {}  # map of element -> genparticles
    unmatched_gp = []
    for gp in sorted(all_genparticles, key=lambda x: g.nodes[x]["e"], reverse=True):
        elems = [e for e in g.successors(gp)]

        # sort elements by energy deposit from genparticle
        elems_sorted = sorted(
            [(g.edges[gp, e]["weight"], e) for e in elems],
            key=lambda x: x[0],
            reverse=True,
        )

        chosen_elem = None
        for weight, elem in elems_sorted:
            if not (elem in elem_to_gp):
                chosen_elem = elem
                elem_to_gp[elem] = []
                break

        if chosen_elem is None:
            unmatched_gp += [gp]
        else:
            elem_to_gp[elem] += [gp]

    # assign unmatched genparticles to best element, allowing for overlaps
    for gp in sorted(unmatched_gp, key=lambda x: g.nodes[x]["e"], reverse=True):
        elems = [e for e in g.successors(gp)]
        elems_sorted = sorted(
            [(g.edges[gp, e]["weight"], e) for e in elems],
            key=lambda x: x[0],
            reverse=True,
        )
        _, elem = elems_sorted[0]
        elem_to_gp[elem] += [gp]

    unmatched_cand = []
    elem_to_cand = {}

    # Find primary element for each PFCandidate
    for cand in sorted(all_pfcandidates, key=lambda x: g.nodes[x]["e"], reverse=True):
        tp = g.nodes[cand]["typ"]
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
            # neighbors = [n for n in neighbors if g.nodes[n]["typ"] in [4,5,8,9,10]]
            # sorted_neighbors = sorted(neighbors, key=lambda x: g.nodes[x]["e"], reverse=True)
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
    ygen = np.recarray(
        (len(all_elements),),
        dtype=[(name, np.float32) for name in target_branches],
    )
    ygen.fill(0.0)
    ycand = np.recarray(
        (len(all_elements),),
        dtype=[(name, np.float32) for name in target_branches],
    )
    ycand.fill(0.0)

    for ielem, elem in enumerate(all_elements):
        elem_type = g.nodes[elem]["typ"]
        genparticles = sorted(
            elem_to_gp.get(elem, []),
            key=lambda x: g.edges[(x, elem)]["weight"],
            reverse=True,
        )
        # genparticles = [gp for gp in genparticles if g.nodes[gp]["e"] > genparticle_energy_threshold]
        candidate = elem_to_cand.get(elem, None)

        for j in range(len(elem_branches)):
            Xelem[elem_branches[j]][ielem] = g.nodes[elem][elem_branches[j]]

        if not (candidate is None):
            for j in range(len(target_branches)):
                ycand[target_branches[j]][ielem] = g.nodes[candidate][target_branches[j]]

        lv = vector.obj(x=0, y=0, z=0, t=0)

        # if several CaloParticles/TrackingParticles are associated to ONLY this element, merge them, as they are not reconstructable separately
        if len(genparticles) > 0:

            orig_pid = [(g.nodes[gp]["pid"], g.nodes[gp]["e"]) for gp in genparticles]
            orig_pid = sorted(orig_pid, key=lambda x: x[1], reverse=True)
            orig_pid = orig_pid[0][0]

            pid = g.nodes[genparticles[0]]["remap_pid"]
            charge = g.nodes[genparticles[0]]["charge"]

            sum_pu = 0.0
            sum_tot = 0.0
            for gp in genparticles:
                lv += vector.obj(
                    pt=g.nodes[gp]["pt"],
                    eta=g.nodes[gp]["eta"],
                    phi=g.nodes[gp]["phi"],
                    e=g.nodes[gp]["e"],
                )
                sum_pu += g.nodes[gp]["ispu"] * g.nodes[gp]["e"]
                sum_tot += g.nodes[gp]["e"]

            # remap PID in case of HCAL cluster to neutral
            if elem_type == 5 and (pid == 22 or pid == 11):
                pid = 130

            # remap forward region to HFHAD or HFEM
            if elem_type in [8, 9]:
                if pid == 130:
                    pid = 1
                elif pid == 22:
                    pid = 2

            # Remap HF candidates to neutral hadron or photon in case not matched to HF
            if elem_type in [2, 3, 4, 5]:
                if pid == 1:
                    pid = 130
                elif pid == 2:
                    pid = 22

            gp = {
                "pt": lv.rho,
                "eta": lv.eta,
                "sin_phi": np.sin(lv.phi),
                "cos_phi": np.cos(lv.phi),
                "e": lv.t,
                "typ": pid,
                "orig_pid": orig_pid,
                "px": lv.x,
                "py": lv.y,
                "pz": lv.z,
                "ispu": sum_pu / sum_tot,
                "charge": charge if pid in [211, 11, 13] else 0,
            }
            # print("  mlpf: type={} E={:.2f} eta={:.2f} phi={:.2f} q={}".format(pid, lv.t, lv.eta, lv.phi, gp["charge"]))

            for j in range(len(target_branches)):
                ygen[target_branches[j]][ielem] = gp[target_branches[j]]
    px = np.sum(ygen["pt"] * ygen["cos_phi"])
    py = np.sum(ygen["pt"] * ygen["sin_phi"])
    met = np.sqrt(px**2 + py**2)
    print("normalized, met={:.2f}".format(met))

    return Xelem, ycand, ygen


# end of prepare_normalized_table


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
            e=element_e[iobj],
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
            typ=gen_pdgid[iobj],
            pt=gen_pt[iobj],
            e=gen_e[iobj],
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
            pid=trackingparticle_pid[iobj],
            charge=trackingparticle_charge[iobj],
            pt=trackingparticle_pt[iobj],
            e=trackingparticle_e[iobj],
            eta=trackingparticle_eta[iobj],
            phi=trackingparticle_phi[iobj],
            ispu=float(trackingparticle_ev[iobj] != 0),
        )

    # CaloParticles
    for iobj in range(len(caloparticle_pid)):
        if abs(caloparticle_pid[iobj]) == 15:
            print(
                "tau caloparticle pt={}, this will introduce fake MET due to inclusion of neutrino in the caloparticle".format(caloparticle_pt[iobj])
            )
        g.add_node(
            ("cp", iobj),
            pid=caloparticle_pid[iobj],
            charge=caloparticle_charge[iobj],
            pt=caloparticle_pt[iobj],
            e=caloparticle_e[iobj],
            eta=caloparticle_eta[iobj],
            phi=caloparticle_phi[iobj],
            ispu=float(caloparticle_ev[iobj] != 0),
        )

    # baseline PF for cross-checks
    for iobj in range(len(pfcandidate_pdgid)):
        g.add_node(
            ("pfcand", iobj),
            typ=abs(pfcandidate_pdgid[iobj]),
            pt=pfcandidate_pt[iobj],
            e=pfcandidate_e[iobj],
            eta=pfcandidate_eta[iobj],
            sin_phi=np.sin(pfcandidate_phi[iobj]),
            cos_phi=np.cos(pfcandidate_phi[iobj]),
            charge=get_charge(pfcandidate_pdgid[iobj]),
            ispu=0.0,  # for PF candidates, we don't know if it was PU or not
            orig_pid=0,  # placeholder to match processed gp
        )

    trackingparticle_to_element_first = ev["trackingparticle_to_element.first"][iev]
    trackingparticle_to_element_second = ev["trackingparticle_to_element.second"][iev]
    trackingparticle_to_element_cmp = ev["trackingparticle_to_element_cmp"][iev]
    # for trackingparticles associated to elements, set a very high edge weight
    for tp, elem, c in zip(
        trackingparticle_to_element_first,
        trackingparticle_to_element_second,
        trackingparticle_to_element_cmp,
    ):
        # ignore BREM, because the TrackingParticle is already linked to GSF
        if g.nodes[("elem", elem)]["typ"] in [7]:
            continue
        g.add_edge(("tp", tp), ("elem", elem), weight=c)

    caloparticle_to_element_first = ev["caloparticle_to_element.first"][iev]
    caloparticle_to_element_second = ev["caloparticle_to_element.second"][iev]
    caloparticle_to_element_cmp = ev["caloparticle_to_element_cmp"][iev]
    for sc, elem, c in zip(
        caloparticle_to_element_first,
        caloparticle_to_element_second,
        caloparticle_to_element_cmp,
    ):
        if not (g.nodes[("elem", elem)]["typ"] in [7]):
            g.add_edge(("cp", sc), ("elem", elem), weight=c)

    print("make_graph init, met={:.2f}".format(compute_gen_met(g)))

    # merge caloparticles and trackingparticles that refer to the same particle
    nodes_to_remove = []
    for idx_cp, idx_tp in enumerate(caloparticle_idx_trackingparticle):
        if idx_tp != -1:

            # add all the edges from the trackingparticle to the caloparticle
            for elem in g.neighbors(("tp", idx_tp)):
                g.add_edge(
                    ("cp", idx_cp),
                    elem,
                    weight=g.edges[("tp", idx_tp), elem]["weight"],
                )
            # remove the trackingparticle, keep the caloparticle
            nodes_to_remove += [("tp", idx_tp)]
    g.remove_nodes_from(nodes_to_remove)
    print("make_graph duplicates removed, met={:.2f}".format(compute_gen_met(g)))

    # merge_closeby_particles(g)
    # print("cleanup done, met={:.2f}".format(compute_gen_met(g)))

    element_to_candidate_first = ev["element_to_candidate.first"][iev]
    element_to_candidate_second = ev["element_to_candidate.second"][iev]
    for elem, pfcand in zip(element_to_candidate_first, element_to_candidate_second):
        g.add_edge(("elem", elem), ("pfcand", pfcand), weight=1.0)

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
        g = cleanup_graph(g)

        # associate target particles to input elements
        Xelem, ycand, ygen = prepare_normalized_table(g)
        data = {}

        # produce a list of stable pythia particles for downstream validation
        # stable: status=1 (typical) or status=2 and no daughters (B hadrons)
        ptcls_pythia = [
            n
            for n in g.nodes
            if n[0] == "gen" and ((g.nodes[n]["status"] == 1) or ((g.nodes[n]["status"] == 2) and g.nodes[n]["num_daughters"] == 0))
        ]
        feats = ["typ", "pt", "eta", "phi", "e"]
        arr_ptcls_pythia = np.array([[g.nodes[n][f] for f in feats] for n in ptcls_pythia])

        # produce pythia-level genjets and genmet
        genjet_pt = ev["genjet_pt"][iev]
        genjet_eta = ev["genjet_eta"][iev]
        genjet_phi = ev["genjet_phi"][iev]
        genjet_energy = ev["genjet_energy"][iev]
        genjet = np.stack(
            [awkward.to_numpy(genjet_pt), awkward.to_numpy(genjet_eta), awkward.to_numpy(genjet_phi), awkward.to_numpy(genjet_energy)], axis=-1
        )

        genmet_pt = ev["genmet_pt"][iev]
        genmet_phi = ev["genmet_phi"][iev]
        genmet = np.stack([genmet_pt, genmet_phi], axis=-1)

        data = {
            "Xelem": Xelem,
            "ycand": ycand,
            "ygen": ygen,
            "pythia": arr_ptcls_pythia,
            "genjet": genjet,
            "genmet": genmet,
        }

        # print("trk", ygen[Xelem["typ"] == 1]["typ"])
        # print("ecal", ygen[Xelem["typ"] == 4]["typ"])
        # print("hcal", ygen[Xelem["typ"] == 4]["typ"])

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
