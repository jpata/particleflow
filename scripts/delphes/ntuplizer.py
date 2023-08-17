import bz2
import math
import multiprocessing
import pickle
import sys

import networkx as nx
import numpy as np
import ROOT
import uproot_methods

ROOT.gSystem.Load("libDelphes.so")
ROOT.gInterpreter.Declare('#include "classes/DelphesClasses.h"')

# for debugging
save_full_graphs = False

# 0 - nothing associated
# 1 - charged hadron
# 2 - neutral hadron
# 3 - photon
# 4 - electron
# 5 - muon
gen_pid_encoding = {
    211: 1,
    130: 2,
    22: 3,
    11: 4,
    13: 5,
}


# check if a genparticle has an associated reco track
def particle_has_track(g, particle):
    for e in g.edges(particle):
        if e[1][0] == "track":
            return True
    return False


# go through all the genparticles associated in the tower that do not have a track
# returns the sum of energies by PID and the list of these genparticles
def get_tower_gen_fracs(g, tower):
    e_130 = 0.0
    e_211 = 0.0
    e_22 = 0.0
    e_11 = 0.0
    ptcls = []
    for e in g.edges(tower):
        if e[1][0] == "particle":
            if not particle_has_track(g, e[1]):
                ptcls.append(e[1])
                pid = abs(g.nodes[e[1]]["pid"])
                ch = abs(g.nodes[e[1]]["charge"])
                e = g.nodes[e[1]]["energy"]
                if pid in [211]:
                    e_211 += e
                elif pid in [130]:
                    e_130 += e
                elif pid == 22:
                    e_22 += e
                elif pid == 11:
                    e_11 += e
                else:
                    if ch == 1:
                        e_211 += e
                    else:
                        e_130 += e
    return ptcls, (e_130, e_211, e_22, e_11)


# creates the feature vector for calorimeter towers
def make_tower_array(tower_dict):
    return np.array(
        [
            1,  # tower is denoted with ID 1
            tower_dict["et"],
            tower_dict["eta"],
            np.sin(tower_dict["phi"]),
            np.cos(tower_dict["phi"]),
            tower_dict["energy"],
            tower_dict["eem"],
            tower_dict["ehad"],
            # padding
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )


# creates the feature vector for tracks
def make_track_array(track_dict):
    return np.array(
        [
            2,  # track is denoted with ID 2
            track_dict["pt"],
            track_dict["eta"],
            np.sin(track_dict["phi"]),
            np.cos(track_dict["phi"]),
            track_dict["p"],
            track_dict["eta_outer"],
            np.sin(track_dict["phi_outer"]),
            np.cos(track_dict["phi_outer"]),
            track_dict["charge"],
            track_dict["is_gen_muon"],  # muon bit set from generator to mimic PFDelphes
            track_dict["is_gen_electron"],  # electron bit set from generator to mimic PFDelphes
        ]
    )


# creates the target vector for gen-level particles
def make_gen_array(gen_dict):
    if not gen_dict:
        return np.zeros(7)

    encoded_pid = gen_pid_encoding.get(abs(gen_dict["pid"]), 1)
    charge = math.copysign(1, gen_dict["pid"]) if encoded_pid in [1, 4, 5] else 0

    return np.array(
        [
            encoded_pid,
            charge,
            gen_dict["pt"],
            gen_dict["eta"],
            np.sin(gen_dict["phi"]),
            np.cos(gen_dict["phi"]),
            gen_dict["energy"],
        ]
    )


# creates the output vector for delphes PFCandidates
def make_cand_array(cand_dict):
    if not cand_dict:
        return np.zeros(7)

    encoded_pid = gen_pid_encoding.get(abs(cand_dict["pid"]), 1)
    return np.array(
        [
            encoded_pid,
            cand_dict["charge"],
            cand_dict.get("pt", 0),
            cand_dict["eta"],
            np.sin(cand_dict["phi"]),
            np.cos(cand_dict["phi"]),
            cand_dict.get("energy", 0),
        ]
    )


# make (reco, gen, cand) triplets from tracks and towers
# also return genparticles that were not associated to any reco object
def make_triplets(g, tracks, towers, particles, pfparticles):
    triplets = []
    remaining_particles = set(particles)
    remaining_pfcandidates = set(pfparticles)

    # loop over all reco tracks
    for t in tracks:

        # for each track, find the associated GenParticle
        ptcl = None
        for e in g.edges(t):
            if e[1][0] == "particle":
                ptcl = e[1]
                break

        # for each track, find the associated PFCandidate.
        # The track does not store the PFCandidate links directly.
        # Instead, we need to get the links to PFCandidates from the GenParticle found above.
        # We should only look for charged PFCandidates,
        # we assume the track makes only one genparticle, and the GenParticle makes only one charged PFCandidate
        pf_ptcl = None
        for e in g.edges(ptcl):
            if e[1][0] in ["pfcharged", "pfel", "pfmu"] and e[1] in remaining_pfcandidates:
                pf_ptcl = e[1]
                break

        remaining_particles.remove(ptcl)

        if pf_ptcl:
            remaining_pfcandidates.remove(pf_ptcl)

        triplets.append((t, ptcl, pf_ptcl))

    # now loop over all the reco calo towers
    for t in towers:

        # get all the genparticles in the tower
        ptcls, fracs = get_tower_gen_fracs(g, t)

        # get the index of the highest energy deposit in the array (neutral hadron, charged hadron, photon, electron)
        imax = np.argmax(fracs)

        # determine the PID based on which energy deposit is maximal
        if len(ptcls) > 0:
            if imax == 0:
                pid = 130
            elif imax == 1:
                pid = 211
            elif imax == 2:
                pid = 22
            elif imax == 3:
                pid = 11
            for ptcl in ptcls:
                if ptcl in remaining_particles:
                    remaining_particles.remove(ptcl)

        # add up the genparticles in the tower
        lvs = []
        for ptcl in ptcls:
            lv = uproot_methods.TLorentzVector.from_ptetaphie(
                g.nodes[ptcl]["pt"],
                g.nodes[ptcl]["eta"],
                g.nodes[ptcl]["phi"],
                g.nodes[ptcl]["energy"],
            )
            lvs.append(lv)

        lv = None
        gen_ptcl = None

        # determine the GenParticle to reconstruct from this tower
        if len(lvs) > 0:
            lv = sum(lvs[1:], lvs[0])
            gen_ptcl = {
                "pid": pid,
                "pt": lv.pt,
                "eta": lv.eta,
                "phi": lv.phi,
                "energy": lv.energy,
            }

            # charged gen particles outside the tracker acceptance should be reconstructed as neutrals
            if gen_ptcl["pid"] == 211 and abs(gen_ptcl["eta"]) > 2.5:
                gen_ptcl["pid"] = 130

            # we don't want to reconstruct neutral genparticles that have too low energy.
            # the threshold is set according to the delphes PFCandidate energy distribution
            if gen_ptcl["pid"] == 130 and gen_ptcl["energy"] < 9.0:
                gen_ptcl = None

        # find the PFCandidate matched to this tower.
        # again, we need to loop over the GenParticles that are associated to the tower.
        found_pf = False
        for pf_ptcl in remaining_pfcandidates:
            if (g.nodes[pf_ptcl]["eta"] == g.nodes[t]["eta"]) and (g.nodes[pf_ptcl]["phi"] == g.nodes[t]["phi"]):
                found_pf = True
                break

        if found_pf:
            remaining_pfcandidates.remove(pf_ptcl)
        else:
            pf_ptcl = None

        triplets.append((t, gen_ptcl, pf_ptcl))
    return (
        triplets,
        list(remaining_particles),
        list(remaining_pfcandidates),
    )


def process_chunk(infile, ev_start, ev_stop, outfile):
    f = ROOT.TFile.Open(infile)
    tree = f.Get("Delphes")

    X_all = []
    ygen_all = []
    ygen_remaining_all = []
    ycand_all = []

    for iev in range(ev_start, ev_stop):
        print("event {}/{} out of {} in the full file".format(iev, ev_stop, tree.GetEntries()))

        tree.GetEntry(iev)
        pileupmix = list(tree.PileUpMix)
        pileupmix_idxdict = {}
        for ip, p in enumerate(pileupmix):
            pileupmix_idxdict[p] = ip

        towers = list(tree.Tower)
        tracks = list(tree.Track)

        pf_charged = list(tree.PFChargedHadron)
        pf_neutral = list(tree.PFNeutralHadron)
        pf_photon = list(tree.PFPhoton)
        pf_el = list(tree.PFElectron)
        pf_mu = list(tree.PFMuon)

        # Create a graph with particles, tracks and towers as nodes and gen-level information as edges
        graph = nx.Graph()
        for i in range(len(pileupmix)):
            node = ("particle", i)
            graph.add_node(node)
            graph.nodes[node]["pid"] = pileupmix[i].PID
            graph.nodes[node]["eta"] = pileupmix[i].Eta
            graph.nodes[node]["phi"] = pileupmix[i].Phi
            graph.nodes[node]["pt"] = pileupmix[i].PT
            graph.nodes[node]["charge"] = pileupmix[i].Charge
            graph.nodes[node]["energy"] = pileupmix[i].E
            graph.nodes[node]["is_pu"] = pileupmix[i].IsPU

        for i in range(len(towers)):
            node = ("tower", i)
            graph.add_node(node)
            graph.nodes[node]["eta"] = towers[i].Eta
            graph.nodes[node]["phi"] = towers[i].Phi
            graph.nodes[node]["energy"] = towers[i].E
            graph.nodes[node]["et"] = towers[i].ET
            graph.nodes[node]["eem"] = towers[i].Eem
            graph.nodes[node]["ehad"] = towers[i].Ehad
            for ptcl in towers[i].Particles:
                ip = pileupmix_idxdict[ptcl]
                graph.add_edge(("tower", i), ("particle", ip))

        for i in range(len(tracks)):
            node = ("track", i)
            graph.add_node(node)
            graph.nodes[node]["p"] = tracks[i].PT * np.cosh(tracks[i].Eta)  # tracks[i].P
            graph.nodes[node]["eta"] = tracks[i].Eta
            graph.nodes[node]["phi"] = tracks[i].Phi
            graph.nodes[node]["eta_outer"] = tracks[i].EtaOuter
            graph.nodes[node]["phi_outer"] = tracks[i].PhiOuter
            graph.nodes[node]["pt"] = tracks[i].PT
            graph.nodes[node]["pid"] = tracks[i].PID
            graph.nodes[node]["charge"] = tracks[i].Charge
            ip = pileupmix_idxdict[tracks[i].Particle.GetObject()]
            graph.add_edge(("track", i), ("particle", ip))

        for i in range(len(pf_charged)):
            node = ("pfcharged", i)
            graph.add_node(node)
            graph.nodes[node]["pid"] = pf_charged[i].PID
            graph.nodes[node]["eta"] = pf_charged[i].Eta
            # print(pf_charged[i].Eta, pf_charged[i].CtgTheta)
            graph.nodes[node]["phi"] = pf_charged[i].Phi
            graph.nodes[node]["pt"] = pf_charged[i].PT
            graph.nodes[node]["charge"] = pf_charged[i].Charge
            ip = pileupmix_idxdict[pf_charged[i].Particle.GetObject()]
            graph.add_edge(("pfcharged", i), ("particle", ip))

        for i in range(len(pf_el)):
            node = ("pfel", i)
            graph.add_node(node)
            graph.nodes[node]["pid"] = 11
            graph.nodes[node]["eta"] = pf_el[i].Eta
            graph.nodes[node]["phi"] = pf_el[i].Phi
            graph.nodes[node]["pt"] = pf_el[i].PT
            graph.nodes[node]["charge"] = pf_el[i].Charge
            ip = pileupmix_idxdict[pf_el[i].Particle.GetObject()]
            graph.add_edge(("pfel", i), ("particle", ip))

        for i in range(len(pf_mu)):
            node = ("pfmu", i)
            graph.add_node(node)
            graph.nodes[node]["pid"] = 13
            graph.nodes[node]["eta"] = pf_mu[i].Eta
            graph.nodes[node]["phi"] = pf_mu[i].Phi
            graph.nodes[node]["pt"] = pf_mu[i].PT
            graph.nodes[node]["charge"] = pf_mu[i].Charge
            ip = pileupmix_idxdict[pf_mu[i].Particle.GetObject()]
            graph.add_edge(("pfmu", i), ("particle", ip))

        for i in range(len(pf_neutral)):
            node = ("pfneutral", i)
            graph.add_node(node)
            graph.nodes[node]["pid"] = 130
            graph.nodes[node]["eta"] = pf_neutral[i].Eta
            graph.nodes[node]["phi"] = pf_neutral[i].Phi
            graph.nodes[node]["energy"] = pf_neutral[i].E
            graph.nodes[node]["charge"] = 0
            for ptcl in pf_neutral[i].Particles:
                ip = pileupmix_idxdict[ptcl]
                graph.add_edge(("pfneutral", i), ("particle", ip))

        for i in range(len(pf_photon)):
            node = ("pfphoton", i)
            graph.add_node(node)
            graph.nodes[node]["pid"] = 22
            graph.nodes[node]["eta"] = pf_photon[i].Eta
            graph.nodes[node]["phi"] = pf_photon[i].Phi
            graph.nodes[node]["energy"] = pf_photon[i].E
            graph.nodes[node]["charge"] = 0
            for ptcl in pf_photon[i].Particles:
                ip = pileupmix_idxdict[ptcl]
                graph.add_edge(("pfphoton", i), ("particle", ip))

        # write the full graph, mainly for study purposes
        if iev < 10 and save_full_graphs:
            nx.readwrite.write_gpickle(
                graph,
                outfile.replace(".pkl.bz2", "_graph_{}.pkl".format(iev)),
            )

        # now clean up the graph, keeping only reconstructable genparticles
        # we also merge neutral genparticles within towers, as they are otherwise not reconstructable
        particles = [n for n in graph.nodes if n[0] == "particle"]
        pfcand = [n for n in graph.nodes if n[0].startswith("pf")]

        tracks = [n for n in graph.nodes if n[0] == "track"]
        towers = [n for n in graph.nodes if n[0] == "tower"]

        (
            triplets,
            remaining_particles,
            remaining_pfcandidates,
        ) = make_triplets(graph, tracks, towers, particles, pfcand)
        print("remaining PF", len(remaining_pfcandidates))
        for pf in remaining_pfcandidates:
            print(pf, graph.nodes[pf])

        X = []
        ygen = []
        ygen_remaining = []
        ycand = []
        for triplet in triplets:
            reco, gen, cand = triplet
            if reco[0] == "track":
                track_dict = graph.nodes[reco]
                gen_dict = graph.nodes[gen]

                # delphes PF reconstructs electrons and muons based on generator info,
                # so if a track was associated with a gen-level electron or muon,
                # we embed this information so that MLPF would have access to the same low-level info
                if abs(gen_dict["pid"]) == 13:
                    track_dict["is_gen_muon"] = 1.0
                else:
                    track_dict["is_gen_muon"] = 0.0

                if abs(gen_dict["pid"]) == 11:
                    track_dict["is_gen_electron"] = 1.0
                else:
                    track_dict["is_gen_electron"] = 0.0

                X.append(make_track_array(track_dict))
                ygen.append(make_gen_array(gen_dict))
            else:
                X.append(make_tower_array(graph.nodes[reco]))
                ygen.append(make_gen_array(gen))

            ycand.append(make_cand_array(graph.nodes[cand] if cand else None))

        for prt in remaining_particles:
            ygen_remaining.append(make_gen_array(graph.nodes[prt]))

        X = np.stack(X)
        ygen = np.stack(ygen)
        ygen_remaining = np.stack(ygen_remaining)
        ycand = np.stack(ycand)
        print(
            "X",
            X.shape,
            "ygen",
            ygen.shape,
            "ygen_remaining",
            ygen_remaining.shape,
            "ycand",
            ycand.shape,
        )

        X_all.append(X)
        ygen_all.append(ygen)
        ygen_remaining_all.append(ygen_remaining)
        ycand_all.append(ycand)

    with bz2.BZ2File(outfile, "wb") as fi:
        pickle.dump({"X": X_all, "ygen": ygen_all, "ycand": ycand_all}, fi)


def process_chunk_args(args):
    process_chunk(*args)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


if __name__ == "__main__":
    pool = multiprocessing.Pool(24)

    infile = sys.argv[1]
    f = ROOT.TFile.Open(infile)
    tree = f.Get("Delphes")
    num_evs = tree.GetEntries()

    arg_list = []
    ichunk = 0

    for chunk in chunks(range(num_evs), 100):
        outfile = sys.argv[2].replace(".pkl.bz2", "_{}.pkl.bz2".format(ichunk))
        # print(chunk[0], chunk[-1]+1)
        arg_list.append((infile, chunk[0], chunk[-1] + 1, outfile))
        ichunk += 1

    pool.map(process_chunk_args, arg_list)
    # for arg in arg_list:
    #    process_chunk_args(arg)
