import ROOT
import numpy as np
import networkx as nx
from collections import Counter
import uproot_methods
import math
import pickle

ROOT.gSystem.Load("libDelphes.so")
ROOT.gInterpreter.Declare('#include "classes/DelphesClasses.h"')

#for debugging
save_full_graphs = True

#check if a genparticle has an associated reco track and return the index (-1 otherwise)
def particle_track_id(g, particle):
    for edge in g.edges(particle):
        if edge[1][0] == "track":
            return edge[1][1]
    return -1

#check if a genparticle has associated reco objects (tracks or towers)
def has_reco(g, particle):
    for edge in g.edges(particle):
        if edge[1][0] in ["tower", "track"]:
            return True
    return False

#the gen particle is constructed from the sum of all neutral genparticles in the tower.
#It will be a photon if any of them are gen photons, otherwise, it will be a neutral hadron.
def get_tower_genparticle(g, tower_node):
    particles_without_track = []
    is_photon = False
    
    for edge in g.edges(tower_node):
        if edge[1][0] == "particle":
            particle = edge[1]
            if g.nodes[particle]["pid"] == 22:
                is_photon = True
            trk_id = particle_track_id(g, particle)
            if trk_id == -1:
                particles_without_track.append(particle)
    lvs = []

    for p in particles_without_track:
        lv = uproot_methods.TLorentzVector.from_ptetaphie(
            g.nodes[p]["pt"],
            g.nodes[p]["eta"],
            g.nodes[p]["phi"],
            g.nodes[p]["energy"],
        )
        lvs.append(lv)

    if len(lvs) > 0:
        lv = sum(lvs[1:], lvs[0])
        return {
            "pt": lv.pt, "eta": lv.eta, "phi": lv.phi, "energy": lv.energy, "pid": 22 if is_photon else 130
        }

    else:
        return None

#for a given reco track, get the genparticle and corresponding PF particle.
#all tracks have an associated charged genparticle by construction.
#the PFCandidate just copies the track information per Delphes-PF
def get_track_truth(g, track):
    particle = None
    pfcandidate = None
    for edge in g.edges(track):
        if edge[1][0] == "particle":
            particle = edge[1]
            for edge2 in g.edges(particle):
                if edge2[1][0] == "pfcharged":
                    pfcandidate = g.nodes[edge2[1]]
                    break
            if pfcandidate:
                break
    return (g.nodes[track], g.nodes[particle], pfcandidate)

#for a given tower, get the cleaned genparticle and corresponding PF particle.
#the genparticle is constructed according to the tower genparticle cleaning algo in get_tower_genparticle
#the PF particle will be either a neutral hadron or a photon.
def get_tower_truth(g, tower):
    particle = None
    pfcandidate = None

    reco_objs = []
    for edge in g.edges(tower):        
        for particle_edge in g.edges(edge[1]):
            if particle_edge[1][0] in ["pfneutral", "pfphoton"]:
                reco_objs.append(particle_edge[1])

    #get the first reco-pf object in case there were several (rare, but to be understood)
    reco_objs = list(set(reco_objs))
    reco_obj = g.nodes[reco_objs[0]] if len(reco_objs)>0 else None
    
    cleaned_genparticle = get_tower_genparticle(g, tower)
    return (g.nodes[tower], cleaned_genparticle, reco_obj)

def make_tower_array(tower_dict):
    return np.array([
        1, #tower is denoted with ID 1
        tower_dict["et"],
        tower_dict["eta"],
        tower_dict["phi"],
        tower_dict["energy"],
        tower_dict["eem"],
        tower_dict["ehad"],
    ])

def make_track_array(track_dict):
    return np.array([
        2, #track is denoted with ID 2
        track_dict["pt"],
        track_dict["eta"],
        track_dict["phi"],
        track_dict["p"],
        track_dict["eta_outer"],
        track_dict["phi_outer"],
    ])

#0 - nothing associated
#1 - charged hadron
#2 - neutral hadron
#3 - photon
#4 - muon
#5 - electron
gen_pid_encoding = {
    211: 1,
    130: 2,
    22: 3,
    13: 4,
    11: 5,
}

def make_gen_array(gen_dict):
    if not gen_dict:
        return np.zeros(6)

    encoded_pid = gen_pid_encoding.get(abs(gen_dict["pid"]), 1)
    charge = math.copysign(1, gen_dict["pid"]) if encoded_pid in [1,4,5] else 0

    return np.array([
        encoded_pid,
        charge,
        gen_dict["pt"],
        gen_dict["eta"],
        gen_dict["phi"],
        gen_dict["energy"]
    ])

def make_cand_array(cand_dict):
    if not cand_dict:
        return np.zeros(6)

    encoded_pid = gen_pid_encoding.get(abs(cand_dict["pid"]), 1)
    return np.array([
        encoded_pid,
        cand_dict["charge"],
        cand_dict.get("pt", 0),
        cand_dict["eta"],
        cand_dict["phi"],
        cand_dict.get("energy", 0)
    ])

if __name__ == "__main__":
    f = ROOT.TFile("out.root")
    tree = f.Get("Delphes")


    X_all = []
    ygen_all = []
    ycand_all = []

    for iev in range(tree.GetEntries()):
        print("event {}/{}".format(iev, tree.GetEntries()))

        tree.GetEntry(iev)
        pileupmix = list(tree.PileUpMix)
        pileupmix_idxdict = {}
        for ip, p in enumerate(pileupmix):
            pileupmix_idxdict[p] = ip
        towers = list(tree.Tower)
        tracks = list(tree.Track)
        pf_charged = list(tree.PFCharged)
        pf_photon = list(tree.PFPhoton)
        pf_neutral = list(tree.PFNeutralHadron)

        #Create a graph with particles, tracks and towers as nodes and gen-level information as edges
        graph = nx.Graph()
        for i in range(len(pileupmix)):
            node = ("particle", i)
            graph.add_node(node)
            graph.nodes[node]["pid"] = pileupmix[i].PID
            graph.nodes[node]["eta"] = pileupmix[i].Eta
            graph.nodes[node]["phi"] = pileupmix[i].Phi
            graph.nodes[node]["pt"] = pileupmix[i].PT
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
            graph.nodes[node]["p"] = tracks[i].P
            graph.nodes[node]["eta"] = tracks[i].Eta
            graph.nodes[node]["phi"] = tracks[i].Phi
            graph.nodes[node]["eta_outer"] = tracks[i].EtaOuter
            graph.nodes[node]["phi_outer"] = tracks[i].PhiOuter
            graph.nodes[node]["pt"] = tracks[i].PT
            ip = pileupmix_idxdict[tracks[i].Particle.GetObject()]
            graph.add_edge(("track", i), ("particle", ip))

        for i in range(len(pf_charged)):
            node = ("pfcharged", i)
            graph.add_node(node)
            graph.nodes[node]["pid"] = pf_charged[i].PID
            graph.nodes[node]["eta"] = pf_charged[i].Eta
            graph.nodes[node]["phi"] = pf_charged[i].Phi
            graph.nodes[node]["pt"] = pf_charged[i].PT
            graph.nodes[node]["charge"] = pf_charged[i].Charge
            ip = pileupmix_idxdict[pf_charged[i].Particle.GetObject()]
            graph.add_edge(("pfcharged", i), ("particle", ip))
        
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

        #write the full graph, mainly for study purposes
        if iev<10 and save_full_graphs:
            nx.readwrite.write_gpickle(graph, "graph_{}.pkl".format(iev))

        #now clean up the graph, keeping only reconstructable genparticles
        #we also merge neutral genparticles within towers, as they are otherwise not reconstructable
        particles = [n for n in graph.nodes if n[0] == "particle"]
        particles_with_reco = [n for n in particles if has_reco(graph, n)]

        tracks = [n for n in graph.nodes if n[0] == "track"]
        towers = [n for n in graph.nodes if n[0] == "tower"]

        X = []
        ygen = []
        ycand = []

        #create matrices
        for track in tracks:
            truth = get_track_truth(graph, track)
            X.append(make_track_array(truth[0]))
            ygen.append(make_gen_array(truth[1]))
            ycand.append(make_cand_array(truth[2]))

        for tower in towers:
            truth = get_tower_truth(graph, tower)
            X.append(make_tower_array(truth[0]))
            ygen.append(make_gen_array(truth[1]))
            ycand.append(make_cand_array(truth[2]))

        X = np.stack(X)
        ygen = np.stack(ygen)
        ycand = np.stack(ycand)
        print("X.shape", X.shape)

        X_all.append(X)
        ygen_all.append(ygen)
        ycand_all.append(ycand)

    with open("out.pkl", "wb") as fi:
        pickle.dump({"X": X_all, "ygen": ygen_all, "ycand": ycand_all}, fi)