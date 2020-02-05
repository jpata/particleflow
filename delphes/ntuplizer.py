import ROOT
import numpy as np
import networkx as nx
from collections import Counter
import numba

ROOT.gSystem.Load("libDelphes.so")
ROOT.gInterpreter.Declare('#include "classes/DelphesClasses.h"')

#Bins for tiling to create graph edges
bins_eta = np.linspace(-8, 8, 50)
bins_phi = np.linspace(-4, 4, 50)

maxparticles_per_tower = 5

@numba.njit
def fill_adj_matrix(adj_matrix, inds_tile_eta, inds_tile_phi):
    n = len(inds_tile_eta)
    for iel1 in range(n):
        ind1_eta = inds_tile_eta[iel1]
        ind1_phi = inds_tile_phi[iel1]
        for iel2 in range(iel1+1, n): 
            ind2_eta = inds_tile_eta[iel2]
            ind2_phi = inds_tile_phi[iel2]
            if abs(ind1_eta - ind2_eta) <= 1:
                if abs(ind1_phi - ind2_phi) <= 1:
                    adj_matrix[iel1, iel2] = 1

class Output:
    def __init__(self, outfile):
        self.tfile = ROOT.TFile(outfile, "RECREATE")
        self.tree = ROOT.TTree("tree", "tree")
        
        self.maxparticles = 20000
        self.maxtowers = 10000
        self.maxtracks = 10000
        
        self.nparticles = np.zeros(1, dtype=np.uint32)
        self.tree.Branch("nparticles", self.nparticles, "nparticles/i")
        self.particles_pt = np.zeros(self.maxparticles, dtype=np.float32)
        self.tree.Branch("particles_pt", self.particles_pt, "particles_pt[nparticles]/F")
        self.particles_eta = np.zeros(self.maxparticles, dtype=np.float32)
        self.tree.Branch("particles_eta", self.particles_eta, "particles_eta[nparticles]/F")
        self.particles_phi = np.zeros(self.maxparticles, dtype=np.float32)
        self.tree.Branch("particles_phi", self.particles_phi, "particles_phi[nparticles]/F")
        self.particles_mass = np.zeros(self.maxparticles, dtype=np.float32)
        self.tree.Branch("particles_mass", self.particles_mass, "particles_mass[nparticles]/F")
        self.particles_pid = np.zeros(self.maxparticles, dtype=np.int32)
        self.tree.Branch("particles_pid", self.particles_pid, "particles_pid[nparticles]/I")
        self.particles_iblock = np.zeros(self.maxparticles, dtype=np.int32)
        self.tree.Branch("particles_iblock", self.particles_iblock, "particles_iblock[nparticles]/I")

        self.ntowers = np.zeros(1, dtype=np.uint32)
        self.tree.Branch("ntowers", self.ntowers, "ntowers/i")
        self.towers_e = np.zeros(self.maxtowers, dtype=np.float32)
        self.tree.Branch("towers_e", self.towers_e, "towers_e[ntowers]/F")
        self.towers_eta = np.zeros(self.maxtowers, dtype=np.float32)
        self.tree.Branch("towers_eta", self.towers_eta, "towers_eta[ntowers]/F")
        self.towers_phi = np.zeros(self.maxtowers, dtype=np.float32)
        self.tree.Branch("towers_phi", self.towers_phi, "towers_phi[ntowers]/F")
        self.towers_eem = np.zeros(self.maxtowers, dtype=np.float32)
        self.tree.Branch("towers_eem", self.towers_eem, "towers_eem[ntowers]/F")
        self.towers_ehad = np.zeros(self.maxtowers, dtype=np.float32)
        self.tree.Branch("towers_ehad", self.towers_ehad, "towers_ehad[ntowers]/F")
        self.towers_iblock = np.zeros(self.maxtowers, dtype=np.int32)
        self.tree.Branch("towers_iblock", self.towers_iblock, "towers_iblock[ntowers]/I")
 
        self.ntracks = np.zeros(1, dtype=np.uint32)
        self.tree.Branch("ntracks", self.ntracks, "ntracks/i")
        self.tracks_charge = np.zeros(self.maxtracks, dtype=np.float32)
        self.tree.Branch("tracks_charge", self.tracks_charge, "tracks_charge[ntracks]/F")
        self.tracks_pt = np.zeros(self.maxtracks, dtype=np.float32)
        self.tree.Branch("tracks_pt", self.tracks_pt, "tracks_pt[ntracks]/F")
        self.tracks_eta = np.zeros(self.maxtracks, dtype=np.float32)
        self.tree.Branch("tracks_eta", self.tracks_eta, "tracks_eta[ntracks]/F")
        self.tracks_phi = np.zeros(self.maxtracks, dtype=np.float32)
        self.tree.Branch("tracks_phi", self.tracks_phi, "tracks_phi[ntracks]/F")
        self.tracks_ctgtheta = np.zeros(self.maxtracks, dtype=np.float32)
        self.tree.Branch("tracks_ctgtheta", self.tracks_ctgtheta, "tracks_ctgtheta[ntracks]/F")
        self.tracks_etaouter = np.zeros(self.maxtracks, dtype=np.float32)
        self.tree.Branch("tracks_etaouter", self.tracks_etaouter, "tracks_etaouter[ntracks]/F")
        self.tracks_phiouter = np.zeros(self.maxtracks, dtype=np.float32)
        self.tree.Branch("tracks_phiouter", self.tracks_phiouter, "tracks_phiouter[ntracks]/F")
        self.tracks_iblock = np.zeros(self.maxtracks, dtype=np.int32)
        self.tree.Branch("tracks_iblock", self.tracks_iblock, "tracks_iblock[ntracks]/I")

    def clear(self):
        self.nparticles[:] = 0        
        self.particles_pt[:] = 0
        self.particles_eta[:] = 0
        self.particles_phi[:] = 0
        self.particles_mass[:] = 0
        self.particles_pid[:] = 0
        self.particles_iblock[:] = 0

        self.ntowers[:] = 0
        self.towers_e[:] = 0 
        self.towers_eta[:] = 0 
        self.towers_phi[:] = 0 
        self.towers_eem[:] = 0 
        self.towers_ehad[:] = 0
        self.towers_iblock[:] = 0
        
        self.ntracks[:] = 0
        self.tracks_charge[:] = 0 
        self.tracks_pt[:] = 0 
        self.tracks_eta[:] = 0 
        self.tracks_phi[:] = 0 
        self.tracks_ctgtheta[:] = 0 
        self.tracks_etaouter[:] = 0 
        self.tracks_phiouter[:] = 0 
        self.tracks_iblock[:] = 0 
 
    def close(self):
        self.tfile.Write()
        self.tfile.Close()

if __name__ == "__main__":
    f = ROOT.TFile("out.root")
    tree = f.Get("Delphes")

    #out = Output("out_flat.root")    
    for iev in range(tree.GetEntries()):
        print(iev)
        out.clear()

        tree.GetEntry(iev)
        particles = list(tree.Particle)
        pileupmix = list(tree.PileUpMix)
        pileupmix_idxdict = {}
        for ip, p in enumerate(pileupmix):
            pileupmix_idxdict[p] = ip
        towers = list(tree.Tower)
        tracks = list(tree.Track)

        #Create a graph with particles, tracks and towers as nodes and gen-level information as edges
        graph = nx.Graph()
        for i in range(len(pileupmix)):
            graph.add_node(("particle", i))
        for i in range(len(towers)):
            graph.add_node(("tower", i))
            for ptcl in towers[i].Particles:
                ip = pileupmix_idxdict[ptcl]
                graph.add_edge(("tower", i), ("particle", ip))
        for i in range(len(tracks)):
            graph.add_node(("track", i))
            ip = pileupmix_idxdict[tracks[i].Particle.GetObject()]
            graph.add_edge(("track", i), ("particle", ip))

        #Assign a unique ID to each connected subset of tracks, towers and particles
        isg = 0
        truncated_particles = 0
        all_sources = []
        all_targets = []
        for sg in nx.connected_components(graph):
            for node in sg:
                graph.nodes[node]["iblock"] = isg
            track_nodes = [n for n in sg if n[0] == "track"]
            tower_nodes = [n for n in sg if n[0] == "tower"]
            particle_nodes = [n for n in sg if n[0] == "particle"]
            sources = []
            targets = []
            if len(track_nodes) + len(tower_nodes) > 0:
                for t in track_nodes:
                    matched_gp = pileupmix_idxdict[tracks[t[1]].Particle.GetObject()]
                    sources += [t]
                    targets += [matched_gp]
                for t in tower_nodes:
                    matched_gps = [pileupmix_idxdict[p] for p in towers[t[1]].Particles]

                    #Each tower is assumed to produce up to maxparticles_per_tower particles, truncate the rest
                    matched_gps_trunc = matched_gps[:maxparticles_per_tower]
                    truncated_particles += len(matched_gps) - len(matched_gps_trunc)
                    src = maxparticles_per_tower * [t]
                    tgt = maxparticles_per_tower * [None]
                    tgt[:len(matched_gps_trunc)] = matched_gps_trunc[:]
                    sources += src
                    targets += tgt
            all_sources += sources
            all_targets += targets
            isg += 1
        print("truncated candidates from towers {:.4f}%".format(100.0*truncated_particles/len(pileupmix)))

        #convert to flat numpy arrays
        src_array = np.zeros((len(all_sources), 6))
        tgt_array = np.zeros((len(all_targets), 4))

        #source conversion
        for i, s in enumerate(all_sources):
            if s[0] == "tower":
                tower = towers[s[1]]
                src_array[i] = np.array([0, tower.E, tower.Eta, tower.Phi, tower.Eem, tower.Ehad])  
            elif s[0] == "track":
                track = tracks[s[1]]
                src_array[i] = np.array([1, track.Charge/track.PT, track.Eta, track.Phi, track.EtaOuter, track.PhiOuter])
        inds_tile_eta = np.searchsorted(bins_eta, src_array[:, 2])
        inds_tile_phi = np.searchsorted(bins_phi, src_array[:, 3])

        #Target conversion
        for i, t in enumerate(all_targets):
            if t is None:
                continue
            assert(isinstance(t, int))
            ptcl = pileupmix[t]
            tgt_array[i] = np.array([ptcl.PID, ptcl.PT, ptcl.Eta, ptcl.Phi])  
        
        #Create edges between elements in the same tile or neighbouring tiles
        n = len(inds_tile_eta)
        adj_matrix = np.zeros((n, n))
        fill_adj_matrix(adj_matrix, inds_tile_eta, inds_tile_phi)

        np.savez_compressed("raw/ev_{}.npz".format(iev), X=src_array, y=tgt_array, adj=adj_matrix)

        #all_particles = pileupmix 
        #itgt = 0
        #for isrc in range(len(all_particles)):
        #    if all_particles[isrc].PT > 0.1 and all_particles[isrc].Status==1:
        #        out.particles_pt[itgt] = all_particles[isrc].PT 
        #        out.particles_eta[itgt] = all_particles[isrc].Eta
        #        out.particles_phi[itgt] = all_particles[isrc].Phi
        #        out.particles_mass[itgt] = all_particles[isrc].Mass
        #        out.particles_pid[itgt] = all_particles[isrc].PID
        #        out.particles_iblock[itgt] = graph.nodes[("particle", isrc)]["iblock"]
        #        itgt += 1
        #inds = np.argsort(out.particles_iblock[:itgt])
        #out.particles_pt[:len(inds)] = out.particles_pt[inds][:]
        #out.particles_eta[:len(inds)] = out.particles_eta[inds][:]
        #out.particles_phi[:len(inds)] = out.particles_phi[inds][:]
        #out.particles_mass[:len(inds)] = out.particles_mass[inds][:] 
        #out.particles_pid[:len(inds)] = out.particles_pid[inds][:] 
        #out.particles_iblock[:len(inds)] = out.particles_iblock[inds][:] 
        #out.nparticles[0] = itgt
        #
        #itgt = 0
        #for isrc in range(len(towers)):
        #    out.towers_e[itgt] = towers[isrc].E
        #    out.towers_eta[itgt] = towers[isrc].Eta
        #    out.towers_phi[itgt] = towers[isrc].Phi
        #    out.towers_eem[itgt] = towers[isrc].Eem
        #    out.towers_ehad[itgt] = towers[isrc].Ehad
        #    out.towers_iblock[itgt] = graph.nodes[("tower", isrc)]["iblock"]
        #    itgt += 1
        #inds = np.argsort(out.towers_iblock[:itgt])
        #out.towers_e[:len(inds)] = out.towers_e[inds][:] 
        #out.towers_eta[:len(inds)] = out.towers_eta[inds][:] 
        #out.towers_phi[:len(inds)] = out.towers_phi[inds][:] 
        #out.towers_eem[:len(inds)] = out.towers_eem[inds][:] 
        #out.towers_ehad[:len(inds)] = out.towers_ehad[inds][:] 
        #out.towers_iblock[:len(inds)] = out.towers_iblock[inds][:] 
        #out.ntowers[0] = itgt
        #
        #itgt = 0
        #for isrc in range(len(tracks)):
        #    out.tracks_charge[itgt] = tracks[isrc].Charge
        #    out.tracks_pt[itgt] = tracks[isrc].PT
        #    out.tracks_eta[itgt] = tracks[isrc].Eta
        #    out.tracks_phi[itgt] = tracks[isrc].Phi
        #    out.tracks_ctgtheta[itgt] = tracks[isrc].CtgTheta
        #    out.tracks_etaouter[itgt] = tracks[isrc].EtaOuter
        #    out.tracks_phiouter[itgt] = tracks[isrc].PhiOuter
        #    out.tracks_iblock[itgt] = graph.nodes[("track", isrc)]["iblock"]
        #    itgt += 1
        #inds = np.argsort(out.tracks_iblock[:itgt])
        #out.tracks_charge[:len(inds)] = out.tracks_charge[inds][:] 
        #out.tracks_pt[:len(inds)] = out.tracks_pt[inds][:] 
        #out.tracks_eta[:len(inds)] = out.tracks_eta[inds][:] 
        #out.tracks_phi[:len(inds)] = out.tracks_phi[inds][:] 
        #out.tracks_ctgtheta[:len(inds)] = out.tracks_ctgtheta[inds][:] 
        #out.tracks_etaouter[:len(inds)] = out.tracks_etaouter[inds][:] 
        #out.tracks_phiouter[:len(inds)] = out.tracks_phiouter[inds][:] 
        #out.tracks_iblock[:len(inds)] = out.tracks_iblock[inds][:] 
        #out.ntracks[0] = itgt

        #out.tree.Fill()
    #out.close()
