import ROOT
import numpy as np
import networkx as nx
from collections import Counter
import numba

ROOT.gSystem.Load("libDelphes.so")
ROOT.gInterpreter.Declare('#include "classes/DelphesClasses.h"')

#Bins for tiling to create graph edges
bins_eta = np.linspace(-8, 8, 50, dtype=np.float32)
bins_phi = np.linspace(-4, 4, 50, dtype=np.float32)

@numba.njit
def fill_adj_matrix(src, bins_eta, bins_phi, adj_matrix):
    n = len(adj_matrix)

    for iel1 in range(n):
        t1 = src[iel1, 0]
        bin_eta1 = np.searchsorted(bins_eta, src[iel1, 1])  
        bin_phi1 = np.searchsorted(bins_phi, src[iel1, 2])  
        for iel2 in range(iel1+1, n): 
            t2 = src[iel2, 0]
            #tower vs track, use track outer position
            if t1 == 0 and t2 == 1:
                bin_eta2 = np.searchsorted(bins_eta, src[iel2, 5]) 
                bin_phi2 = np.searchsorted(bins_phi, src[iel2, 6])
            else:
                bin_eta2 = np.searchsorted(bins_eta, src[iel2, 1]) 
                bin_phi2 = np.searchsorted(bins_phi, src[iel2, 2])

            if abs(bin_eta1 - bin_eta2) <= 1:
                if np.mod(abs(bin_phi1 - bin_phi2), len(bins_phi)) <= 1:
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
        self.particles_e = np.zeros(self.maxparticles, dtype=np.float32)
        self.tree.Branch("particles_e", self.particles_e, "particles_e[nparticles]/F")
        self.particles_eta = np.zeros(self.maxparticles, dtype=np.float32)
        self.tree.Branch("particles_eta", self.particles_eta, "particles_eta[nparticles]/F")
        self.particles_phi = np.zeros(self.maxparticles, dtype=np.float32)
        self.tree.Branch("particles_phi", self.particles_phi, "particles_phi[nparticles]/F")
        self.particles_mass = np.zeros(self.maxparticles, dtype=np.float32)
        self.tree.Branch("particles_mass", self.particles_mass, "particles_mass[nparticles]/F")
        self.particles_pid = np.zeros(self.maxparticles, dtype=np.int32)
        self.tree.Branch("particles_pid", self.particles_pid, "particles_pid[nparticles]/I")
        self.particles_nelem_tower = np.zeros(self.maxparticles, dtype=np.int32)
        self.tree.Branch("particles_nelem_tower", self.particles_nelem_tower, "particles_nelem_tower[nparticles]/I")
        self.particles_nelem_track = np.zeros(self.maxparticles, dtype=np.int32)
        self.tree.Branch("particles_nelem_track", self.particles_nelem_track, "particles_nelem_track[nparticles]/I")
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
        self.towers_nparticles = np.zeros(self.maxtowers, dtype=np.int32)
        self.tree.Branch("towers_nparticles", self.towers_nparticles, "towers_nparticles[ntowers]/I")
 
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
        self.particles_e[:] = 0
        self.particles_eta[:] = 0
        self.particles_phi[:] = 0
        self.particles_mass[:] = 0
        self.particles_pid[:] = 0
        self.particles_nelem_tower[:] = 0
        self.particles_nelem_track[:] = 0
        self.particles_iblock[:] = 0

        self.ntowers[:] = 0
        self.towers_e[:] = 0 
        self.towers_eta[:] = 0 
        self.towers_phi[:] = 0 
        self.towers_eem[:] = 0 
        self.towers_ehad[:] = 0
        self.towers_iblock[:] = 0
        self.towers_nparticles[:] = 0
        
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

class FakeParticle:
    pass

if __name__ == "__main__":
    f = ROOT.TFile("out.root")
    tree = f.Get("Delphes")

    out = Output("out_flat.root")    
    for iev in range(tree.GetEntries()):
        #if iev > 100:
        #    break
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
            node = ("particle", i)
            graph.add_node(node)
            graph.nodes[node]["pid"] = pileupmix[i].PID

        for i in range(len(towers)):
            graph.add_node(("tower", i))
            for ptcl in towers[i].Particles:
                ip = pileupmix_idxdict[ptcl]
                graph.add_edge(("tower", i), ("particle", ip))
        for i in range(len(tracks)):
            graph.add_node(("track", i))
            ip = pileupmix_idxdict[tracks[i].Particle.GetObject()]
            graph.add_edge(("track", i), ("particle", ip))
        if iev < 10:
            nx.readwrite.write_gpickle(graph, "graph_{}.pkl".format(iev))
        #Assign a unique ID to each connected subset of tracks, towers and particles
        isg = 0
        truncated_particles = 0
        all_sources_trk = []
        all_sources_tower = []
        all_targets_trk = []
        all_targets_tower = []
        tower_matched_particles = np.zeros(len(towers))
        particles_matched_nelem_tower = np.zeros(len(pileupmix))
        particles_matched_nelem_track = np.zeros(len(pileupmix))

        for sg in nx.connected_components(graph):
            for node in sg:
                graph.nodes[node]["iblock"] = isg
            track_nodes = [n for n in sg if n[0] == "track"]
            tower_nodes = [n for n in sg if n[0] == "tower"]
            particle_nodes = [n for n in sg if n[0] == "particle"]
     
            targets_trk = []
            targets_tower = []

            if len(track_nodes) + len(tower_nodes) > 0:
                matched_gp_from_tracks = []
                matched_gp_from_towers = []
                for t in track_nodes:
                    matched_gp = pileupmix_idxdict[tracks[t[1]].Particle.GetObject()]
                    all_sources_trk += [t]
                    all_targets_trk += [matched_gp]
                    particles_matched_nelem_track[matched_gp] += 1

                for t in tower_nodes:
                    #find particles matched to tower
                    matched_gps = [pileupmix_idxdict[p] for p in towers[t[1]].Particles]

                    #remove particles that were already matched to tracks
                    matched_gps = [p for p in matched_gps if not (p in matched_gp_from_tracks)]

                    #sort according to energy
                    matched_gps = sorted(matched_gps, key=lambda p, pileupmix=pileupmix: pileupmix[p].E, reverse=True)

                    #keep only stable particles
                    matched_gps = [p for p in matched_gps if pileupmix[p].Status==1]

                    #remove particles already matched to another tower
                    #matched_gps = [p for p in matched_gps if not p in matched_gp_from_towers]

                    #print("tower", isg, towers[t[1]].Eem, towers[t[1]].Ehad)
                    #for gp in matched_gps:
                    #    print("p", pileupmix[gp].E, pileupmix[gp].PID)
                    #import pdb;pdb.set_trace()     

                    #keep track of how many particles were attached to this tower, and how many towers to each particle            
                    tower_matched_particles[t[1]] += len(matched_gps)
                    for matched_gp in matched_gps:
                        particles_matched_nelem_tower[matched_gp] += 1
                    
                    matched_gp_from_towers += matched_gps
                    all_sources_tower += [t]
                    all_targets_tower += [matched_gps]

            isg += 1

        #convert to flat numpy arrays
        src_array_trk = np.zeros((len(all_sources_trk), 10))
        src_array_tower = np.zeros((len(all_sources_tower), 10))
        tgt_array_trk = np.zeros((len(all_targets_trk), 4))
        tgt_array_tower = np.zeros((len(all_targets_tower), 4))

        #source conversion
        for i, s in enumerate(all_sources_tower):
            tower = towers[s[1]]
            src_array_tower[i, 0] = 0
            src_array_tower[i, 1:5] = np.array([
                tower.Eta, tower.Phi, tower.Eem, tower.Ehad
            ])
        for i, s in enumerate(all_sources_trk):
            track = tracks[s[1]]
            src_array_trk[i, 0] = 1
            src_array_trk[i, 1:] = np.array([
                track.Eta, track.Phi,
                track.Charge, track.PT,
                track.EtaOuter, track.PhiOuter, track.CtgTheta,
                track.D0, track.DZ
            ])

        #Target array conversion
        for i, targets_per_source in enumerate(all_targets_tower):
            nt = len(targets_per_source)
            if nt > 0:
                ptcl = pileupmix[targets_per_source[0]]
                etot = sum([pileupmix[t].E for t in targets_per_source])
                pid = ptcl.PID
                if pid != 22:
                    pid = 130 
                tgt_array_tower[i] = np.array([pid, etot, ptcl.Eta, ptcl.Phi])
  
        for i, t in enumerate(all_targets_trk):
            ptcl = pileupmix[t]
            tgt_array_trk[i] = np.array([ptcl.PID, ptcl.E, ptcl.Eta, ptcl.Phi])  
       
        src_array = np.concatenate([src_array_tower, src_array_trk], axis=0)
  
        #Create edges between elements in the same tile or neighbouring tiles
        n = len(src_array)
        adj_matrix = np.zeros((n, n))
        fill_adj_matrix(src_array, bins_eta, bins_phi, adj_matrix)

        np.savez_compressed("raw2/ev_{}.npz".format(iev), X=src_array, y_trk=tgt_array_trk, y_tower=tgt_array_tower, adj=adj_matrix)

        all_particles = pileupmix 
        itgt = 0
        for isrc in range(len(all_particles)):
            if all_particles[isrc].Status==1:
                out.particles_pt[itgt] = all_particles[isrc].PT 
                out.particles_e[itgt] = all_particles[isrc].E
                out.particles_eta[itgt] = all_particles[isrc].Eta
                out.particles_phi[itgt] = all_particles[isrc].Phi
                out.particles_mass[itgt] = all_particles[isrc].Mass
                out.particles_pid[itgt] = all_particles[isrc].PID
                out.particles_nelem_tower[itgt] = particles_matched_nelem_tower[isrc]
                out.particles_nelem_track[itgt] = particles_matched_nelem_track[isrc]
                out.particles_iblock[itgt] = graph.nodes[("particle", isrc)]["iblock"]
                itgt += 1
        inds = np.argsort(out.particles_iblock[:itgt])
        out.particles_pt[:len(inds)] = out.particles_pt[inds][:]
        out.particles_e[:len(inds)] = out.particles_e[inds][:]
        out.particles_eta[:len(inds)] = out.particles_eta[inds][:]
        out.particles_phi[:len(inds)] = out.particles_phi[inds][:]
        out.particles_mass[:len(inds)] = out.particles_mass[inds][:] 
        out.particles_pid[:len(inds)] = out.particles_pid[inds][:] 
        out.particles_nelem_tower[:len(inds)] = out.particles_nelem_tower[inds][:] 
        out.particles_nelem_track[:len(inds)] = out.particles_nelem_track[inds][:] 
        out.particles_iblock[:len(inds)] = out.particles_iblock[inds][:] 
        out.nparticles[0] = itgt
        
        itgt = 0
        for isrc in range(len(towers)):
            out.towers_e[itgt] = towers[isrc].E
            out.towers_eta[itgt] = towers[isrc].Eta
            out.towers_phi[itgt] = towers[isrc].Phi
            out.towers_eem[itgt] = towers[isrc].Eem
            out.towers_ehad[itgt] = towers[isrc].Ehad
            out.towers_nparticles[itgt] = tower_matched_particles[isrc]
            out.towers_iblock[itgt] = graph.nodes[("tower", isrc)]["iblock"]
            itgt += 1
        inds = np.argsort(out.towers_iblock[:itgt])
        out.towers_e[:len(inds)] = out.towers_e[inds][:] 
        out.towers_eta[:len(inds)] = out.towers_eta[inds][:] 
        out.towers_phi[:len(inds)] = out.towers_phi[inds][:] 
        out.towers_eem[:len(inds)] = out.towers_eem[inds][:] 
        out.towers_ehad[:len(inds)] = out.towers_ehad[inds][:] 
        out.towers_iblock[:len(inds)] = out.towers_iblock[inds][:] 
        out.towers_nparticles[:len(inds)] = out.towers_nparticles[inds][:]
        out.ntowers[0] = itgt
        
        itgt = 0
        for isrc in range(len(tracks)):
            out.tracks_charge[itgt] = tracks[isrc].Charge
            out.tracks_pt[itgt] = tracks[isrc].PT
            out.tracks_eta[itgt] = tracks[isrc].Eta
            out.tracks_phi[itgt] = tracks[isrc].Phi
            out.tracks_ctgtheta[itgt] = tracks[isrc].CtgTheta
            out.tracks_etaouter[itgt] = tracks[isrc].EtaOuter
            out.tracks_phiouter[itgt] = tracks[isrc].PhiOuter
            out.tracks_iblock[itgt] = graph.nodes[("track", isrc)]["iblock"]
            itgt += 1
        inds = np.argsort(out.tracks_iblock[:itgt])
        out.tracks_charge[:len(inds)] = out.tracks_charge[inds][:] 
        out.tracks_pt[:len(inds)] = out.tracks_pt[inds][:] 
        out.tracks_eta[:len(inds)] = out.tracks_eta[inds][:] 
        out.tracks_phi[:len(inds)] = out.tracks_phi[inds][:] 
        out.tracks_ctgtheta[:len(inds)] = out.tracks_ctgtheta[inds][:] 
        out.tracks_etaouter[:len(inds)] = out.tracks_etaouter[inds][:] 
        out.tracks_phiouter[:len(inds)] = out.tracks_phiouter[inds][:] 
        out.tracks_iblock[:len(inds)] = out.tracks_iblock[inds][:] 
        out.ntracks[0] = itgt

        out.tree.Fill()
    out.close()
