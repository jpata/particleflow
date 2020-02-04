import ROOT
import numpy as np

ROOT.gSystem.Load("libDelphes.so")
ROOT.gInterpreter.Declare('#include "classes/DelphesClasses.h"')

class Output:
    def __init__(self, outfile):
        self.tfile = ROOT.TFile(outfile, "RECREATE")
        self.tree = ROOT.TTree("tree", "tree")
        
        self.maxparticles = 20000
        
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
    
    def clear(self):
        self.nparticles[0] = 0        
        self.particles_pt[:] = 0
        self.particles_eta[:] = 0
        self.particles_phi[:] = 0
        self.particles_mass[:] = 0
        self.particles_pid[:] = 0
    
    def close(self):
        self.tfile.Write()
        self.tfile.Close()

if __name__ == "__main__":
    f = ROOT.TFile("out.root")
    tree = f.Get("Delphes")

    out = Output("out_flat.root")    
    for iev in range(tree.GetEntries()):
        if iev>10:
            break
        out.clear()

        tree.GetEntry(iev)
        particles = list(tree.Particle)
        pileupmix = list(tree.PileUpMix)
        print(iev, len(particles), len(pileupmix))
   
        all_particles = particles + pileupmix 
        itgt = 0
        for isrc in range(len(all_particles)):
            if all_particles[isrc].PT > 0:
                out.particles_pt[itgt] = all_particles[isrc].PT 
                out.particles_eta[itgt] = all_particles[isrc].Eta
                out.particles_phi[itgt] = all_particles[isrc].Phi
                out.particles_mass[itgt] = all_particles[isrc].Mass
                out.particles_pid[itgt] = all_particles[isrc].PID
                itgt += 1
        out.nparticles[0] = itgt

        out.tree.Fill()
    out.close()
