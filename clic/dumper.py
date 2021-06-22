from org.lcsim.lcio import LCIOReader
from hep.io.sio import SIOReader
from hep.lcio.implementation.sio import SIOLCReader
from hep.lcio.implementation.io import LCFactory
from hep.lcio.event import * 
from hep.lcio.io import *
from jhplot import *    # import graphics
from hephysics.particle import LParticle
import math
import json
import bz2

save_calohits = True

def genParticleToDict(par):
    mom = par.getMomentum()
    parent_pdgid = 0
    if len(par.getParents()) > 0:
        parent_pdgid = par.getParents()[0].getPDG()
 
    vec = {
        "pdgid": par.getPDG(),
        "status": par.getGeneratorStatus(),
        "mass": par.getMass(),
        "charge": par.getCharge(),
        "pdgid_parent0": parent_pdgid,
        "px": mom[0],
        "py": mom[1],
        "pz": mom[2],
        "energy": par.getEnergy()
    }
    return vec

def pfParticleToDict(par):
    mom = par.getMomentum()
    vec = {
        "type": par.getType(),
        "px": mom[0],
        "py": mom[1],
        "pz": mom[2],
        "energy": par.getEnergy()
    }
    return vec

def clusterToDict(par):
    pos = par.getPosition()

    vec = {
        "type": par.getType(),
        "x": pos[0],
        "y": pos[1],
        "z": pos[2],
        "energy": par.getEnergy(),
        "gp_contributions": [],
        "nhits_ecal": 0,
        "nhits_hcal": 0
    }
    for recohit in par.getCalorimeterHits():
        if recohit in set_hcal_hits:
            vec["nhits_hcal"] += 1
        elif recohit in set_ecal_hits:
            vec["nhits_ecal"] += 1
    return vec

def trackHitToDict(par):
    pos = par.getPosition()
    vec = {
        "x": pos[0],
        "y": pos[1],
        "z": pos[2],
    }

    gps = {}
    if par in sim_trackhit_to_gen:
        for gp in sim_trackhit_to_gen[par]:
            gpid = genparticle_dict[gp]
            if not gpid in gps:
                gps[gpid] = 0
            gps[gpid] += 1

    gp_contributions = sorted(gps.items(), key=lambda x: x[1], reverse=True)
    vec["gp_contributions"] = gp_contributions

    return vec

def trackToDict(par):
    ts = par.getTrackStates()[0]
    vec = {
        "d0": ts.getD0(),
        "z0": ts.getZ0(),
        "omega": ts.getOmega(),
        "phi": ts.getPhi(),
        "tan_lambda": ts.getTanLambda(),
        "nhits": len(par.getTrackerHits())
    }

    #for each hit in the track, find the associated genparticle
    gps = {}
    for hit in par.getTrackerHits():
        if hit in sim_trackhit_to_gen:
            for gp in sim_trackhit_to_gen[hit]:
                gpid = genparticle_dict[gp]
                if not gpid in gps:
                    gps[gpid] = 0
                gps[gpid] += 1

    #assign the track to the genparticle which was associated to the most hits
    gp_contributions = sorted(gps.items(), key=lambda x: x[1], reverse=True)
    vec["gp_contributions"] = {c[0]: c[1] for c in gp_contributions}

    return vec

def caloHitToDict(par, calohit_to_cluster, genparticle_dict, calohit_recotosim):
    pos = par.getPosition()
    vec = {
       "x": pos[0],
       "y": pos[1],
       "z": pos[2],
       "energy": par.getEnergy(),
       "cluster_idx": calohit_to_cluster.get(par, -1)
    }

    #get the simhit corresponding to this calohit
    simhit = calohit_recotosim[par]
    gp_contributions = []
    for iptcl in range(simhit.getNMCParticles()):
        ptcl = simhit.getParticleCont(iptcl)
        idx_ptcl = genparticle_dict[ptcl]
        energy_cont = par.getEnergy() * (simhit.getEnergyCont(iptcl)/simhit.getEnergy())
        gp_contributions.append([idx_ptcl, energy_cont])

    gp_contributions = sorted(gp_contributions, key=lambda x: x[1], reverse=True)
    vec["gp_contributions"] = gp_contributions

    return vec

if __name__ == "__main__":
    infile = "/home/joosep/Downloads/pythia6_ttbar_0001_pandora.slcio"

    factory = LCFactory.getInstance()
    reader = factory.createLCReader()
    reader.open(infile)
    event_data = []
    
    nEvent = 0
    while True:
        evt = reader.readNextEvent()
        if (evt == None):
            print("EOF at event %d" % nEvent)
            break
    
        col = evt.getCollection("MCParticle")
        colPF = evt.getCollection("PandoraPFOCollection")
        colCl = evt.getCollection("ReconClusters")
        colTr = evt.getCollection("Tracks")

        simTrackHits = evt.getCollection("HelicalTrackHits")
        simTrackHitToReco = evt.getCollection("HelicalTrackHitRelations")
        simTrackHitToGen = evt.getCollection("HelicalTrackMCRelations")

        reco_trackhit_to_sim = {}
        sim_trackhit_to_gen = {}
        for shr in simTrackHitToReco:
            sh = shr.from
            rh = shr.to
            if not (rh in reco_trackhit_to_sim):
                reco_trackhit_to_sim[rh] = []
            reco_trackhit_to_sim[rh].append(sh)

        for shg in simTrackHitToGen:
            sh = shg.from
            gp = shg.to
            if not (sh in sim_trackhit_to_gen):
                sim_trackhit_to_gen[sh] = []
            sim_trackhit_to_gen[sh].append(gp)

        colHCB = evt.getCollection("HAD_BARREL")
        colHCE = evt.getCollection("HAD_ENDCAP")
        set_hcal_hits = set(list(colHCB) + list(colHCE))

        colECB = evt.getCollection("EM_BARREL")
        colECE = evt.getCollection("EM_ENDCAP")
        set_ecal_hits = set(list(colECB) + list(colECE))

        nMc=col.getNumberOfElements()
        nPF=colPF.getNumberOfElements()
        nCl=colCl.getNumberOfElements()
        nTr=colTr.getNumberOfElements()
        nHCB=colHCB.getNumberOfElements()
        nHCE=colHCE.getNumberOfElements()
        nECB=colECB.getNumberOfElements()
        nECE=colECE.getNumberOfElements()
        
        calohit_relations = evt.getCollection("CalorimeterHitRelations")
        calohit_recotosim = {}
        for c in calohit_relations:
            recohit = c.from
            simhit = c.to
            assert(not (recohit in calohit_recotosim))
            calohit_recotosim[recohit] = simhit
 
        print "Event %d, nGen=%d, nPF=%d, nClusters=%d, nTracks=%d, nHCAL=%d, nECAL=%d" % (nEvent, nMc, nPF, nCl, nTr, nHCB+nHCE, nECB+nECE)
    
        genparticles = []
        genparticle_dict = {}
        for i in range(nMc): # loop over all particles 
            par=col.getElementAt(i)
            genparticle_dict[par] = i
            vec = genParticleToDict(par)
            genparticles.append(vec)
    
        clusters = []
        cluster_dict = {}
        calohit_to_cluster = {}
        for i in range(nCl):
            parCl = colCl.getElementAt(i)
            pos = parCl.getPosition()
            cluster_dict[parCl] = i
            vec = clusterToDict(parCl)
            clusters.append(vec)
            for hit in parCl.getCalorimeterHits():
                calohit_to_cluster[hit] = i
    
        tracks = [] 
        track_dict = {}
        for i in range(nTr):
            parTr = colTr.getElementAt(i)
            track_dict[parTr] = i
            vec = trackToDict(parTr)
            tracks.append(vec)
    
        pfs = []
        for i in range(nPF): # loop over all particles
            parPF = colPF.getElementAt(i) 
            vec = pfParticleToDict(parPF)

            cluster_index = -1
            assert(len(parPF.getClusters())<=1)
            for cl in parPF.getClusters():
                cluster_index = cluster_dict[cl]
                break
          
            track_index = -1
            for tr in parPF.getTracks():
                track_index = track_dict[tr]
                break
 
            vec["cluster_idx"] = cluster_index
            vec["track_idx"] = track_index

            pfs.append(vec)


        track_hits = []
        for i in range(len(simTrackHits)):
            par = simTrackHits.getElementAt(i)
            track_hits.append(trackHitToDict(par))

        hcal_hits = []
        for i in range(nHCB):
            par = colHCB.getElementAt(i)
            hcal_hits.append(caloHitToDict(par, calohit_to_cluster, genparticle_dict, calohit_recotosim))
        for i in range(nHCE):
            par = colHCE.getElementAt(i)
            hcal_hits.append(caloHitToDict(par, calohit_to_cluster, genparticle_dict, calohit_recotosim))

        ecal_hits = []
        for i in range(nECB):
            par = colECB.getElementAt(i);
            ecal_hits.append(caloHitToDict(par, calohit_to_cluster, genparticle_dict, calohit_recotosim))
        for i in range(nECE):
            par = colECE.getElementAt(i);
            ecal_hits.append(caloHitToDict(par, calohit_to_cluster, genparticle_dict, calohit_recotosim))

        for hit in hcal_hits + ecal_hits:
            clidx = hit["cluster_idx"]
            gps = hit.pop("gp_contributions")
            if clidx >= 0:
                for gp_contrib in gps:
                    clusters[clidx]["gp_contributions"] += gps
            hit["gp_contributions"] = {c[0]: c[1] for c in gps}

        for icl in range(len(clusters)):
            gps = clusters[icl].pop("gp_contributions")
            gps_d = {}
            for gp, energy in gps:
                if not (gp in gps_d):
                    gps_d[gp] = 0.0
                gps_d[gp] += energy
            clusters[icl]["gp_contributions"] = gps_d

        event = {
            "track_hits": track_hits,
            "genparticles": genparticles,
            "clusters": clusters,
            "tracks": tracks,
            "pfs": pfs,
            "hcal_hits": hcal_hits,
            "ecal_hits": ecal_hits,
        }
 
        event_data.append(event)
        nEvent += 1
   
    #save the event data to a file 
    ofi = bz2.BZ2File(infile.replace(".slcio", ".json.bz2"), "w")
    json.dump(event_data, ofi, indent=2, sort_keys=True)
    ofi.close()
    
    reader.close() # close the file
