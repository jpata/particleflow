import ROOT
import sys
import numpy as np
from DataFormats.FWLite import Events, Handle

filename = sys.argv[1]
events = Events(filename)

class HandleLabel:
    def __init__(self, dtype, label):
        self.handle = Handle(dtype)
        self.label = (label, )

    def getByLabel(self, event):
        event.getByLabel(self.label, self.handle)

    def product(self):
        return self.handle.product()

class EventDesc:
    def __init__(self):
        self.genparticle = HandleLabel("std::vector<reco::GenParticle>", "prunedGenParticles")
        self.pfblock = HandleLabel("std::vector<reco::PFBlock>", "particleFlowBlock")
        self.pfcand = HandleLabel("std::vector<reco::PFCandidate>", "particleFlow")
        self.tracks = HandleLabel("std::vector<reco::Track>", "generalTracks")

    def get(self, event):
        self.genparticle.getByLabel(event) 
        self.pfcand.getByLabel(event) 
        self.tracks.getByLabel(event) 
        self.pfblock.getByLabel(event)

class Output:
    def __init__(self):
        self.tfile = ROOT.TFile(sys.argv[2], "RECREATE")
        
        self.pftree = ROOT.TTree("pftree", "pftree")
        self.linktree = ROOT.TTree("linktree", "linktree")

        #http://cmsdoxygen.web.cern.ch/cmsdoxygen/CMSSW_10_6_2/doc/html/d6/dd4/classreco_1_1PFCluster.html
        self.nclusters = np.zeros(1, dtype=np.uint32)
        self.maxclusters = 5000
        self.clusters_iblock = np.zeros(self.maxclusters, dtype=np.uint32)
        self.clusters_ielem = np.zeros(self.maxclusters, dtype=np.uint32)
        self.clusters_layer = np.zeros(self.maxclusters, dtype=np.int32)
        self.clusters_depth = np.zeros(self.maxclusters, dtype=np.int32)
        self.clusters_type = np.zeros(self.maxclusters, dtype=np.int32)
        self.clusters_energy = np.zeros(self.maxclusters, dtype=np.float32)
        self.clusters_eta = np.zeros(self.maxclusters, dtype=np.float32)
        self.clusters_phi = np.zeros(self.maxclusters, dtype=np.float32)
        self.clusters_x = np.zeros(self.maxclusters, dtype=np.float32)
        self.clusters_y = np.zeros(self.maxclusters, dtype=np.float32)
        self.clusters_z = np.zeros(self.maxclusters, dtype=np.float32)
        
        self.pftree.Branch("nclusters", self.nclusters, "nclusters/i")
        self.pftree.Branch("clusters_iblock", self.clusters_iblock, "clusters_iblock[nclusters]/i")
        self.pftree.Branch("clusters_ielem", self.clusters_ielem, "clusters_ielem[nclusters]/i")
        self.pftree.Branch("clusters_layer", self.clusters_layer, "clusters_layer[nclusters]/I")
        self.pftree.Branch("clusters_depth", self.clusters_depth, "clusters_depth[nclusters]/I")
        self.pftree.Branch("clusters_type", self.clusters_type, "clusters_type[nclusters]/I")
        self.pftree.Branch("clusters_energy", self.clusters_energy, "clusters_energy[nclusters]/F")
        self.pftree.Branch("clusters_x", self.clusters_x, "clusters_x[nclusters]/F")
        self.pftree.Branch("clusters_y", self.clusters_y, "clusters_y[nclusters]/F")
        self.pftree.Branch("clusters_z", self.clusters_z, "clusters_z[nclusters]/F")
        self.pftree.Branch("clusters_eta", self.clusters_eta, "clusters_eta[nclusters]/F")
        self.pftree.Branch("clusters_phi", self.clusters_phi, "clusters_phi[nclusters]/F")
        
        self.ngenparticles = np.zeros(1, dtype=np.uint32)
        self.maxgenparticles = 1000
        self.genparticles_pt = np.zeros(self.maxgenparticles, dtype=np.float32)
        self.genparticles_eta = np.zeros(self.maxgenparticles, dtype=np.float32)
        self.genparticles_phi = np.zeros(self.maxgenparticles, dtype=np.float32)
        self.genparticles_x = np.zeros(self.maxgenparticles, dtype=np.float32)
        self.genparticles_y = np.zeros(self.maxgenparticles, dtype=np.float32)
        self.genparticles_z = np.zeros(self.maxgenparticles, dtype=np.float32)
        self.genparticles_pdgid = np.zeros(self.maxgenparticles, dtype=np.int32)
        
        self.pftree.Branch("ngenparticles", self.ngenparticles, "ngenparticles/i")
        self.pftree.Branch("genparticles_pt", self.genparticles_pt, "genparticles_pt[ngenparticles]/F")
        self.pftree.Branch("genparticles_eta", self.genparticles_eta, "genparticles_eta[ngenparticles]/F")
        self.pftree.Branch("genparticles_phi", self.genparticles_phi, "genparticles_phi[ngenparticles]/F")
        self.pftree.Branch("genparticles_x", self.genparticles_x, "genparticles_x[ngenparticles]/F")
        self.pftree.Branch("genparticles_y", self.genparticles_y, "genparticles_y[ngenparticles]/F")
        self.pftree.Branch("genparticles_z", self.genparticles_z, "genparticles_z[ngenparticles]/F")
        self.pftree.Branch("genparticles_pdgid", self.genparticles_pdgid, "genparticles_pdgid[ngenparticles]/I")
       
        #http://cmsdoxygen.web.cern.ch/cmsdoxygen/CMSSW_10_6_2/doc/html/dd/d5b/classreco_1_1Track.html 
        self.ntracks = np.zeros(1, dtype=np.uint32)
        self.maxtracks = 5000
        self.tracks_iblock = np.zeros(self.maxtracks, dtype=np.uint32)
        self.tracks_ielem = np.zeros(self.maxtracks, dtype=np.uint32)
        self.tracks_qoverp = np.zeros(self.maxtracks, dtype=np.float32)
        self.tracks_lambda = np.zeros(self.maxtracks, dtype=np.float32)
        self.tracks_phi = np.zeros(self.maxtracks, dtype=np.float32)
        self.tracks_dxy = np.zeros(self.maxtracks, dtype=np.float32)
        self.tracks_dsz = np.zeros(self.maxtracks, dtype=np.float32)
        self.tracks_inner_eta = np.zeros(self.maxtracks, dtype=np.float32)
        self.tracks_inner_phi = np.zeros(self.maxtracks, dtype=np.float32)
        self.tracks_outer_eta = np.zeros(self.maxtracks, dtype=np.float32)
        self.tracks_outer_phi = np.zeros(self.maxtracks, dtype=np.float32)
        self.tracks_inner_eta = np.zeros(self.maxtracks, dtype=np.float32)
        self.tracks_inner_phi = np.zeros(self.maxtracks, dtype=np.float32)
        
        self.pftree.Branch("ntracks", self.ntracks, "ntracks/i")
        self.pftree.Branch("tracks_iblock", self.tracks_iblock, "tracks_iblock[ntracks]/i")
        self.pftree.Branch("tracks_ielem", self.tracks_ielem, "tracks_ielem[ntracks]/i")
        self.pftree.Branch("tracks_qoverp", self.tracks_qoverp, "tracks_qoverp[ntracks]/F")
        self.pftree.Branch("tracks_lambda", self.tracks_lambda, "tracks_lambda[ntracks]/F")
        self.pftree.Branch("tracks_phi", self.tracks_phi, "tracks_phi[ntracks]/F")
        self.pftree.Branch("tracks_dxy", self.tracks_dxy, "tracks_dxy[ntracks]/F")
        self.pftree.Branch("tracks_dsz", self.tracks_dsz, "tracks_dsz[ntracks]/F")
        self.pftree.Branch("tracks_outer_eta", self.tracks_outer_eta, "tracks_outer_eta[ntracks]/F")
        self.pftree.Branch("tracks_outer_phi", self.tracks_outer_phi, "tracks_outer_phi[ntracks]/F")
        self.pftree.Branch("tracks_inner_eta", self.tracks_inner_eta, "tracks_inner_eta[ntracks]/F")
        self.pftree.Branch("tracks_inner_phi", self.tracks_inner_phi, "tracks_inner_phi[ntracks]/F")
       
        #http://cmsdoxygen.web.cern.ch/cmsdoxygen/CMSSW_10_6_2/doc/html/dc/d55/classreco_1_1PFCandidate.html 
        self.npfcands = np.zeros(1, dtype=np.uint32)
        self.maxpfcands = 2000
        self.pfcands_pt = np.zeros(self.maxpfcands, dtype=np.float32)
        self.pfcands_eta = np.zeros(self.maxpfcands, dtype=np.float32)
        self.pfcands_phi = np.zeros(self.maxpfcands, dtype=np.float32)
        self.pfcands_charge = np.zeros(self.maxpfcands, dtype=np.float32)
        self.pfcands_energy = np.zeros(self.maxpfcands, dtype=np.float32)
        self.pfcands_pdgid = np.zeros(self.maxpfcands, dtype=np.int32)
        self.pfcands_iblock = np.zeros(self.maxpfcands, dtype=np.int32)
        self.pfcands_ielem = np.zeros(self.maxpfcands, dtype=np.int32)
        
        self.pftree.Branch("npfcands", self.npfcands, "npfcands/i")
        self.pftree.Branch("pfcands_pt", self.pfcands_pt, "pfcands_pt[npfcands]/F")
        self.pftree.Branch("pfcands_eta", self.pfcands_eta, "pfcands_eta[npfcands]/F")
        self.pftree.Branch("pfcands_phi", self.pfcands_phi, "pfcands_phi[npfcands]/F")
        self.pftree.Branch("pfcands_charge", self.pfcands_charge, "pfcands_charge[npfcands]/F")
        self.pftree.Branch("pfcands_energy", self.pfcands_energy, "pfcands_energy[npfcands]/F")
        self.pftree.Branch("pfcands_pdgid", self.pfcands_pdgid, "pfcands_pdgid[npfcands]/I")
        self.pftree.Branch("pfcands_iblock", self.pfcands_iblock, "pfcands_iblock[npfcands]/I")
        self.pftree.Branch("pfcands_ielem", self.pfcands_ielem, "pfcands_ielem[npfcands]/I")
       
        self.maxlinkdata = 50000
        self.nlinkdata = np.zeros(1, dtype=np.uint32)
        self.linkdata_k = np.zeros(self.maxlinkdata, dtype=np.uint32)
        self.linkdata_distance = np.zeros(self.maxlinkdata, dtype=np.float32) 
        self.linkdata_test = np.zeros(self.maxlinkdata, dtype=np.int32)
        self.linkdata_iev = np.zeros(self.maxlinkdata, dtype=np.uint32)
        self.linkdata_iblock = np.zeros(self.maxlinkdata, dtype=np.uint32) 
        self.linkdata_nelem = np.zeros(self.maxlinkdata, dtype=np.uint32) 
        self.linktree.Branch("nlinkdata", self.nlinkdata, "nlinkdata/i")
        self.linktree.Branch("linkdata_k", self.linkdata_k, "linkdata_k[nlinkdata]/i")
        self.linktree.Branch("linkdata_distance", self.linkdata_distance, "linkdata_distance[nlinkdata]/F")
        self.linktree.Branch("linkdata_test", self.linkdata_test, "linkdata_test[nlinkdata]/I")
        self.linktree.Branch("linkdata_iev", self.linkdata_iev, "linkdata_iev[nlinkdata]/i")
        self.linktree.Branch("linkdata_iblock", self.linkdata_iblock, "linkdata_iblock[nlinkdata]/i")
        self.linktree.Branch("linkdata_nelem", self.linkdata_nelem, "linkdata_nelem[nlinkdata]/i")

    def close(self):
        self.tfile.Write()
        self.tfile.Close()

    def clear(self):
        self.nclusters[0] = 0        
        self.clusters_iblock[:] = 0
        self.clusters_ielem[:] = 0
        self.clusters_layer[:] = 0
        self.clusters_depth[:] = 0
        self.clusters_type[:] = 0
        self.clusters_energy[:] = 0
        self.clusters_eta[:] = 0
        self.clusters_phi[:] = 0
        self.clusters_x[:] = 0
        self.clusters_y[:] = 0
        self.clusters_z[:] = 0
        
        self.ngenparticles[0] = 0
        self.genparticles_pt[:] = 0 
        self.genparticles_eta[:] = 0
        self.genparticles_phi[:] = 0
        self.genparticles_x[:] = 0
        self.genparticles_y[:] = 0
        self.genparticles_z[:] = 0
        self.genparticles_pdgid[:] = 0
        
        self.ntracks[0] = 0
        self.tracks_iblock[:] = 0
        self.tracks_ielem[:] = 0
        self.tracks_qoverp[:] = 0
        self.tracks_lambda[:] = 0
        self.tracks_phi[:] = 0
        self.tracks_dxy[:] = 0
        self.tracks_dsz[:] = 0
        self.tracks_outer_eta[:] = 0
        self.tracks_outer_phi[:] = 0
        self.tracks_inner_eta[:] = 0
        self.tracks_inner_phi[:] = 0
        
        self.npfcands[0] = 0
        self.pfcands_pt[:] = 0
        self.pfcands_eta[:] = 0
        self.pfcands_phi[:] = 0
        self.pfcands_charge[:] = 0
        self.pfcands_energy[:] = 0
        self.pfcands_pdgid[:] = 0
        self.pfcands_iblock[:] = 0
        self.pfcands_ielem[:] = 0
        
        self.nlinkdata[0] = 0
        self.linkdata_k[:] = 0
        self.linkdata_distance[:] = 0
        self.linkdata_test[:] = 0
        self.linkdata_iev[:] = 0
        self.linkdata_iblock[:] = 0
        self.linkdata_nelem[:] = 0

if __name__ == "__main__":
    evdesc = EventDesc()
    output = Output()
    
    # loop over events
    for iev, event in enumerate(events):
        eid = event.object().id()
        eventId = (eid.run(), eid.luminosityBlock(), int(eid.event()))
    
        evdesc.get(event)
        output.clear()
    
        genpart = evdesc.genparticle.product()
        ngenparticles = 0
        for gp in sorted(genpart, key=lambda x: x.pt(), reverse=True):
            if gp.pt() < 1:
                continue
            output.genparticles_pt[ngenparticles] = gp.pt() 
            output.genparticles_eta[ngenparticles] = gp.eta() 
            output.genparticles_phi[ngenparticles] = gp.phi() 
            output.genparticles_pdgid[ngenparticles] = gp.pdgId() 
            output.genparticles_x[ngenparticles] = gp.px() 
            output.genparticles_y[ngenparticles] = gp.py() 
            output.genparticles_z[ngenparticles] = gp.pz() 
            ngenparticles += 1
        output.ngenparticles[0] = ngenparticles

        blocks = {}
        pfcand = evdesc.pfcand.product()
        npfcands = 0

        #create initial list of pf candidates
        pfcands_to_analyze = []
        
        npfcands = 0
        pfcand_to_block_element = {}
        
        for c in sorted(pfcand, key=lambda x: x.pt(), reverse=True):
            if c.pt() < 1:
                continue
            pfcands_to_analyze += [c]
            pfcand_to_block_element[npfcands] = []
            for el in c.elementsInBlocks():
                blocks[el.first.index()] = el.first
                pfcand_to_block_element[npfcands] += [(el.first.index(), el.second)] 
            npfcands += 1
        
        #map block product index to array index
        blidx_to_iblock = {}
        for ibl, k in enumerate(sorted(blocks.keys())):
            blidx_to_iblock[k] = ibl
        
        #get list of blocks to save from all pfcandidates
        blocks = [blocks[bl].get() for bl in sorted(blocks.keys())]
 
        #now save PF candidates
        npfcands = 0
        for c in pfcands_to_analyze:
            output.pfcands_pt[npfcands] = c.pt()
            output.pfcands_eta[npfcands] = c.eta()
            output.pfcands_phi[npfcands] = c.phi()
	    output.pfcands_charge[npfcands] = c.charge()
	    output.pfcands_energy[npfcands] = c.energy()
            output.pfcands_pdgid[npfcands] = c.pdgId()
            #take only the first block/element pair
            if len(pfcand_to_block_element[npfcands]) > 0:
                blidx, iel = pfcand_to_block_element[npfcands][0]
                output.pfcands_iblock[npfcands] = blidx_to_iblock[blidx]
                output.pfcands_ielem[npfcands] = int(iel)

            npfcands += 1

        output.npfcands[0] = npfcands
        #save blocks
        nblocks = 0
        nclusters = 0
        ntracks = 0
        nlinkdata = 0
        for iblock, bl in enumerate(blocks):
            for ielem, el in enumerate(bl.elements()):
                tp = el.type()
                if (tp == ROOT.reco.PFBlockElement.ECAL or
                   tp == ROOT.reco.PFBlockElement.PS1 or
                   tp == ROOT.reco.PFBlockElement.PS2 or
                   tp == ROOT.reco.PFBlockElement.HCAL or
                   tp == ROOT.reco.PFBlockElement.GSF or
                   tp == ROOT.reco.PFBlockElement.HO or
                   tp == ROOT.reco.PFBlockElement.HFHAD or
                   tp == ROOT.reco.PFBlockElement.HFEM):
                    clref = el.clusterRef()
                    if clref.isNonnull():
                        cl = clref.get()
                        output.clusters_layer[nclusters] = int(cl.layer())
                        output.clusters_depth[nclusters] = cl.depth()
                        output.clusters_energy[nclusters] = cl.energy()
                        output.clusters_x[nclusters] = cl.x()
                        output.clusters_y[nclusters] = cl.y()
                        output.clusters_z[nclusters] = cl.z()
                        output.clusters_eta[nclusters] = cl.eta()
                        output.clusters_phi[nclusters] = cl.phi()
                        output.clusters_type[nclusters] = int(tp)
                        output.clusters_iblock[nclusters] = iblock
                        output.clusters_ielem[nclusters] = ielem
                        nclusters += 1
                elif (tp == ROOT.reco.PFBlockElement.TRACK):
                   c = el.trackRef().get()
                   output.tracks_qoverp[ntracks] = c.qoverp()
                   output.tracks_lambda[ntracks] = getattr(c, "lambda")() #lambda is a reserved word in python, so we need to use a proxy
                   output.tracks_phi[ntracks] = c.phi()
                   output.tracks_dxy[ntracks] = c.dxy()
                   output.tracks_dsz[ntracks] = c.dsz()
                   output.tracks_outer_eta[ntracks] = c.outerPosition().eta()
                   output.tracks_outer_phi[ntracks] = c.outerPosition().phi()
                   output.tracks_inner_eta[ntracks] = c.innerPosition().eta()
                   output.tracks_inner_phi[ntracks] = c.innerPosition().phi()
                   output.tracks_iblock[ntracks] = iblock
                   output.tracks_ielem[ntracks] = ielem
                   ntracks += 1

            #save links for each block
            linkdata = {int(kv.first): (kv.second.distance, ord(kv.second.test)) for kv in bl.linkData()}
            for k, (v0, v1) in linkdata.items():
                output.linkdata_k[nlinkdata] = k
                output.linkdata_distance[nlinkdata] = 1000.0*v0 
                output.linkdata_test[nlinkdata] = v1 
                output.linkdata_iev[nlinkdata] = iev
                output.linkdata_iblock[nlinkdata] = iblock
                output.linkdata_nelem[nlinkdata] = len(bl.elements())
                nlinkdata += 1

        output.nclusters[0] = nclusters
        output.ntracks[0] = ntracks
        output.nlinkdata[0] = nlinkdata

        output.pftree.Fill()
        output.linktree.Fill()
    pass 
    #end of event loop 

    output.close()
