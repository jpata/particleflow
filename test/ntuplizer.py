from __future__ import print_function
import ROOT
import sys, os
import numpy as np
from DataFormats.FWLite import Events, Handle
import numba

#encode a 2d upper-triangular index (i,j) in a 1d vector as per CMSSW
@numba.njit
def get_index_triu_vector(i, j, vecsize):
    k = j - i - 1
    k += i*vecsize
    missing = int(i*(i+1)/2)
    k -= missing
    return k

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
        self.genparticle = HandleLabel("std::vector<reco::GenParticle>", "genParticles")
        self.simtrack = HandleLabel("std::vector<SimTrack>", "g4SimHits")
        self.pfblock = HandleLabel("std::vector<reco::PFBlock>", "particleFlowBlock")
        self.pfcand = HandleLabel("std::vector<reco::PFCandidate>", "particleFlow")
        self.tracks = HandleLabel("std::vector<reco::PFRecTrack>", "pfTrack")

    def get(self, event):
        self.genparticle.getByLabel(event) 
        self.simtrack.getByLabel(event) 
        self.pfcand.getByLabel(event) 
        self.tracks.getByLabel(event) 
        self.pfblock.getByLabel(event)

class Output:
    def __init__(self, outfile):
        self.tfile = ROOT.TFile(outfile, "RECREATE")
        
        self.pftree = ROOT.TTree("pftree", "pftree")
        self.linktree = ROOT.TTree("linktree", "linktree for elements in block")
        self.linktree_elemtocand = ROOT.TTree("linktree_elemtocand", "element to candidate link data")

        #http://cmsdoxygen.web.cern.ch/cmsdoxygen/CMSSW_10_6_2/doc/html/d6/dd4/classreco_1_1PFCluster.html
        self.nclusters = np.zeros(1, dtype=np.uint32)
        self.maxclusters = 5000
        self.clusters_iblock = np.zeros(self.maxclusters, dtype=np.uint32)
        self.clusters_ielem = np.zeros(self.maxclusters, dtype=np.uint32)
        self.clusters_layer = np.zeros(self.maxclusters, dtype=np.int32)
        self.clusters_depth = np.zeros(self.maxclusters, dtype=np.int32)
        self.clusters_type = np.zeros(self.maxclusters, dtype=np.int32)
        self.clusters_ecalIso = np.zeros(self.maxclusters, dtype=np.float32)
        self.clusters_hcalIso = np.zeros(self.maxclusters, dtype=np.float32)
        self.clusters_trackIso = np.zeros(self.maxclusters, dtype=np.float32)
        self.clusters_phiWidth = np.zeros(self.maxclusters, dtype=np.float32)
        self.clusters_etaWidth = np.zeros(self.maxclusters, dtype=np.float32)
        self.clusters_preshowerEnergy = np.zeros(self.maxclusters, dtype=np.float32)
        self.clusters_energy = np.zeros(self.maxclusters, dtype=np.float32)
        self.clusters_correctedEnergy = np.zeros(self.maxclusters, dtype=np.float32)
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
        self.pftree.Branch("clusters_ecalIso", self.clusters_ecalIso, "clusters_ecalIso[nclusters]/F")
        self.pftree.Branch("clusters_hcalIso", self.clusters_hcalIso, "clusters_hcalIso[nclusters]/F")
        self.pftree.Branch("clusters_trackIso", self.clusters_trackIso, "clusters_trackIso[nclusters]/F")
        self.pftree.Branch("clusters_phiWidth", self.clusters_phiWidth, "clusters_phiWidth[nclusters]/F")
        self.pftree.Branch("clusters_etaWidth", self.clusters_etaWidth, "clusters_etaWidth[nclusters]/F")
        self.pftree.Branch("clusters_preshowerEnergy", self.clusters_preshowerEnergy, "clusters_preshowerEnergy[nclusters]/F")
        self.pftree.Branch("clusters_energy", self.clusters_energy, "clusters_energy[nclusters]/F")
        self.pftree.Branch("clusters_correctedEnergy", self.clusters_correctedEnergy, "clusters_correctedEnergy[nclusters]/F")
        self.pftree.Branch("clusters_x", self.clusters_x, "clusters_x[nclusters]/F")
        self.pftree.Branch("clusters_y", self.clusters_y, "clusters_y[nclusters]/F")
        self.pftree.Branch("clusters_z", self.clusters_z, "clusters_z[nclusters]/F")
        self.pftree.Branch("clusters_eta", self.clusters_eta, "clusters_eta[nclusters]/F")
        self.pftree.Branch("clusters_phi", self.clusters_phi, "clusters_phi[nclusters]/F")
        
        self.ngenparticles = np.zeros(1, dtype=np.uint32)
        self.maxgenparticles = 100000
        self.genparticles_pt = np.zeros(self.maxgenparticles, dtype=np.float32)
        self.genparticles_eta = np.zeros(self.maxgenparticles, dtype=np.float32)
        self.genparticles_phi = np.zeros(self.maxgenparticles, dtype=np.float32)
        self.genparticles_x = np.zeros(self.maxgenparticles, dtype=np.float32)
        self.genparticles_y = np.zeros(self.maxgenparticles, dtype=np.float32)
        self.genparticles_z = np.zeros(self.maxgenparticles, dtype=np.float32)
        self.genparticles_pdgid = np.zeros(self.maxgenparticles, dtype=np.int32)
        self.genparticles_status = np.zeros(self.maxgenparticles, dtype=np.int32)
        
        self.pftree.Branch("ngenparticles", self.ngenparticles, "ngenparticles/i")
        self.pftree.Branch("genparticles_pt", self.genparticles_pt, "genparticles_pt[ngenparticles]/F")
        self.pftree.Branch("genparticles_eta", self.genparticles_eta, "genparticles_eta[ngenparticles]/F")
        self.pftree.Branch("genparticles_phi", self.genparticles_phi, "genparticles_phi[ngenparticles]/F")
        self.pftree.Branch("genparticles_x", self.genparticles_x, "genparticles_x[ngenparticles]/F")
        self.pftree.Branch("genparticles_y", self.genparticles_y, "genparticles_y[ngenparticles]/F")
        self.pftree.Branch("genparticles_z", self.genparticles_z, "genparticles_z[ngenparticles]/F")
        self.pftree.Branch("genparticles_pdgid", self.genparticles_pdgid, "genparticles_pdgid[ngenparticles]/I")
        self.pftree.Branch("genparticles_status", self.genparticles_status, "genparticles_status[ngenparticles]/I")
       
        #http://cmsdoxygen.web.cern.ch/cmsdoxygen/CMSSW_10_6_2/doc/html/dd/d5b/classreco_1_1Track.html 
        self.ntracks = np.zeros(1, dtype=np.uint32)
        self.maxtracks = 5000
        self.tracks_iblock = np.zeros(self.maxtracks, dtype=np.uint32)
        self.tracks_ielem = np.zeros(self.maxtracks, dtype=np.uint32)
        self.tracks_qoverp = np.zeros(self.maxtracks, dtype=np.float32)
        self.tracks_lambda = np.zeros(self.maxtracks, dtype=np.float32)
        self.tracks_phi = np.zeros(self.maxtracks, dtype=np.float32)
        self.tracks_eta = np.zeros(self.maxtracks, dtype=np.float32)
        self.tracks_dxy = np.zeros(self.maxtracks, dtype=np.float32)
        self.tracks_dsz = np.zeros(self.maxtracks, dtype=np.float32)
        self.tracks_inner_eta = np.zeros(self.maxtracks, dtype=np.float32)
        self.tracks_inner_phi = np.zeros(self.maxtracks, dtype=np.float32)
        self.tracks_outer_eta = np.zeros(self.maxtracks, dtype=np.float32)
        self.tracks_outer_phi = np.zeros(self.maxtracks, dtype=np.float32)
        
        self.pftree.Branch("ntracks", self.ntracks, "ntracks/i")
        self.pftree.Branch("tracks_iblock", self.tracks_iblock, "tracks_iblock[ntracks]/i")
        self.pftree.Branch("tracks_ielem", self.tracks_ielem, "tracks_ielem[ntracks]/i")
        self.pftree.Branch("tracks_qoverp", self.tracks_qoverp, "tracks_qoverp[ntracks]/F")
        self.pftree.Branch("tracks_lambda", self.tracks_lambda, "tracks_lambda[ntracks]/F")
        self.pftree.Branch("tracks_phi", self.tracks_phi, "tracks_phi[ntracks]/F")
        self.pftree.Branch("tracks_eta", self.tracks_eta, "tracks_eta[ntracks]/F")
        self.pftree.Branch("tracks_dxy", self.tracks_dxy, "tracks_dxy[ntracks]/F")
        self.pftree.Branch("tracks_dsz", self.tracks_dsz, "tracks_dsz[ntracks]/F")
        self.pftree.Branch("tracks_outer_eta", self.tracks_outer_eta, "tracks_outer_eta[ntracks]/F")
        self.pftree.Branch("tracks_outer_phi", self.tracks_outer_phi, "tracks_outer_phi[ntracks]/F")
        self.pftree.Branch("tracks_inner_eta", self.tracks_inner_eta, "tracks_inner_eta[ntracks]/F")
        self.pftree.Branch("tracks_inner_phi", self.tracks_inner_phi, "tracks_inner_phi[ntracks]/F")
       
        #http://cmsdoxygen.web.cern.ch/cmsdoxygen/CMSSW_10_6_2/doc/html/dc/d55/classreco_1_1PFCandidate.html 
        self.npfcands = np.zeros(1, dtype=np.uint32)
        self.maxpfcands = 5000
        self.pfcands_pt = np.zeros(self.maxpfcands, dtype=np.float32)
        self.pfcands_eta = np.zeros(self.maxpfcands, dtype=np.float32)
        self.pfcands_phi = np.zeros(self.maxpfcands, dtype=np.float32)
        self.pfcands_charge = np.zeros(self.maxpfcands, dtype=np.float32)
        self.pfcands_energy = np.zeros(self.maxpfcands, dtype=np.float32)
        self.pfcands_pdgid = np.zeros(self.maxpfcands, dtype=np.int32)
        self.pfcands_nelem = np.zeros(self.maxpfcands, dtype=np.int32)
        self.pfcands_ielem0 = np.zeros(self.maxpfcands, dtype=np.int32)
        self.pfcands_ielem1 = np.zeros(self.maxpfcands, dtype=np.int32)
        self.pfcands_ielem2 = np.zeros(self.maxpfcands, dtype=np.int32)
        self.pfcands_ielem3 = np.zeros(self.maxpfcands, dtype=np.int32)
        self.pfcands_iblock = np.zeros(self.maxpfcands, dtype=np.int32)
        
        self.pftree.Branch("npfcands", self.npfcands, "npfcands/i")
        self.pftree.Branch("pfcands_pt", self.pfcands_pt, "pfcands_pt[npfcands]/F")
        self.pftree.Branch("pfcands_eta", self.pfcands_eta, "pfcands_eta[npfcands]/F")
        self.pftree.Branch("pfcands_phi", self.pfcands_phi, "pfcands_phi[npfcands]/F")
        self.pftree.Branch("pfcands_charge", self.pfcands_charge, "pfcands_charge[npfcands]/F")
        self.pftree.Branch("pfcands_energy", self.pfcands_energy, "pfcands_energy[npfcands]/F")
        self.pftree.Branch("pfcands_pdgid", self.pfcands_pdgid, "pfcands_pdgid[npfcands]/I")
        self.pftree.Branch("pfcands_nelem", self.pfcands_nelem, "pfcands_nelem[npfcands]/I")
        self.pftree.Branch("pfcands_ielem0", self.pfcands_ielem0, "pfcands_ielem0[npfcands]/I")
        self.pftree.Branch("pfcands_ielem1", self.pfcands_ielem1, "pfcands_ielem0[npfcands]/I")
        self.pftree.Branch("pfcands_ielem2", self.pfcands_ielem2, "pfcands_ielem0[npfcands]/I")
        self.pftree.Branch("pfcands_ielem3", self.pfcands_ielem3, "pfcands_ielem0[npfcands]/I")
        self.pftree.Branch("pfcands_iblock", self.pfcands_iblock, "pfcands_iblock[npfcands]/I")
       
        self.maxlinkdata = 50000
        self.nlinkdata = np.zeros(1, dtype=np.uint32)
        self.linkdata_distance = np.zeros(self.maxlinkdata, dtype=np.float32) 
        self.linkdata_iev = np.zeros(self.maxlinkdata, dtype=np.uint32)
        self.linkdata_iblock = np.zeros(self.maxlinkdata, dtype=np.uint32) 
        self.linkdata_ielem = np.zeros(self.maxlinkdata, dtype=np.uint32) 
        self.linkdata_jelem = np.zeros(self.maxlinkdata, dtype=np.uint32) 
        self.linktree.Branch("nlinkdata", self.nlinkdata, "nlinkdata/i")
        self.linktree.Branch("linkdata_distance", self.linkdata_distance, "linkdata_distance[nlinkdata]/F")
        self.linktree.Branch("linkdata_iev", self.linkdata_iev, "linkdata_iev[nlinkdata]/i")
        self.linktree.Branch("linkdata_iblock", self.linkdata_iblock, "linkdata_iblock[nlinkdata]/i")
        self.linktree.Branch("linkdata_ielem", self.linkdata_ielem, "linkdata_ielem[nlinkdata]/i")
        self.linktree.Branch("linkdata_jelem", self.linkdata_jelem, "linkdata_jelem[nlinkdata]/i")
        
        self.maxlinkdata_elemtocand = 50000
        self.nlinkdata_elemtocand = np.zeros(1, dtype=np.uint32)
        self.linkdata_elemtocand_iev = np.zeros(self.maxlinkdata_elemtocand, dtype=np.uint32)
        self.linkdata_elemtocand_iblock = np.zeros(self.maxlinkdata_elemtocand, dtype=np.uint32)
        self.linkdata_elemtocand_ielem = np.zeros(self.maxlinkdata_elemtocand, dtype=np.uint32)
        self.linkdata_elemtocand_icand = np.zeros(self.maxlinkdata_elemtocand, dtype=np.uint32)
        self.linktree_elemtocand.Branch("nlinkdata_elemtocand", self.nlinkdata_elemtocand, "nlinkdata_elemtocand/i")
        self.linktree_elemtocand.Branch("linkdata_elemtocand_iev", self.linkdata_elemtocand_iev, "linkdata_elemtocand_iev[nlinkdata_elemtocand]/i")
        self.linktree_elemtocand.Branch("linkdata_elemtocand_iblock", self.linkdata_elemtocand_iblock, "linkdata_elemtocand_iblock[nlinkdata_elemtocand]/i")
        self.linktree_elemtocand.Branch("linkdata_elemtocand_ielem", self.linkdata_elemtocand_ielem, "linkdata_elemtocand_ielem[nlinkdata_elemtocand]/i")
        self.linktree_elemtocand.Branch("linkdata_elemtocand_icand", self.linkdata_elemtocand_icand, "linkdata_elemtocand_icand[nlinkdata_elemtocand]/i")
    
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
        self.clusters_ecalIso[:] = 0
        self.clusters_hcalIso[:] = 0
        self.clusters_trackIso[:] = 0
        self.clusters_phiWidth[:] = 0
        self.clusters_etaWidth[:] = 0
        self.clusters_preshowerEnergy[:] = 0
        self.clusters_energy[:] = 0
        self.clusters_correctedEnergy[:] = 0
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
        self.genparticles_status[:] = 0
        
        self.ntracks[0] = 0
        self.tracks_iblock[:] = 0
        self.tracks_ielem[:] = 0
        self.tracks_qoverp[:] = 0
        self.tracks_lambda[:] = 0
        self.tracks_phi[:] = 0
        self.tracks_eta[:] = 0
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
        
        self.nlinkdata[0] = 0
        self.linkdata_distance[:] = 0
        self.linkdata_iev[:] = 0
        self.linkdata_iblock[:] = 0
        self.linkdata_ielem[:] = 0
        self.linkdata_jelem[:] = 0
        
        self.nlinkdata_elemtocand[0] = 0
        self.linkdata_elemtocand_iev[:] = 0
        self.linkdata_elemtocand_iblock[:] = 0
        self.linkdata_elemtocand_ielem[:] = 0
        self.linkdata_elemtocand_icand[:] = 0

if __name__ == "__main__":

    filename = sys.argv[2]
    outpath = sys.argv[1]
    outfile = os.path.join(outpath, os.path.basename(filename).replace("AOD", "ntuple"))
    if os.path.isfile(outfile):
        print("Output file {0} exists, exiting".format(outfile), file=sys.stderr)
        sys.exit(0)

    events = Events(filename)
    print("Reading input file {0}".format(filename))
    print("Saving output to file {0}".format(outfile))

    evdesc = EventDesc()
    output = Output(outfile)

    num_events = events.size()
    
    # loop over events
    for iev, event in enumerate(events):
        #if iev > 10:
        #    break
        eid = event.object().id()
        if iev%10 == 0:
            print("Event {0}/{1}".format(iev, num_events))
        eventId = (eid.run(), eid.luminosityBlock(), int(eid.event()))
    
        output.clear()
        evdesc.get(event)

        simtrack = evdesc.simtrack.product()
        print(simtrack.size())
        #genjets = evdesc.genjet.product()
        #genjet_daughters = []
        #for gj in genjets:
        #    nd = gj.numberOfDaughters()
        #    for idaughter in range(nd):
        #        genjet_daughters += [gj.daughter(idaughter)]
        
        genpart = evdesc.genparticle.product()
        ngenparticles = 0
        for gp in sorted(genpart, key=lambda x: x.pt(), reverse=True):
            output.genparticles_pt[ngenparticles] = gp.pt() 
            output.genparticles_eta[ngenparticles] = gp.eta() 
            output.genparticles_phi[ngenparticles] = gp.phi() 
            output.genparticles_pdgid[ngenparticles] = gp.pdgId() 
            output.genparticles_status[ngenparticles] = gp.status() 
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
       
        #for each PF candidate, create (block, elindex) 
        for c in sorted(pfcand, key=lambda x: x.pt(), reverse=True):
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
        blocks = [(bl, blocks[bl].get()) for bl in sorted(blocks.keys())]
 
        #now save PF candidates
        npfcands = 0
        blidx_ielem_to_pfcand = {}
        pfcand_elem_pairs = []
        for c in pfcands_to_analyze:
            output.pfcands_pt[npfcands] = c.pt()
            output.pfcands_eta[npfcands] = c.eta()
            output.pfcands_phi[npfcands] = c.phi()
            output.pfcands_charge[npfcands] = c.charge()
            output.pfcands_energy[npfcands] = c.energy()
            output.pfcands_pdgid[npfcands] = c.pdgId()
            output.pfcands_nelem[npfcands] = len(pfcand_to_block_element[npfcands])
            #fill the map of element -> pfcandidate 
            if len(pfcand_to_block_element[npfcands]) > 0:
                blidx_ = -1
                for ipf_block_elem in range(len(pfcand_to_block_element[npfcands])):
                    blidx, iel = pfcand_to_block_element[npfcands][ipf_block_elem]
                    if ipf_block_elem == 0:
                        blidx_ = blidx
                    else:
                        assert(blidx == blidx_)
                    if ipf_block_elem < 4:
                        getattr(output, "pfcands_ielem{0}".format(ipf_block_elem))[npfcands] = iel
                    k = (int(blidx), int(iel))
                    pfcand_elem_pairs += [(int(blidx), int(iel), npfcands)]
                    if not k in blidx_ielem_to_pfcand:
                        blidx_ielem_to_pfcand[k] = []
                    blidx_ielem_to_pfcand[k] += [npfcands]
                output.pfcands_iblock[npfcands] = blidx_ 
            npfcands += 1
        
        pftracks = evdesc.tracks.product()
        pftracks_dict = {t.trackRef().qoverp(): t for t in pftracks}

        output.npfcands[0] = npfcands
        nblocks = 0
        nclusters = 0
        ntracks = 0
        nlinkdata = 0
        nlinkdata_elemtocand = 0

        #save blocks
        for iblock, bl in blocks:
            for ielem, el in enumerate(bl.elements()):
                tp = el.type()
                matched_pfcands = blidx_ielem_to_pfcand.get((int(iblock), int(ielem)), [])

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
                    matched_pftrack = el.trackRefPF().get()

                    #matched_pftrack = pftracks_dict.get(el.trackRef().qoverp(), None)
                    if not matched_pftrack is None:
                        assert(matched_pftrack.trackRef().qoverp() == el.trackRef().qoverp())
                        assert(matched_pftrack.trackRef() == el.trackRef())
                        atECAL = matched_pftrack.extrapolatedPoint(ROOT.reco.PFTrajectoryPoint.ECALShowerMax)  
                        atHCAL = matched_pftrack.extrapolatedPoint(ROOT.reco.PFTrajectoryPoint.HCALEntrance)  
                        if atHCAL.isValid():
                            output.tracks_outer_eta[ntracks] = atHCAL.positionREP().eta()
                            output.tracks_outer_phi[ntracks] = atHCAL.positionREP().phi()
                        if atECAL.isValid():
                            output.tracks_inner_eta[ntracks] = atECAL.positionREP().eta()
                            output.tracks_inner_phi[ntracks] = atECAL.positionREP().phi()
                        output.tracks_eta[ntracks] = c.momentum().eta()
                        output.tracks_phi[ntracks] = c.momentum().phi()
                    
                    #mr = el.muonRef()
                    #if mr and mr.isNonnull():
                    #    muon = mr.get()
                    #    print("MU", muon.isGlobalMuon(), muon.isStandAloneMuon(), muon.isTrackerMuon(), muon.numberOfMatches())

                    output.tracks_qoverp[ntracks] = c.qoverp()
                    output.tracks_lambda[ntracks] = getattr(c, "lambda")() #lambda is a reserved word in python, so we need to use a proxy
                    output.tracks_dxy[ntracks] = c.dxy()
                    output.tracks_dsz[ntracks] = c.dsz()
                    output.tracks_iblock[ntracks] = iblock
                    output.tracks_ielem[ntracks] = ielem
                    ntracks += 1
                elif (tp == ROOT.reco.PFBlockElement.BREM):
                    matched_pftrack = el.trackPF()
                    momentum = matched_pftrack.innermostMeasurement().momentum()
                    atECAL = matched_pftrack.extrapolatedPoint(ROOT.reco.PFTrajectoryPoint.ECALShowerMax)  
                    atHCAL = matched_pftrack.extrapolatedPoint(ROOT.reco.PFTrajectoryPoint.HCALEntrance)  
                    if atHCAL.isValid():
                        output.tracks_outer_eta[ntracks] = atHCAL.positionREP().eta()
                        output.tracks_outer_phi[ntracks] = atHCAL.positionREP().phi()
                    if atECAL.isValid():
                        output.tracks_inner_eta[ntracks] = atECAL.positionREP().eta()
                        output.tracks_inner_phi[ntracks] = atECAL.positionREP().phi()
                    output.tracks_eta[ntracks] = momentum.eta()
                    output.tracks_phi[ntracks] = momentum.phi()
                    
                    p = momentum.P()
                    output.tracks_qoverp[ntracks] = 0
                    if p > 0:
                        output.tracks_qoverp[ntracks] = matched_pftrack.charge() / p
                    output.tracks_dxy[ntracks] = 0
                    output.tracks_dsz[ntracks] = 0
                    output.tracks_iblock[ntracks] = iblock
                    output.tracks_ielem[ntracks] = ielem
                    ntracks += 1
                elif (tp == ROOT.reco.PFBlockElement.SC):
                    scref = el.superClusterRef()
                    if scref.isNonnull():
                        cl = scref.get()
                        output.clusters_layer[nclusters] = 0
                        output.clusters_depth[nclusters] = 0
                        output.clusters_energy[nclusters] = cl.energy()
                        output.clusters_correctedEnergy[nclusters] = cl.correctedEnergy()
                        output.clusters_x[nclusters] = cl.x()
                        output.clusters_y[nclusters] = cl.y()
                        output.clusters_z[nclusters] = cl.z()
                        output.clusters_eta[nclusters] = cl.eta()
                        output.clusters_phi[nclusters] = cl.phi()
                        output.clusters_type[nclusters] = int(tp)
                        output.clusters_ecalIso[nclusters] = el.ecalIso()
                        output.clusters_hcalIso[nclusters] = el.hcalIso()
                        output.clusters_trackIso[nclusters] = el.trackIso()
                        output.clusters_phiWidth[nclusters] = cl.phiWidth()
                        output.clusters_etaWidth[nclusters] = cl.etaWidth()
                        output.clusters_preshowerEnergy[nclusters] = cl.preshowerEnergy()
                        output.clusters_iblock[nclusters] = iblock
                        output.clusters_ielem[nclusters] = ielem
                else:
                    print("unknown type: {0}".format(tp))
                    

            #get link data for each block
            linkdata = {int(kv.first): (kv.second.distance, ord(kv.second.test)) for kv in bl.linkData()}
            vecsize = len(bl.elements())
            #get all the pairs of indices of an upper-triangular matrix
            inds_triu = np.triu_indices(n=vecsize, m=vecsize, k=1)
            #encode the 2d indices into a 1d vector
            inds_triu_encoded = {
                get_index_triu_vector(i, j, vecsize): (i,j) for i, j in zip(inds_triu[0], inds_triu[1])
            }
            #unencode the 1d vector stored by PF into 2d indices and get the distance 
            for k in linkdata.keys():
                i, j = inds_triu_encoded[k]
                output.linkdata_iev[nlinkdata] = iev
                output.linkdata_iblock[nlinkdata] = iblock
                output.linkdata_ielem[nlinkdata] = i
                output.linkdata_jelem[nlinkdata] = j
                output.linkdata_distance[nlinkdata] = linkdata[k][0]
                nlinkdata += 1
        #end of block loop
   
        #store the element to candidate links 
        for iblock, ielem, icand in sorted(pfcand_elem_pairs):
            output.linkdata_elemtocand_iev[nlinkdata_elemtocand] = iev
            output.linkdata_elemtocand_iblock[nlinkdata_elemtocand] = iblock
            output.linkdata_elemtocand_ielem[nlinkdata_elemtocand] = ielem
            output.linkdata_elemtocand_icand[nlinkdata_elemtocand] = icand
            nlinkdata_elemtocand += 1

        output.nclusters[0] = nclusters
        output.ntracks[0] = ntracks
        output.nlinkdata[0] = nlinkdata
        output.nlinkdata_elemtocand[0] = nlinkdata_elemtocand

        output.pftree.Fill()
        output.linktree.Fill()
        output.linktree_elemtocand.Fill()
    pass 
    #end of event loop 

    output.close()
