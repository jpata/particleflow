#!/usr/bin/env python3
"""Extract same-event metric comparisons with FWLite; run in CMSSW."""
import argparse
import collections
import json
from pathlib import Path
import ROOT
from DataFormats.FWLite import Events,Handle
ROOT.gInterpreter.Declare('#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"')

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('input',type=Path)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--skip-legacy',action='store_true',help='Export graph weights without requiring legacy track maps')
parser.add_argument('--sparse',action='store_true',help='Export only overlapping calorimeter pairs; requires --skip-legacy')
args=parser.parse_args()
if args.sparse and not args.skip_legacy: parser.error('--sparse requires --skip-legacy')
handles={}
def get(event,kind,label,instance=''):
    key=(kind,label,instance)
    if key not in handles: handles[key]=Handle(kind)
    event.getByLabel(label,instance,handles[key])
    return handles[key].product()

def channel_hits(index,root,channel):
    hits=collections.defaultdict(float)
    if index.sharedSubgraphStore():
        ranges=index.subgraphRanges(root)
        for i in range(ranges.size()):
            values=index.rangeHits(channel,ranges[i])
            for j in range(values.size()):
                h=values[j]
                hits[int(h.detId)]+=float(h.energy)
    else:
        values=index.subgraphHits(channel,root)
        for j in range(values.size()):
            h=values[j]
            hits[int(h.detId)]+=float(h.energy)
    return hits

def calo_values(fractions,branch,total,reco):
    old=shared=score_num=denom=0.
    shared_reco=score_reco_num=denom_reco=0.
    sim_branch_denom=sum(e*e for e in branch.values())
    reco_branch={cell:(energy/total[cell]*reco.get(cell,0.) if total[cell]>0 else 0.) for cell,energy in branch.items()}
    reco_branch_denom=sum(e*e for e in reco_branch.values())
    captured_sim=captured_reco=0.
    for cell,fraction in fractions.items():
        s=total[cell];b=branch.get(cell,0.);e=reco.get(cell,0.)
        old+=fraction*e*(b/s if s>0 else 0.)
        r=fraction*s
        shared+=min(r,b);score_num+=max(0.,r-b)**2;denom+=r*r
        captured_sim+=b*b-max(0.,b-r)**2
        rr=fraction*e;br=reco_branch.get(cell,0.)
        shared_reco+=min(rr,br);score_reco_num+=max(0.,rr-br)**2;denom_reco+=rr*rr
        captured_reco+=br*br-max(0.,br-rr)**2
    return dict(old_graph=old,sim_shared=shared,sim_score=score_num/denom if denom>0 else 1.,
                sim_reverse=max(0.,sim_branch_denom-captured_sim)/sim_branch_denom if sim_branch_denom>0 else 1.,
                reco_shared=shared_reco,reco_score=score_reco_num/denom_reco if denom_reco>0 else 1.,
                reco_reverse=max(0.,reco_branch_denom-captured_reco)/reco_branch_denom if reco_branch_denom>0 else 1.)

out=[];metadata=[]
for event_index,event in enumerate(Events(str(args.input.resolve()))):
    number=int(event.eventAuxiliary().event())
    graph=get(event,'truth::Graph','truthLogicalGraphProducer')
    index=get(event,'truth::LogicalGraphHitIndex','truthLogicalGraphHitIndexProducer')
    raw=get(event,'TruthGraph','mix')
    particles=list(graph.particles())
    roots=[i for i,p in enumerate(particles) if p.isAtLevel(ROOT.truth.LevelFlag.PFReconstructable)]
    covered=set();stack=list(roots)
    while stack:
        node=stack.pop()
        if node in covered: continue
        covered.add(node)
        vertices=graph.decayVertices(node)
        for j in range(vertices.size()):
            children=graph.outgoingParticles(vertices[j])
            stack.extend(int(children[k]) for k in range(children.size()))
    detected=[i for i,p in enumerate(particles) if p.hasTrackerSimHit or p.hasCaloSimHit or p.hasMuonSimHit]
    cps=get(event,'std::vector<CaloParticle>','mix','MergedCaloTruth')
    tps=get(event,'std::vector<TrackingParticle>','mix','MergedTrackTruth')
    def sim_key(p):
        return (int(raw.nodes()[p.simNode].key),int(p.bunchCrossing()),int(p.eventIndex())) if p.simNode>=0 else None
    def track_key(t):return(int(t.trackId()),int(t.eventId().bunchCrossing()),int(t.eventId().event()))
    cpmap={};tpmap={};cp_by_sim={};tp_by_sim={}
    for i,c in enumerate(cps):
        if len(c.g4Tracks()): cp_by_sim.setdefault(track_key(c.g4Tracks()[0]),i)
    for i,t in enumerate(tps):
        if len(t.g4Tracks()): tp_by_sim.setdefault(track_key(t.g4Tracks()[0]),i)
    for row,root in enumerate(roots):
        key=sim_key(particles[root])
        cpmap[row]=cp_by_sim.get(key,-1)
        tpmap[row]=tp_by_sim.get(key,-1)
    cpfracs=[]
    for cp in cps:
        fractions=collections.defaultdict(float)
        if not args.sparse:
            for ref in cp.simClusters():
                sc=ref.get()
                for pair in sc.hits_and_fractions(): fractions[int(pair.first)]+=float(pair.second)
        cpfracs.append(fractions)
    channel=ROOT.truth.HitChannel.Calo
    total=collections.defaultdict(float)
    for h in index.channel(channel).directHits: total[int(h.detId)]+=float(h.energy)
    branch=[channel_hits(index,root,channel) for root in roots]
    roots_by_cell=collections.defaultdict(set)
    if args.sparse:
        for row,hits in enumerate(branch):
            for cell in hits: roots_by_cell[cell].add(row)
    reco={}
    for label in ['particleFlowRecHitECAL','particleFlowRecHitHBHE']:
        for hit in get(event,'std::vector<reco::PFRecHit>',label):reco[int(hit.detId())]=float(hit.energy())
    for instance in ['HGCEERecHits','HGCHEFRecHits','HGCHEBRecHits']:
        for hit in get(event,'std::vector<io_v1::HGCRecHit>','HGCalRecHit',instance):reco[int(hit.id().rawId())]=float(hit.energy())
    objects={}
    for name,label in [('ECAL','particleFlowClusterECAL'),('HCAL','particleFlowClusterHCAL')]:
        values=[]
        for cluster in get(event,'std::vector<reco::PFCluster>',label):
            fractions=collections.defaultdict(float)
            for pair in cluster.hitsAndFractions(): fractions[int(pair.first.rawId())]+=float(pair.second)
            legacy=collections.defaultdict(float)
            for rh in cluster.recHitFractions():
                hit=rh.recHitRef().get()
                legacy[int(hit.detId())]+=float(hit.energy())*float(rh.fraction())
            values.append((dict(fractions),float(cluster.energy()),dict(legacy)))
        objects[name]=values
    lcs=get(event,'std::vector<reco::CaloCluster>','hgcalMergeLayerClusters')
    for name in ['ticlTrackstersCLUE3DHigh','ticlTracksterLinks']:
        values=[]
        for ts in get(event,'std::vector<ticl::Trackster>',name):
            fractions=collections.defaultdict(float)
            for lc,mult in zip(ts.vertices(),ts.vertex_multiplicity()):
                mult=float(mult) if mult>0 else 1.
                for pair in lcs[int(lc)].hitsAndFractions():fractions[int(pair.first.rawId())]+=float(pair.second)/mult
            legacy={cell:fraction*reco.get(cell,0.) for cell,fraction in fractions.items()}
            values.append((dict(fractions),float(ts.raw_energy()),legacy))
        objects[name]=values
    for collection,values in objects.items():
        for obj,(fractions,energy,legacy) in enumerate(values):
            candidates=sorted(set().union(*(roots_by_cell[cell] for cell in fractions))) if args.sparse else range(len(roots))
            for row in candidates:
                root=roots[row]
                cp=cpmap[row];old_cp=None if args.sparse else (sum(e*cpfracs[cp].get(cell,0.) for cell,e in legacy.items()) if cp>=0 else 0.)
                link=bool(set(fractions)&set(branch[row]))
                metrics=calo_values(fractions,branch[row],total,reco)
                if args.sparse and metrics['old_graph']<=0: continue
                out.append(dict(event=number,collection=collection,object=obj,root=row,cp=cp,tp=tpmap[row],
                                graph_link=link,old_cp=old_cp,cp_link=old_cp>0 if old_cp is not None else None,reco_energy=energy,
                                truth_energy=float(particles[root].momentum.energy()),**metrics))
    # AssociationMap's interactive iterator cache can persist across FWLite entries.
    # Open a fresh event reader before dereferencing these association maps.
    legacy_event=None
    if not args.skip_legacy:
        legacy_event=Events(str(args.input.resolve()))
        legacy_event.to(event_index)
    for collection,label,kind,assoc in [('Track','generalTracks','std::vector<reco::Track>','trackingParticleRecoTrackAsssociation'),('GSF','electronGsfTracks','std::vector<reco::GsfTrack>','trackingParticleGsfTrackAssociation')]:
        if args.sparse: continue  # Track links are already stored in NanoAOD.
        quality={}
        if legacy_event is not None:
            for entry in get(legacy_event,'reco::RecoToSimCollection',assoc):
                for match in entry.val:quality[(int(entry.key.key()),int(match.first.key()))]=float(match.second)
        for obj,track in enumerate(get(event,kind,label)):
            for row,root in enumerate(roots):
                old=quality.get((obj,tpmap[row]),-1.)
                out.append(dict(event=number,collection=collection,object=obj,root=row,tp=tpmap[row],cp=cpmap[row],old_quality=old,old_track_link=old>=0,reco_energy=float(track.p()),truth_energy=float(particles[root].momentum.energy())))
    gen_status1=[];gen_last_copy_tops=[]
    for i,p in enumerate(get(event,'std::vector<reco::GenParticle>','genParticles')):
        if abs(int(p.pdgId()))==6 and p.statusFlags().isLastCopy():
            gen_last_copy_tops.append(dict(index=i,pdg_id=int(p.pdgId()),status=int(p.status()),
                                          p4=[float(p.px()),float(p.py()),float(p.pz()),float(p.energy())]))
        if p.status()==1:
            gen_status1.append(dict(index=i,pdg_id=int(p.pdgId()),charge=float(p.charge()),
                                    p4=[float(p.px()),float(p.py()),float(p.pz()),float(p.energy())]))
    def gen_ancestor_indices(root):
        found=set();seen=set();stack=[root]
        while stack:
            node=stack.pop()
            if node in seen:continue
            seen.add(node);particle=particles[node]
            if particle.hasGen() and particle.status==1:
                p4=[particle.momentum.px(),particle.momentum.py(),particle.momentum.pz(),particle.momentum.energy()]
                for gen in gen_status1:
                    if gen['pdg_id']==particle.pdgId and all(abs(a-b)<1e-5*max(1.,abs(a)) for a,b in zip(p4,gen['p4'])):
                        found.add(gen['index'])
                continue
            vertices=graph.productionVertices(node)
            for j in range(vertices.size()):
                parents=graph.incomingParticles(vertices[j])
                stack.extend(int(parents[k]) for k in range(parents.size()))
        return sorted(found)
    root_p4=[[float(particles[r].momentum.px()),float(particles[r].momentum.py()),float(particles[r].momentum.pz()),float(particles[r].momentum.energy())] for r in roots]
    metadata.append(dict(event=number,run=int(event.eventAuxiliary().run()),lumi=int(event.eventAuxiliary().luminosityBlock()),
                         roots=roots,root_pdg_ids=[int(particles[r].pdgId) for r in roots],root_p4=root_p4,
                         root_bunch_crossing=[int(particles[r].bunchCrossing()) for r in roots],
                         root_sim_event=[int(particles[r].eventIndex()) for r in roots],
                         detected_particle_count=len(detected),uncovered_detected_particles=[i for i in detected if i not in covered],
                         signal_seeds=[dict(index=i,pdg_id=int(p.pdgId),status=int(p.status)) for i,p in enumerate(particles)
                                       if p.isAtLevel(ROOT.truth.LevelFlag.Signal)],
                         gen_last_copy_tops=gen_last_copy_tops,
                         gen_status1=gen_status1,gen_ancestors=[gen_ancestor_indices(r) for r in roots],
                         cp_map=cpmap,tp_map=tpmap,caloparticles=len(cps),trackingparticles=len(tps)))
    if args.sparse:
        print('Event',number,'CP',len(cps),'TP',len(tps),'PF roots',len(roots),'uncovered detected',len(metadata[-1]['uncovered_detected_particles']),flush=True)
    else:
        print('Event',number,'CP',len(cps),'TP',len(tps),'roots',roots,'CP map',cpmap,flush=True)
args.output.write_text(json.dumps(dict(rows=out,events=metadata,sparse=args.sparse),indent=2,allow_nan=False)+'\n')
print('Saved',len(out),'rows to',args.output)
