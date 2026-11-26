#!/usr/bin/env python3
"""Validate extracted metrics against NanoAOD and plot paired distributions."""
import argparse
import collections
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory',type=Path)
args=parser.parse_args();out=args.directory
payload=json.loads((out/'pairs.json').read_text());rows=payload['rows']
links=json.loads((out/'nano_links.json').read_text())
errors=[];max_residual=collections.defaultdict(float);checked=collections.Counter()
for row in rows:
    link=links[f'{row["event"]}:{row["collection"]}'].get(f'{row["object"]}:{row["root"]}')
    if row['collection'] in ('Track','GSF'):
        row['graph_link']=link is not None
        row['shared_hits']=link['shared'] if link else 0.
        row['track_score']=link['score'] if link else 1.
        row['track_reverse']=link['reverse'] if link else 1.
    else:
        if row['graph_link']!=(link is not None):errors.append(('link set',row))
        if link:
            for key,field in [('shared','sim_shared'),('score','sim_score'),('reverse','sim_reverse')]:
                residual=abs(row[field]-link[key])
                max_residual[field]=max(max_residual[field],residual)
                if not np.isclose(row[field],link[key],rtol=2e-5,atol=2e-6):errors.append((field,row[field],link[key],row['collection'],row['event'],row['object']))
            checked[row['collection']]+=1
if errors:
    (out/'crosscheck_errors.json').write_text(json.dumps(errors,indent=2)+'\n')
    raise RuntimeError(f'{len(errors)} metric reproduction errors; see crosscheck_errors.json')
fields=sorted(set().union(*(r.keys() for r in rows)))
with (out/'pairs.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
def stats(x):
    x=np.asarray(x,dtype=float)
    if not len(x):return {'n':0}
    return dict(n=len(x),min=float(x.min()),p10=float(np.quantile(x,.1)),median=float(np.median(x)),p90=float(np.quantile(x,.9)),max=float(x.max()))
summary={'events':len(payload['events']),'nano_crosscheck':dict(links_checked=dict(checked),max_absolute_residual=dict(max_residual)),'collections':{}}
collections_order=['ECAL','HCAL','ticlTrackstersCLUE3DHigh','ticlTracksterLinks','Track','GSF']
for collection in collections_order:
    selected=[r for r in rows if r['collection']==collection]
    graph=[r for r in selected if r['graph_link']]
    item={'objects':len({(r['event'],r['object']) for r in selected}),'graph_links':len(graph)}
    if collection in ('Track','GSF'):
        old=[r for r in selected if r['old_track_link']]
        common=[r for r in old if r['graph_link']]
        item.update(old_links=len(old),common_links=len(common),old_purity=stats([r['old_quality'] for r in old]),legacy_postprocessing_weight=stats([r['old_quality']*r['reco_energy'] for r in old]),new_purity=stats([1-r['track_score'] for r in graph]),shared_hits=stats([r['shared_hits'] for r in graph]),reverse_score=stats([r['track_reverse'] for r in graph]),purity_difference=stats([1-r['track_score']-r['old_quality'] for r in common]))
    else:
        item.update(old_cp_links=sum(r['old_cp']>0 for r in selected),cp_available_roots=len({(r['event'],r['root']) for r in selected if r['cp']>=0}))
        for field in ['old_cp','old_graph','sim_shared','reco_shared','sim_score','sim_reverse','reco_score','reco_reverse']:
            item[field]=stats([r[field] for r in graph])
        common=[r for r in graph if r['old_graph']>0]
        item['sim_over_product']=stats([r['sim_shared']/r['old_graph'] for r in common])
        item['reco_min_over_product']=stats([r['reco_shared']/r['old_graph'] for r in common])
        item['new_score_below_025']=sum(r['sim_score']<.25 for r in graph)
        item['best_root_changes']={}
        objects=collections.defaultdict(list)
        for r in graph:objects[(r['event'],r['object'])].append(r)
        for field,sign in [('sim_shared',1),('sim_score',-1),('reco_shared',1)]:
            changes=0
            for candidates in objects.values():
                baseline=max(candidates,key=lambda r:r['old_graph'])['root']
                choice=max(candidates,key=lambda r:sign*r[field])['root']
                changes+=choice!=baseline
            item['best_root_changes'][field]=changes
    summary['collections'][collection]=item
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def ecdf(ax,values,label,color):
    ax.set_ylim(0,1.05)
    values=np.sort(np.asarray(values,float));values=values[values>0]
    if len(values):ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',label=label,color=color)
colors=['#1764ab','#dd6b20','#13845b']
fig,axes=plt.subplots(3,4,figsize=(17,10),constrained_layout=True)
for j,collection in enumerate(collections_order[:4]):
    data=[r for r in rows if r['collection']==collection and r['graph_link']]
    title={'ticlTrackstersCLUE3DHigh':'HGCal CLUE3D','ticlTracksterLinks':'HGCal linked'}.get(collection,collection)
    for key,label,color in zip(['old_graph','sim_shared','reco_shared'],['Old product, graph fractions','New minimum, sim energy','New minimum, reco energy'],colors):ecdf(axes[0,j],[r[key] for r in data],label,color)
    axes[0,j].set(xscale='log',title=f'{title}: {len(data)} linked pairs',xlabel='Shared weight [GeV]',ylabel='ECDF')
    axes[0,j].legend(fontsize=7)
    for key,label,color in [('sim_shared','Sim minimum / old product',colors[1]),('reco_shared','Reco minimum / old product',colors[2])]:
        ecdf(axes[1,j],[r[key]/r['old_graph'] for r in data if r['old_graph']>0],label,color)
    axes[1,j].set(xscale='log',xlabel='Paired ratio',ylabel='ECDF');axes[1,j].axvline(1,color='0.5',ls=':');axes[1,j].legend(fontsize=7)
    axes[2,j].scatter([r['sim_score'] for r in data],[r['sim_reverse'] for r in data],s=32,color=colors[1],label='Sim weighting')
    axes[2,j].scatter([r['reco_score'] for r in data],[r['reco_reverse'] for r in data],s=22,marker='x',color=colors[2],label='Reco weighting')
    axes[2,j].plot([0,1],[0,1],color='0.75',ls=':');axes[2,j].set(xlim=(-.03,1.03),ylim=(-.03,1.03),xlabel='Reco-normalized score (lower better)',ylabel='Truth-normalized score (lower better)');axes[2,j].legend(fontsize=7)
fig.suptitle('Same ten pion-gun events: calorimeter metric comparisons\nOld-product curves use graph truth fractions to isolate the formula; legacy CaloParticles have different detector coverage.',fontsize=13)
fig.savefig(out/'calorimeter_distributions.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
for j,collection in enumerate(['Track','GSF']):
    data=[r for r in rows if r['collection']==collection]
    graph=[r for r in data if r['graph_link']];old=[r for r in data if r['old_track_link']];common=[r for r in old if r['graph_link']]
    axes[0,j].scatter([r['old_quality'] for r in common],[1-r['track_score'] for r in common],s=45,color=colors[0]);axes[0,j].plot([0,1],[0,1],color='0.7',ls=':')
    axes[0,j].set(xlim=(0,1.05),ylim=(0,1.05),xlabel='Legacy RecoToSim purity',ylabel='New purity = 1 - forward score',title=f'{collection}: {len(common)} common links')
    ecdf(axes[1,j],[r['old_quality'] for r in old],f'Legacy purity (n={len(old)})',colors[0])
    ecdf(axes[1,j],[1-r['track_score'] for r in graph],f'New purity (n={len(graph)})',colors[1])
    axes[1,j].set(xlim=(0,1.05),xlabel='Purity',ylabel='ECDF');axes[1,j].legend()
fig.suptitle('Track association distributions: primary-particle identity aligned by simulation track and event IDs')
fig.savefig(out/'track_distributions.png',dpi=160);plt.close(fig)
lines=['Metric distribution audit: ten pion-gun events','',f'NanoAOD reproduction cross-check: {summary["nano_crosscheck"]}','','Values below are conditional on an existing graph link. Unlinked pairs remain in pairs.csv.','The legacy PFAnalysis run crashed in getElementProperties. Legacy track maps were produced successfully without that feature extractor.','Actual legacy CP weights are reconstructed from stored SimCluster fractions and cluster rechit references. For HGCal tracksters the old formula is adapted to their constituent hits; the old ntuplizer does not process these trackster objects.','','Collections:']
for collection,item in summary['collections'].items():lines+=[collection,json.dumps(item,indent=2)]
(out/'report.txt').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
