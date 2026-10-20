#!/usr/bin/env python3
"""Train simple calibration controls and compare paired frozen-readout results.

No new backbone inference. All coefficients fit train data, ridge selection uses
validation, and reused test results are explicitly exploratory.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import torch
from scripts.studies.train_frozen_tracker_readout import (
    ARMS, PROCS, read_cache, charged, physical, write_json, TrackerMomentumReadout,
)
from scripts.studies.frozen_tracker_filtering import match, momentum, stats, jet_metrics

BOUND = math.log(1.5)


def design(base, kind):
    if kind == 'constant':
        return np.ones((*base.shape[:-1], 1), dtype=np.float64)
    # Fixed scales, predicted quantities only; no fitting on validation/test.
    z = np.clip(base[..., 1], -4, 6).astype(np.float64)/3
    a = np.clip(np.abs(base[..., 2]), 0, 5).astype(np.float64)/2
    return np.stack([np.ones_like(z), z, a, z*z, z*a, a*a], -1)


def support(cache, arm):
    i = ARMS.index(arm)
    # Cached geometric summaries: log(1 + effective count) in each radial bin.
    logs = np.asarray(cache['summary'][:, i, :, 256:]).reshape(-1,256,4,11)[...,-1]
    return np.expm1(logs).sum(-1).clip(0,1).astype(np.float64)


def fit_controls(cache, gate, kind, ridges):
    base = np.asarray(cache['base']); target = np.asarray(cache['truth'])
    mapping = np.asarray(cache['match_index']); valid = (mapping >= 0) & charged(base[...,0]) & (gate > 0)
    tlog = np.take_along_axis(target[...,1], mapping.clip(0), axis=1)
    x = design(base, kind)[valid]*gate[valid,None]
    y = (tlog-base[...,1])[valid].astype(np.float64)
    weights = np.exp(tlog[valid].astype(np.float64)/2)
    gram = x.T@(weights[:,None]*x)/weights.sum()
    rhs = x.T@(weights*y)/weights.sum()
    penalty = np.eye(x.shape[1]); penalty[0,0] = 0
    return [{'kind':kind,'ridge':r,'coefficients':np.linalg.solve(gram+r*penalty,rhs).tolist(),
             'training_matched_slots':int(valid.sum())} for r in ridges]


def calibration(cache, gate, model):
    raw = design(cache['base'], model['kind'])@np.asarray(model['coefficients'])
    return np.clip(raw, -BOUND, BOUND)*gate*charged(cache['base'][...,0])


def fixed_residuals(cache, corrections):
    values, events = [], []
    for e in range(len(cache['base'])):
        base=cache['base'][e]; truth=cache['truth'][e]; mapping=cache['match_index'][e]
        slots=np.flatnonzero((mapping>=0)&cache['active'][e]); targets=mapping[slots]
        dr=np.hypot(base[slots,2]-truth[targets,2],np.arctan2(np.sin(base[slots,3]-truth[targets,3]),np.cos(base[slots,3]-truth[targets,3])))
        keep=(dr<.1)&(truth[targets,0]==1)
        r=np.expm1(base[slots[keep],1]+corrections[e,slots[keep]]-truth[targets[keep],1])
        values.extend(r);events.extend([e]*len(r))
    return np.asarray(values),np.asarray(events,dtype=np.int64)


def head_corrections(source, cache, arm):
    model=TrackerMomentumReadout(256,BOUND).cuda().eval()
    checkpoint=torch.load(source/'checkpoints'/f'best_{arm}.pth',map_location='cuda',weights_only=False)
    model.load_state_dict(checkpoint['head'])
    result=[]
    with torch.no_grad():
        for start in range(0,len(cache['base']),128):
            summary=torch.tensor(np.array(cache['summary'][start:start+128,ARMS.index(arm)]),device='cuda')
            result.append(model(summary).cpu().numpy())
    return np.concatenate(result)*charged(cache['base'][...,0])


def evaluate(cache, correction):
    n=len(cache['base']); rows=np.zeros((n,8)); residuals=[]; eventids=[]
    collections={p:[] for p in PROCS}
    for e in range(n):
        base=np.array(cache['base'][e]);base[:,1]+=correction[e]
        truth=physical(cache['truth'][e][cache['truth'][e,:,0]!=0]);pred=physical(base[cache['active'][e]])
        pi,ti,angular,relative=match(pred,truth)
        accepted=angular&(abs(relative)<.5);hadron=truth[ti,0]==1
        r=relative[angular&hadron];residuals.extend(r);eventids.extend([e]*len(r))
        rows[e]=[accepted.sum(),len(pred)+len(truth),(accepted&hadron).sum(),(truth[:,0]==1).sum(),
                 (abs(r)>.5).sum(),len(r),np.linalg.norm(momentum(pred).sum(0)-momentum(truth).sum(0)),0]
        collections[PROCS[e%3]].append((truth,pred))
    fixed,fixids=fixed_residuals(cache,correction)
    result={'f1':float(np.mean([2*rows[i::3,0].sum()/rows[i::3,1].sum() for i in range(3)])),
            'charged_hadron_efficiency':float(rows[:,2].sum()/rows[:,3].sum()),
            'charged_hadron':stats(residuals),'fixed_charged_hadron':stats(fixed),
            'p3_error_mean':float(rows[:,6].mean()),
            'jets_by_process':{p:jet_metrics(c) for p,c in collections.items()}}
    return result,dict(rows=rows,residuals=np.asarray(residuals),events=np.asarray(eventids,dtype=np.int64),fixed=fixed,fixevents=fixids)


def repeated_quantile(sorted_values, repetitions, q):
    """Exact numpy linear quantile of a conceptual repeated-particle array."""
    cumulative=np.cumsum(repetitions)
    position=q*(cumulative[-1]-1)
    lo=int(np.floor(position));hi=int(np.ceil(position))
    a=sorted_values[np.searchsorted(cumulative,lo+1)]
    b=sorted_values[np.searchsorted(cumulative,hi+1)]
    return a+(b-a)*(position-lo)


def bootstrap(data, weights):
    rows=data['rows']; samples=[]
    orders={k:np.argsort(data[k]) for k in ['residuals','fixed']}
    for w in weights:
        f1=np.mean([2*(rows[i::3,0]*w[i::3]).sum()/(rows[i::3,1]*w[i::3]).sum() for i in range(3)])
        efficiency=(rows[:,2]*w).sum()/(rows[:,3]*w).sum()
        tails=(rows[:,4]*w).sum()/(rows[:,5]*w).sum()
        widths=[]
        for k,ids in [('residuals','events'),('fixed','fixevents')]:
            order=orders[k];v=data[k][order];counts=w[data[ids][order]]
            widths.append((repeated_quantile(v,counts,.84)-repeated_quantile(v,counts,.16))/2)
        samples.append([f1,efficiency,tails,*widths])
    return np.asarray(samples)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('experiments/cld_pf_heptv2_frozen_readout_local_5k'))
    p.add_argument('--output',type=Path,default=Path('experiments/cld_pf_heptv2_frozen_calibration_local'))
    p.add_argument('--bootstrap',type=int,default=1000)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    shutil.copy2(__file__,args.output/'analysis_source.py')
    manifest={'source':str(args.source),'source_manifest_sha256':hashlib.sha256((args.source/'manifest.json').read_bytes()).hexdigest(),
              'method':'sqrt(target pT)-weighted log-pT least squares; correction clipped to +/-log(1.5); same charged/support gates as each readout',
              'polynomial_features':['1','log(predicted pT)/3','abs(predicted eta)/2','z^2','z*a','a^2'],
              'ridge_candidates':[0.,.001,.01,.1,1.], 'selection':'minimum validation fixed-cohort charged-hadron width, including no correction',
              'bootstrap':args.bootstrap,'seed':12345,
              'limitations':['Reuses previously inspected test events: exploratory paired comparison, not new untouched confirmation.',
                'Event bootstrap excludes training-seed and model-selection uncertainty. Oracle diagnostic is not deployable.',
                'Calibration sees predicted pT/eta and support only; backbone and other outputs frozen.']}
    write_json(args.output/'manifest.json',manifest)
    # Enforce the original train/validation/test event separation before fitting.
    event_manifest={s:json.loads((args.source/'cache'/s/'events.json').read_text()) for s in ['train','valid','test']}
    assert event_manifest['train']['tfds_split']=='train'
    for i in range(3):
        assert not set(event_manifest['valid']['indices'][i])&set(event_manifest['test']['indices'][i])
    cache_args=argparse.Namespace(output=args.source)
    train=read_cache(cache_args,'train');valid=read_cache(cache_args,'valid')
    selected={};sweep={}
    for gate_name in ['hard','soft']:
        train_gate=support(train,gate_name);valid_gate=support(valid,gate_name)
        for kind in ['constant','polynomial']:
            candidates=fit_controls(train,train_gate,kind,[0.] if kind=='constant' else manifest['ridge_candidates'])
            candidates.insert(0,{'kind':kind,'ridge':None,'coefficients':[0.]*(1 if kind=='constant' else 6),'training_matched_slots':0})
            name=f'{kind}_{gate_name}'
            scored=[]
            for c in candidates:
                corr=calibration(valid,valid_gate,c);r,_=fixed_residuals(valid,corr)
                scored.append(dict(c,validation=stats(r)))
            selected[name]=min(scored,key=lambda x:x['validation']['half68']);sweep[name]=scored
            print('SELECT',name,json.dumps(selected[name]),flush=True)
    write_json(args.output/'calibration_selection.json',dict(selected=selected,sweep=sweep))
    del train,valid
    test=read_cache(cache_args,'test');n=len(test['base'])
    corrections={'baseline':np.zeros((n,256))}
    for name,c in selected.items():corrections[name]=calibration(test,support(test,name.split('_')[-1]),c)
    for a in ARMS:corrections['readout_'+a]=head_corrections(args.source,test,a)
    results={};data={}
    for name,c in corrections.items():
        results[name],data[name]=evaluate(test,c)
        results[name]['oracle_diagnostic_only']=name.endswith('oracle')
        np.savez_compressed(args.output/(name+'_events.npz'),**data[name])
        print('RESULT',name,json.dumps(results[name]),flush=True)
    # Verify learned-head reproduction against the completed source experiment.
    previous=json.loads((args.source/'history/test_step_5000.json').read_text())
    for name in ['baseline',*ARMS]:
        r=results[name if name=='baseline' else 'readout_'+name]
        assert abs(r['f1']-previous[name]['macro_f1'])<1e-8
        assert abs(r['charged_hadron']['half68']-previous[name]['momentum']['charged_hadron']['half68'])<1e-6
    write_json(args.output/'results.json',results)
    rng=np.random.default_rng(12345);weights=np.zeros((args.bootstrap,n),dtype=np.int32)
    for i in range(3):
        ids=np.arange(i,n,3)
        for b in range(args.bootstrap):weights[b,ids]=np.bincount(rng.integers(0,len(ids),len(ids)),minlength=len(ids))
    boot={}
    for name,d in data.items():
        boot[name]=bootstrap(d,weights);print('BOOTSTRAP',name,'done',flush=True)
    pairs=[(x,'baseline') for x in results if x!='baseline']
    pairs += [('readout_'+a,k+'_'+a) for a in ['hard','soft'] for k in ['constant','polynomial']]
    pairs += [('readout_soft','readout_hard'),('readout_oracle','readout_soft')]
    intervals={}
    metrics=['f1_delta_pp','charged_hadron_efficiency_delta_pp','charged_hadron_tail_delta_pp','charged_hadron_width_delta_pp','fixed_width_delta_pp']
    for v,b in pairs:
        ci=np.quantile(100*(boot[v]-boot[b]),[.025,.975],axis=0)
        intervals[f'{v} minus {b}']={m:ci[:,i].tolist() for i,m in enumerate(metrics)}
    write_json(args.output/'paired_intervals.json',intervals)
    write_json(args.output/'completed.json',{'evaluated_events':n,'bootstrap_replicates':args.bootstrap})
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
