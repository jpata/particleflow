#!/usr/bin/env python3
"""Frozen hit-assignment filtering: development selection followed by held-out evaluation.

No truth labels enter filtering. Momentum substitution changes charged pT only;
energy, direction, presence and PID remain frozen. Jet results therefore test
this particular pT-only intervention, not a complete physical track readout.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def stats(a):
    a = np.asarray(a, dtype=float)
    if not len(a):
        return {'n': 0, 'half68': None, 'mae': None, 'tail50': None, 'median': None}
    q = np.quantile(a, [.16, .5, .84])
    return dict(n=len(a), half68=float((q[2]-q[0])/2), mae=float(abs(a).mean()),
                tail50=float((abs(a)>.5).mean()), median=float(q[1]))


def circle(x, field=2.):
    """Numpy equivalent of production three-iteration circle fit and gates."""
    xyz = np.asarray(x[:, 6:9], dtype=float)*.001
    finite = np.isfinite(xyz).all(1)
    xyz, x = xyz[finite], x[finite]
    n = len(x)
    if n < 4:
        return None
    mean = xyz[:, :2].mean(0)
    scale = max(np.sqrt(np.mean(np.sum((xyz[:, :2]-mean)**2, axis=1))), 1e-6)
    u = (xyz[:, :2]-mean)/scale
    design = np.column_stack([2*u, np.ones(n)])
    rhs = np.sum(u*u, axis=1)
    weights = np.ones(n)
    for _ in range(3):
        gram = design.T@(weights[:, None]*design)
        pars = np.linalg.solve(gram+1e-10*n*np.eye(3), design.T@(weights*rhs))
        radius = np.sqrt(max(np.sum(pars[:2]**2)+pars[2], 0))*scale
        center = mean+pars[:2]*scale
        residual = np.linalg.norm(xyz[:, :2]-center, axis=1)-radius
        rms = np.sqrt(np.mean(residual**2))
        weights = np.minimum(1, max(2*rms, 1e-5)/np.maximum(abs(residual), 1e-12))
    span = np.ptp(np.linalg.norm(xyz[:, :2], axis=1))
    eig = np.linalg.eigvalsh(gram)
    condition = max(eig[0], 0)/max(eig[-1], 1e-12)
    pt = .299792458*field*radius
    valid = (len(np.unique(x[:, 12:15], axis=0)) >= 3 and span >= .1 and condition > 1e-7
             and rms/max(span, 1e-6) < .02 and np.isfinite(pt) and .1 < pt < 1e4)
    if not valid:
        return None
    return dict(pt=float(pt), center=center, radius=radius, relative_residual=rms/max(span, 1e-6))


def geometry_mask(x, seed, tolerance, hypotheses=24):
    """Deterministic short-arc circle/z-line consensus, then reject and refit twice."""
    xyz = np.asarray(x[:, 6:9], dtype=float)*.001
    finite = np.isfinite(xyz).all(1)
    pool = np.flatnonzero(seed & finite)
    if len(pool) < 3:
        return np.zeros(len(x), dtype=bool)
    rng = np.random.default_rng(12345)
    radial = pool[np.argsort(np.linalg.norm(xyz[pool, :2], axis=1))]
    triples = [radial[[0, len(radial)//2, -1]]]
    triples += [rng.choice(pool, 3, replace=False) for _ in range(hypotheses)]
    best = np.zeros(len(x), dtype=bool)
    best_score = (-1, -np.inf)

    def inliers(center, radius, fit_indices):
        theta = np.arctan2(xyz[:, 1]-center[1], xyz[:, 0]-center[0])
        ref = theta[fit_indices[0]]
        arc = radius*np.arctan2(np.sin(theta-ref), np.cos(theta-ref))
        matrix = np.column_stack([arc, np.ones(len(x))])
        pars = np.linalg.lstsq(matrix[fit_indices], xyz[fit_indices, 2], rcond=None)[0]
        transverse = abs(np.linalg.norm(xyz[:, :2]-center, axis=1)-radius)
        longitudinal = abs(xyz[:, 2]-matrix@pars)
        keep = finite & (transverse < tolerance) & (longitudinal < 5*tolerance)
        return keep, transverse+longitudinal/5

    for ids in triples:
        if len(np.unique(x[ids, 12:15], axis=0)) < 3 or np.ptp(np.linalg.norm(xyz[ids, :2], axis=1)) < .1:
            continue
        xy = xyz[ids, :2]
        u = xy[1:]-xy[0]
        if abs(np.linalg.det(u)) < 1e-10:
            continue
        center = xy[0]+np.linalg.solve(2*u, np.sum(u*u, axis=1))
        radius = np.linalg.norm(xy[0]-center)
        keep, residual = inliers(center, radius, ids)
        score = (int(keep.sum()), -float(residual[keep].sum()))
        if score > best_score:
            best, best_score = keep, score
    for _ in range(2):
        fit = circle(x[best])
        if fit is None:
            return np.zeros(len(x), dtype=bool)
        best, _ = inliers(fit['center'], fit['radius'], np.flatnonzero(best))
    return best


def filter_fit(x, probability, margin, config):
    if config['family'] == 'learned':
        return None, np.zeros(len(x), dtype=bool)
    seed = np.ones(len(x), dtype=bool)
    if config.get('score') == 'probability':
        seed = probability >= config['threshold']
    elif config.get('score') == 'margin':
        seed = margin >= config['threshold']
    keep = geometry_mask(x, seed, config['tolerance_m']) if 'tolerance_m' in config else seed
    return circle(x[keep]), keep


def candidates():
    result = [dict(name='learned', family='learned'), dict(name='unfiltered_fit', family='unfiltered')]
    for score, thresholds in [('probability', [.2, .4, .6, .8]), ('margin', [.5, 1., 2., 3.])]:
        for threshold in thresholds:
            result.append(dict(name=f'{score}_{threshold}', family='confidence', score=score, threshold=threshold))
    for tol in [.001, .003, .01]:
        result.append(dict(name=f'geometry_{tol}', family='geometry', tolerance_m=tol))
        for threshold in [.4, .6]:
            result.append(dict(name=f'combined_p{threshold}_{tol}', family='combined', score='probability', threshold=threshold, tolerance_m=tol))
    return result


def extract(args):
    import torch
    import yaml
    from mlpf.conf import MLPFConfig, Y_FEATURES
    from mlpf.model.mlpf import MLPF
    from mlpf.model.losses import make_task_loss_weighter
    from mlpf.model.PFDataset import PFDataset
    torch.set_num_threads(4)
    torch.manual_seed(12345)
    cfg = MLPFConfig.model_validate(yaml.safe_load((args.run/'train-config.yaml').read_text()))
    cfg.model.attention.use_flash_attn_varlen = False
    model = MLPF(cfg)
    model.task_loss_weighter = make_task_loss_weighter(cfg.task_loss_weights.model_dump())
    checkpoint = args.run/'checkpoints/checkpoint-50000.pth'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model_state_dict'], strict=True)
    del state
    model.requires_grad_(False).eval().cuda()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.file_digest(checkpoint.open('rb'), 'sha256').hexdigest(),
        device=torch.cuda.get_device_name(), config='1:3.2.1', dev_events_per_process=args.dev, test_events_per_process=args.test,
        development_indices=[0,args.dev], test_indices=[args.dev,args.dev+args.test], intervals='half-open', seed=12345,
        no_training=True, inference_dtype='bfloat16', magnetic_field=cfg.model.set_decoder.tracker_group_magnetic_field,
        presence_threshold=cfg.model.set_decoder.presence_threshold)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    started = time.monotonic()
    for proc in ['ttbar', 'ww_fullhad', 'qq']:
        ds = PFDataset(args.data_dir, f'cld_edm_{proc}_hits/1:3.2.1', 'test', pad_to_multiple=cfg.pad_to_multiple_elements, feature_dim=cfg.input_dim, max_open_readers=4, build_target_set=True).ds
        assert len(ds) >= args.dev+args.test
        for event in range(args.dev+args.test):
            rec = ds[event]; x = np.asarray(rec['X'], dtype=np.float32)
            xt = torch.as_tensor(x, device='cuda')[None]
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(xt, xt[..., 0] != 0)
                keys, slots, bg = out.hit_grouping
                winners, probabilities, margins = [], [], []
                for start in range(0,len(x),1024):
                    logits = torch.cat([10*(keys[0,start:start+1024]@slots[0].T), bg[0,start:start+1024]],-1).float()
                    top, ids = logits.topk(2,dim=-1)
                    winners.append(ids[:,0].cpu().numpy())
                    probabilities.append(logits.softmax(-1).gather(1,ids[:,:1])[:,0].cpu().numpy())
                    margins.append((top[:,0]-top[:,1]).cpu().numpy())
            presence, pid, mom, _ = [v[0].float().cpu().numpy() for v in out]
            selected = np.flatnonzero(1/(1+np.exp(np.clip(presence[:,0]-presence[:,1],-80,80))) >= cfg.model.set_decoder.presence_threshold)
            # Particle columns: class, pT, eta, phi, energy.
            pred = np.column_stack([pid[:,1:].argmax(-1)+1, np.exp(np.clip(mom[:,0],-20,20)),mom[:,1],np.arctan2(mom[:,2],mom[:,3]),np.exp(np.clip(mom[:,4],-20,20))])[selected]
            t = rec['ytarget_set']
            truth = np.column_stack([t[:,0],np.exp(t[:,2]),t[:,3],np.arctan2(t[:,4],t[:,5]),np.exp(t[:,6])])
            tracker = x[:,0]==1
            folder = args.output/('dev' if event<args.dev else 'test')/proc
            folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/f'{event:04d}.npz', x=x[tracker], winner=np.concatenate(winners)[tracker], probability=np.concatenate(probabilities)[tracker], margin=np.concatenate(margins)[tracker],
                hit_pn=rec['ytarget'][tracker,Y_FEATURES.index('particle_number')], target_pn=t[:,Y_FEATURES.index('particle_number')], truth=truth, pred=pred, slots=selected)
            if (event+1)%32==0:print(f'extract {proc} {event+1}/{args.dev+args.test}, {time.monotonic()-started:.1f}s',flush=True)
            del out, xt, keys, slots, bg


def match(pred, truth):
    dr = np.hypot(pred[:,2,None]-truth[:,2],np.arctan2(np.sin(pred[:,3,None]-truth[:,3]),np.cos(pred[:,3,None]-truth[:,3])))
    pi,ti = linear_sum_assignment((dr/.1)**2+(np.log(np.maximum(pred[:,1,None],1e-8)/np.maximum(truth[:,1],1e-8))/np.log(2))**2)
    rel = pred[pi,1]/truth[ti,1]-1
    return pi,ti,dr[pi,ti]<.1,rel


def momentum(p):
    return np.column_stack([p[:,1]*np.cos(p[:,3]),p[:,1]*np.sin(p[:,3]),p[:,1]*np.sinh(p[:,2])])


def jet_metrics(events):
    import fastjet
    import awkward as ak
    from mlpf.conf import Dataset
    from mlpf.jet_utils import get_jet_config, _match_jets_event, jet_matching_metrics
    # Dataset enum name is resolved from its value for portability.
    jetdef, cut, drcut = get_jet_config(Dataset('cld_hits'))
    counts = [0,0]; ratios=[]
    for target,pred in events:
        jets=[]
        for p in [target,pred]:
            q=momentum(p)
            vec=ak.zip({k:ak.Array([v]) for k,v in {'px':q[:,0], 'py':q[:,1], 'pz':q[:,2], 'E':p[:,4]}.items()})
            cs=fastjet.ClusterSequence(vec,jetdef)
            j=cs.inclusive_jets(min_pt=cut)
            px,py,pz=[ak.to_numpy(j[k][0]) for k in ['px','py','pz']]
            pt=np.hypot(px,py)
            jets.append((pt,np.arcsinh(pz/np.maximum(pt,1e-12)),np.arctan2(py,px)))
        a,b=jets; counts[0]+=len(a[0]);counts[1]+=len(b[0])
        i,j=_match_jets_event(a[1],a[2],b[1],b[2],drcut)
        ratios.extend((b[0][j]/a[0][i]).tolist())
    result=jet_matching_metrics(ratios,*counts)
    qualified=np.array(ratios);qualified=qualified[abs(qualified-1)<.5]
    result['response_iqr']=float(np.diff(np.quantile(qualified,[.25,.75]))[0]) if len(qualified) else None
    return result


def evaluate(paths, config, jets=False):
    charged_rel=[];neutral_rel=[];changed_rel=[];changed_baseline=[];fit_rel=[];purity=[]
    fixed_by_pt={f'{lo}-{hi}':[] for lo,hi in [(0,1),(1,3),(3,10),(10,30),(30,1e6)]}
    ntarget=0; ncharged=0; naccepted=0; naccepted_charged=0; npred=0; changed=0
    p3=[];ptvec=[];jet_events=[];records=[]; selected_hits=0;total_hits=0
    started=time.monotonic()
    for path in paths:
        with np.load(path) as e:
            pred=e['pred'].copy();truth=e['truth'];before=pred.copy()
            fits={}; switched=np.zeros(len(pred),dtype=bool);masks={}
            for j,slot in enumerate(e['slots']):
                if pred[j,0] not in [1,4,5]:continue
                group=e['winner']==slot
                fit,keep=filter_fit(e['x'][group],e['probability'][group],e['margin'][group],config)
                total_hits+=int(group.sum());selected_hits+=int(keep.sum())
                if fit is not None:
                    pred[j,1]=fit['pt'];switched[j]=True;fits[j]=fit
                    masks[j]=e['hit_pn'][group][keep]
            # Fixed baseline correspondence for direct residual comparison.
            pi,ti,angular,original_rel=match(before,truth)
            rel=pred[pi,1]/truth[ti,1]-1
            ch=np.isin(truth[ti,0],[1,4,5]);fixed=angular&ch
            charged_rel.extend(rel[fixed]);neutral_rel.extend(rel[angular&~ch])
            use=fixed&switched[pi]
            changed_rel.extend(rel[use]);changed_baseline.extend(original_rel[use]);fit_rel.extend(rel[use])
            for i,t in zip(pi[use],ti[use]):
                purity.append(float(np.mean(masks[i]==e['target_pn'][t])))
            for lo,hi in [(0,1),(1,3),(3,10),(10,30),(30,1e6)]:
                fixed_by_pt[f'{lo}-{hi}'].extend(rel[fixed&(truth[ti,1]>=lo)&(truth[ti,1]<hi)])
            # Re-match complete output to measure reconstruction efficiency/F1.
            p2,t2,a2,r2=match(pred,truth); good=a2&(abs(r2)<.5)
            ntarget+=len(truth);npred+=len(pred);ncharged+=int(np.isin(truth[:,0],[1,4,5]).sum())
            naccepted+=int(good.sum());naccepted_charged+=int((good&np.isin(truth[t2,0],[1,4,5])).sum())
            delta=momentum(pred).sum(0)-momentum(truth).sum(0)
            p3.append(float(np.linalg.norm(delta)));ptvec.append(float(np.linalg.norm(delta[:2])));changed+=int(switched.sum())
            if jets:jet_events.append((truth,pred))
            records.append(dict(process=path.parent.name,event=path.stem,charged_residuals=rel[fixed].tolist(),changed=int(switched.sum())))
    result=dict(config=config,events=len(paths),charged_targets=ncharged,charged_efficiency=naccepted_charged/max(ncharged,1),particle_f1=2*naccepted/max(ntarget+npred,1),
        charged_fixed_matches=stats(charged_rel),neutral_fixed_matches=stats(neutral_rel),fit_subset=stats(changed_rel),learned_same_subset=stats(changed_baseline),
        fit_coverage_on_angular_matches=len(changed_rel)/max(len(charged_rel),1),changed_predictions=changed,mean_retained_hit_purity=float(np.mean(purity)) if purity else None,
        selected_hits=selected_hits,candidate_hits=total_hits,pt_bins={k:stats(v) for k,v in fixed_by_pt.items()},event_p3_mean=float(np.mean(p3)),event_pt_vector_mean=float(np.mean(ptvec)),seconds=time.monotonic()-started)
    if jets:
        result['jets']=jet_metrics(jet_events)
        result['by_process_jets']={proc:jet_metrics([e for e,path in zip(jet_events,paths) if path.parent.name==proc]) for proc in ['ttbar','ww_fullhad','qq']}
        result['event_records']=records
    return result


def tune(args):
    paths=sorted((args.output/'dev').glob('*/*.npz'))
    assert paths
    results=[]
    for c in candidates():
        r=evaluate(paths,c);results.append(r)
        (args.output/'development.json').write_text(json.dumps(results,indent=2)+'\n')
        print(c['name'],r['charged_fixed_matches'],'coverage',r['fit_coverage_on_angular_matches'],flush=True)
    baseline=results[0];selected=[]
    for family in ['confidence','geometry','combined']:
        pool=[r for r in results if r['config']['family']==family]
        safe=[r for r in pool if r['fit_coverage_on_angular_matches']>=.05 and r['charged_fixed_matches']['tail50']<=baseline['charged_fixed_matches']['tail50']+.001 and r['charged_efficiency']>=baseline['charged_efficiency']-.002]
        winner=min(safe or pool,key=lambda r:r['charged_fixed_matches']['mae'])
        selected.append(dict(config=winner['config'],passes_development_safety=bool(safe)))
    frozen=dict(rule='Within each family, minimize charged fixed-correspondence MAE with >=5% coverage, tail50 <= baseline+0.001 and efficiency >= baseline-0.002. If none qualify, retain lowest-MAE candidate as diagnostic only.',selected=selected)
    (args.output/'frozen_selection.json').write_text(json.dumps(frozen,indent=2)+'\n')
    print(json.dumps(frozen,indent=2),flush=True)


def test(args):
    frozen=json.loads((args.output/'frozen_selection.json').read_text())
    paths=sorted((args.output/'test').glob('*/*.npz'));assert paths
    configs=candidates()[:2]+[x['config'] for x in frozen['selected']]
    results=[]
    for c in configs:
        r=evaluate(paths,c,jets=True);results.append(r)
        (args.output/'heldout.json').write_text(json.dumps(results,indent=2)+'\n')
        print(c['name'],r['charged_fixed_matches'],'coverage',r['fit_coverage_on_angular_matches'],'eff',r['charged_efficiency'],'p3',r['event_p3_mean'],flush=True)


def report(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results=json.loads((args.output/'heldout.json').read_text())
    baseline=results[0]
    # Paired event bootstrap, stratified by process. Each resample retains all
    # charged residuals in an event rather than treating particles as independent.
    groups={proc:[i for i,e in enumerate(baseline['event_records']) if e['process']==proc]
            for proc in ['ttbar','ww_fullhad','qq']}
    rng=np.random.default_rng(12345)
    draws=[np.concatenate([rng.choice(ids,len(ids),replace=True) for ids in groups.values()]) for _ in range(500)]
    base_arrays=[np.asarray(e['charged_residuals']) for e in baseline['event_records']]
    base_stats=[stats(np.concatenate([base_arrays[i] for i in draw])) for draw in draws]
    intervals={}
    for result in results[1:]:
        arrays=[np.asarray(e['charged_residuals']) for e in result['event_records']]
        assert all(len(a)==len(b) for a,b in zip(arrays,base_arrays))
        differences={k:[] for k in ['half68','mae','tail50']}
        for draw,b in zip(draws,base_stats):
            current=stats(np.concatenate([arrays[i] for i in draw]))
            for k in differences:differences[k].append(current[k]-b[k])
        intervals[result['config']['name']]={k:np.quantile(v,[.025,.975]).tolist() for k,v in differences.items()}
    (args.output/'bootstrap.json').write_text(json.dumps(dict(method='500 paired event bootstraps, stratified by process; delta versus learned baseline',intervals=intervals),indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(13,4.4))
    labels=['Learned','Unfiltered fit','Confidence','Geometry','Combined']
    for ax,key,title in zip(axes,['half68','tail50','coverage'],['Charged pT residual width','Charged errors above 50%','Fit replacement coverage']):
        values=[100*(r['fit_coverage_on_angular_matches'] if key=='coverage' else r['charged_fixed_matches'][key]) for r in results]
        ax.bar(range(len(results)),values,color=['#596A73','#DF6655','#277DA1','#C58A21','#13A895'])
        ax.set(xticks=range(len(results)),xticklabels=labels,title=title,ylabel='Percent')
        ax.tick_params(axis='x',labelrotation=30)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle(f"{baseline['events']} held-out events; frozen thresholds; ordinary regression fallback",fontsize=12)
    fig.tight_layout();fig.savefig(args.output/'comparison.png',dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['extract','tune','test','report'])
    p.add_argument('--run',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-dir',default='/mnt/work/mlpf/tensorflow_datasets/cld')
    p.add_argument('--dev',type=int,default=32)
    p.add_argument('--test',type=int,default=128)
    args=p.parse_args()
    {'extract':extract,'tune':tune,'test':test,'report':report}[args.action](args)


if __name__=='__main__':main()
