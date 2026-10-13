#!/usr/bin/env python3
"""Inference-only oracle study of the production circle fitter and hit ownership.

Directions use a separate diagnostic short-arc helix fit: increasing radial
order, an outward trajectory and the transverse point of closest approach.
They are not production track parameters or material-aware fits.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment

from mlpf.conf import MLPFConfig, Y_FEATURES
from mlpf.model.losses import make_task_loss_weighter
from mlpf.model.mlpf import MLPF
from mlpf.model.PFDataset import PFDataset
from mlpf.model.tracker_group_readout import tracker_group_geometry


def direction_fit(features):
    """Centered robust circle followed by robust z(s); all lengths in metres."""
    xyz = np.asarray(features[:, 6:9], dtype=np.float64) * .001
    xyz = xyz[np.isfinite(xyz).all(1)]
    if len(xyz) < 4:
        return None
    xy = xyz[:, :2]
    mean = xy.mean(0)
    scale = np.sqrt(np.mean(np.sum((xy-mean)**2, axis=1)))
    if scale < 1e-6:
        return None
    u = (xy-mean)/scale
    design = np.column_stack([2*u, np.ones(len(u))])
    rhs = np.sum(u*u, axis=1)
    w = np.ones(len(u))
    for _ in range(3):
        pars = np.linalg.solve(design.T@(w[:, None]*design)+1e-10*len(u)*np.eye(3), design.T@(w*rhs))
        radius = np.sqrt(max(np.sum(pars[:2]**2)+pars[2], 0))*scale
        center = mean+pars[:2]*scale
        residual = np.linalg.norm(xy-center, axis=1)-radius
        w = np.minimum(1, max(2*np.sqrt(np.mean(residual**2)), 1e-5)/np.maximum(abs(residual), 1e-12))
    order = np.argsort(np.linalg.norm(xy, axis=1))
    theta = np.unwrap(np.arctan2(xy[order, 1]-center[1], xy[order, 0]-center[0]))
    turn = np.sign(theta[-1]-theta[0])
    path = radius*turn*(theta-theta[0])
    # Reject looping/backtracking groups for this simple direction diagnostic.
    if not turn or np.ptp(theta) > np.pi or np.any(np.diff(path) < -1e-4) or np.ptp(path) < .1:
        return None
    design = np.column_stack([path-path.mean(), np.ones(len(path))])
    z = xyz[order, 2]
    w = np.ones(len(z))
    for _ in range(3):
        pars = np.linalg.lstsq(design*np.sqrt(w[:, None]), z*np.sqrt(w), rcond=None)[0]
        residual = z-design@pars
        w = np.minimum(1, max(2*np.sqrt(np.mean(residual**2)), 1e-5)/np.maximum(abs(residual), 1e-12))
    phi = np.arctan2(-center[1], -center[0])+turn*np.pi/2
    return dict(eta=float(np.arcsinh(pars[0])), phi=float(np.arctan2(np.sin(phi), np.cos(phi))),
                z_rms_m=float(np.sqrt(np.mean(residual**2))))


def fit_groups(x, groups, n, field):
    summary, logpt, valid = tracker_group_geometry(torch.as_tensor(x), torch.as_tensor(groups, dtype=torch.long), n, field)
    results = []
    for i in range(n):
        use = groups == i
        results.append(dict(valid=bool(valid[i]), hits=int(use.sum()),
                            pt=float(logpt[i].exp()) if valid[i] else None,
                            residual_over_span=float(summary[i, 6]),
                            direction=direction_fit(x[use]) if valid[i] else None))
    return results


def residual_summary(values):
    a = np.asarray(values)
    if not len(a):
        return dict(n=0)
    q = np.quantile(a, [.16, .5, .84])
    return dict(n=len(a), median=float(q[1]), half68=float((q[2]-q[0])/2),
                mae=float(abs(a).mean()), tail50=float((abs(a)>.5).mean()))


def summarize(rows):
    result = dict(charged_targets=len(rows), targets_with_tracker_hits=sum(r['oracle']['hits'] > 0 for r in rows), methods={})
    angular = [r for r in rows if r['angular_match']]
    paired = [r for r in angular if r['oracle']['valid'] and r['predicted']['valid'] and r['cleaned']['valid']]
    for method in ['oracle', 'predicted', 'cleaned']:
        population = rows if method == 'oracle' else angular
        good = [r for r in population if r[method]['valid']]
        result['methods'][method] = dict(denominator=len(population), valid=len(good),
            pt=residual_summary([r[method]['pt']/r['truth_pt']-1 for r in good]),
            pt_bins={f'{lo}-{hi}':residual_summary([r[method]['pt']/r['truth_pt']-1 for r in good if lo <= r['truth_pt'] < hi]) for lo, hi in [(0,1),(1,3),(3,10),(10,30),(30,1e6)]},
            direction_delta_r=residual_summary([np.hypot(r[method]['direction']['eta']-r['truth_eta'], np.arctan2(np.sin(r[method]['direction']['phi']-r['truth_phi']),np.cos(r[method]['direction']['phi']-r['truth_phi']))) for r in good if r[method]['direction'] is not None]))
    result['angular_matches'] = len(angular)
    result['learned_angular_pt'] = residual_summary([r['learned_pt']/r['truth_pt']-1 for r in angular])
    result['common_valid_angular_subset'] = {method:residual_summary([(r['learned_pt'] if method=='learned' else r[method]['pt'])/r['truth_pt']-1 for r in paired]) for method in ['learned','oracle','predicted','cleaned']}
    result['predicted_fit_by_purity'] = {f'{lo}-{hi}':residual_summary([r['predicted']['pt']/r['truth_pt']-1 for r in angular if r['predicted']['valid'] and lo <= r['purity'] <= hi and (lo==0 or r['purity']>lo)]) for lo,hi in [(0,.5),(.5,.9),(.9,1)]}
    return result


def plot_results(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    paired = result['summary']['common_valid_angular_subset']
    purity = result['summary']['predicted_fit_by_purity']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    keys = ['learned', 'oracle', 'predicted', 'cleaned']
    axes[0].bar(range(4), [100*paired[k]['half68'] for k in keys], color=['#596A73', '#13A895', '#DF6655', '#277DA1'])
    axes[0].set(xticks=range(4), xticklabels=['Learned', 'Truth-hit\nfit', 'Predicted\ngroup fit', 'Truth-cleaned\ngroup fit'],
                title=f"Common valid subset: {paired['learned']['n']} particles", ylabel='Relative pT half central 68% [%]')
    axes[1].bar(range(3), [100*purity[k]['half68'] for k in purity], color='#277DA1')
    axes[1].set(xticks=range(3), xticklabels=['≤50%', '50–90%', '>90%'], xlabel='Truth purity of predicted group',
                title='Accepted predicted fits by group purity', ylabel='Relative pT half central 68% [%]')
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Oracle diagnostic: selected subsets, no pT acceptance cut', fontsize=11)
    fig.tight_layout()
    fig.savefig(output/'comparison.png', dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-dir', default='/mnt/work/mlpf/tensorflow_datasets/cld')
    p.add_argument('--events-per-process', type=int, default=32)
    p.add_argument('--processes', nargs='+', default=['ttbar','ww_fullhad','qq'])
    p.add_argument('--step', type=int, default=50000)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(12345)
    config = MLPFConfig.model_validate(yaml.safe_load((args.run/'train-config.yaml').read_text()))
    config.model.attention.use_flash_attn_varlen = False
    model = MLPF(config)
    model.task_loss_weighter = make_task_loss_weighter(config.task_loss_weights.model_dump())
    checkpoint = args.run/'checkpoints'/f'checkpoint-{args.step}.pth'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model_state_dict'], strict=True)
    del state
    model.requires_grad_(False).eval().to(args.device)
    manifest = dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        checkpoint_sha256=hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest(),
        device=torch.cuda.get_device_name() if args.device.startswith('cuda') else args.device,
        event_selection='First N test events per process, config 1 v3.2.1; no training or tuning.',
        matching='Presence-selected slots; Hungarian (dr/.1)^2+(log pT ratio/log2)^2, then dr<.1; no pT or PID acceptance cut.',
        fit='Production circle fitter, 2T from configuration; predicted groups include all tracker hits in the matched slot, no PID or calorimeter-anchor eligibility gate.',
        cleaned='Truth filtering of predicted groups; oracle diagnostic only, unavailable at inference.',
        direction='Diagnostic short outward arc in increasing rho, z versus arc length, tangent at transverse closest approach; no material model or displaced/curling-track treatment.')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    rows=[]; started=time.monotonic(); pnidx=Y_FEATURES.index('particle_number')
    for proc in args.processes:
        ds=PFDataset(args.data_dir,f'cld_edm_{proc}_hits/1:3.2.1','test',pad_to_multiple=config.pad_to_multiple_elements,feature_dim=config.input_dim,max_open_readers=4,build_target_set=True).ds
        for event in range(min(args.events_per_process,len(ds))):
            rec=ds[event]; x=np.asarray(rec['X'],dtype=np.float32); target=rec['ytarget_set']; hitpn=rec['ytarget'][:,pnidx].astype(int)
            xt=torch.as_tensor(x,device=args.device)[None];mask=xt[...,0]!=0
            with torch.inference_mode(), torch.autocast(device_type=args.device.split(':')[0],dtype=torch.bfloat16,enabled=args.device.startswith('cuda')):
                out=model(xt,mask)
                keys,slots,bg=out.hit_grouping
                winners=[]
                for start in range(0,len(x),1024):
                    scores=10*(keys[0,start:start+1024]@slots[0].T)
                    winners.append(torch.cat([scores,bg[0,start:start+1024]],-1).argmax(-1).cpu())
            winner=torch.cat(winners).numpy(); nslots=slots.shape[1]
            pres,pid,mom,_=[v[0].float().cpu().numpy() for v in out]
            # Stable two-class softmax, matching inference's presence threshold.
            selected=np.flatnonzero(1/(1+np.exp(np.clip(pres[:,0]-pres[:,1],-80,80)))>=config.model.set_decoder.presence_threshold)
            predpt=np.exp(np.clip(mom[:,0],-20,20)); predphi=np.arctan2(mom[:,2],mom[:,3])
            truthpt=np.exp(target[:,2]);truthphi=np.arctan2(target[:,4],target[:,5])
            dr=np.hypot(mom[selected,1,None]-target[:,3], np.arctan2(np.sin(predphi[selected,None]-truthphi),np.cos(predphi[selected,None]-truthphi)))
            si,ti=linear_sum_assignment((dr/.1)**2+(np.log(predpt[selected,None]/truthpt)/np.log(2))**2)
            matched={int(t):(int(selected[i]),float(dr[i,t])) for i,t in zip(si,ti)}
            tracker=x[:,0]==1; tx=x[tracker]; tpn=hitpn[tracker]; tw=winner[tracker]
            targetpn=target[:,pnidx].astype(int); pnmap={v:i for i,v in enumerate(targetpn)}
            tg=np.array([pnmap.get(v,-1) for v in tpn]); owned=tg>=0
            oracle=fit_groups(tx[owned],tg[owned],len(target),config.model.set_decoder.tracker_group_magnetic_field)
            fg=tw<nslots; predicted=fit_groups(tx[fg],tw[fg],nslots,config.model.set_decoder.tracker_group_magnetic_field)
            cleanmask=np.zeros(len(tx),dtype=bool)
            for t,(slot,_) in matched.items():cleanmask|=(tw==slot)&(tpn==targetpn[t])
            cleaned=fit_groups(tx[cleanmask],tw[cleanmask],nslots,config.model.set_decoder.tracker_group_magnetic_field)
            for t in np.flatnonzero(np.isin(target[:,0],[1,4,5])):
                slot,dist=matched.get(int(t),(-1,float('inf'))); pure=(tw==slot)&(tpn==targetpn[t]); count=int((tw==slot).sum())
                empty=dict(valid=False,hits=0,pt=None,direction=None)
                rows.append(dict(process=proc,event=event,particle_number=int(targetpn[t]),cls=int(target[t,0]),truth_pt=float(truthpt[t]),truth_eta=float(target[t,3]),truth_phi=float(truthphi[t]),angular_match=dist<.1,learned_pt=float(predpt[slot]) if slot>=0 else None,purity=float(pure.sum()/max(count,1)),completeness=float(pure.sum()/max((tpn==targetpn[t]).sum(),1)),oracle=oracle[t],predicted=predicted[slot] if slot>=0 else empty,cleaned=cleaned[slot] if slot>=0 else empty))
            print(f'{proc} {event+1}/{args.events_per_process}; charged targets={len(rows)}; elapsed={time.monotonic()-started:.1f}s',flush=True)
            del out,xt,keys,slots,bg
    result=dict(elapsed_seconds=time.monotonic()-started,summary=summarize(rows),by_species={str(c):summarize([r for r in rows if r['cls']==c]) for c in [1,4,5]},by_process={proc:summarize([r for r in rows if r['process']==proc]) for proc in args.processes},rows=rows)
    (args.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    plot_results(result, args.output)
    print(json.dumps(result['summary'],indent=2),flush=True)


if __name__=='__main__':main()
