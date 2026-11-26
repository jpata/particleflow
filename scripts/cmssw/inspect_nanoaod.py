#!/usr/bin/env python3
"""Inspect HGCal NanoAOD contents and check stored array/link consistency."""
import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot


def inspect(path):
    with uproot.open(path) as root:
        tree = root['Events']
        arrays = tree.arrays(library='ak', how=dict)
        report = dict(file=str(path.resolve()), bytes=path.stat().st_size,
                      trees={k: {'class': v.classname, 'entries': v.num_entries}
                             for k, v in root.items() if hasattr(v, 'num_entries')},
                      events=tree.num_entries, branches=len(tree.keys()), collections={},
                      errors=[], nonfinite={}, sentinels_minus999={}, int32_min_values={}, schema=[])
        for name, branch in tree.items():
            report['schema'].append(dict(name=name, type=branch.typename, description=branch.title))
            if branch.count_branch is not None:
                expected = arrays[branch.count_branch.name]
                if not bool(ak.all(ak.num(arrays[name], axis=1) == expected)):
                    report['errors'].append(f'{name}: array length differs from {branch.count_branch.name}')
            values = np.asarray(ak.to_numpy(ak.flatten(arrays[name], axis=None)))
            if np.issubdtype(values.dtype, np.floating):
                bad = int(np.count_nonzero(~np.isfinite(values)))
                if bad:
                    report['nonfinite'][name] = bad
                sentinel = int(np.count_nonzero(values == -999))
                if sentinel:
                    report['sentinels_minus999'][name] = sentinel
            if np.issubdtype(values.dtype, np.signedinteger):
                extreme = int(np.count_nonzero(values == -(2**31)))
                if extreme:
                    report['int32_min_values'][name] = extreme
            if name.startswith('n') and '_' not in name and values.ndim == 1:
                counts = values.astype(int)
                report['collections'][name[1:]] = dict(per_event=counts.tolist(), total=int(counts.sum()),
                                                       nonempty_events=int(np.count_nonzero(counts)),
                                                       min=int(counts.min()), max=int(counts.max()))
        def index_check(branch, target, allow_missing=False):
            if branch not in arrays:
                return
            for event, (indices, count) in enumerate(zip(ak.to_list(arrays[branch]), ak.to_list(arrays['n'+target]))):
                bad = [i for i in indices if not (0 <= i < count or (allow_missing and i == -1))]
                if bad:
                    report['errors'].append(f'{branch}: event {event+1} invalid indices {bad} for {target} ({count})')
        pairs = [('Track','GeneralTrack'), ('GSFTrack','GSFTrack'),
                 ('PFClusterECAL','PFClusterECAL'), ('PFClusterHCAL','PFClusterHCAL'),
                 ('TracksterCLUE3DHigh','ticlTrackstersCLUE3DHigh'), ('TracksterLinks','ticlTracksterLinks')]
        report['truth_associations'] = {}
        for label, collection in pairs:
            forward, reverse = label+'2TruthBranch', 'TruthBranch2'+label
            edge_sets = []
            for parent, source in [(forward, collection), (reverse, 'TruthBranch')]:
                child = parent+'Links'
                counts = ak.to_list(arrays[parent+'_n'+child])
                if not bool(ak.all(ak.sum(arrays[parent+'_n'+child], axis=1) == arrays['n'+child])):
                    report['errors'].append(f'{parent}: link counts do not sum to link table size')
                if not bool(ak.all(arrays['n'+parent] == arrays['n'+source])):
                    report['errors'].append(f'{parent}: parent count differs from {source}')
                target = 'TruthBranch' if parent == forward else collection
                index_check(child+'_index', target)
                all_edges = []
                for nrows, indices in zip(counts, ak.to_list(arrays[child+'_index'])):
                    edges, offset = set(), 0
                    for row, n in enumerate(nrows):
                        for col in indices[offset:offset+n]:
                            edges.add((row, col) if parent == forward else (col, row))
                        offset += n
                    all_edges.append(edges)
                edge_sets.append(all_edges)
            if edge_sets[0] != edge_sets[1]:
                report['errors'].append(f'{forward}: forward and reverse links differ')
            report['truth_associations'][collection] = dict(links=report['collections'][forward+'Links']['total'],
                                                          reciprocal=edge_sets[0] == edge_sets[1])
        for collection in ['ticlTrackstersCLUE3DHigh', 'ticlTracksterLinks', 'ticlCandidate',
                           'ticlTracksterLinksSuperclusteringDNN', 'ticlSimTracksters', 'ticlSimTrackstersfromCPs']:
            child = collection+'vertices'
            count, offset = arrays[collection+'_n'+child], arrays[collection+'_o'+child]
            if not bool(ak.all(ak.sum(count, axis=1) == arrays['n'+child])):
                report['errors'].append(f'{collection}: vertex counts differ from vertex table size')
            expected = ak.to_list(count)
            for event, (counts, offsets) in enumerate(zip(expected, ak.to_list(offset))):
                starts = np.cumsum([0]+counts[:-1]).tolist() if counts else []
                if offsets != starts:
                    report['errors'].append(f'{collection}: invalid vertex offsets in event {event+1}')
            index_check(child+'_vertices', 'HGCalLayerClusters')
        for calo, hits in [('ECAL', 'PFRecHitECAL'), ('HCAL', 'PFRecHitHBHE')]:
            prefix = 'PFCluster'+calo
            if not bool(ak.all(ak.sum(arrays[prefix+'_nRecHits'], axis=1) == arrays['n'+prefix+'RecHits'])):
                report['errors'].append(f'{prefix}: rechit counts differ from link table size')
            index_check(prefix+'RecHits_recHitIndex', hits)
        for branch, target in [('GsfElectron_gsfTrackIndex','GSFTrack'),
                               ('TICLCandidates_trackIdx','GeneralTrack'),
                               ('SimTICLCandidates_trackIdx','GeneralTrack'),
                               ('SimTICLCandidates_gsftrackIdx','GSFTrack'),
                               ('SimTICLCandidatesTrackIdxs_trackIndex','GeneralTrack'),
                               ('SimTICLCandidatesGsfTrackIdxs_trackIndex','GSFTrack'),
                               ('TICLCandidatesTrackIdxs_trackIndex','GeneralTrack'),
                               ('TICLCandidatesGsfTrackIdxs_trackIndex','GSFTrack'),
                               ('linkedTracksters_tracksterIndex','ticlTracksterLinks'),
                               ('TICLCandidatesExtra_tracksterIndex','ticlTracksterLinks')]:
            index_check(branch, target, allow_missing=True)
        report['events_detail'] = []
        for i in range(tree.num_entries):
            event = dict(run=int(arrays['run'][i]), lumi=int(arrays['luminosityBlock'][i]), event=int(arrays['event'][i]))
            for name in ['HGCalGenPart_pdgId', 'HGCalGenPart_pt', 'HGCalGenPart_eta', 'TruthBranch_pdgId',
                         'TruthBranch_isSignal', 'TruthBranch_simEvent', 'TruthBranch_bunchCrossing',
                         'TruthPV_found', 'TruthPV_x', 'TruthPV_y', 'TruthPV_z', 'PV_isValid', 'PV_isFake', 'PV_nTracks']:
                event[name] = ak.to_list(arrays[name][i])
            report['events_detail'].append(event)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    report = inspect(args.file)
    out = args.output_dir or args.file.parent/'nano_inspection'
    out.mkdir(parents=True, exist_ok=True)
    (out/'contents.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (out/'branches.tsv').write_text('name\ttype\tdescription\n'+''.join(
        f'{b["name"]}\t{b["type"]}\t{b["description"]}\n' for b in report['schema']))
    print('Events:',report['events'],'Branches:',report['branches'],'Collections:',len(report['collections']))
    print('Trees:',report['trees'])
    print('Errors:',report['errors'])
    print('Non-finite values:',report['nonfinite'])
    print('Truth associations:',report['truth_associations'])
    print('Details:',out)


if __name__ == '__main__':
    main()
