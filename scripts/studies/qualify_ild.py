#!/usr/bin/env python3
"""Audit native ILD ROOT identities/reconstruction and the corresponding PF targets.

Run with the repository Python environment, independently of the Geant4 runtime.
The report records measurements; passing structural checks alone is not physics qualification.
"""

import argparse
import hashlib
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from mlpf.heptfds.edm4hep_utils.utils_ild import PROVENANCE


def summary(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0}
    return dict(
        count=len(values),
        min=float(values.min()),
        median=float(np.median(values)),
        p90=float(np.quantile(values, 0.9)),
        max=float(values.max()),
    )


def target_direction_metrics(data):
    """Separate source charge using hit representatives, before cluster neutralization."""
    groups = {
        "track_pt_lt_1": [],
        "track_pt_1_to_5": [],
        "track_pt_ge_5": [],
        "cluster_charged_source": [],
        "cluster_neutral_source": [],
        "cluster_unknown_source": [],
    }
    for event in range(len(data["X_track"])):
        source_charge = {}
        for field in ("ytarget_hit_tracker", "ytarget_hit_calo"):
            for row in ak.to_numpy(data[field][event]).reshape(-1, 14):
                if row[0] != 0:
                    source_charge[int(row[13])] = row[1]
        for kind in ("track", "cluster"):
            x = ak.to_numpy(data[f"X_{kind}"][event])
            y = ak.to_numpy(data[f"ytarget_{kind}"][event]).reshape(-1, 14)
            for features, target in zip(x, y):
                if target[0] == 0:
                    continue
                if kind == "cluster":
                    eta = np.arcsinh(features[8] / max(np.hypot(features[6], features[7]), 1e-12))
                    phi = np.arctan2(features[7], features[6])
                    charge = source_charge.get(int(target[13]))
                    group = "cluster_unknown_source" if charge is None else ("cluster_charged_source" if charge else "cluster_neutral_source")
                else:
                    eta, phi = features[2], np.arctan2(features[3], features[4])
                    group = "track_pt_lt_1" if target[2] < 1 else ("track_pt_1_to_5" if target[2] < 5 else "track_pt_ge_5")
                delta = phi - np.arctan2(target[4], target[5])
                groups[group].append(float(np.hypot(eta - target[3], np.arctan2(np.sin(delta), np.cos(delta)))))
    return {
        name: {
            **summary(values),
            "fraction_aligned": float(np.mean(np.asarray(values) < (0.1 if name.startswith("track") else 0.3))) if values else None,
        }
        for name, values in groups.items()
    }


def audit(root_path, parquet_path):
    root = uproot.open(root_path, handler=uproot.source.file.MemmapSource)
    tree = root["events"]
    prefix = "events___CollectionTypeInfo/events___CollectionTypeInfo."
    info = root["podio_metadata"]
    ids = dict(
        zip(
            ak.to_list(info[prefix + "name"].array()[0]),
            ak.to_list(info[prefix + "collectionID"].array()[0]),
        )
    )
    names = {value: key for key, value in ids.items()}
    data = ak.from_parquet(parquet_path)
    errors = []
    digest = hashlib.sha256()
    with open(root_path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if "metadata_input_sha256" not in data.fields or not bool(ak.all(data["metadata_input_sha256"] == digest.hexdigest())):
        errors.append("Parquet source ROOT checksum is missing or mismatched")
    manifest = Path(str(root_path) + ".software.txt")
    expected_events = None
    if manifest.exists():
        entries = dict(line.split("=", 1) for line in manifest.read_text().splitlines() if "=" in line)
        expected_events = int(entries["events"])
        if expected_events != tree.num_entries:
            errors.append("Requested/generated event counts differ")
    if len(data["X_track"]) != tree.num_entries:
        errors.append("ROOT/Parquet event counts differ")
    for field, expected in PROVENANCE.items():
        if field not in data.fields or not bool(ak.all(data[field] == expected)):
            errors.append(f"Invalid {field}")
    cache = {}

    def array(branch, event):
        if branch not in cache:
            cache[branch] = tree[branch].array()
        return ak.to_numpy(cache[branch][event])

    def check_refs(relation, event):
        for side in ("from", "to"):
            prefix = f"_{relation}_{side}/_{relation}_{side}"
            indices, collections = (
                array(prefix + ".index", event),
                array(prefix + ".collectionID", event),
            )
            for cid in np.unique(collections):
                collection = names.get(int(cid))
                if collection is None:
                    errors.append(f"{event}: {relation} unknown collection {cid}")
                    continue
                branch = tree[collection].keys()[0]
                size = len(array(f"{collection}/{branch}", event))
                values = indices[collections == cid]
                if np.any((values < 0) | (values >= size)):
                    errors.append(f"{event}: {relation} invalid {side} reference")

    track_counts, cluster_counts, pfo_counts = [], [], []
    direct_ratios, direct_dr, direct_purity, wrong_charge = [], [], [], []
    primary_reco_response, primary_target_response = [], []
    target_loss, target_response, premerge_counts = [], [], []
    invalid_tracks = 0
    upstream_link_warnings = []
    native_weights = []
    target_jet_errors = []
    multitrack_pfos = 0
    single_primary_events = []
    acceptance_edges = [0.0, 0.8, 0.95, 0.99, 1.000001]
    truth_counts = np.zeros(4, dtype=int)
    matched_counts = np.zeros(4, dtype=int)
    for event in range(tree.num_entries):
        tracks = array("MarlinTrkTracks/MarlinTrkTracks.chi2", event)
        ntracks = len(tracks)
        nclusters = len(array("PandoraClusters/PandoraClusters.energy", event))
        track_counts.append(ntracks)
        cluster_counts.append(nclusters)
        pfo_px = array("PandoraPFOs/PandoraPFOs.momentum.x", event)
        pfo_py = array("PandoraPFOs/PandoraPFOs.momentum.y", event)
        pfo_pz = array("PandoraPFOs/PandoraPFOs.momentum.z", event)
        pfo_mass = array("PandoraPFOs/PandoraPFOs.mass", event)
        pfo_counts.append(len(pfo_px))
        multitrack_pfos += int(np.sum(array("PandoraPFOs/PandoraPFOs.tracks_end", event) - array("PandoraPFOs/PandoraPFOs.tracks_begin", event) > 1))
        for relation in ("MarlinTrkTracksMCTruthLink", "CalohitMCTruthLink"):
            check_refs(relation, event)
        weights = array("MarlinTrkTracksMCTruthLink/MarlinTrkTracksMCTruthLink.weight", event)
        native_weights.extend(weights.tolist())
        if np.any(~np.isfinite(weights) | (weights < 0) | (weights > 1.00001)):
            upstream_link_warnings.append(f"{event}: native track truth weights outside [0,1]; targets use independent distinct-hit votes")
        links_from = array(
            "_MarlinTrkTracksMCTruthLink_from/_MarlinTrkTracksMCTruthLink_from.index",
            event,
        )
        links_to = array("_MarlinTrkTracksMCTruthLink_to/_MarlinTrkTracksMCTruthLink_to.index", event)
        mc_px = array("MCParticles/MCParticles.momentum.x", event)
        mc_py = array("MCParticles/MCParticles.momentum.y", event)
        mc_pz = array("MCParticles/MCParticles.momentum.z", event)
        mc_mass = array("MCParticles/MCParticles.mass", event)
        mc_status = array("MCParticles/MCParticles.generatorStatus", event)
        mc_charge = array("MCParticles/MCParticles.charge", event)
        mc_pt = np.hypot(mc_px, mc_py)
        mc_p = np.hypot(mc_pt, mc_pz)
        eligible = (mc_status == 1) & (mc_charge != 0) & (mc_pt > 1.0)
        cos_theta = np.divide(np.abs(mc_pz), mc_p, out=np.zeros_like(mc_p), where=mc_p > 0)
        truth_counts += np.histogram(cos_theta[eligible], bins=acceptance_edges)[0]
        matched_gp = set()
        ndof = array("MarlinTrkTracks/MarlinTrkTracks.ndf", event)
        begins = array("MarlinTrkTracks/MarlinTrkTracks.trackStates_begin", event)
        ends = array("MarlinTrkTracks/MarlinTrkTracks.trackStates_end", event)
        locations = array("_MarlinTrkTracks_trackStates/_MarlinTrkTracks_trackStates.location", event)
        omega = array("_MarlinTrkTracks_trackStates/_MarlinTrkTracks_trackStates.omega", event)
        phi = array("_MarlinTrkTracks_trackStates/_MarlinTrkTracks_trackStates.phi", event)
        tanlambda = array("_MarlinTrkTracks_trackStates/_MarlinTrkTracks_trackStates.tanLambda", event)
        for track, (begin, end) in enumerate(zip(begins, ends)):
            ip = np.flatnonzero(locations[begin:end] == 1) + begin
            valid = len(ip) == 1 and np.isfinite(tracks[track]) and tracks[track] >= 0 and ndof[track] > 0
            if valid:
                state = ip[0]
                valid = np.all(np.isfinite([omega[state], phi[state], tanlambda[state]])) and omega[state] != 0
            if not valid:
                invalid_tracks += 1
                continue
            # The converter must use the IP state, not merely the first stored state.
            x = ak.to_numpy(data["X_track"][event])[track]
            if not np.isclose(x[1], 0.0006 / abs(omega[state]), rtol=1e-5):
                errors.append(f"{event}:{track}: stored input is not the fitted IP momentum")
            links = np.flatnonzero(links_from == track)
            if not len(links):
                continue
            best = links[np.argmax(weights[links])]
            gp = links_to[best]
            if mc_status[gp] != 1 or not mc_charge[gp] or weights[best] <= 0.5:
                continue
            pt = np.hypot(mc_px[gp], mc_py[gp])
            if pt == 0:
                continue
            dphi = np.arctan2(
                np.sin(phi[state] - np.arctan2(mc_py[gp], mc_px[gp])),
                np.cos(phi[state] - np.arctan2(mc_py[gp], mc_px[gp])),
            )
            direct_ratios.append(0.0006 / abs(omega[state]) / pt)
            direct_dr.append(np.hypot(np.arcsinh(tanlambda[state]) - np.arcsinh(mc_pz[gp] / pt), dphi))
            direct_purity.append(weights[best])
            wrong_charge.append(np.sign(omega[state]) != np.sign(mc_charge[gp]))
            if eligible[gp]:
                matched_gp.add(int(gp))
        matched_counts += np.histogram(cos_theta[sorted(matched_gp)], bins=acceptance_edges)[0]
        if event >= len(data["X_track"]):
            continue
        ys = [ak.to_numpy(data[name][event]).reshape(-1, 14) for name in ("ytarget_track", "ytarget_cluster")]
        xs = [ak.to_numpy(data[name][event]) for name in ("X_track", "X_cluster")]
        for x, y in zip(xs, ys):
            if len(x) != len(y) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
                errors.append(f"{event}: invalid PF matrices")
        target = np.concatenate(ys)
        representatives = target[target[:, 0] != 0]
        if len(np.unique(representatives[:, 13])) != len(representatives):
            errors.append(f"{event}: duplicate target particle representatives")
        if np.any(ys[1][:, 1] != 0):
            errors.append(f"{event}: charged cluster targets")
        charged_reps = ys[0][ys[0][:, 0] != 0]
        if len(charged_reps) and (np.any(charged_reps[:, 1] == 0) or np.any(charged_reps[:, 10] < 0.75) or np.any(charged_reps[:, 10] > 1.00001)):
            errors.append(f"{event}: track targets violate direct charged-majority policy")
        for jet_index, jet in enumerate(ak.to_numpy(data["targetjet"][event]).reshape(-1, 4)):
            constituents = representatives[representatives[:, 12] == jet_index]
            actual = np.array(
                [
                    np.sum(constituents[:, 2] * constituents[:, 5]),
                    np.sum(constituents[:, 2] * constituents[:, 4]),
                    np.sum(constituents[:, 2] * np.sinh(constituents[:, 3])),
                    np.sum(constituents[:, 6]),
                ]
            )
            expected = np.array([jet[0] * np.cos(jet[2]), jet[0] * np.sin(jet[2]), jet[0] * np.sinh(jet[1]), jet[3]])
            target_jet_errors.append(float(np.max(np.abs(actual - expected))))
            if not np.allclose(actual, expected, rtol=1e-5, atol=1e-4):
                errors.append(f"{event}: target jet constituent four-vectors do not close")
        candidate = np.concatenate([ak.to_numpy(data[name][event]).reshape(-1, 14) for name in ("ycand_track", "ycand_cluster")])
        pfo_energy = float(np.sum(np.sqrt(pfo_px**2 + pfo_py**2 + pfo_pz**2 + pfo_mass**2)))
        if not np.isclose(np.sum(candidate[:, 6]), pfo_energy, rtol=1e-5, atol=1e-5):
            errors.append(f"{event}: baseline does not preserve native PFO energy exactly once")
        if np.count_nonzero(candidate[:, 0]) != len(pfo_px):
            errors.append(f"{event}: baseline does not preserve native PFO count")
        pre = float(data["metadata_premerge_energy"][event])
        post = float(np.sum(representatives[:, 6]))
        target_loss.append(pre - post)
        if pre > 0:
            target_response.append(post / pre)
        premerge_counts.append(int(data["metadata_premerge_particles"][event]))
        primary = mc_status == 1
        primary_energy = np.sqrt(mc_px**2 + mc_py**2 + mc_pz**2 + mc_mass**2)[primary]
        if primary.sum() == 1:
            primary_reco_response.append(float(np.sum(np.sqrt(pfo_px**2 + pfo_py**2 + pfo_pz**2 + pfo_mass**2))) / primary_energy[0])
            primary_target_response.append(post / primary_energy[0])
            gp = int(np.flatnonzero(primary)[0])
            single_primary_events.append(
                dict(
                    event=event,
                    abs_cos_theta=float(cos_theta[gp]),
                    tracks=ntracks,
                    pandora_response=primary_reco_response[-1],
                    target_response=primary_target_response[-1],
                )
            )
    return {
        "root": str(Path(root_path).resolve()),
        "parquet": str(Path(parquet_path).resolve()),
        "events": tree.num_entries,
        "requested_events": expected_events,
        "structural_errors": sorted(set(errors)),
        "invalid_fitted_tracks": invalid_tracks,
        "upstream_link_warnings": upstream_link_warnings,
        "native_track_truth_weights": summary(native_weights),
        "target_jet_p4_closure_error_GeV": summary(target_jet_errors),
        "multitrack_pfos": multitrack_pfos,
        "charged_primary_tracking_pt_gt_1GeV": [
            dict(abs_cos_theta=[acceptance_edges[i], min(1.0, acceptance_edges[i + 1])], truth=int(truth_counts[i]), matched=int(matched_counts[i]))
            for i in range(4)
        ],
        "single_primary_events": single_primary_events,
        "tracks_per_event": summary(track_counts),
        "clusters_per_event": summary(cluster_counts),
        "pfos_per_event": summary(pfo_counts),
        "direct_primary_track_pt_ratio": summary(direct_ratios),
        "direct_primary_track_delta_r": summary(direct_dr),
        "direct_primary_track_fraction_dr_lt_0p1": float(np.mean(np.asarray(direct_dr) < 0.1)) if direct_dr else None,
        "direct_primary_track_wrong_charge": int(np.sum(wrong_charge)),
        "direct_primary_track_purity": summary(direct_purity),
        "single_primary_pandora_energy_response": summary(primary_reco_response),
        "single_primary_target_energy_response": summary(primary_target_response),
        "target_over_premerge_energy": summary(target_response),
        "premerge_minus_target_energy_GeV": summary(target_loss),
        "visible_particles_before_merge": summary(premerge_counts),
        "target_direction_decomposition": target_direction_metrics(data),
        "physics_qualified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("parquet")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.root, args.parquet)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return int(bool(result["structural_errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
