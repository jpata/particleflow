import os
import torch
import uproot
import awkward as ak
import numpy as np
import tqdm
import yaml
import pickle as pkl
import argparse
import vector

from mlpf.model.mlpf import MLPF
from mlpf.conf import MLPFConfig, CLASS_LABELS
from mlpf.model.utils import load_checkpoint, unpack_predictions

# Import needed functions from postprocessing
from mlpf.data.key4hep.postprocessing import (
    track_coll,
    mc_coll,
    tracker_hit_relations,
    tracker_hit_sim,
    get_hit_matrix_and_genadj,
    hit_cluster_adj,
    cluster_to_features,
    track_to_features,
    get_feature_matrix,
    sanitize,
    track_feature_order,
    cluster_feature_order,
    hit_feature_order,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input ROOT file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--config", type=str, help="Model config (YAML or model_kwargs.pkl)")
    parser.add_argument("--outpath", type=str, default="eval_results.parquet", help="Output Parquet file")
    parser.add_argument("--detector", type=str, default="clic", choices=["clic", "cld", "clic_hits", "cld_hits"], help="Detector type")
    parser.add_argument("--num-events", type=int, default=-1, help="Number of events to process")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu or cuda)")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"], help="Data type for inference")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")

    # Load config
    if args.config:
        if args.config.endswith(".yaml"):
            with open(args.config, "r") as f:
                config_dict = yaml.safe_load(f)
            config = MLPFConfig(**config_dict)
        elif args.config.endswith(".pkl"):
            with open(args.config, "rb") as f:
                config = pkl.load(f)
    else:
        # Try to find config near checkpoint
        config_path = os.path.join(os.path.dirname(args.checkpoint), "model_kwargs.pkl")
        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                config = pkl.load(f)
        else:
            raise ValueError("Config not provided and model_kwargs.pkl not found near checkpoint.")

    # Initialize model
    model = MLPF(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    load_checkpoint(checkpoint, model, None, strict=False)
    model.to(device)
    model.eval()

    print(f"Loading {args.input}")
    fi = uproot.open(args.input)
    tree = fi["events"]

    # map collection ID name to numerical key
    collectionIDs = {
        k: v
        for k, v in zip(
            fi.get("podio_metadata").arrays("events___CollectionTypeInfo/events___CollectionTypeInfo.name")[
                "events___CollectionTypeInfo/events___CollectionTypeInfo.name"
            ][0],
            fi.get("podio_metadata").arrays("events___CollectionTypeInfo/events___CollectionTypeInfo.collectionID")[
                "events___CollectionTypeInfo/events___CollectionTypeInfo.collectionID"
            ][0],
        )
    }
    collectionIDs_reverse = {v: k for (k, v) in collectionIDs.items()}
    mcp_id = collectionIDs["MCParticles"]

    # branches to read
    branches = [
        mc_coll,
        "MCParticles.PDG",
        "MCParticles.momentum.x",
        "MCParticles.momentum.y",
        "MCParticles.momentum.z",
        "MCParticles.mass",
        "MCParticles.generatorStatus",
        track_coll,
        "_SiTracks_Refitted_trackStates",
        "PandoraClusters",
        "_PandoraClusters_hits/_PandoraClusters_hits.index",
        "_PandoraClusters_hits/_PandoraClusters_hits.collectionID",
        "SiTracks_Refitted_dQdx",
    ]

    calohit_branches = [
        "CalohitMCTruthLink.weight",
        "_CalohitMCTruthLink_to/_CalohitMCTruthLink_to.collectionID",
        "_CalohitMCTruthLink_to/_CalohitMCTruthLink_to.index",
        "_CalohitMCTruthLink_from/_CalohitMCTruthLink_from.collectionID",
        "_CalohitMCTruthLink_from/_CalohitMCTruthLink_from.index",
    ]

    tracker_link_branches = []
    for rel_name in tracker_hit_relations.values():
        tracker_link_branches.extend(
            [
                f"_{rel_name}_from/_{rel_name}_from.index",
                f"_{rel_name}_to/_{rel_name}_to.index",
            ]
        )
    for sim_name in tracker_hit_sim.values():
        tracker_link_branches.extend(
            [
                f"_{sim_name}_particle/_{sim_name}_particle.index",
                f"_{sim_name}_particle/_{sim_name}_particle.collectionID",
            ]
        )

    if args.detector.startswith("clic"):
        hit_collections = [
            "ECALBarrel",
            "ECALEndcap",
            "ECALOther",
            "HCALBarrel",
            "HCALEndcap",
            "HCALOther",
            "MUON",
            "LumiCal_Hits",
            "ITrackerHits",
            "ITrackerEndcapHits",
            "OTrackerHits",
            "OTrackerEndcapHits",
            "VXDTrackerHits",
            "VXDEndcapTrackerHits",
        ]
    else:
        hit_collections = [
            "ECALBarrel",
            "ECALEndcap",
            "HCALBarrel",
            "HCALEndcap",
            "HCALOther",
            "MUON",
            "ITrackerHits",
            "ITrackerEndcapHits",
            "OTrackerHits",
            "OTrackerEndcapHits",
            "VXDTrackerHits",
            "VXDEndcapTrackerHits",
        ]

    # Filter available branches
    hit_collections = [k for k in hit_collections if k in tree]

    num_entries = tree.num_entries
    if args.num_events > 0:
        num_entries = min(num_entries, args.num_events)

    print(f"Reading {num_entries} events")
    prop_data = tree.arrays(branches, entry_stop=num_entries)
    calohit_links = tree.arrays(calohit_branches, entry_stop=num_entries)
    tracker_links = tree.arrays(tracker_link_branches, entry_stop=num_entries)

    hit_data = {}
    for k in hit_collections:
        hit_data[k] = tree[k].array(entry_stop=num_entries)

    results = []

    det_key = "clic" if "clic" in args.detector else "cld"
    class_labels = np.array(CLASS_LABELS[det_key])

    dtype = getattr(torch, args.dtype)

    for iev in tqdm.tqdm(range(num_entries)):
        # Get status 1 MC particles (excluding neutrinos)
        mc_pdg_vals = np.abs(prop_data[mc_coll + ".PDG"][iev])
        mc_st1_mask = (prop_data[mc_coll + ".generatorStatus"][iev] == 1) & (mc_pdg_vals != 12) & (mc_pdg_vals != 14) & (mc_pdg_vals != 16)

        true_pdg = ak.to_numpy(prop_data[mc_coll + ".PDG"][iev][mc_st1_mask])
        true_p4 = vector.awk(
            ak.zip(
                {
                    "px": prop_data[mc_coll + ".momentum.x"][iev][mc_st1_mask],
                    "py": prop_data[mc_coll + ".momentum.y"][iev][mc_st1_mask],
                    "pz": prop_data[mc_coll + ".momentum.z"][iev][mc_st1_mask],
                    "mass": prop_data[mc_coll + ".mass"][iev][mc_st1_mask],
                }
            )
        )

        # Prepare input features X
        if "hits" in args.detector:
            hit_features, _, _ = get_hit_matrix_and_genadj(hit_data, calohit_links, tracker_links, iev, collectionIDs, mcp_id)
            X = get_feature_matrix(hit_features, hit_feature_order)
        else:
            # track/cluster based
            track_features = track_to_features(prop_data, iev)
            X_track = get_feature_matrix(track_features, track_feature_order)

            # Need hit features for clusters
            hit_features, _, hit_idx_local_to_global = get_hit_matrix_and_genadj(hit_data, calohit_links, tracker_links, iev, collectionIDs, mcp_id)
            hit_to_cluster = hit_cluster_adj(prop_data, hit_idx_local_to_global, iev, collectionIDs_reverse)
            cluster_features = cluster_to_features(prop_data, hit_features, hit_to_cluster, iev)
            X_cluster = get_feature_matrix(cluster_features, cluster_feature_order)

            # Pad and concatenate
            max_len = max(X_track.shape[1], X_cluster.shape[1])
            X_track = np.pad(X_track, ((0, 0), (0, max_len - X_track.shape[1])))
            X_cluster = np.pad(X_cluster, ((0, 0), (0, max_len - X_cluster.shape[1])))
            X = np.concatenate([X_track, X_cluster], axis=0)

        sanitize(X)
        X_torch = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)
        mask_torch = torch.ones(X_torch.shape[:2], dtype=torch.float32).to(device)

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
                preds = model(X_torch, mask_torch)

            # convert all outputs to float32 in case running in float16 or bfloat16
            preds = tuple([y.to(torch.float32) for y in preds])

            # transform log (pt/elempt) -> pt
            pred_id = torch.argmax(preds[0], axis=-1)
            preds[2][..., 0] = torch.exp(preds[2][..., 0]) * X_torch[..., 1]
            preds[2][..., 0][pred_id == 0] = 0

            # transform log (E/elemE) -> E
            preds[2][..., 4] = torch.exp(preds[2][..., 4]) * X_torch[..., 5]
            preds[2][..., 4][pred_id == 0] = 0

            preds_unpacked = unpack_predictions(preds)

        # Filter out null particles (ID=0)
        pred_id = preds_unpacked["cls_id"][0].cpu().numpy()
        valid_mask = pred_id != 0

        pred_pdg = class_labels[pred_id[valid_mask]]
        pred_p4 = preds_unpacked["p4"][0][valid_mask].cpu().numpy()

        results.append(
            {
                "true_pdg": true_pdg,
                "true_pt": ak.to_numpy(true_p4.pt),
                "true_eta": ak.to_numpy(true_p4.eta),
                "true_phi": ak.to_numpy(true_p4.phi),
                "true_energy": ak.to_numpy(true_p4.energy),
                "pred_pdg": pred_pdg,
                "pred_pt": pred_p4[:, 0],
                "pred_eta": pred_p4[:, 1],
                "pred_phi": pred_p4[:, 2],
                "pred_energy": pred_p4[:, 3],
            }
        )

    print(f"Saving results to {args.outpath}")
    ak.to_parquet(ak.Array(results), args.outpath)


if __name__ == "__main__":
    main()
