import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import os
import sys
import numpy as np
import fastjet
import vector
import awkward as ak
import yaml
import argparse

# Import standalone model
from mlpf.standalone.train import MLPF, train

# Import library modules
from mlpf.model.PFDataset import Collater, PFDataset
from mlpf.logger import _logger, _configLogger
from mlpf.conf import MLPFConfig
from mlpf.jet_utils import match_jets

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None, help="Path to tfds directory")
    return parser.parse_args()

def cluster_jets(p4s):
    # p4s: awkward array of vectors
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
    cluster = fastjet.ClusterSequence(p4s, jetdef)
    jets = cluster.inclusive_jets(ptmin=5.0)
    return jets

def med_iqr(arr):
    if len(arr) == 0:
        return 0.0, 0.0
    p25 = np.percentile(arr, 25)
    p50 = np.percentile(arr, 50)
    p75 = np.percentile(arr, 75)
    iqr = p75 - p25
    return p50, iqr

def evaluate(model, loader, device):
    model.eval()
    all_pred_p4 = []
    all_target_p4 = []
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i > 100: # Limit evaluation to 100 batches for speed
                break
                
            X = batch.X.to(device)
            mask = batch.mask.to(device)
            
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits_binary, logits_pid, preds_momentum = model(X, mask)
            
            # Predicted
            pred_id = torch.argmax(logits_pid, dim=-1)
            mask_pred = (pred_id != 0) & mask
            
            pred_pt = torch.exp(preds_momentum[..., 0]) * X[..., 1]
            pred_eta = preds_momentum[..., 1]
            pred_sin_phi = preds_momentum[..., 2]
            pred_cos_phi = preds_momentum[..., 3]
            pred_phi = torch.atan2(pred_sin_phi, pred_cos_phi)
            pred_e = torch.exp(preds_momentum[..., 4]) * X[..., 5]
            
            # Target
            target_id = batch.ytarget[:, :, 0].to(device)
            mask_target = (target_id != 0) & mask
            
            target_pt = torch.exp(batch.ytarget[:, :, 2].to(device)) * X[..., 1]
            target_eta = batch.ytarget[:, :, 3].to(device)
            target_phi = torch.atan2(batch.ytarget[:, :, 4].to(device), batch.ytarget[:, :, 5].to(device))
            target_e = torch.exp(batch.ytarget[:, :, 6].to(device)) * X[..., 5]

            # Collect particles for jet clustering
            for b in range(X.shape[0]):
                p_pred = vector.awk([
                    {"pt": pt, "eta": eta, "phi": phi, "e": e}
                    for pt, eta, phi, e, m in zip(pred_pt[b], pred_eta[b], pred_phi[b], pred_e[b], mask_pred[b])
                    if m
                ])
                all_pred_p4.append(p_pred)
                
                p_target = vector.awk([
                    {"pt": pt, "eta": eta, "phi": phi, "e": e}
                    for pt, eta, phi, e, m in zip(target_pt[b], target_eta[b], target_phi[b], target_e[b], mask_target[b])
                    if m
                ])
                all_target_p4.append(p_target)

    # Convert to awkward for fastjet
    all_pred_p4 = ak.from_iter(all_pred_p4)
    all_target_p4 = ak.from_iter(all_target_p4)
    
    # Cluster jets
    jets_pred = cluster_jets(all_pred_p4)
    jets_target = cluster_jets(all_target_p4)
    
    # Match jets
    jet_match_dr = 0.4
    j1_idx, j2_idx = match_jets(jets_target, jets_pred, jet_match_dr)
    
    # Compute response
    responses = []
    for ev in range(len(jets_target)):
        for i_target, i_pred in zip(j1_idx[ev], j2_idx[ev]):
            pt_target = jets_target[ev][i_target].pt
            pt_pred = jets_pred[ev][i_pred].pt
            responses.append(pt_pred / pt_target)
            
    res_med, res_iqr = med_iqr(responses)
    return res_iqr

if __name__ == "__main__":
    _configLogger("mlpf", 0)
    args = get_args()
    data_dir = args.data_dir
    print(f"Data directory: {data_dir}")
    
    # Load dataset
    ds_train = PFDataset(data_dir, "cms_pf_ttbar/1:3.0.0", "train", num_samples=1000).ds
    ds_valid = PFDataset(data_dir, "cms_pf_ttbar/1:3.0.0", "test", num_samples=200).ds
    
    collater = Collater(["X", "ytarget"], ["genmet"])
    train_loader = DataLoader(ds_train, batch_size=4, collate_fn=collater, shuffle=True)
    valid_loader = DataLoader(ds_valid, batch_size=4, collate_fn=collater)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 55 features for CMS
    model = MLPF(input_dim=55, num_classes=8, embedding_dim=128, width=128, num_convs=3, num_heads=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
    
    # Record start time
    start_total = time.time()
    
    # Train for 2 minutes
    avg_loss, num_steps = train(model, train_loader, optimizer, device, duration_seconds=120)
    
    training_seconds = time.time() - start_total
    
    # Evaluate
    print("Evaluating...")
    val_jet_iqr = evaluate(model, valid_loader, device)
    
    total_seconds = time.time() - start_total
    
    # Peak VRAM
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        peak_vram_mb = 0.0
        
    # Model info
    num_params_M = sum(p.numel() for p in model.parameters()) / 1e6
    depth = 3 # num_convs
    
    # Final output
    print("---")
    print(f"val_jet_iqr:      {val_jet_iqr:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {num_steps}")
    print(f"num_params_M:     {num_params_M:.1f}")
    print(f"depth:            {depth}")
