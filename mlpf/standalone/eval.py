import torch
from torch.utils.data import DataLoader
import time
import numpy as np
import fastjet
import vector
import awkward as ak
import argparse
import sys
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Import standalone model
from mlpf.standalone.train import MLPF, train, validate

# Import library modules
from mlpf.model.PFDataset import Collater, PFDataset
from mlpf.logger import _configLogger
from mlpf.jet_utils import match_jets


from mlpf.standalone.dsl import parse_dsl


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None, help="Path to tfds directory")
    parser.add_argument(
        "--attention-type",
        type=str,
        default="global",
        choices=["hept", "global", "standard", "fastformer"],
        help="Attention type (ignored if --dsl is used)",
    )
    parser.add_argument("--dsl", type=str, default=None, help="Model architecture DSL string")
    parser.add_argument("--show-attention", action="store_true", help="Save attention visualization")
    return parser.parse_args()


def cluster_jets(p4s):
    # p4s: awkward array of vectors
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
    cluster = fastjet.ClusterSequence(p4s, jetdef)
    jets = cluster.inclusive_jets()
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
    all_pred_particles = []
    all_target_particles = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i % 10 == 0:
                print("eval batch {}".format(i))
            if i > 10:  # Limit evaluation for speed
                break

            X = batch.X.to(device)
            mask = batch.mask.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits_binary, logits_pid, logits_pu, preds_momentum = model(X, mask)

            # Predicted
            pred_id = torch.argmax(logits_pid, dim=-1)

            pt = (torch.exp(preds_momentum[..., 0]) * X[..., 1]).detach().cpu().numpy()
            eta = preds_momentum[..., 1].detach().cpu().numpy()
            sin_phi = preds_momentum[..., 2].detach().cpu().numpy()
            cos_phi = preds_momentum[..., 3].detach().cpu().numpy()
            phi = np.arctan2(sin_phi, cos_phi)
            energy = (torch.exp(preds_momentum[..., 4]) * X[..., 5]).detach().cpu().numpy()

            # Target
            target_id = batch.ytarget[:, :, 0].detach().cpu().numpy()
            target_pt_log = batch.ytarget[:, :, 2].detach().cpu().numpy()
            target_eta_val = batch.ytarget[:, :, 3].detach().cpu().numpy()
            target_sin_phi = batch.ytarget[:, :, 4].detach().cpu().numpy()
            target_cos_phi = batch.ytarget[:, :, 5].detach().cpu().numpy()
            target_energy_log = batch.ytarget[:, :, 6].detach().cpu().numpy()

            X_pt = X[..., 1].detach().cpu().numpy()
            X_e = X[..., 5].detach().cpu().numpy()
            mask_np = mask.detach().cpu().numpy()
            pred_id_np = pred_id.detach().cpu().numpy()

            # Collect particles for jet clustering
            for b in range(X.shape[0]):
                p_pred = []
                for j in range(X.shape[1]):
                    if mask_np[b, j] and pred_id_np[b, j] != 0:
                        p_pred.append({"pt": pt[b, j], "eta": eta[b, j], "phi": phi[b, j], "e": energy[b, j]})
                if len(p_pred) > 0:
                    all_pred_particles.append(vector.awk(ak.from_iter(p_pred)))
                else:
                    all_pred_particles.append(None)

                p_target = []
                for j in range(X.shape[1]):
                    if mask_np[b, j] and target_id[b, j] != 0:
                        t_pt = np.exp(target_pt_log[b, j]) * X_pt[b, j]
                        t_phi = np.arctan2(target_sin_phi[b, j], target_cos_phi[b, j])
                        t_e = np.exp(target_energy_log[b, j]) * X_e[b, j]
                        p_target.append({"pt": t_pt, "eta": target_eta_val[b, j], "phi": t_phi, "e": t_e})
                if len(p_target) > 0:
                    all_target_particles.append(vector.awk(ak.from_iter(p_target)))
                else:
                    all_target_particles.append(None)

    # Compute response by clustering event-by-event
    responses = []
    jet_match_dr = 0.4
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)

    total_jets_pred = 0
    total_matched_jets_pred = 0

    for ev in range(len(all_target_particles)):
        if all_target_particles[ev] is not None and all_pred_particles[ev] is not None:
            if np.sum(all_target_particles[ev].pt > 0) > 0 and np.sum(all_pred_particles[ev].pt > 0) > 0:
                # Cluster target jets
                cluster_target = fastjet.ClusterSequence(all_target_particles[ev], jetdef)
                jets_target = cluster_target.inclusive_jets(min_pt=5.0)

                # Cluster predicted jets
                cluster_pred = fastjet.ClusterSequence(all_pred_particles[ev], jetdef)
                jets_pred = cluster_pred.inclusive_jets(min_pt=5.0)

                if len(jets_target) > 0 and len(jets_pred) > 0:
                    total_jets_pred += len(jets_pred)

                    # Prepare jets for matching by converting to spherical awkward array
                    jets_target_v = vector.awk(ak.zip({"px": jets_target.px, "py": jets_target.py, "pz": jets_target.pz, "E": jets_target.E}))
                    jets_pred_v = vector.awk(ak.zip({"px": jets_pred.px, "py": jets_pred.py, "pz": jets_pred.pz, "E": jets_pred.E}))

                    jets_target_sph = ak.zip({"pt": jets_target_v.pt, "eta": jets_target_v.eta, "phi": jets_target_v.phi, "e": jets_target_v.E})
                    jets_pred_sph = ak.zip({"pt": jets_pred_v.pt, "eta": jets_pred_v.eta, "phi": jets_pred_v.phi, "e": jets_pred_v.E})

                    # Match jets
                    j1_idx, j2_idx = match_jets(ak.Array([jets_target_sph]), ak.Array([jets_pred_sph]), jet_match_dr)

                    if len(j1_idx[0]) > 0:
                        total_matched_jets_pred += len(j1_idx[0])
                        for i_target, i_pred in zip(j1_idx[0], j2_idx[0]):
                            pt_target = jets_target_sph[i_target].pt
                            pt_pred = jets_pred_sph[i_pred].pt
                            if pt_target > 0:
                                responses.append(pt_pred / pt_target)

    if len(responses) == 0:
        print("No matched jets found.")
        return 0.0, 0.0

    res_med, res_iqr = med_iqr(responses)
    matched_fraction = total_matched_jets_pred / total_jets_pred if total_jets_pred > 0 else 0.0
    return res_iqr, matched_fraction


def save_attention_visualization(model, batch, device, output_dir="plots"):
    model.eval()
    X = batch.X.to(device)
    mask = batch.mask.to(device)

    # We must run with batch size one for visualization to be interpretable
    X = X[0:1]
    mask = mask[0:1]
    assert X.shape[0] == 1

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            _, _, _, _, attns = model(X, mask, return_attn=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Take the last layer's attention for visualization
    attn = attns[-1]

    if attn is None:
        print("No attention weights available for this model type.")
        return

    if isinstance(attn, torch.Tensor):
        plt.figure(figsize=(10, 8))
        # Standard attention: [B=1, heads, N, N]
        # Average over heads
        mat = attn[0].mean(dim=0).cpu().numpy()
        plt.imshow(mat, cmap="viridis", interpolation="nearest")
        plt.colorbar()
        plt.title("Standard Attention Matrix (Mean over heads)")
    elif isinstance(attn, tuple) and len(attn) == 3:
        # HEPT: (qk, q_positions, k_positions)
        # qk: [hashes, heads, nbuckets, bsz, bsz]
        qk, q_pos, k_pos = attn
        n = X.shape[1]
        hashes, heads, nbuckets, bsz, _ = qk.shape

        # 1. Natural Order reconstruction
        full_matrix = torch.zeros((n, n), device=qk.device)
        for i in range(hashes):
            for j in range(heads):
                for b in range(nbuckets):
                    q_idx = q_pos[i, j, b * bsz : (b + 1) * bsz]
                    k_idx = k_pos[i, j, b * bsz : (b + 1) * bsz]
                    q_mask = q_idx < n
                    k_mask = k_idx < n

                    valid_qk = qk[i, j, b][q_mask][:, k_mask]
                    valid_q_idx = q_idx[q_mask]
                    valid_k_idx = k_idx[k_mask]

                    if len(valid_q_idx) > 0 and len(valid_k_idx) > 0:
                        for row_local, row_global in enumerate(valid_q_idx):
                            full_matrix[row_global, valid_k_idx] += valid_qk[row_local]

        full_matrix /= hashes * heads

        # 2. LSH-Sorted Order reconstruction (for the first hash)
        sorted_matrix = torch.zeros((nbuckets * bsz, nbuckets * bsz), device=qk.device)
        sorted_blocks = qk[0].mean(dim=0)  # Average over heads for the first hash
        for b in range(nbuckets):
            sorted_matrix[b * bsz : (b + 1) * bsz, b * bsz : (b + 1) * bsz] = sorted_blocks[b]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        im1 = ax1.imshow(full_matrix.cpu().numpy(), cmap="viridis", interpolation="nearest")
        fig.colorbar(im1, ax=ax1)
        ax1.set_title("HEPT Natural Order\n(Input sequence indices)")

        im2 = ax2.imshow(sorted_matrix.cpu().numpy(), cmap="viridis", interpolation="nearest")
        fig.colorbar(im2, ax=ax2)
        ax2.set_title("HEPT LSH-Sorted Order\n(Block-diagonal structure)")

    elif isinstance(attn, tuple) and len(attn) == 2:
        plt.figure(figsize=(10, 8))
        # Global or Fastformer: (alpha, beta)
        alpha, beta = attn
        # alpha: [B=1, N, heads]
        mat = alpha[0].mean(dim=-1).cpu().numpy()
        plt.plot(mat)
        plt.title("Global Attention Weights (alpha, mean over heads)")

    plt.savefig(os.path.join(output_dir, "attention_matrix.png"))
    plt.close()
    print(f"Attention visualization saved to {output_dir}/attention_matrix.png")


if __name__ == "__main__":
    _configLogger("mlpf", 0)
    args = get_args()
    data_dir = args.data_dir
    print(f"Data directory: {data_dir}")

    # Load dataset
    ds_train = PFDataset(data_dir, "cms_pf_qcd/1:3.0.0", "train", num_samples=1000).ds
    ds_valid = PFDataset(data_dir, "cms_pf_qcd/1:3.0.0", "test", num_samples=200).ds

    collater = Collater(["X", "ytarget"], ["genmet"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results = []

    # Run training 3 times
    for i in range(1):
        print(f"\n--- Run {i+1}/3 ---")

        if args.dsl:
            print(f"Using DSL: {args.dsl}")
            config = parse_dsl(args.dsl)
            model = MLPF(config=config).to(device)
        else:
            # 73 features for CMS, re-initialize model for each run
            model = MLPF(
                input_dim=73,
                num_classes=8,
                embedding_dim=128,
                width=128,
                num_convs=3,
                num_heads=16,
                attention_type=args.attention_type,
            ).to(device)

        if i == 0:
            print(model)

        model.train()

        # Fresh loaders for each run (especially for shuffling)
        train_loader = DataLoader(ds_train, batch_size=8, collate_fn=collater, shuffle=True, num_workers=1, persistent_workers=True)
        valid_loader = DataLoader(ds_valid, batch_size=8, collate_fn=collater, num_workers=1, persistent_workers=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Record start time
        start_total = time.time()

        # Train for a fixed time
        train_loss, train_loss_binary, train_loss_pid, train_loss_kinematics, train_loss_pu, num_steps = train(model, train_loader, optimizer, device, duration_seconds=30)

        training_seconds = time.time() - start_total

        # Save attention visualization for one event
        if args.show_attention:
            print("Saving attention visualization...")
            one_batch = next(iter(valid_loader))
            save_attention_visualization(model, one_batch, device)

        # Validation loss
        print("Computing validation loss...")
        val_loss, val_loss_binary, val_loss_pid, val_loss_kinematics, val_loss_pu = validate(model, valid_loader, device)

        # Evaluate jet metrics
        print("Evaluating jet metrics...")
        val_jet_iqr, val_jet_matched_frac = evaluate(model, valid_loader, device)

        total_seconds = time.time() - start_total

        # Peak VRAM
        if torch.cuda.is_available():
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            torch.cuda.reset_peak_memory_stats()
        else:
            peak_vram_mb = 0.0

        # Benchmarking
        model.eval()
        sample_input = torch.randn(1, 4096, 73)
        sample_mask = torch.ones(1, 4096).bool()

        # CPU runtime
        try:
            model_cpu = model.to("cpu")
            model_cpu.compile()
            model_cpu(sample_input, sample_mask)
            cpu_times = []
            with torch.no_grad():
                for _ in range(10):
                    start = time.time()
                    _ = model_cpu(sample_input, sample_mask)
                    cpu_times.append((time.time() - start) * 1000)
            runtime_cpu_ms = np.median(cpu_times)
        except Exception as e:
            print(f"CPU benchmarking failed: {e}")
            runtime_cpu_ms = 10000.0

        # GPU runtime
        if torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model_gpu = model.to("cuda")
                sample_input_gpu = sample_input.to("cuda")
                sample_mask_gpu = sample_mask.to("cuda")
                model_gpu.compile()
                model_gpu(sample_input_gpu, sample_mask_gpu)
                gpu_times = []
                # Warmup
                for _ in range(5):
                    _ = model_gpu(sample_input_gpu, sample_mask_gpu)
                with torch.no_grad():
                    for _ in range(10):
                        torch.cuda.synchronize()
                        start = time.time()
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            _ = model_gpu(sample_input_gpu, sample_mask_gpu)
                        torch.cuda.synchronize()
                        gpu_times.append((time.time() - start) * 1000)
                runtime_gpu_ms = np.median(gpu_times)
        else:
            runtime_gpu_ms = 0.0

        all_results.append(
            {
                "train_loss": train_loss,
                "train_loss_binary": train_loss_binary,
                "train_loss_pid": train_loss_pid,
                "train_loss_kinematics": train_loss_kinematics,
                "train_loss_pu": train_loss_pu,
                "val_loss": val_loss,
                "val_loss_binary": val_loss_binary,
                "val_loss_pid": val_loss_pid,
                "val_loss_kinematics": val_loss_kinematics,
                "val_loss_pu": val_loss_pu,
                "val_jet_iqr": val_jet_iqr,
                "val_jet_matched_frac": val_jet_matched_frac,
                "training_seconds": training_seconds,
                "total_seconds": total_seconds,
                "peak_vram_mb": peak_vram_mb,
                "num_steps": num_steps,
                "runtime_cpu_ms": runtime_cpu_ms,
                "runtime_gpu_ms": runtime_gpu_ms,
            }
        )

    # Model info
    num_params_M = sum(p.numel() for p in model.parameters()) / 1e6

    # Final output
    print("\n--- Final Results (3 runs) ---")
    for key in all_results[0].keys():
        values = [res[key] for res in all_results]
        mean = np.mean(values)
        variance = np.var(values)
        print(f"{key:16}: {mean:.6f} ± {variance:.6f} (var)")

    print(f"num_params_M:     {num_params_M:.1f}")
