import torch
from torch.utils.data import DataLoader
import time
import numpy as np
import fastjet
import vector
import awkward as ak
import argparse
import sys

# Ensure unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Import standalone model
from mlpf.standalone.train import MLPF, train

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
                logits_binary, logits_pid, preds_momentum = model(X, mask)

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


if __name__ == "__main__":
    _configLogger("mlpf", 0)
    args = get_args()
    data_dir = args.data_dir
    print(f"Data directory: {data_dir}")

    # Load dataset
    ds_train = PFDataset(data_dir, "cms_pf_ttbar/1:3.0.0", "train", num_samples=1000).ds
    ds_valid = PFDataset(data_dir, "cms_pf_ttbar/1:3.0.0", "test", num_samples=200).ds

    collater = Collater(["X", "ytarget"], ["genmet"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results = []

    # Run training 3 times
    for i in range(3):
        print(f"\n--- Run {i+1}/3 ---")

        if args.dsl:
            print(f"Using DSL: {args.dsl}")
            config = parse_dsl(args.dsl)
            model = MLPF(config=config).to(device)
        else:
            # 55 features for CMS, re-initialize model for each run
            model = MLPF(
                input_dim=55,
                num_classes=8,
                embedding_dim=128,
                width=128,
                num_convs=6,
                num_heads=16,
                attention_type=args.attention_type,
            ).to(device)

        if i == 0:
            print(model)

        model.train()

        # Fresh loaders for each run (especially for shuffling)
        train_loader = DataLoader(ds_train, batch_size=4, collate_fn=collater, shuffle=True, num_workers=1, persistent_workers=True)
        valid_loader = DataLoader(ds_valid, batch_size=4, collate_fn=collater, num_workers=1, persistent_workers=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Record start time
        start_total = time.time()

        # Train for a fixed time
        avg_loss, num_steps = train(model, train_loader, optimizer, device, duration_seconds=120)

        training_seconds = time.time() - start_total

        # Evaluate
        print("Evaluating...")
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
        sample_input = torch.randn(1, 4096, 55)
        sample_mask = torch.ones(1, 4096).bool()

        # CPU runtime
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
