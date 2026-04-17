import torch
from torch.utils.data import DataLoader
import numpy as np
import fastjet
import vector
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
from mlpf.standalone.train import MLPF, validate

# Import library modules
from mlpf.model.PFDataset import Collater, PFDataset
from mlpf.logger import _configLogger


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
    return cluster.inclusive_jets()


def save_attention_plots(model, ds, device, output_dir="plots/attention"):
    # ds is a dataset, take one event
    loader = DataLoader(ds, batch_size=1, collate_fn=Collater())
    batch = next(iter(loader))
    X = batch.X.to(device)
    mask = batch.mask.to(device)

    model.eval()
    with torch.no_grad():
        # Check if we can use AMP
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            _, _, _, attns = model(X, mask, return_attn=True)

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
                    valid_q = q_idx[q_mask]
                    valid_k = k_idx[k_mask]
                    full_matrix[valid_q.unsqueeze(-1), valid_k] += qk[i, j, b, : len(valid_q), : len(valid_k)]

        mat = (full_matrix / (hashes * heads)).cpu().numpy()
        plt.figure(figsize=(10, 8))
        plt.imshow(mat, cmap="viridis", interpolation="nearest")
        plt.colorbar()
        plt.title("HEPT Reconstructed Attention Matrix")

    plt.savefig(f"{output_dir}/attention_matrix.png")
    print(f"Saved attention visualization to {output_dir}/attention_matrix.png")


def main():
    args = get_args()
    _configLogger("eval")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Setup Model
    if args.dsl:
        print(f"Using DSL: {args.dsl}")
        config = parse_dsl(args.dsl)
        model = MLPF(config=config).to(device)
    else:
        print(f"Using manual config with attention_type={args.attention_type}")
        model = MLPF(attention_type=args.attention_type).to(device)

    # 2. Setup Data
    ds_test = PFDataset(args.data_dir, "cms_pf_ttbar", "test", num_samples=100)
    loader = DataLoader(ds_test, batch_size=4, collate_fn=Collater())

    # 3. Basic Validation
    val_loss = validate(model, loader, device)
    print(f"Validation Loss (random weights): {val_loss:.4f}")

    # 4. Jet Clustering Test
    print("Testing jet clustering...")
    batch = next(iter(loader))
    X = batch.X.to(device)
    mask = batch.mask.to(device)
    y = batch.y.to(device)

    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            _, preds_id, preds_momentum, _ = model(X, mask)

    # Convert to momentum vectors
    # preds_momentum: [B, N, 5] (pt, eta, sin_phi, cos_phi, energy)
    # y: [B, N, 7] (pid, charge, pt, eta, sin_phi, cos_phi, energy)

    for i in range(X.shape[0]):
        # Target jets
        y_i = y[i][mask[i].bool()]
        p4_target = vector.awk(
            {
                "pt": y_i[:, 2].cpu().numpy(),
                "eta": y_i[:, 3].cpu().numpy(),
                "phi": np.arctan2(y_i[:, 4].cpu().numpy(), y_i[:, 5].cpu().numpy()),
                "e": y_i[:, 6].cpu().numpy(),
            }
        )
        jets_target = cluster_jets(p4_target)

        # Predicted jets
        p_i = preds_momentum[i][mask[i].bool()]
        p4_pred = vector.awk(
            {
                "pt": p_i[:, 0].cpu().numpy(),
                "eta": p_i[:, 1].cpu().numpy(),
                "phi": np.arctan2(p_i[:, 2].cpu().numpy(), p_i[:, 3].cpu().numpy()),
                "e": p_i[:, 4].cpu().numpy(),
            }
        )
        jets_pred = cluster_jets(p4_pred)

        print(f"Event {i}: Target jets: {len(jets_target)}, Pred jets: {len(jets_pred)}")

    # 5. Attention visualization
    if args.show_attention:
        save_attention_plots(model, ds_test, device)


if __name__ == "__main__":
    main()
