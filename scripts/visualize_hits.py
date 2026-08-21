import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import umap
from sklearn.cluster import KMeans
import json

from mlpf.model.mlpf import MLPF
from mlpf.model.PFDataset import PFDataset, Collater
from mlpf.conf import MLPFConfig


def load_model(checkpoint_path, config):
    model = MLPF(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def get_embeddings_and_preds(model, batch, dtype):
    X = batch.X.to(dtype)
    mask = batch.mask

    # Run the model's internal logic
    with torch.no_grad():
        Xfeat_normed = X
        B, S, _ = Xfeat_normed.shape
        num_types = len(model.elemtypes_nonzero)

        # input encoding
        all_id = model.nn0_id(Xfeat_normed).view(B, S, num_types, -1)
        all_reg = model.nn0_reg(Xfeat_normed).view(B, S, num_types, -1)

        elemtype_mask = torch.cat([X[..., 0:1] == elemtype for elemtype in model.elemtypes_nonzero], axis=-1)

        embedding_id = torch.sum(all_id * elemtype_mask.unsqueeze(-1), axis=2)
        embedding_reg = torch.sum(all_reg * elemtype_mask.unsqueeze(-1), axis=2)

        embeddings_id = []
        embeddings_reg = []

        if model.num_convs != 0:
            for num, conv in enumerate(model.conv_id):
                conv_input = embedding_id if num == 0 else embeddings_id[-1]
                out_padded = conv(conv_input, mask, embedding_id)
                embeddings_id.append(out_padded)
            for num, conv in enumerate(model.conv_reg):
                conv_input = embedding_reg if num == 0 else embeddings_reg[-1]
                out_padded = conv(conv_input, mask, embedding_reg)
                embeddings_reg.append(out_padded)
        else:
            embeddings_id.append(embedding_id)
            embeddings_reg.append(embedding_reg)

        final_embedding_id = embeddings_id[-1]
        if model.use_pre_layernorm:
            final_embedding_id = model.final_norm_id(final_embedding_id)

        preds_binary_particle = model.nn_binary_particle(final_embedding_id)

        return embeddings_id[-1], embeddings_reg[-1], preds_binary_particle


def process_embeddings(embeddings_raw, mask, n_clusters=10):
    embeddings = embeddings_raw[0].cpu().float().numpy()[mask]

    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_embeddings = reducer.fit_transform(embeddings)

    print("Clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    return umap_embeddings, labels


def main():
    checkpoint_path = "experiments/pyg-cld-hits-v1_cld_20260328_101205_900871/checkpoints/checkpoint-80000.pth"
    hyperparams_path = "experiments/pyg-cld-hits-v1_cld_20260328_101205_900871/hyperparameters.json"

    with open(hyperparams_path, "r") as f:
        config_dict = json.load(f)

    valid_keys = MLPFConfig.model_fields.keys()
    filtered_config_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
    config = MLPFConfig.model_validate(filtered_config_dict)

    print("Loading model...")
    model = load_model(checkpoint_path, config)

    dtype = torch.float32
    if config.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif config.dtype == "float16":
        dtype = torch.float16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.to(dtype)

    print("Loading dataset...")
    ds = PFDataset("data/tfds/tensorflow_datasets/cld", "cld_edm_ttbar_hits/1:3.1.0", "test", num_samples=1).ds
    loader = DataLoader(ds, batch_size=1, collate_fn=Collater(["X", "ytarget"], ["genmet"]))

    batch = next(iter(loader))
    batch = batch.to(device, dtype=dtype)

    print("Extracting embeddings and predictions...")
    emb_id, emb_reg, preds_binary = get_embeddings_and_preds(model, batch, dtype)

    mask = batch.mask[0].cpu().numpy()
    X_masked = batch.X[0].cpu().float().numpy()[mask]
    pos_x, pos_y, pos_z = X_masked[:, 6], X_masked[:, 7], X_masked[:, 8]

    # Calculate number of predicted particles
    # preds_binary is [B, S, 2], we take the argmax over the last dimension
    pred_is_particle = torch.argmax(preds_binary[0], dim=-1).cpu().numpy()[mask]
    num_pred_particles = np.sum(pred_is_particle == 1)

    # Calculate true number of particles
    # ytarget is [B, S, 13], index 0 is PID. PID 0 is none/padded.
    ytarget_pid = batch.ytarget[0].cpu().float().numpy()[mask][:, 0]
    num_true_particles = np.sum(ytarget_pid != 0)

    print(f"Total hits: {len(X_masked)}")
    print(f"Number of predicted particles: {num_pred_particles}")
    print(f"Number of true particles: {num_true_particles}")

    n_clusters = max(1, int(num_pred_particles))
    print(f"Using n_clusters={n_clusters} for KMeans.")

    # Hit types for markers
    # subdetector is at index 10 in EDM4HEP.HitFeatures
    subdetector = X_masked[:, 10].astype(int)
    tracker_mask = subdetector == 3
    calo_mask = subdetector != 3

    # Energy-based marker sizes
    # energy is at index 5 in EDM4HEP.HitFeatures
    energies = X_masked[:, 5]
    min_energy = np.percentile(energies, 5)
    max_energy = np.percentile(energies, 95)
    # Ensure min_energy > 0 for log scaling and avoid division by zero
    min_energy = max(min_energy, 1e-9)
    scaling = np.log10(np.clip(energies, min_energy, max_energy) / min_energy) + 1

    s_tracker = 2 * scaling[tracker_mask]
    s_calo = 5 * scaling[calo_mask]

    print(f"Number of tracker hits: {np.sum(tracker_mask)}")
    print(f"Number of calo hits: {np.sum(calo_mask)}")

    print("\nProcessing ID embeddings...")
    umap_id, labels_id = process_embeddings(emb_id, mask, n_clusters=n_clusters)

    print("\nProcessing REG embeddings...")
    umap_reg, labels_reg = process_embeddings(emb_reg, mask, n_clusters=n_clusters)

    print("\nCreating visualization...")
    fig = plt.figure(figsize=(30, 20))
    fig.suptitle(f"CLD Hits Visualization\nPredicted Particles: {num_pred_particles}, True Particles: {num_true_particles}", fontsize=24)

    cmap = plt.get_cmap("tab20") if n_clusters <= 20 else plt.get_cmap("gist_ncar")

    # --- ROW 1: conv_id based ---

    # Plot 1: 3D Hits colored by conv_id clusters
    ax11 = fig.add_subplot(231, projection="3d")
    ax11.scatter(
        pos_x[tracker_mask], pos_y[tracker_mask], pos_z[tracker_mask], c=labels_id[tracker_mask], cmap=cmap, marker="o", s=s_tracker, label="Tracker"
    )
    ax11.scatter(pos_x[calo_mask], pos_y[calo_mask], pos_z[calo_mask], c=labels_id[calo_mask], cmap=cmap, marker="s", s=s_calo, label="Calo")
    ax11.set_title("Hits in 3D Space (Colored by conv_id Clusters)")
    ax11.set_xlabel("X")
    ax11.set_ylabel("Y")
    ax11.set_zlabel("Z")
    ax11.legend()

    # Plot 2: UMAP of conv_id embeddings
    ax12 = fig.add_subplot(232)
    ax12.scatter(umap_id[tracker_mask, 0], umap_id[tracker_mask, 1], c=labels_id[tracker_mask], cmap=cmap, s=s_tracker, marker="o")
    scatter12_c = ax12.scatter(umap_id[calo_mask, 0], umap_id[calo_mask, 1], c=labels_id[calo_mask], cmap=cmap, s=s_calo, marker="s")
    ax12.set_title("UMAP of conv_id Embeddings")
    ax12.set_xlabel("UMAP 1")
    ax12.set_ylabel("UMAP 2")
    plt.colorbar(scatter12_c, ax=ax12, label="Cluster ID")

    # Plot 3: 3D Hits with target-based opacity
    ax13 = fig.add_subplot(233, projection="3d")
    is_target = ytarget_pid != 0
    for msk, alpha, label_prefix in [(is_target, 1.0, "Target"), (~is_target, 0.05, "None")]:
        tm = tracker_mask & msk
        cm = calo_mask & msk
        if np.any(tm):
            ax13.scatter(
                pos_x[tm],
                pos_y[tm],
                pos_z[tm],
                c=labels_id[tm],
                cmap=cmap,
                marker="o",
                s=s_tracker[msk[tracker_mask]],
                alpha=alpha,
                label=f"{label_prefix} Tracker",
            )
        if np.any(cm):
            ax13.scatter(
                pos_x[cm],
                pos_y[cm],
                pos_z[cm],
                c=labels_id[cm],
                cmap=cmap,
                marker="s",
                s=s_calo[msk[calo_mask]],
                alpha=alpha,
                label=f"{label_prefix} Calo",
            )
    ax13.set_title("Hits in 3D Space (Target Opacity)")
    ax13.set_xlabel("X")
    ax13.set_ylabel("Y")
    ax13.set_zlabel("Z")

    # --- ROW 2: conv_reg based ---

    # Plot 4: 3D Hits colored by conv_reg clusters
    ax21 = fig.add_subplot(234, projection="3d")
    ax21.scatter(
        pos_x[tracker_mask], pos_y[tracker_mask], pos_z[tracker_mask], c=labels_reg[tracker_mask], cmap=cmap, marker="o", s=s_tracker, label="Tracker"
    )
    ax21.scatter(pos_x[calo_mask], pos_y[calo_mask], pos_z[calo_mask], c=labels_reg[calo_mask], cmap=cmap, marker="s", s=s_calo, label="Calo")
    ax21.set_title("Hits in 3D Space (Colored by conv_reg Clusters)")
    ax21.set_xlabel("X")
    ax21.set_ylabel("Y")
    ax21.set_zlabel("Z")
    ax21.legend()

    # Plot 5: UMAP of conv_reg embeddings
    ax22 = fig.add_subplot(235)
    ax22.scatter(umap_reg[tracker_mask, 0], umap_reg[tracker_mask, 1], c=labels_reg[tracker_mask], cmap=cmap, s=s_tracker, marker="o")
    scatter22_c = ax22.scatter(umap_reg[calo_mask, 0], umap_reg[calo_mask, 1], c=labels_reg[calo_mask], cmap=cmap, s=s_calo, marker="s")
    ax22.set_title("UMAP of conv_reg Embeddings")
    ax22.set_xlabel("UMAP 1")
    ax22.set_ylabel("UMAP 2")
    plt.colorbar(scatter22_c, ax=ax22, label="Cluster ID")

    # Plot 6: 3D Hits with prediction-based opacity
    ax23 = fig.add_subplot(236, projection="3d")
    is_pred = pred_is_particle == 1
    for msk, alpha, label_prefix in [(is_pred, 1.0, "Pred"), (~is_pred, 0.05, "None")]:
        tm = tracker_mask & msk
        cm = calo_mask & msk
        if np.any(tm):
            ax23.scatter(
                pos_x[tm],
                pos_y[tm],
                pos_z[tm],
                c=labels_reg[tm],
                cmap=cmap,
                marker="o",
                s=s_tracker[msk[tracker_mask]],
                alpha=alpha,
                label=f"{label_prefix} Tracker",
            )
        if np.any(cm):
            ax23.scatter(
                pos_x[cm],
                pos_y[cm],
                pos_z[cm],
                c=labels_reg[cm],
                cmap=cmap,
                marker="s",
                s=s_calo[msk[calo_mask]],
                alpha=alpha,
                label=f"{label_prefix} Calo",
            )
    ax23.set_title("Hits in 3D Space (Prediction Opacity)")
    ax23.set_xlabel("X")
    ax23.set_ylabel("Y")
    ax23.set_zlabel("Z")

    output_path = "hits_visualization_2x3.png"
    plt.savefig(output_path)
    print(f"\nVisualization saved to {output_path}")


if __name__ == "__main__":
    main()
