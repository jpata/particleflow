"""
Spec: 3D visualization tool for Particle Number (PN) assignments. Plots tracker/calo hits, tracks, and clusters colored by PN using matplotlib 3D. Used for qualitative assessment of clustering logic. Requires an external '.parquet' file (e.g., from 'scripts/fetch_test_data_cld.sh' or 'scripts/local_test_cld.sh').
"""
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def visualize_event(file_path, iev=0):
    print(f"Loading {file_path} (Event {iev})...")
    data = ak.from_parquet(file_path)

    num_events = len(data["X_hit_tracker"])
    if iev >= num_events:
        print(f"Error: Event index {iev} out of range (max {num_events-1})")
        return

    # 1. Extract Hits and their PN (Particle Number)
    x_trk_h = ak.to_numpy(data["X_hit_tracker"][iev])
    yt_trk_h = ak.to_numpy(data["ytarget_hit_tracker"][iev])
    x_calo_h = ak.to_numpy(data["X_hit_calo"][iev])
    yt_calo_h = ak.to_numpy(data["ytarget_hit_calo"][iev])

    mask_trk_h = x_trk_h[:, 0] != 0
    mask_calo_h = x_calo_h[:, 0] != 0

    pos_trk_h = x_trk_h[mask_trk_h, 6:9]
    pn_trk_h = yt_trk_h[mask_trk_h, -1]

    pos_calo_h = x_calo_h[mask_calo_h, 6:9]
    pn_calo_h = yt_calo_h[mask_calo_h, -1]

    # 2. Extract Tracks and Clusters
    x_trk = ak.to_numpy(data["X_track"][iev])
    yt_trk = ak.to_numpy(data["ytarget_track"][iev])
    x_cl = ak.to_numpy(data["X_cluster"][iev])
    yt_cl = ak.to_numpy(data["ytarget_cluster"][iev])

    mask_trk = x_trk[:, 0] != 0
    mask_cl = x_cl[:, 0] != 0

    # Track position approximation: use innermost radius, phi, and Z0
    # X_track features: ..., 3: sin_phi, 4: cos_phi, ..., 10: radiusOfInnermostHit, ..., 14: Z0
    r_trk = x_trk[mask_trk, 10]
    sin_phi_trk = x_trk[mask_trk, 3]
    cos_phi_trk = x_trk[mask_trk, 4]
    z0_trk = x_trk[mask_trk, 14]
    pos_trk = np.stack([r_trk * cos_phi_trk, r_trk * sin_phi_trk, z0_trk], axis=1)
    pn_trk = yt_trk[mask_trk, -1]

    # Cluster positions: 6, 7, 8 are X, Y, Z
    pos_cl = x_cl[mask_cl, 6:9]
    pn_cl = yt_cl[mask_cl, -1]

    print(f"Plotting {len(pos_trk_h)} hits and {len(pos_trk) + len(pos_cl)} high-level objects.")

    # 3. Plotting
    fig = plt.figure(figsize=(20, 10))
    cmap = plt.get_cmap("tab20")

    # Ensure consistent color mapping across both subplots
    all_pns = np.concatenate([pn_trk_h, pn_calo_h, pn_trk, pn_cl])
    vmin, vmax = np.min(all_pns), np.max(all_pns)

    # Subplot 1: Hits
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(pos_trk_h[:, 0], pos_trk_h[:, 1], pos_trk_h[:, 2], c=pn_trk_h, cmap=cmap, vmin=vmin, vmax=vmax, s=2, label="Tracker Hits", alpha=0.5)
    ax1.scatter(
        pos_calo_h[:, 0],
        pos_calo_h[:, 1],
        pos_calo_h[:, 2],
        c=pn_calo_h,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=15,
        marker="s",
        label="Calo Hits",
        alpha=0.7,
    )

    ax1.set_xlabel("X [mm]")
    ax1.set_ylabel("Y [mm]")
    ax1.set_zlabel("Z [mm]")
    ax1.set_title(f"Hits Colored by PN (Event {iev})")
    ax1.view_init(elev=20, azim=45)

    # Subplot 2: Tracks and Clusters (with colored low-alpha hits for context)
    ax2 = fig.add_subplot(122, projection="3d")

    # Context Hits (very low alpha, colored by PN)
    ax2.scatter(pos_trk_h[:, 0], pos_trk_h[:, 1], pos_trk_h[:, 2], c=pn_trk_h, cmap=cmap, vmin=vmin, vmax=vmax, s=1, alpha=0.03)
    ax2.scatter(pos_calo_h[:, 0], pos_calo_h[:, 1], pos_calo_h[:, 2], c=pn_calo_h, cmap=cmap, vmin=vmin, vmax=vmax, s=2, marker="s", alpha=0.03)

    ax2.scatter(
        pos_trk[:, 0],
        pos_trk[:, 1],
        pos_trk[:, 2],
        c=pn_trk,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=40,
        marker="o",
        edgecolors="black",
        label="Tracks",
        alpha=1.0,
    )
    ax2.scatter(
        pos_cl[:, 0],
        pos_cl[:, 1],
        pos_cl[:, 2],
        c=pn_cl,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=60,
        marker="D",
        edgecolors="black",
        label="Clusters",
        alpha=1.0,
    )

    ax2.set_xlabel("X [mm]")
    ax2.set_ylabel("Y [mm]")
    ax2.set_zlabel("Z [mm]")
    ax2.set_title(f"Tracks & Clusters Colored by PN (Event {iev})")
    ax2.view_init(elev=20, azim=45)

    # Sync axis limits
    for ax in [ax1, ax2]:
        ax.set_xlim(-2500, 2500)
        ax.set_ylim(-2500, 2500)
        ax.set_zset_xlim = ax.set_xlim  # Dummy to remind of consistency
        ax.set_zlim(-2500, 2500)

    plt.suptitle(f"PN Validation: {os.path.basename(file_path)}")

    # Save
    prefix = "cld" if "cld" in file_path.lower() else "clic" if "clic" in file_path.lower() else "det"
    out_name = f"pn_validation_side_{prefix}_ev{iev}.png"
    plt.savefig(out_name, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {out_name}")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_pn.py <file_path> [event_index]")
        sys.exit(1)

    fpath = sys.argv[1]
    event_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    visualize_event(fpath, event_idx)
