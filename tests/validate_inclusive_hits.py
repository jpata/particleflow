import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import argparse


def fit_circle(x, y, max_iters=50, threshold=10.0):
    """
    Fits a circle to points (x, y) constrained to pass through the origin using RANSAC.
    Returns the radius R.
    """
    best_inliers_count = 0
    best_params = None

    if len(x) < 3:
        return None

    # We need 2 points to define a circle passing through the origin
    for _ in range(max_iters):
        idx = np.random.choice(len(x), 2, replace=False)
        xs, ys = x[idx], y[idx]

        z = xs**2 + ys**2
        A = np.stack([2 * xs, 2 * ys], axis=1)
        try:
            params = np.linalg.solve(A, z)
            xc, yc = params
            R = np.sqrt(xc**2 + yc**2)

            # Distance from point to circle passing through origin
            distances = np.abs(np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - R)
            inliers = distances < threshold
            inliers_count = np.sum(inliers)

            if inliers_count > best_inliers_count:
                best_inliers_count = inliers_count

                # Re-fit with all inliers using least squares
                A_in = np.stack([2 * x[inliers], 2 * y[inliers]], axis=1)
                z_in = x[inliers] ** 2 + y[inliers] ** 2
                best_params, _, _, _ = np.linalg.lstsq(A_in, z_in, rcond=None)
        except Exception:
            continue

    if best_params is not None:
        xc, yc = best_params
        R = np.sqrt(xc**2 + yc**2)
        return R
    return None


def validate_inclusive(file_path, b_field):
    print(f"Loading {file_path}...")
    data = ak.from_parquet(file_path)

    # Track results
    track_results = {"pT_truth": [], "pT_fit": []}
    # Calo results
    calo_results = {"E_truth": [], "E_hit_sum": []}

    B_FIELD = b_field  # Tesla
    print(f"Using B-Field: {B_FIELD} T")

    num_events = len(data["X_hit_tracker"])
    print(f"Processing {num_events} events...")

    for iev in tqdm(range(num_events)):
        # 1. Collect gen-particle truth by PN
        # We look at both ytarget_track and ytarget_cluster
        yt_trk = ak.to_numpy(data["ytarget_track"][iev])
        yt_cl = ak.to_numpy(data["ytarget_cluster"][iev])

        # PN is at -1, pt at 2, energy at 6
        pn_to_truth = {}

        # Helper to extract truth from a branch
        def extract_truth(yt):
            for row in yt:
                pn = int(row[-1])
                if pn > 0 and row[0] > 0:  # If it's a representative (PID > 0)
                    pn_to_truth[pn] = {"pt": row[2], "energy": row[6]}

        extract_truth(yt_trk)
        extract_truth(yt_cl)

        # 2. Process Tracker Hits
        x_trk_h = ak.to_numpy(data["X_hit_tracker"][iev])
        yt_trk_h = ak.to_numpy(data["ytarget_hit_tracker"][iev])
        mask_trk_h = x_trk_h[:, 0] != 0
        if np.any(mask_trk_h):
            pos_trk_h = x_trk_h[mask_trk_h, 6:8]  # X, Y
            pns_trk_h = yt_trk_h[mask_trk_h, -1]

            for pn in np.unique(pns_trk_h):
                if pn == 0 or pn not in pn_to_truth:
                    continue
                hits = pos_trk_h[pns_trk_h == pn]
                if len(hits) >= 5:
                    R = fit_circle(hits[:, 0], hits[:, 1])
                    if R and R < 1e5:
                        pT_fit = 0.0003 * B_FIELD * R
                        track_results["pT_truth"].append(pn_to_truth[pn]["pt"])
                        track_results["pT_fit"].append(pT_fit)

                        if pT_fit / pn_to_truth[pn]["pt"] < 0.5 and len(track_results.get("bad_tracks", [])) < 5:
                            if "bad_tracks" not in track_results:
                                track_results["bad_tracks"] = []
                            track_results["bad_tracks"].append(1)
                            print("\n--- Bad Fit Found ---")
                            print(f"Event: {iev}, PN: {pn}")
                            print(f"pT_truth: {pn_to_truth[pn]['pt']:.3f} GeV, pT_fit: {pT_fit:.3f} GeV")
                            print(f"R: {R:.3f} mm")
                            print("Hit Coordinates (X, Y):")
                            for h in hits:
                                print(f"  {h[0]:.3f}, {h[1]:.3f}")

        # 3. Process Calo Hits
        x_calo_h = ak.to_numpy(data["X_hit_calo"][iev])
        yt_calo_h = ak.to_numpy(data["ytarget_hit_calo"][iev])
        mask_calo_h = x_calo_h[:, 0] != 0
        if np.any(mask_calo_h):
            e_calo_h = x_calo_h[mask_calo_h, 5]
            pns_calo_h = yt_calo_h[mask_calo_h, -1]

            for pn in np.unique(pns_calo_h):
                if pn == 0 or pn not in pn_to_truth:
                    continue
                e_sum = np.sum(e_calo_h[pns_calo_h == pn])
                calo_results["E_truth"].append(pn_to_truth[pn]["energy"])
                calo_results["E_hit_sum"].append(e_sum)

    # Convert to numpy
    for k in track_results:
        track_results[k] = np.array(track_results[k])
    for k in calo_results:
        calo_results[k] = np.array(calo_results[k])

    # 4. Plotting
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 15))

    # 4.1 Track Scatter
    ax1.scatter(track_results["pT_truth"], track_results["pT_fit"], alpha=0.3, s=10)
    max_pt = max(np.max(track_results["pT_truth"]), 100)
    ax1.plot([0, max_pt], [0, max_pt], color="red", linestyle="--")
    ax1.set_xlabel("Gen Particle pT [GeV]")
    ax1.set_ylabel("Fitted pT from Hits [GeV]")
    ax1.set_title("Tracker Hits pT vs Gen pT")
    ax1.set_xlim(0, max_pt)
    ax1.set_ylim(0, max_pt)

    # 4.2 Track Ratio
    ratio_pt = track_results["pT_fit"] / track_results["pT_truth"]
    ax2.hist(ratio_pt, bins=100, range=(0, 2), alpha=0.7, color="darkorange", edgecolor="black")
    ax2.axvline(1.0, color="red", linestyle="--")
    ax2.set_xlabel("pT_fit / pT_truth")
    ax2.set_title(f"Momentum Ratio (Mean: {np.mean(ratio_pt):.3f})")

    # 4.3 Calo Scatter
    ax3.scatter(calo_results["E_truth"], calo_results["E_hit_sum"], alpha=0.3, s=10)
    max_e = max(np.max(calo_results["E_truth"]), 100)
    ax3.plot([0, max_e], [0, max_e], color="red", linestyle="--")
    ax3.set_xlabel("Gen Particle Energy [GeV]")
    ax3.set_ylabel("Sum of Hit Energies [GeV]")
    ax3.set_title("Calo Hits Energy vs Gen Energy")
    ax3.set_xlim(0, max_e)
    ax3.set_ylim(0, max_e)

    # 4.4 Calo Ratio
    ratio_e = calo_results["E_hit_sum"] / calo_results["E_truth"]
    ax4.hist(ratio_e, bins=100, range=(0, 2), alpha=0.7, color="steelblue", edgecolor="black")
    ax4.axvline(1.0, color="red", linestyle="--")
    ax4.set_xlabel("Sum(E_hits) / E_gen")
    ax4.set_title(f"Energy Ratio (Mean: {np.mean(ratio_e):.3f})")

    print(f"Momentum Ratio Mean: {np.mean(ratio_pt):.3f}")
    print(f"Energy Ratio Mean: {np.mean(ratio_e):.3f}")

    plt.suptitle(f"Unified Truth Consistency Validation\nFile: {os.path.basename(file_path)}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_file = "unified_validation.png"
    plt.savefig(out_file, dpi=150)
    print(f"Saved unified validation plot to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate inclusive hit labeling")
    parser.add_argument("file_path", help="Path to the parquet file to validate")
    parser.add_argument("--bfield", type=float, default=4.0, help="Magnetic field in Tesla (4.0 for CLIC, 2.0 fo CLD)")
    args = parser.parse_args()

    validate_inclusive(args.file_path, args.bfield)
