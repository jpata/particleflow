import awkward
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# Feature indices from mlpf/data/key4hep/postprocessing.py
# track_feature_order = ["elemtype", "pt", "eta", "sin_phi", "cos_phi", ...]
# cluster_feature_order = ["elemtype", "et", "eta", "sin_phi", "cos_phi", ...]
# hit_feature_order = ["elemtype", "et", "eta", "sin_phi", "cos_phi", "energy", "position.x", "position.y", "position.z", "time", "subdetector", "type"]

TRACK_ETA_IDX = 2
TRACK_SIN_PHI_IDX = 3
TRACK_COS_PHI_IDX = 4

CLUSTER_ETA_IDX = 2
CLUSTER_SIN_PHI_IDX = 3
CLUSTER_COS_PHI_IDX = 4

HIT_ETA_IDX = 2
HIT_SIN_PHI_IDX = 3
HIT_COS_PHI_IDX = 4
HIT_SUBDETECTOR_IDX = 10
HIT_TYPE_IDX = 11

# Particle feature indices
PARTICLE_PDG_IDX = 0
PARTICLE_PT_IDX = 2
PARTICLE_ETA_IDX = 3
PARTICLE_SIN_PHI_IDX = 4
PARTICLE_COS_PHI_IDX = 5

def plot_distributions(event_data, output_dir):
    print("Collecting distributions across all events...")
    
    pt_track_cluster = []
    eta_track_cluster = []
    pt_hit = []
    eta_hit = []
    
    genmet = []
    target_met = []
    
    # Lists for correlation plots
    corr_data = {
        "track": {"eta": ([], []), "phi": ([], [])},
        "cluster": {"eta": ([], []), "phi": ([], [])},
        "hit": {"eta": ([], []), "phi": ([], [])}
    }
    
    # In some awkward versions or depending on how it's saved, from_parquet may return a Record
    if isinstance(event_data, awkward.Record):
        fields = event_data.fields
        num_events = len(event_data[fields[0]])
    else:
        num_events = len(event_data)

    for i in range(num_events):
        # Calculate target MET
        t_px, t_py = 0, 0
        
        # Collections to process: (key_X, key_Y, type_name)
        collections = [
            ("X_track", "ytarget_track", "track"),
            ("X_cluster", "ytarget_cluster", "cluster"),
            ("X_hit", "ytarget_hit", "hit")
        ]
        
        for k_x, k_y, name in collections:
            x = event_data[k_x][i]
            y = event_data[k_y][i]
            
            if len(y) > 0:
                mask = y[:, PARTICLE_PDG_IDX] != 0
                
                # For target MET (from track + cluster only)
                if name in ["track", "cluster"]:
                    t_px += np.sum(y[mask, PARTICLE_PT_IDX] * y[mask, PARTICLE_COS_PHI_IDX])
                    t_py += np.sum(y[mask, PARTICLE_PT_IDX] * y[mask, PARTICLE_SIN_PHI_IDX])

                # For 1D plots
                if name in ["track", "cluster"]:
                    pt_track_cluster.extend(y[mask, PARTICLE_PT_IDX])
                    eta_track_cluster.extend(y[mask, PARTICLE_ETA_IDX])
                else:
                    pt_hit.extend(y[mask, PARTICLE_PT_IDX])
                    eta_hit.extend(y[mask, PARTICLE_ETA_IDX])
                
                # For 2D correlation plots
                # X indices are same for all: ETA=2, SIN_PHI=3, COS_PHI=4
                # Y indices: ETA=3, SIN_PHI=4, COS_PHI=5
                x_eta = x[mask, 2]
                y_eta = y[mask, 3]
                x_phi = np.arctan2(x[mask, 3], x[mask, 4])
                y_phi = np.arctan2(y[mask, 4], y[mask, 5])
                
                corr_data[name]["eta"][0].extend(x_eta)
                corr_data[name]["eta"][1].extend(y_eta)
                corr_data[name]["phi"][0].extend(x_phi)
                corr_data[name]["phi"][1].extend(y_phi)

        genmet.append(event_data["genmet"][i])
        target_met.append(np.sqrt(t_px**2 + t_py**2))

    # 1D Plots
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(pt_track_cluster, bins=50, histtype='step', label='Track+Cluster targets', density=True)
    plt.hist(pt_hit, bins=50, histtype='step', label='Hit targets', density=True)
    plt.xlabel('pT [GeV]')
    plt.ylabel('Normalized counts')
    plt.yscale('log')
    plt.legend()
    plt.title('pT Distribution')

    plt.subplot(1, 2, 2)
    plt.hist(eta_track_cluster, bins=50, histtype='step', label='Track+Cluster targets', density=True)
    plt.hist(eta_hit, bins=50, histtype='step', label='Hit targets', density=True)
    plt.xlabel('Eta')
    plt.ylabel('Normalized counts')
    plt.legend()
    plt.title('Eta Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distributions_1d.png'))
    plt.close()

    # 2D Correlation Plots (Eta, Phi)
    for var in ["eta", "phi"]:
        plt.figure(figsize=(18, 5))
        for i, name in enumerate(["track", "cluster", "hit"], 1):
            plt.subplot(1, 3, i)
            x_vals, y_vals = corr_data[name][var]
            if len(x_vals) > 0:
                plt.hist2d(x_vals, y_vals, bins=50, cmap='viridis', cmin=1)
                plt.colorbar(label='Counts')
                
                # Identity line
                mi = min(np.min(x_vals), np.min(y_vals))
                ma = max(np.max(x_vals), np.max(y_vals))
                plt.plot([mi, ma], [mi, ma], color='red', linestyle='--', alpha=0.5)
                
            plt.xlabel(f'X {name} {var}')
            plt.ylabel(f'ytarget {name} {var}')
            plt.title(f'{name.capitalize()} {var} Correlation')
            plt.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'correlation_2d_{var}.png'))
        print(f"Saved 2D {var} correlation plot to {output_dir}/correlation_2d_{var}.png")
        plt.close()

    # MET Correlation Plot
    plt.figure(figsize=(8, 7))
    if len(genmet) > 0:
        plt.hist2d(genmet, target_met, bins=50, cmap='viridis', cmin=1)
        plt.colorbar(label='Counts')
        mi = min(np.min(genmet), np.min(target_met))
        ma = max(np.max(genmet), np.max(target_met))
        plt.plot([mi, ma], [mi, ma], color='red', linestyle='--', alpha=0.5)
    plt.xlabel('genmet [GeV]')
    plt.ylabel('target met [GeV]')
    plt.title('MET Correlation')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_2d_met.png'))
    print(f"Saved 2D MET correlation plot to {output_dir}/correlation_2d_met.png")
    plt.close()

def plot_event(event_data, event_idx, output_dir):
    plt.figure(figsize=(10, 8))
    
    # Extract Tracks
    x_track = event_data["X_track"][event_idx]
    if len(x_track) > 0:
        eta = x_track[:, TRACK_ETA_IDX]
        phi = np.arctan2(x_track[:, TRACK_SIN_PHI_IDX], x_track[:, TRACK_COS_PHI_IDX])
        plt.scatter(eta, phi, marker='x', color='black', label='Tracks', alpha=0.5, s=20)
        
    # Extract Clusters
    x_cluster = event_data["X_cluster"][event_idx]
    if len(x_cluster) > 0:
        eta = x_cluster[:, CLUSTER_ETA_IDX]
        phi = np.arctan2(x_cluster[:, CLUSTER_SIN_PHI_IDX], x_cluster[:, CLUSTER_COS_PHI_IDX])
        plt.scatter(eta, phi, marker='o', facecolors='none', edgecolors='blue', label='Clusters', alpha=0.5, s=50)

    # Extract Hits
    x_hit = event_data["X_hit"][event_idx]
    if len(x_hit) > 0:
        subdet = x_hit[:, HIT_SUBDETECTOR_IDX]
        eta = x_hit[:, HIT_ETA_IDX]
        phi = np.arctan2(x_hit[:, HIT_SIN_PHI_IDX], x_hit[:, HIT_COS_PHI_IDX])
        
        # Plot hits by subdetector
        # 0: ECAL, 1: HCAL, 3: Tracker, 2: Others
        subdet_names = {0: "ECAL", 1: "HCAL", 3: "Tracker", 2: "Other"}
        colors = {0: "green", 1: "red", 3: "cyan", 2: "gray"}
        markers = {0: 's', 1: '^', 3: '.', 2: 'v'}
        
        for sd in np.unique(subdet):
            mask = subdet == sd
            name = subdet_names.get(sd, f"Subdet {sd}")
            plt.scatter(eta[mask], phi[mask], marker=markers.get(sd, '.'), color=colors.get(sd, 'gray'), 
                        label=f'Hits ({name})', alpha=0.3, s=10)

    plt.xlabel('Eta')
    plt.ylabel('Phi')
    plt.title(f'Event {event_idx}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    out_file = os.path.join(output_dir, f'event_{event_idx}.png')
    plt.savefig(out_file)
    print(f"Saved plot to {out_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot events from MLPF post-processed parquet file.")
    parser.add_argument("--input", type=str, nargs='+', required=True, help="Input parquet file(s)")
    parser.add_argument("--num-events", type=int, default=3, help="Number of events to plot")
    parser.add_argument("--output-dir", type=str, default="plots", help="Output directory for plots")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    print(f"Loading {args.input}")
    
    data_list = []
    for f in args.input:
        data_list.append(awkward.from_parquet(f))
        
    if len(data_list) == 1:
        data = data_list[0]
    else:
        # Check if they are records
        if all(isinstance(d, awkward.Record) for d in data_list):
            # Concatenate fields
            fields = data_list[0].fields
            concatenated_data = {}
            for field in fields:
                concatenated_data[field] = awkward.concatenate([d[field] for d in data_list])
            data = awkward.Array(concatenated_data)
        else:
            data = awkward.concatenate(data_list)
    
    # Plot distributions across all events
    plot_distributions(data, args.output_dir)

    # In some awkward versions or depending on how it's saved, from_parquet may return a Record
    if isinstance(data, awkward.Record):
        fields = data.fields
        num_available = len(data[fields[0]])
    else:
        num_available = len(data)
    
    num_to_plot = min(args.num_events, num_available)
    
    print(f"Ploting {num_to_plot} individual events...")
    for i in range(num_to_plot):
        plot_event(data, i, args.output_dir)

if __name__ == "__main__":
    main()
