## Contents

  - `clusters_best_tuned_gnn_clic_v130`:
    - MLPF GNN model after hypertuning
    - inputs are reconstructed tracks and Pandora clusters
    - outputs are reconstructed PF candidates
    - trained on tt and qq v1.3.0 (1M events each)
  - `hits`:
    - MLPF GNN model
    - inputs are reconstructed tracks and calorimeter hits
    - outputs are reconstructed PF candidates
    - trained on tt, qq and gun samples (K0L, gamma, pi+-, pi0, neutron, ele, mu) v1.2.0
    - training is was restarted several times from previous checkpoints
  - `hypertuning`:
    - GNN and transformer model before and after hypertuning
  - `timing`
    - scaling study of baseline PF with number of gun particles on CPU
    - scaling study of GNN model with number of input elements on GPU
