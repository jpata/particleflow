# Set-output inference calibration

Checkpoint: `/home/joosep/particleflow/experiments/cld_hits_output_comparison/set_seed12345_20260906_012757_938629/checkpoints/checkpoint-20000.pth`

Dataset: `cld_edm_ttbar_hits/10:3.2.1` test split; first 400 events for calibration and the remaining 400 for holdout.

| Selection | Policy | Split | Count bias | Efficiency | Purity | F1 | Duplicate | Energy response | Jet median | Jet IQR | Jet match |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | `threshold_0.500` | calibration | -3.32 | 0.546 | 0.565 | 0.555 | 0.242 | 1.022 | 1.020 | 0.238 | 0.907 |
| baseline | `threshold_0.500` | holdout | -2.88 | 0.553 | 0.570 | 0.561 | 0.235 | 1.019 | 1.029 | 0.236 | 0.905 |
| best_particle_f1 | `threshold_0.725` | calibration | -26.33 | 0.563 | 0.775 | 0.653 | 0.087 | 0.840 | 0.860 | 0.222 | 0.891 |
| best_particle_f1 | `threshold_0.725` | holdout | -26.46 | 0.568 | 0.788 | 0.660 | 0.077 | 0.830 | 0.856 | 0.226 | 0.882 |
| best_plain_threshold | `threshold_0.725` | calibration | -26.33 | 0.563 | 0.775 | 0.653 | 0.087 | 0.840 | 0.860 | 0.222 | 0.891 |
| best_plain_threshold | `threshold_0.725` | holdout | -26.46 | 0.568 | 0.788 | 0.660 | 0.077 | 0.830 | 0.856 | 0.226 | 0.882 |
| best_f1_duplicate_le_0p15 | `threshold_0.725` | calibration | -26.33 | 0.563 | 0.775 | 0.653 | 0.087 | 0.840 | 0.860 | 0.222 | 0.891 |
| best_f1_duplicate_le_0p15 | `threshold_0.725` | holdout | -26.46 | 0.568 | 0.788 | 0.660 | 0.077 | 0.830 | 0.856 | 0.226 | 0.882 |
| topk_soft_count | `topk_soft_count` | calibration | -0.24 | 0.541 | 0.543 | 0.542 | 0.257 | 1.055 | 1.045 | 0.239 | 0.904 |
| topk_soft_count | `topk_soft_count` | holdout | +0.08 | 0.541 | 0.540 | 0.541 | 0.257 | 1.045 | 1.051 | 0.242 | 0.909 |
| topk_soft_count_nms_dr0.020 | `topk_soft_count_nms_dr0.020` | calibration | -0.24 | 0.542 | 0.543 | 0.542 | 0.241 | 1.014 | 1.013 | 0.240 | 0.906 |
| topk_soft_count_nms_dr0.020 | `topk_soft_count_nms_dr0.020` | holdout | +0.08 | 0.540 | 0.540 | 0.540 | 0.242 | 1.008 | 1.018 | 0.239 | 0.909 |
| topk_soft_count_nms_dr0.050 | `topk_soft_count_nms_dr0.050` | calibration | -0.24 | 0.501 | 0.502 | 0.502 | 0.231 | 0.930 | 0.931 | 0.284 | 0.904 |
| topk_soft_count_nms_dr0.050 | `topk_soft_count_nms_dr0.050` | holdout | +0.08 | 0.498 | 0.498 | 0.498 | 0.233 | 0.924 | 0.941 | 0.293 | 0.904 |
| topk_soft_count_nms_dr0.100 | `topk_soft_count_nms_dr0.100` | calibration | -0.24 | 0.394 | 0.395 | 0.394 | 0.238 | 0.827 | 0.817 | 0.373 | 0.880 |
| topk_soft_count_nms_dr0.100 | `topk_soft_count_nms_dr0.100` | holdout | +0.08 | 0.391 | 0.391 | 0.391 | 0.241 | 0.821 | 0.824 | 0.385 | 0.884 |
| best_threshold_nms_dr0.02 | `threshold_0.725_nms_dr0.020` | calibration | -28.03 | 0.557 | 0.786 | 0.652 | 0.074 | 0.807 | 0.829 | 0.227 | 0.891 |
| best_threshold_nms_dr0.02 | `threshold_0.725_nms_dr0.020` | holdout | -28.10 | 0.561 | 0.798 | 0.659 | 0.066 | 0.798 | 0.826 | 0.240 | 0.883 |
| best_threshold_nms_dr0.05 | `threshold_0.650_nms_dr0.050` | calibration | -26.20 | 0.556 | 0.764 | 0.644 | 0.068 | 0.772 | 0.793 | 0.257 | 0.894 |
| best_threshold_nms_dr0.05 | `threshold_0.650_nms_dr0.050` | holdout | -25.80 | 0.557 | 0.765 | 0.645 | 0.067 | 0.766 | 0.795 | 0.266 | 0.888 |
| best_threshold_nms_dr0.10 | `threshold_0.625_nms_dr0.100` | calibration | -32.88 | 0.509 | 0.773 | 0.614 | 0.040 | 0.673 | 0.684 | 0.315 | 0.877 |
| best_threshold_nms_dr0.10 | `threshold_0.625_nms_dr0.100` | holdout | -32.26 | 0.510 | 0.772 | 0.614 | 0.041 | 0.664 | 0.683 | 0.333 | 0.876 |
| best_plain_jet_score | `threshold_0.650` | calibration | -18.20 | 0.579 | 0.714 | 0.639 | 0.130 | 0.905 | 0.919 | 0.219 | 0.898 |
| best_plain_jet_score | `threshold_0.650` | holdout | -17.79 | 0.586 | 0.721 | 0.646 | 0.123 | 0.898 | 0.922 | 0.221 | 0.898 |
| best_f1_within_2pct_closure | `threshold_0.550` | calibration | -8.23 | 0.558 | 0.610 | 0.583 | 0.209 | 0.984 | 0.984 | 0.231 | 0.903 |
| best_f1_within_2pct_closure | `threshold_0.550` | holdout | -7.66 | 0.564 | 0.613 | 0.588 | 0.204 | 0.979 | 0.994 | 0.225 | 0.905 |
