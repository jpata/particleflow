# Prediction-level diagnostics: cld_edm_qq_hits

## set_hits

- events: 512, targets: 23010, predictions: 21216
- efficiency 0.610, purity 0.662, duplicates 0.176 of predictions, fakes 0.162
- efficiency track-associated 0.740 (11331 targets) vs no track 0.484 (11679 targets)
- mean count bias -3.50, mean energy response 0.905, mean MET abs error 6.79 GeV
- per-class efficiency: charged hadron 0.743 (n=11078), neutral hadron 0.293 (n=2130), photon 0.526 (n=9561), electron 0.553 (n=152), muon 0.831 (n=89)

## elementwise_hits

- events: 512, targets: 23010, predictions: 18960
- efficiency 0.526, purity 0.639, duplicates 0.151 of predictions, fakes 0.211
- efficiency track-associated 0.481 (11331 targets) vs no track 0.570 (11679 targets)
- mean count bias -7.91, mean energy response 0.828, mean MET abs error 12.72 GeV
- per-class efficiency: charged hadron 0.481 (n=11078), neutral hadron 0.329 (n=2130), photon 0.623 (n=9561), electron 0.421 (n=152), muon 0.562 (n=89)

Figures: efficiency_vs_pt.png, efficiency_vs_eta_and_track.png, resolution_vs_pt.png, pid_confusion.png, unaccepted_predictions.png, jet_response.png.
Tables: efficiency.csv, resolution.csv, pid_confusion.csv, unaccepted_predictions.csv, jet_response.csv.
