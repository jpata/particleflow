# Prediction-level diagnostics: cld_edm_ww_fullhad_hits

## set_hits

- events: 512, targets: 36385, predictions: 33806
- efficiency 0.598, purity 0.643, duplicates 0.192 of predictions, fakes 0.165
- efficiency track-associated 0.718 (17974 targets) vs no track 0.481 (18411 targets)
- mean count bias -5.04, mean energy response 0.952, mean MET abs error 7.66 GeV
- per-class efficiency: charged hadron 0.722 (n=17683), neutral hadron 0.328 (n=3160), photon 0.512 (n=15263), electron 0.428 (n=229), muon 0.760 (n=50)

## elementwise_hits

- events: 512, targets: 36385, predictions: 30973
- efficiency 0.544, purity 0.639, duplicates 0.166 of predictions, fakes 0.196
- efficiency track-associated 0.502 (17974 targets) vs no track 0.584 (18411 targets)
- mean count bias -10.57, mean energy response 0.842, mean MET abs error 12.91 GeV
- per-class efficiency: charged hadron 0.504 (n=17683), neutral hadron 0.358 (n=3160), photon 0.631 (n=15263), electron 0.297 (n=229), muon 0.660 (n=50)

Figures: efficiency_vs_pt.png, efficiency_vs_eta_and_track.png, resolution_vs_pt.png, pid_confusion.png, unaccepted_predictions.png, jet_response.png.
Tables: efficiency.csv, resolution.csv, pid_confusion.csv, unaccepted_predictions.csv, jet_response.csv.
