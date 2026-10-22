# Prediction-level diagnostics: cld_edm_ttbar_hits

## set_hits

- events: 512, targets: 49013, predictions: 46867
- efficiency 0.608, purity 0.635, duplicates 0.202 of predictions, fakes 0.163
- efficiency track-associated 0.715 (24667 targets) vs no track 0.498 (24346 targets)
- mean count bias -4.19, mean energy response 0.986, mean MET abs error 7.90 GeV
- per-class efficiency: charged hadron 0.715 (n=23651), neutral hadron 0.362 (n=3537), photon 0.521 (n=20823), electron 0.645 (n=619), muon 0.859 (n=383)

## elementwise_hits

- events: 512, targets: 49013, predictions: 42595
- efficiency 0.584, purity 0.672, duplicates 0.161 of predictions, fakes 0.167
- efficiency track-associated 0.539 (24667 targets) vs no track 0.630 (24346 targets)
- mean count bias -12.54, mean energy response 0.896, mean MET abs error 11.60 GeV
- per-class efficiency: charged hadron 0.536 (n=23651), neutral hadron 0.430 (n=3537), photon 0.663 (n=20823), electron 0.586 (n=619), muon 0.679 (n=383)

Figures: efficiency_vs_pt.png, efficiency_vs_eta_and_track.png, resolution_vs_pt.png, pid_confusion.png, unaccepted_predictions.png, jet_response.png.
Tables: efficiency.csv, resolution.csv, pid_confusion.csv, unaccepted_predictions.csv, jet_response.csv.
