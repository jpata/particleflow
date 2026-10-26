# CLD photon ECAL readout: 1,000 paired events

Both productions contain ten 100-event reconstructed files. The paired
generator-status-1 photon momentum is identical in all 1,000 event pairs.
The gun energy is 10 GeV. The baseline cell pitch is 5.1 mm; the fine pitch
is 2 mm. `summary.json` contains the full set of distribution summaries and
paired differences. `hit_cluster_distributions.png` plots the distributions.

| Quantity (per event unless stated) | 5.1 mm | 2 mm |
| --- | ---: | ---: |
| Mean simulated ECAL hits | 344.9 | 561.5 |
| Mean simulated ECAL deposited energy [GeV] | 0.248386 | 0.248386 |
| Mean digitized ECAL hits | 311.3 | 509.4 |
| Mean digitized ECAL energy [GeV] | 9.395 | 9.379 |
| Median digitized hit energy [GeV] | 0.01391 | 0.01121 |
| Mean Pandora clusters | 1.075 | 1.114 |
| Events with multiple Pandora clusters | 9.1% | 12.2% |
| Mean total Pandora cluster energy [GeV] | 9.510 | 9.303 |
| Median leading cluster energy [GeV] | 10.045 | 9.804 |
| Mean hits in leading cluster | 252.2 | 400.8 |
| Mean leading photon PFO energy [GeV] | 8.893 | 8.678 |

The 2 mm arm has 64% more digitized ECAL hits. Simulated deposited energy
agrees to floating-point precision, so the changed hit count is a readout
effect. Digitized energy differs by only 0.016 GeV on average, while the
Pandora cluster and photon PFO energies are lower by about 0.21 GeV. This
suggests that the unchanged digitization and Pandora calibration or clustering
settings need tuning for the finer readout; it is not evidence that the finer
readout intrinsically worsens photon energy resolution. Approximately 6% of
events have no Pandora cluster in either arm, so the raw means include those
zero-energy events.

`events/pair_event_{0,38,81}_cluster_zoom.png` shows paired, cluster-centered
transverse event displays for seed 810000, made with
`scripts/visualize_key4hep.py`. Event 0 has a lower leading cluster in the
2 mm arm (10.36 to 8.32 GeV); event 38 has a higher one (6.54 to 9.89 GeV).
These are illustrative event-level clustering changes, not population
performance estimates.
