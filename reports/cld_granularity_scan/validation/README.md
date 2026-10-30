# CLD granularity scan validation

The validation reads collection lengths from all 1,200 reconstructed EDM4hep
ROOT files: 10,000 events at each of 12 pitches. Simulated hit counts use
`ECalBarrelCollection + ECalEndcapCollection` and
`HCalBarrelCollection + HCalEndcapCollection + HCalRingCollection`.
Reconstructed hit counts use `ECALBarrel + ECALEndcap` and
`HCALBarrel + HCALEndcap + HCALOther`. Track, cluster and PFO counts use
`SiTracks_Refitted`, `PandoraClusters` and `PandoraPFOs` respectively.
Photon PFOs have PDG 22. These are per-event collection counts, not
truth-matched object efficiencies. Error bars on mean-count plots are
standard errors over events.

## ECAL scan: Z→ττ

| ECAL pitch [mm] | Sim ECAL hits | Reco ECAL hits | Sim HCAL hits | Tracks | Pandora clusters | PFOs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5.1 | 1,271 | 1,020 | 1,274 | 3.12 | 6.22 | 6.25 |
| 4 | 1,410 | 1,141 | 1,275 | 3.11 | 6.22 | 6.25 |
| 3 | 1,575 | 1,290 | 1,270 | 3.10 | 6.25 | 6.27 |
| 2.5 | 1,706 | 1,404 | 1,277 | 3.12 | 6.36 | 6.38 |
| 2 | 1,865 | 1,542 | 1,276 | 3.11 | 6.44 | 6.45 |
| 1.5 | 2,108 | 1,751 | 1,272 | 3.12 | 6.56 | 6.58 |

From 5.1 to 1.5 mm, the mean simulated ECAL hit count rises by **65.8%**
and the reconstructed ECAL hit count by **71.7%**. HCAL hits and refitted
tracks remain stable. Pandora clusters and PFOs rise by about 5%.

- [Mean counts versus ECAL pitch](ecal_count_trends.png)
- [Baseline versus finest event-count distributions](ecal_count_distributions.png)

## HCAL scan: ttbar

| HCAL pitch [mm] | Sim HCAL hits | Reco HCAL hits | Sim ECAL hits | Tracks | Pandora clusters | PFOs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 10,441 | 1,189 | 10,047 | 53.12 | 94.67 | 95.47 |
| 25 | 10,978 | 1,233 | 10,050 | 53.04 | 94.51 | 95.31 |
| 20 | 11,595 | 1,287 | 10,049 | 53.08 | 94.51 | 95.29 |
| 15 | 12,335 | 1,359 | 10,050 | 53.01 | 94.16 | 94.95 |
| 12 | 12,902 | 1,419 | 10,044 | 53.07 | 94.13 | 94.92 |
| 10 | 13,357 | 1,468 | 10,045 | 53.03 | 93.72 | 94.50 |

From 30 to 10 mm, the mean simulated HCAL hit count rises by **27.9%**
and the reconstructed HCAL hit count by **23.5%**. ECAL hits and tracks remain
stable. Pandora clusters and PFOs fall by about 1% despite the greater HCAL
hit count; this is a reconstruction effect to investigate alongside the jet
response shift.

- [Mean counts versus HCAL pitch](hcal_count_trends.png)
- [Baseline versus finest event-count distributions](hcal_count_distributions.png)

Both changed-calorimeter hit counts increase monotonically as cell pitch
shrinks, while the other calorimeter and tracker provide stable controls.
The increase need not follow inverse cell area: these counts depend on
shower occupancy, thresholds, merging and digitization. Event displays are
in `../events/`, and Pandora PF performance plots are in `../performance/`.

Per-point JSON contains means, standard errors, medians, p10/p90 and maximum;
the NPZ files contain all per-event counts.

Reproduce with:

```bash
sbatch scripts/cld_ecal_readout/validate_granularity_scan.sbatch
uv run --no-sync python scripts/cld_ecal_readout/validate_granularity_scan.py --plot
```

The full validation analysis used local baseline processing and Slurm array
`61816489` for the other 11 points.
