# Pandora PF performance across the CLD granularity scan

All 12 points have 100 reconstructed ROOT files and 10,000 events. The ECAL
scan uses Z→ττ at 91 GeV; the HCAL scan uses ttbar at 365 GeV. The same
generator seeds are used within each scan. These are results from the existing,
unchanged digitization and Pandora configuration.

The analysis reuses the photon matching and jet definitions from
`scripts/cld_ecal_readout/compare_ttbar_pf_photons.py` and
`compare_ttbar_pf_jets.py`. Each matched response has **no response cut**.
The 68% half-width is `(p84-p16)/2` of the matched response distribution.

## ECAL scan

Truth photons are propagated generator-status-1 PDG 22 particles with
`|cosθ|<0.98` and E≥1 GeV. Photon matching uses one-to-one ΔR<0.03 to
Pandora photon PFOs with E≥0.5 GeV. The unmatched PFO fraction uses PFOs
with E≥1 GeV; it is a candidate purity proxy, **not a physical fake rate**.

The π⁰ denominator contains generator π⁰ with exactly two direct photon
daughters, both selected truth photons with E≥1 GeV. A pair is called
resolved when each daughter has a distinct matched Pandora photon. This
measures two-photon reconstruction, but an angular match alone does not prove
that each PFO contains the correct shower energy. The opening angle is the
three-dimensional angle between the truth photons.

| ECAL pitch [mm] | Photon match | Resolved π⁰ pair | Unmatched photon PFO | Median E response | 68% half-width |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5.1 | 80.27% | 66.92% | 29.09% | 1.0028 | 0.0998 |
| 4 | 79.45% | 65.71% | 28.19% | 0.9983 | 0.1002 |
| 3 | 76.68% | 58.92% | 28.63% | 0.9912 | 0.1198 |
| 2.5 | 74.88% | 55.06% | 30.36% | 0.9833 | 0.1324 |
| 2 | 74.02% | 53.00% | 32.03% | 0.9670 | 0.1558 |
| 1.5 | 72.31% | 49.96% | 34.74% | 0.9429 | 0.2009 |

The truth denominators are stable across points: 18,810–18,825 photons and
7,216–7,221 π⁰ pairs. In the opening-angle plot, finer cells lose the most
resolved pairs below about 0.03 rad. For example, in the 0.01–0.02 rad bin,
the 5.1 mm point resolves 1,183/1,842 pairs and the 1.5 mm point resolves
650/1,840. This is an **unretuned Pandora result**, not evidence that fine
ECAL segmentation has intrinsically worse two-shower separation. The
response shift, wider response and lower matching suggest that calibration,
clustering or photon identification need retuning for finer cells.

- [ECAL photon summary](ecal_photon_performance.png)
- [Resolved π⁰ pairs vs opening angle](ecal_pi0_opening_angle.png)

## HCAL scan

Truth jets use propagated visible status-1 particles plus surrogate parents,
excluding neutrinos. Pandora jets are reclustered from nonzero-PDG PFOs with
the Key4hep postprocessor definition: FastJet ee-genkt, R=0.4, p=-1 and
inclusive pT>5 GeV. One-to-one matching requires ΔR<0.1. Response is
Pandora jet pT / truth jet pT.

| HCAL pitch [mm] | Jet match | Median pT response | 68% half-width | Mean PFOs/event |
| ---: | ---: | ---: | ---: | ---: |
| 30 | 91.71% | 0.9950 | 0.0669 | 95.47 |
| 25 | 91.90% | 0.9851 | 0.0649 | 95.31 |
| 20 | 91.94% | 0.9696 | 0.0705 | 95.29 |
| 15 | 91.77% | 0.9537 | 0.0828 | 94.95 |
| 12 | 91.50% | 0.9440 | 0.0930 | 94.92 |
| 10 | 91.42% | 0.9368 | 0.1012 | 94.50 |

The truth-jet count is stable at 59,333–59,339. Matching changes by less
than one percentage point, while the median response falls by about 5.8%
and the response half-width rises by about 51% from 30 to 10 mm. The largest
effect is therefore on energy assignment with the existing Pandora settings,
not on finding a nearby jet.

- [HCAL jet summary](hcal_jet_performance.png)

The fraction error bars in the summary figures are simple binomial counting
errors. They do not account for multiple photons or jets in one event, or
the paired seeds. No uncertainty bars are shown for response quantiles.
Per-point calibration and Pandora retuning are needed before comparing the
intrinsic detector designs. Useful follow-up diagnostics are neutral-hadron
response and charged-to-neutral confusion for the HCAL scan, and a
truth-associated shower-overlap or π⁰ mass measure for the ECAL scan.

## Reproduce

```bash
sbatch scripts/cld_ecal_readout/analyze_granularity_scan.sbatch
uv run --no-sync python scripts/cld_ecal_readout/analyze_granularity_scan.py --plot
```

Per-point machine-readable results are in `ecal_*.json` and `hcal_*.json`.
The full analysis array was Slurm job `61816283`.

Hit, track and cluster count checks across the same scans are in
[`../validation/README.md`](../validation/README.md).
