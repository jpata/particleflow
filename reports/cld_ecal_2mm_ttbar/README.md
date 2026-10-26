# CLD ttbar Pandora jet comparison: 5.1 mm and 2 mm ECAL

Inputs are reconstructed ROOT files for seeds 300000–300009 from the existing
5.1 mm CLD production and the 2 mm ECAL readout production. Each seed has
100 events, giving 1,000 paired ttbar events. The analysis script is
`scripts/cld_ecal_readout/compare_ttbar_pf_jets.py`.

The jet definition follows `mlpf/data/key4hep/postprocessing.py`: FastJet
ee-genkt, R=0.4, p=-1, inclusive jet pT > 5 GeV. Truth jets use propagated,
visible generator-status-1 particles plus surrogate parents, excluding
neutrinos. Pandora jets are reclustered from nonzero-PDG `PandoraPFOs`, using
their momentum and mass as the postprocessor does. The stored
`RefinedVertexJets` are a different reconstruction product; the stored
`targetjet` in the parquet is built from truth-associated targets, so neither
is substituted for Pandora PF jets here. Truth and Pandora jets are matched
one-to-one by maximum cardinality and minimum ΔR within ΔR < 0.1, with no
response cut. Response is Pandora jet pT / truth jet pT. The resolution proxy
is half the 16th–84th percentile response interval. Bootstrap intervals
resample whole events (300 replicates, seed 12345).

| Metric | 5.1 mm | 2 mm |
| --- | ---: | ---: |
| All matched jets | 5,343 | 5,375 |
| Truth jet matching fraction | 91.33% | 91.85% |
| All matched response median | 0.9948 | 0.9898 |
| All matched response 68% half-width | 0.0651 | 0.0680 |
| Common matched truth jets | 5,161 | 5,161 |
| Common-jet response median | 0.9948 | 0.9902 |
| Common-jet response 68% half-width | 0.0636 | 0.0667 |

For the common jets, the median paired response difference (2 mm minus
5.1 mm) is -0.00367, with an event-bootstrap 95% interval of
[-0.00622, -0.00159]. The resolution half-width difference is +0.00317,
with an event-bootstrap 95% interval of [+0.00017, +0.00614]. The fine
readout is therefore modestly lower in response and wider in this
**unretuned Pandora reconstruction**. The photon-gun pilot also showed a
response shift with the existing calibration, so this should not be read as
an intrinsic detector-resolution ranking.

The paired generator-status-1 PDG sequences agree in all 1,000 events. Their
stored momenta are not bitwise identical in 510 events, but the truth-jet
comparison is close: 5,834 of 5,850 baseline truth jets match fine-readout
truth jets within ΔR < 0.01. The median and 16th–84th percentile truth-jet
pT differences are zero. Common-jet response uses the *baseline truth pT*
as the denominator in both arms, so small truth-record differences do not
drive that paired comparison.

Artifacts:

- `pandora_jet_resolution.png`: response distribution, median and 68% spread
  against truth pT and |η|, and matching efficiency.
- `pandora_jet_resolution_width.png`: direct resolution-width profiles against
  truth pT and |η|. Bins with fewer than 30 matched jets are omitted.
- `summary.json`: exact metrics, bin summaries, definitions and pairing checks.
- `events/ttbar_event_{0,1,2}_side_by_side.png`: paired transverse event
  displays for seed 300000, with 5.1 mm on the left and 2 mm on the right.
  These use the same ±2700 mm plotting extent and show up to 2500 hits per
  collection. The cap makes the displays unsuitable for measuring occupancy;
  use the ROOT hit collections for that comparison.

Reproduce with:

```bash
uv run python scripts/cld_ecal_readout/compare_ttbar_pf_jets.py \
  --baseline /local/joosep/mlpf/cld/v1.2.5_key4hep_2025-05-29 \
  --fine /local/joosep/mlpf/cld/ecal_2mm_key4hep_2025-05-29 \
  --output-dir reports/cld_ecal_2mm_ttbar
```

The event displays were made by running `scripts/visualize_key4hep.py` on
each arm separately with `--detector cld --events 0 1 2 --plot-limit 2700
--max-hits 2500` and separate output directories under `events/`. Keeping
the directories separate avoids overwriting images with the same detector,
seed, and event filenames before placing them side by side.

## Pandora photon matching

`pandora_photons.png` compares 1,000 paired ttbar events. Both arms use the
same baseline generator photons: PDG 22, status 1, propagated into the
simulation, and |cos(theta)| < 0.98. Pandora photon PFOs are matched
one-to-one within ΔR < 0.03, without an energy-response cut. Truth photons
are included in the matching from 0.1 GeV and PFOs from 0.5 GeV; plotted
efficiency uses truth E >= 1 GeV and the unmatched-candidate fraction uses
Pandora E >= 1 GeV.

| Metric | 5.1 mm | 2 mm |
| --- | ---: | ---: |
| Matched fraction of 18,362 truth photons, E >= 1 GeV | 78.18% | 75.94% |
| Unmatched fraction of photon PFOs, E >= 1 GeV | 23.37% | 24.55% |
| Matched energy-response median | 1.016 | 0.976 |
| Matched response 68% half-width | 0.127 | 0.152 |

The unmatched PFO fraction is **not** a physical fake-photon rate: a photon
PFO may be a split fragment or originate from a secondary particle absent
from the selected generator-photon collection. In dense ttbar jets, angular
matches can also pair unrelated nearby particles. The figure shows a
reconstruction-level comparison with the current Pandora calibration, not
an intrinsic ECAL resolution measurement. The 2 mm response shift persists
for photons with no other selected truth photon within ΔR = 0.1, although
that selection does not require isolation from charged particles or hadrons.

`photon_summary.json` contains the exact counts and definitions. Reproduce
the figure with:

```bash
uv run python scripts/cld_ecal_readout/compare_ttbar_pf_photons.py \
  --baseline /local/joosep/mlpf/cld/v1.2.5_key4hep_2025-05-29 \
  --fine /local/joosep/mlpf/cld/ecal_2mm_key4hep_2025-05-29 \
  --output-dir reports/cld_ecal_2mm_ttbar
```
