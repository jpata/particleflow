# Paired CLD photon ECAL readout pilot

`particleflow_spec.yaml` defines two generation-only productions,
`cld_photon_5p1` and `cld_photon_2mm`. They use the same 10 GeV photon gun,
seeds 810000–810009, and 100 events per seed. The baseline uses the pinned
Key4hep `CLD_o2_v07` geometry (5.1 mm ECAL cells). The fine view uses the copy
in `mlpf/data/key4hep/geometry/CLD_o2_v07_ecal2mm`; its only change from the
2025-05-29 k4geo release is `ECal_cell_size = 2.0*mm`. That constant controls
both ECAL barrel and endcap readouts.

The photon gun runs inside `ddsim`; this pilot does not generate HepMC. Same
seeds are used in both arms, and `validate_pair.py` checks that the recorded
primary photon momenta match event by event. Geant4 showers need not match.

## Generate

Use the Tallinn Pixi profile and inspect the generated scripts before launch:

```bash
PROD=cld_photon_5p1 pixi run gen --dry-run --printshellcmds
PROD=cld_photon_2mm pixi run gen --dry-run --printshellcmds
```

Then run each arm:

```bash
PROD=cld_photon_5p1 pixi run gen
PROD=cld_photon_2mm pixi run gen
```

Outputs go to separate workspaces under
`/local/joosep/mlpf/cld/photon_ecal_{5p1mm,2mm}_key4hep_2025-05-29/`.
Each seed retains `sim_gun_gamma_10gev_<seed>.root` and
`reco_gun_gamma_10gev_<seed>.root`. The production uses the existing CLD
digitisation and Pandora chain with its matching compact geometry.

## Compare

After the same seeds finish in both arms:

```bash
uv run python scripts/cld_photon_readout/validate_pair.py \
  --baseline /local/joosep/mlpf/cld/photon_ecal_5p1mm_key4hep_2025-05-29 \
  --fine /local/joosep/mlpf/cld/photon_ecal_2mm_key4hep_2025-05-29 \
  --output /local/joosep/mlpf/cld/photon_ecal_readout_comparison.json
```

The comparison fails if seed sets, event counts, or primary photon momenta
disagree. It reports simulated and digitised ECAL hit counts and energy, the
fraction of events with a reconstructed photon, and leading-photon energy
response. These are pilot integrity and single-photon metrics, not a pi-zero
separation result. Follow with two-photon or tau events for the physics study.

The geometry retains the baseline 5 mm maximum calorimeter step length. Before
a large campaign, check whether that limit affects 2 mm cell assignment with
a dedicated small-sample step-limit comparison. Keep any revised step limit
the same in both arms.

## CLD ttbar follow-up

`cld_ecal_2mm` starts with 1,000 `p8_ee_ttbar_ecm365` events as ten 100-event
jobs, using seeds 300000–300009. Those seeds match the first ten files in the
existing 5.1 mm CLD production at
`/local/joosep/mlpf/cld/v1.2.5_key4hep_2025-05-29/gen/p8_ee_ttbar_ecm365/root/`.
It uses the same Pythia card and pinned Key4hep release as the baseline, and
the 2 mm compact XML for both simulation and reconstruction. The new outputs
go under `/local/joosep/mlpf/cld/ecal_2mm_key4hep_2025-05-29/`.
Additional CLD samples can use the same production and workspace, with one
subdirectory under `gen/` per process name.
This validation production has no postprocessing or TFDS mapping and does not
retain separate simulation files; the reconstructed ROOT includes the
simulated ECAL hit collections needed for the first comparison.

```bash
PROD=cld_ecal_2mm pixi run gen --dry-run --printshellcmds
PROD=cld_ecal_2mm pixi run gen
```

Before treating the runs as paired, compare generator-particle records for
every matching seed and event. The baseline was produced earlier, so matching
seeds alone do not prove matching events. If the event records differ, generate
a fresh 5.1 mm arm with the same current setup. The photon pilot found a
roughly 0.2 GeV shift in Pandora photon energy with unchanged calibration;
evaluate ttbar hits and clusters separately from jet or PFO response until
the 2 mm reconstruction is calibrated.

For hit and cluster distributions, run:

```bash
uv run python scripts/cld_photon_readout/compare_distributions.py \
  --baseline /local/joosep/mlpf/cld/photon_ecal_5p1mm_key4hep_2025-05-29 \
  --fine /local/joosep/mlpf/cld/photon_ecal_2mm_key4hep_2025-05-29 \
  --output-dir reports/cld_photon_readout
```

The script writes `summary.json` and `hit_cluster_distributions.png` and
checks that the paired primary photon momenta match. For shower closeups,
run `scripts/visualize_key4hep.py` separately on the two reconstructed ROOT
files with the same seed and event indices, using `--detector cld
--center-on-cluster --plot-limit 200`. Use separate output directories for
the two arms because they share the same detector and seed labels.
