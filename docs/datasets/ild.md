# ILD at FCC-ee

The `ild` production uses **ILD_FCCee_v02** with a 2 T field, full Geant4
simulation, Clupatra pattern recognition with DDKalTest refits
(`MarlinTrkTracks`), and Pandora calorimeter clustering and particle flow.
The TFDS datasets (version **0.2.0**) are experimental: they are intended for
qualifying the detector in MLPF, and the reconstruction has not yet been
validated for physics.

## Physics samples

| Physics sample | Process name | TFDS dataset | Energy |
|---|---|---|---|
| Top-quark pairs | `p8_ee_ttbar_ecm365` | `ild_edm_ttbar_pf` | 365 GeV |
| Fully hadronic WW | `p8_ee_WW_fullhad_ecm365` | `ild_edm_ww_fullhad_pf` | 365 GeV |
| Inclusive quark pairs | `p8_ee_qq_ecm365` | `ild_edm_qq_pf` | 365 GeV |
| Z→qq | `p8_ee_Z_qq_ecm91` | `ild_edm_z_qq_pf` | 91 GeV |
| Z→tautau | `p8_ee_Z_tautau_ecm91` | `ild_edm_z_tautau_pf` | 91 GeV |

All datasets use configuration **1** and ArrayRecord format. The
`pyg-ild-v1` recipe trains on qq; `pyg-ild-key4hep-v140` uses all five samples.
The runner can also produce particle guns (`gun_e_*`, `gun_mu_*`, `gun_pi_*`,
`gun_gamma_*`, `gun_kaon0L_*`) for calibration checks; they have no TFDS
datasets.

## Software and generation

The generator is in `mlpf/data/key4hep/gen/ild`. Its ILDConfig submodule is
pinned to `fc1c9f58f9912b4d1fb10da5defbaca2133d60a0`, and it shares the k4geo
and k4RecTracker builds and the development container with IDEA. Like the other
Key4hep detectors, it runs on the stable **Key4hep 2026-04-08** release with
locally built k4FWCore v1.6 and MarlinReco v1.38.1. Build the software once per
host, following the [build instructions](../../mlpf/data/key4hep/gen/README.md):

```bash
git submodule update --init --recursive
apptainer exec --bind /cvmfs,"$PWD",/tmp --env LD_PRELOAD=/usr/lib64/libffi.so.8 \
  mlpf/data/key4hep/gen/idea/idea-dev.sif bash mlpf/data/key4hep/gen/build_all.sh
bash scripts/check_ild_runtime.sh
```

To generate ten 10 GeV muons by hand:

```bash
apptainer exec --bind /cvmfs,"$PWD",/tmp --env LD_PRELOAD=/usr/lib64/libffi.so.8 \
  mlpf/data/key4hep/gen/idea/idea-dev.sif \
  env NEV=10 OUTDIR="$PWD/local_test_data/ild/gen" WORKDIR="$PWD/local_test_data/ild/mu-work" \
  bash mlpf/data/key4hep/gen/ild/run_sim.sh gun_mu_10gev 600000
```

Use a new `WORKDIR` for each job. When the job succeeds, the runner writes
`OUTDIR/PROCESS/root/reco_PROCESS_SEED.root` together with
`reco_PROCESS_SEED.software.txt`, which records the source commits, software
and geometry checksums, and the exact commands. Work directories under
`/scratch/local/$USER` are removed when the job exits; elsewhere, the logs and
intermediate files are kept.

Collision samples apply the ILDConfig crossing-angle boost; particle guns do
not. Set `GUN_ENERGY`, `GUN_PARTICLE`, `GUN_THETA_MIN` and `GUN_THETA_MAX` to
configure a gun, with DD4hep units for angles (for example `45*degree`). The
gun energy sets the momentum magnitude, so the total energy of a massive
particle is slightly higher.

The output keeps the simulated hits and the digitization relations, so that
target assignments can be traced back to the original particles. The upstream
reconstruction does not store dQ/dx or the tracker cell-ID encoding; the dE/dx
features are therefore zero, and there is no ILD hit dataset.

## Production

The datasets come from the [Key4hep v1.4.0 production](key4hep-50k.md):
50,000 events per sample, split by ROOT file into 45,000 training and 5,000
test events. All five datasets are [published](catalog.md#published-availability)
on the Hub.

## Inputs and targets

Postprocess a ROOT file with:

```bash
PYTHONPATH=. python -m mlpf.data.key4hep.postprocessing \
  --detector ild --input /path/to/reco_p8_ee_qq_ecm365_500000.root \
  --outpath /path/to/post/p8_ee_qq_ecm365
```

Track features come from the fitted state at the interaction point, and the
innermost radius from the `AtFirstHit` state. A track with an invalid state or
fit quality stops the conversion with an error instead of being dropped. As
for the other Key4hep detectors, inputs have 19 features and targets have 14.

The target policy is **`ild-direct-hit-majority-v1`**:

- Each distinct attached tracker measurement votes once, through the
  digitization-to-simulation and simulation-to-MC identities. A measurement
  shared by several particles splits its vote; an unlinked measurement stays
  in the denominator.
- A track labels only its direct charged status-1 majority owner at **75%**
  purity or higher. Secondary, conversion and low-purity tracks remain inputs
  without a label, and never inherit a parent's momentum.
- Calorimeter links use each particle's share of a hit's truth weight times
  the calibrated hit energy; descendant deposits propagate to status-1
  ancestors. The common visibility rule applies (deposit above 10% of the
  particle energy, or above 0.5 GeV for charged particles, or a qualifying
  track).
- Visible particles claim one exclusive track or cluster in descending truth
  energy. An unmatched particle merges into the representative of its
  largest-deposit cluster if that cluster has one; otherwise it is dropped and
  the lost energy is recorded in `metadata_premerge_energy`.
- Cluster representatives are neutral with the matching neutral class, even
  when the source particle was charged; track representatives keep charged
  classes. Generator jets and MET are a separate pre-visibility reference.

Native `MarlinTrkTracksMCTruthLink` weights are retained for audit only, since
they can exceed one. `ycand` is **real Pandora output**: multi-track PFOs get
one representative (charged PFOs their highest-momentum track, neutrals their
highest-energy cluster), and the full PFO momentum is stored once.

## Qualification and TFDS

```bash
PYTHONPATH=. python -m scripts.studies.qualify_ild \
  /path/to/reco.root /path/to/reco.parquet --output native_audit.json
PYTHONPATH=. python tests/validate_parquet.py \
  --detector ild --input /path/to/post/p8_ee_qq_ecm365 --plots-dir validation
```

The native audit checks event preservation, collection identities, fitted IP
momenta, direct-primary tracking, angular acceptance, PFO count and energy
preservation, target provenance, unique representatives, merge energy loss
and target-jet four-vector closure. The validator applies the same direction and energy checks as for the other
detectors.

Known limitations of the reconstruction:

- Tracking coverage ends at about |cosθ| ≈ 0.95.
- Low-pT tracks have direction outliers.
- For clusters from charged particles, the shower direction differs from the
  particle's production direction, because the particle curves in the field.
  The validator therefore reports clusters from neutral and charged particles
  separately.

The builders under `mlpf/heptfds/ild_pf_edm4hep/` require ILD provenance
records, finite arrays and at least two generation jobs per process. Train and
test sets use disjoint ROOT files, and files with duplicate checksums are
rejected.

```bash
tfds build mlpf/heptfds/ild_pf_edm4hep/qq.py --config 1 \
  --manual_dir /path/to/post --data_dir /path/to/tfds
```

## Comparing CLD and ILD

Convert both detectors with the current postprocessor into
`POST/<detector>/<sample>/` (for example `POST/cld/ttbar/` and `POST/ild/ttbar/`),
then run:

```bash
bash scripts/run_detector_feature_comparison.sh --samples POST OUTPUT 200 "cld ild"
```

This writes one report over all samples and one per sample, comparing track and
cluster features, target and generator jet spectra, matching efficiency and
purity, and jet response. Check the generator spectra first: equal cards and
seeds across software versions do not guarantee identical events.
