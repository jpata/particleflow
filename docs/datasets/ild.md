# ILD at FCC-ee

The `ild` production uses **ILD_FCCee_v02** with a 2 T field, full Geant4
simulation, Clupatra pattern recognition with DDKalTest refits
(`MarlinTrkTracks`), and Pandora calorimeter clustering and particle flow.
TFDS version **0.2.0** is an experimental qualification dataset: a successful
conversion or TFDS build does not establish reconstruction performance.

## Physics samples

| Physics sample | Process name | TFDS dataset | Energy |
|---|---|---|---|
| Top-quark pairs | `p8_ee_ttbar_ecm365` | `ild_edm_ttbar_pf` | 365 GeV |
| Fully hadronic WW | `p8_ee_WW_fullhad_ecm365` | `ild_edm_ww_fullhad_pf` | 365 GeV |
| Inclusive quark pairs | `p8_ee_qq_ecm365` | `ild_edm_qq_pf` | 365 GeV |
| Z→qq | `p8_ee_Z_qq_ecm91` | `ild_edm_z_qq_pf` | 91 GeV |
| Z→tautau | `p8_ee_Z_tautau_ecm91` | `ild_edm_z_tautau_pf` | 91 GeV |

All datasets use configuration **1**, ArrayRecord, and the `pyg-ild-v1` recipe
in `particleflow_spec.yaml`. Particle guns (`gun_e_*`, `gun_mu_*`, `gun_pi_*`,
`gun_gamma_*`, `gun_kaon0L_*`) are available through the runner for
calibration checks but are not part of the campaign.

## Software and generation

The generator lives in `mlpf/data/key4hep/gen/ild` (key4hep-sim v1.4.0). Its
ILDConfig submodule pins `fc1c9f58f9912b4d1fb10da5defbaca2133d60a0`, and it
reuses the k4geo and k4RecTracker builds and the development container of the
IDEA recipe. All CLD, CLIC, ILD and IDEA productions use stable
**Key4hep 2026-04-08** with locally built k4FWCore v1.6 and MarlinReco
v1.38.1. Follow the [build instructions](../../mlpf/data/key4hep/gen/README.md)
once per execution host:

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

Use a fresh `WORKDIR` per job. The runner publishes
`OUTDIR/PROCESS/root/reco_PROCESS_SEED.root` atomically together with
`reco_PROCESS_SEED.software.txt`, which records source commits, the stack
checksum, geometry checksums and the exact commands. A work directory under
`/scratch/local/$USER` is removed on exit; any other work directory keeps the
logs and intermediate files. Collision samples use the ILDConfig crossing-angle
boost; guns use none. `GUN_ENERGY`, `GUN_PARTICLE`, `GUN_THETA_MIN` and
`GUN_THETA_MAX` (DD4hep units such as `45*degree`) adjust gun samples.
`--gun.energy` is the momentum magnitude, so a massive particle's total energy
is slightly above the nominal setting.

The reconstruction keeps the native simulation hits and digitization
relations in the output so that target attribution can be audited against the
original identities. Upstream drops dQ/dx and tracker cell-ID encodings, so the
dEdx features are zero and there is no ILD hit-dataset builder.

## Tallinn campaign

The [joint v1.4.0 campaign](key4hep-50k.md) generates **50,000 events per
sample** in 5,000 jobs of 10 events, with seeds 300000–305000 (ttbar),
400000–405000 (WW), 500000–505000 (qq), 600000–605000 (Z→qq) and
700000–705000 (Z→tautau), end exclusive. The workspace is
`/local/joosep/mlpf/ild/v1.4.0/50k` with `gen/`, `post/` and `tfds/`.
ILD jobs request 2,900 MiB for generation and reconstruction, 1,200 MiB for
postprocessing and 8,000 MiB for TFDS, with a 48-hour limit. Each dataset has
**45,000 train / 5,000 test** events split by independent ROOT jobs.

```bash
ln -sfn configs/tallinn/pixi.toml pixi.toml
pixi run init
bash scripts/check_ild_runtime.sh
pixi run datasets_key4hep --jobs 20 --dry-run
pixi run datasets_key4hep --jobs 20
```

## Inputs and targets

Postprocess a ROOT file with:

```bash
PYTHONPATH=. python -m mlpf.data.key4hep.postprocessing \
  --detector ild --input /path/to/reco_p8_ee_qq_ecm365_500000.root \
  --outpath /path/to/post/p8_ee_qq_ecm365
```

Track features come from the unique fitted state at the interaction point and
the first-hit radius from `AtFirstHit`; tracks with invalid states or fit
quality fail conversion rather than being silently dropped. The common feature
width is 19 and the target width is 14.

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
and target-jet four-vector closure. The validator applies the common direction
and energy gates. Known limitations of the current reconstruction are a
forward tracking coverage limit above |cosθ| ≈ 0.95, direction outliers for
low-pT tracks, and charged-source clusters whose production direction differs
from the transported shower direction; the validator's cluster gate therefore
reports neutral- and charged-source clusters separately.

The builders under `mlpf/heptfds/ild_pf_edm4hep/` require compatible ILD
provenance, finite arrays and at least two generation jobs per process. Files
are split into disjoint train and test sets by ROOT job, and duplicate source
checksums are rejected.

```bash
tfds build mlpf/heptfds/ild_pf_edm4hep/qq.py --config 1 \
  --manual_dir /path/to/post --data_dir /path/to/tfds
```

## Comparing CLD and ILD

Convert both detectors with the current postprocessor into
`POST/{cld,ild}/{ttbar,qq,ww_fullhad}/`, then run:

```bash
bash scripts/run_cld_ild_comparison.sh POST OUTPUT 200
```

This drives `scripts/compare_detector_features.py` to compare track and
cluster features, target and generator jet spectra, matching efficiency and
purity, and jet response. Check the generator spectra first: equal cards and
seeds across software versions do not guarantee identical events.
