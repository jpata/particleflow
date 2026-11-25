# Local detector comparison

Two local checks compare the track/cluster inputs of ColliderML, CLIC, CLD, IDEA, and MAIA on ttbar events:

- **Feature comparison report** (`scripts/run_detector_feature_comparison.sh`): converts a small sample from each detector, plots the input and target features side by side, checks their internal consistency, and compares target jets with generator jets.
- **Real ColliderML training check** (`scripts/local_test_colliderml_real.sh`): converts 1,000 real ColliderML events and trains the production model on them.

Both check software and data consistency. Collision types, energies, reconstruction, and target rules differ between detectors, so the plots are **not detector performance rankings**.

## Feature comparison report

### Inputs

| Detector | Source | Events compared |
|---|---|---|
| ColliderML | One `ttbar_pu0` shard of `CERN/ColliderML-Release-1`, downloaded at a pinned revision | 100 |
| CLIC, CLD, IDEA | `reco_p8_ee_ttbar_ecm{380,365}_300000.root` under `$KEY4HEP_ROOT_DIR/<detector>/ttbar/` | 100 |
| MAIA | The local smoke sample from `scripts/local_test_maia.sh` (10 events) | 10 |

The report also reads stored TFDS for comparison. ColliderML and MAIA TFDS are built by the runner at their builders' current versions. CLIC, CLD, and IDEA use existing TFDS version directories set by `CLIC_TFDS`, `CLD_TFDS`, and `IDEA_TFDS`; the defaults point at 3.3.0, 3.3.0, and 0.3.0 under `/mnt/work/mlpf/tensorflow_datasets/`. A TFDS column that is not available is skipped with a message. The CLD/CLIC/IDEA ROOT and TFDS paths are site-local; set the four variables on another machine.

### Run

From a checkout with a prepared `.venv` (`uv sync --locked`):

```bash
KEY4HEP_ROOT_DIR=/data/mlpf/root \
CLD_TFDS=/data/tfds/cld/cld_edm_ttbar_pf/1/3.3.0 \
bash scripts/run_detector_feature_comparison.sh
```

The runner downloads the ColliderML shard if missing, converts the Key4HEP ROOT files and the ColliderML shard, builds the small ColliderML TFDS, runs `scripts/local_test_maia.sh` if the MAIA TFDS is missing (which also trains two CPU steps), renders event displays, and runs `scripts/compare_detector_features.py`. Everything is written under `local_test_data/`:

| Output | Contents |
|---|---|
| `detector_comparison/real_plots/index.html` | Feature sheets, jet matching, checks, and caveats |
| `detector_comparison/real_plots/summary.json` | Inputs, settings, feature-column maps, value ranges, checks, and matching counts |
| `detector_comparison/event_displays/index.html` | Event gallery for all five detectors |
| `detector_comparison/colliderml_source/provenance.json` | Pinned ColliderML revision and shard paths |

`--num-events` of `compare_detector_features.py` only limits plotting; more statistics require converting more events into a new directory.

### Caching

The ColliderML build is stored under `colliderml_builds/<fingerprint>/`, where the fingerprint hashes the converter, target-building, builder, and configuration sources. A code change therefore produces a new build instead of reusing stale features; old builds are kept. CLIC/CLD/IDEA Parquet and the MAIA TFDS are reused if present: after changing their converters, delete them or point the comparison at new directories.

### What the report checks

- **Features** are matched by meaning, not column index: Key4HEP inputs have 19 columns, ColliderML 17 with ACTS-derived track parameters (`tanLambda = sinh(eta)`, `omega` from the nominal 3 T field, innermost radius from the linked hits). Histograms share bins across detectors, use the union of each detector's 0.5–99.5% quantiles as range, and annotate out-of-range fractions.
- **Consistency checks**: finite values, input/target alignment, valid classes and element types, phi range, target jet indices, `p = pt cosh(eta)`, `ET = E / cosh(eta)`, and target energy vs momentum.
- **Round trips**: every Parquet event is passed through the TFDS feature encoder; for ColliderML and MAIA, all native-loader events are also compared with all stored TFDS records by SHA-256 fingerprint. The stored CLIC/CLD/IDEA TFDS are independent of the small ROOT samples, so their comparison is distributional only.
- **Target occupancy**: `target_active_fraction = n_target / (n_track + n_cluster)`. It is label occupancy, not efficiency, since one particle can leave several elements but only one carries its target.
- **Jets**: target jets are matched one-to-one to generator jets within ΔR < 0.1 with `mlpf.jet_utils`, without a response cut, using each detector's own jet algorithm and pT threshold. The report shows response (`targetjet_pt / genjet_pt`), efficiency (matched gen jets), and purity (matched target jets) with 68% Wilson intervals.

### Event displays

`bash scripts/run_detector_event_displays.sh 0 1 2` renders only the gallery for the given event indices; `COLLIDERML_DISPLAY_PARQUET` selects a ColliderML build (the full runner sets it). Views are transverse with a common ±6500 mm extent; hits are sampled up to 800 per collection, and charged truth guides are idealized helices in each detector's nominal field. Equal indices are independent collisions. For reusable published views, see [Detector event views](../datasets/event-displays.md). The same directory also holds decorative multi-detector artwork (`openlab_multi_event_dark*.png`), which is not a comparison.

### Known limitations

- IDEA tracks are truth-seeded and its reference candidates are target oracles; ColliderML has no PF reference.
- MAIA has 10 events and missing CellIDEncoding metadata. Its Parquet validation fails `tracker_hit_momentum`, `calorimeter_hit_energy`, and `baseline_particle_energy`; the report embeds that validation, and the smoke script continues past it.

## Real ColliderML training check

```bash
bash scripts/local_test_colliderml_real.sh
```

This converts the first 1,000 events of the downloaded ColliderML shard, builds TFDS configuration 10 split 900/100 by event, validates the training data, and trains `pyg-colliderml-v1` for 225 steps (one pass) on one GPU in bfloat16 with Flash attention. The 100 held-out events serve both validation and inference; there is no independent test sample. Results go to `local_test_data/colliderml_real/<fingerprint>/`, keyed by the converter code and event count.

| Variable | Effect |
|---|---|
| `GPUS=0` | CPU run in float32 with math attention |
| `NUM_EVENTS`, `NUM_STEPS`, `VAL_FREQ`, `CHECKPOINT_FREQ`, `LOG_FREQ` | Sample size and training schedule |
| `PREPARE_ONLY=1` | Stop after data preparation and validation |

For a five-pass convergence check, start a fresh run (resuming would keep the one-pass schedule) and plot it:

```bash
NUM_STEPS=1125 VAL_FREQ=225 CHECKPOINT_FREQ=225 LOG_FREQ=25 bash scripts/local_test_colliderml_real.sh
PYTHONPATH="$PWD" uv run python3 scripts/plot_training_convergence.py --experiment-dir /path/to/experiment
```

This checks training integration and loss behavior, not physics performance.

## Tests

```bash
PYTHONPATH="$PWD" uv run python3 -m pytest -q \
  tests/test_detector_jet_comparison.py tests/test_detector_plot_layout.py \
  tests/test_detector_event_displays.py tests/test_maia_pipeline.py \
  tests/test_colliderml_adapter.py tests/test_colliderml_postprocessing.py
```

These cover jet matching, plot layout, event-display geometry, MAIA identification, the ColliderML column mapping, smoke-sample splitting, and the native builder contracts.
