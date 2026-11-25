# Validate a dataset

Use `tests/validate_parquet.py` on detector-postprocessed Parquet before building or publishing TFDS. The command writes a JSON report suitable for an automated gate and diagnostic plots for human review.

## Run a bounded check

For the CLD file downloaded by the local data-preparation script:

```bash
uv run python3 tests/validate_parquet.py \
  --input local_test_data/cld/p8_ee_ttbar_ecm365/reco_p8_ee_ttbar_ecm365_300000.parquet \
  --detector cld \
  --max-events 20 \
  --plots-dir plots/validation/cld
```

For CLIC, use the corresponding path and `--detector clic`. For ColliderML, use `--detector colliderml`.

Every invocation runs the same checks and writes plots and `validation_report.json`. Checks that do not apply to the available detector data are marked `SKIP`. Schema errors stop further checks. The exit code is 1 if any check fails, and 0 otherwise. There are no separate alignment or report modes.

The four command-line options are:

| Option | Meaning |
|---|---|
| `--input` | Required Parquet file or directory of files |
| `--detector` | Required detector name |
| `--max-events` | Optional positive event limit per file; by default, use every event |
| `--plots-dir` | Output directory; default `validation_plots` |

For the supported detector names:

```bash
uv run python3 tests/validate_parquet.py --help
```

Run the validator on a directory to cover several shards. A small event limit is useful while developing a postprocessor; a publication gate must sample enough files and event types to expose rare failures.

## What the gates mean

Check names appear in the JSON report, terminal output, and plot filenames. For example, `track_target_direction` produces `track_target_direction.png`.

| Check | Comparison |
|---|---|
| `file_contents`, `required_fields`, `finite_values` | File contents, required arrays, and finite feature values |
| `input_target_counts` | Input and target row counts in each event |
| `particle_labels`, `particle_targets`, `target_charge` | Particle numbers, target particles, and charges |
| `hit_counts`, `track_cluster_counts` | Detector input counts |
| `tracker_hit_momentum`, `calorimeter_hit_energy` | Momentum and energy from hits versus truth |
| `detector_positions`, `hit_cluster_positions` | Detector positions and mean hit positions relative to clusters |
| `element_target_momentum` | Track pT and cluster energy relative to their assigned targets |
| `track_target_direction`, `cluster_target_direction` | Input directions relative to their assigned target particles |
| `target_momentum_agreement`, `target_jet_indices` | Repeated target values and jet indices |
| `target_jet_response`, `deposited_energy` | Target jets and calorimeter deposits relative to truth |
| `muon_energy_deposits`, `neutral_energy_deposits`, `simulation_flags` | Energy deposits and simulation flags |
| `baseline_particle_counts`, `baseline_particle_labels`, `baseline_particle_energy` | Baseline reconstruction versus targets |
| `truth_missing_momentum`, `truth_jet_counts`, `target_truth_energy`, `particle_types` | Truth quantities and target particle types |

Direction checks use the assigned input/target row, without searching for another nearby particle. Tracks use their stored momentum direction and require at least 95% of assigned particles within delta-R < 0.1. Clusters use the direction from the origin to their stored position, with phi = atan2(y, x) and eta = asinh(z / hypot(x, y)), and require at least 80% within delta-R < 0.3. Stored cluster angle features are not used by this check. Nonfinite positions and positions with zero transverse radius count as invalid directions. Invalid directions count against the passing fraction. The JSON records the direction source. The JSON also contains results for hard-scatter particles, pileup particles, and target pT >= 1 and >= 5 GeV. These quality thresholds can flag physically difficult assignments; a failure alone does not establish an indexing error.

The checks cover these relationships:

- the Parquet file loads and contains at least one event;
- required arrays and feature widths match the selected detector representation;
- input, target, and candidate rows and their indices are mutually consistent;
- particle-number assignments refer to valid particles and obey representation rules;
- input and target energy sums satisfy the implemented closure checks; and
- tracker and calorimeter hit metadata and geometry are valid for hit inputs.

Treat every failed gate as a production error until its cause is understood. Relationship and schema gates determine the result even when a diagnostic plot looks plausible.

## Validation between production stages

| Transition | Check before continuing |
|---|---|
| Simulation to postprocessing | ROOT file opens, expected collections exist, event count is plausible |
| Postprocessing to TFDS | Strict Parquet validator passes on representative outputs |
| TFDS to training | TFDS metadata resolves, one event decodes, configured names/partitions/versions match |
| Dataset publication | Repeatable report, provenance, schema version, event counts, and storage manifest are recorded |

The [generation guide](../datasets/generate.md) shows where these gates enter the Pixi/Snakemake workflow. Detector-specific interpretation belongs in the [CMS](../datasets/cms.md), [Key4HEP](../datasets/key4hep.md), and [hit-input](../datasets/hit-based.md) guides.

## Compare target assignment across detectors

`scripts/studies/validate_detector_matching.py` rebuilds the targets of a detector's ROOT file with both its native assignment and `majority_energy`, keeping each detector's visibility, ancestry, and merging policy. Run one job per detector:

```bash
PYTHONPATH=. uv run python3 scripts/studies/validate_detector_matching.py \
  --detector cld \
  --input /path/to/reco_p8_ee_ttbar_ecm365_300000.root \
  --output experiments/cross_detector_matching/cld --events 50
```

Use `clic`, `idea`, or `cms` with the matching input (for CMS, `pfntuple_100000.root` from `scripts/fetch_test_data_cms.sh`). `scripts/studies/report_detector_matching.py --input experiments/cross_detector_matching` combines the outputs into plots and paired event-bootstrap intervals. Results describe target construction, not model predictions, and absolute jet recall is not comparable across detectors because their truth-jet definitions differ.

## Pass/fail discipline

Keep the machine-readable report with the production campaign. Record the repository commit, detector mode, command, event/file selection, and any accepted exception. An intentional invariant change requires an updated focused unit test and a documented scientific reason for the schema or target-definition change.
