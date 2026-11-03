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

Direction checks use the assigned input/target row, without searching for another nearby particle. Tracks require at least 95% of assigned particles within delta-R < 0.1; clusters require at least 80% within delta-R < 0.3. Invalid directions count as failures. The JSON also contains results for hard-scatter particles, pileup particles, and target pT >= 1 and >= 5 GeV. These quality thresholds can flag physically difficult assignments; a failure alone does not establish an indexing error.

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

## Assignment solver checks across detectors

`scripts/studies/validate_detector_matching.py` compares the native assignment, greedy matching on ownership scores, and the joint solver. It preserves each detector's visibility, ancestry and post-assignment policy. Run one job per detector; these jobs can run in parallel on CPU:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
PYTHONPATH=. uv run python3 scripts/studies/validate_detector_matching.py \
  --detector cld \
  --input /path/to/reco_p8_ee_ttbar_ecm365_300000.root \
  --output experiments/cross_detector_matching/cld --events 50
```

Use the matching `clic`, `idea`, or `cms` detector and ROOT path for the other jobs. CMS uses `pfntuple_100000.root` from `scripts/fetch_test_data_cms.sh`, the same input used by `scripts/local_test_cms.sh`. `scripts/studies/report_detector_matching.py --input experiments/cross_detector_matching` combines the four outputs and writes plots and paired event-bootstrap intervals.

The validation used the first 50 ttbar events from the published CLD, CLIC and IDEA `300000` ROOT files and all 50 events in the CMS smoke-test file. The CLD/CLIC/IDEA SHA-256 hashes match the Hugging Face files. Native target momenta and target-bearing input rows match the saved native postprocessing output on every event for all four detectors. All energy-accounting checks pass.

| Detector | Track delta-R > 0.1, native → solver | Target-to-truth jet recall, native → solver |
|---|---:|---:|
| CLD | 3.70% → 3.56% | 97.11% → 97.11% |
| CLIC | 3.50% → 3.46% | 98.97% → 98.97% |
| IDEA | 2.59% → 2.59% | 100% → 100% |
| CMS | 9.65% → 9.64% | 28.74% → 30.84% |

Jet definitions and truth references remain detector-specific: the native Key4HEP ee-genkt R=0.4, pT > 5 GeV reference for CLD/CLIC/IDEA, and the native CMS anti-kt R=0.4, pT > 3 GeV reference. Both use one-to-one delta-R < 0.1 matching. These are target-construction results, not model predictions, and absolute recall should not be compared across detector samples.

Important interpretation details:

- CLD/CLIC ownership uses calibrated hit energies, rather than corrected Pandora cluster energy. Their stored cluster directions differ substantially from centroid directions. Against centroids, the cluster delta-R > 0.3 fraction is 5.29% → 5.36% for CLD and 7.41% → 7.56% for CLIC. High-pT centroid-direction errors remain about 0.98% and 0.47%, respectively. Thus the much larger tail against the stored direction feature is not evidence that most cluster ownership assignments are wrong.
- IDEA supplies truth-seeded tracks. Its current target pool and no-merge policy are preserved; every selected particle remains represented. The solver changes representative inputs but not the set of target four-vectors. Cluster delta-R > 0.3 changes from 2.22% to 2.02%.
- The CMS ntuple supplies association weights, not per-hit truth. Track weights are divided by track energy to recover the recorded association fractions. Across the sample, 725 track columns sum above one and are excluded from solver eligibility, without renormalization. Calorimeter weights are recorded attributed energies. GSF-electron restrictions and CaloParticle splitting remain native.
- CMS fallback can put unmatched particles onto unused inputs. The optional assignment callback in `prepare_normalized_table` preserves that policy and safely handles these inputs. It creates 604 track targets outside the solver's exclusive selection in this sample. Final CMS track-direction tails therefore do not directly test the exclusive majority rule. All selected CaloParticle energy remains accounted for.
- CMS angular jet recall gains 2.10 percentage points over native (95% paired event-bootstrap interval 0.19–4.09), but only 0.55 points over same-score greedy matching (interval -0.43–1.46). Response-qualified recall changes from 16.79% to 17.34%, with an inconclusive interval. There is no convincing evidence here for a solver advantage over same-score greedy on CMS, nor for a jet-recall improvement on CLD, CLIC or IDEA.

Detailed results, source checks, native-output parity checks, and intervals are in `experiments/cross_detector_matching/`. The production assignment defaults remain unchanged.

## Pass/fail discipline

Keep the machine-readable report with the production campaign. Record the repository commit, detector mode, command, event/file selection, and any accepted exception. An intentional invariant change requires an updated focused unit test and a documented scientific reason for the schema or target-definition change.

## Production assignment validation, 2026-11-03

Checked on 2026-11-03 with unchanged validator thresholds. CLD, CLIC, IDEA and ColliderML use 20 ttbar events; MAIA uses its available 10 events. CMS uses all 50 events in the native pickle sample.

ColliderML samples were freshly converted with bipartite matching and the existing CLUE pu0/pu200 presets. Other datasets use their native processed samples.

| Dataset | Events | Track ΔR < 0.1 | Cluster ΔR < 0.3 | Failed checks |
|---|---:|---:|---:|---:|
| cld | 20 | 96.64% | 45.08% | 1 |
| clic | 20 | 96.37% | 45.22% | 1 |
| idea | 20 | 97.37% | 97.31% | 0 |
| maia | 10 | 29.09% | 37.62% | 5 |
| colliderml_pu0 | 20 | 99.74% | 62.55% | 5 |
| colliderml_pu200 | 20 | 99.34% | 51.06% | 4 |

Track and cluster direction checks require 95% and 80%, respectively.

CLD and CLIC fail only the stored cluster-direction check; the centroid comparison above explains the distinction. IDEA passes. MAIA fails track and cluster direction, tracker momentum, calorimeter energy, and baseline particle energy checks. Both ColliderML samples fail tracker-hit momentum, calorimeter-hit energy, hit-to-cluster positions, and cluster-to-target direction checks. PU0 also fails target-to-truth total energy. Full observations are recorded under `experiments/matching_cleanup_validation/`, alongside plots, commands and source paths. The CMS native-format validator reports five inconsistent particle-label groups across 50 events; its pickle schema is not supported by `validate_parquet.py`.
