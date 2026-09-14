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

For CLIC, use the corresponding path and `--detector clic`. To see all supported detector modes and output controls:

```bash
uv run python3 tests/validate_parquet.py --help
```

Run the validator on a directory to cover several shards. A small event limit is useful while developing a postprocessor; a publication gate must sample enough files and event types to expose rare failures.

## What the gates mean

The report groups checks around these invariants:

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

## Pass/fail discipline

Keep the machine-readable report with the production campaign. Record the repository commit, detector mode, command, event/file selection, and any accepted exception. An intentional invariant change requires an updated focused unit test and a documented scientific reason for the schema or target-definition change.
