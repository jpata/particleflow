# Validate CMS reconstruction

CMS validation compares the standard PF and MLPF reconstruction chains in CMSSW. It covers two distinct tasks:

- **simulation validation**, where generator-level references support response and resolution studies;
- **collision-data commissioning**, where observed PF and MLPF distributions are compared after data-quality and luminosity selection.

This is a site-dependent production workflow. It requires CMSSW-compatible inputs, the configured container, substantial storage, and experiment-specific knowledge. Start with the [local CMS smoke test](../getting-started/quickstart.md) if the goal is only to check the software.

## Inputs and stages

```text
CMSSW PF and MLPF outputs
        |
        +-- simulated sample --> compact PF/MLPF Parquet
        |                         |
        |                         +--> derive jet-energy corrections
        |                         +--> AK4/AK8 jet and MET validation
        |
        +-- collision data ------> golden-JSON and luminosity selection
                                  +--> corrected PF/MLPF distributions
```

The checked-in `validation_cms.yaml` defines the workspace, MC and data samples, output directory, correction sample, AK4 and AK8 jets, fiducial regions, center-of-mass energy, golden JSON, luminosity CSV, and batch resources.

## Prepare a scenario

Copy `validation_cms.yaml` for a campaign and review every path and selection. In particular, confirm:

- PF and MLPF files were produced from compatible CMSSW conditions;
- the MC sample used for corrections represents the intended phase space;
- the golden JSON and luminosity table match the collision-data period;
- the AK4/AK8 jet definitions and fiducial regions match the analysis question; and
- the site profile can read inputs and write the output directory from inside the container.

Reuse a correction file only when its recorded sample, reconstruction versions, binning, and derivation commit match the new workflow.

## Preview and run

Generate and inspect a dry-run workflow before submitting jobs:

```bash
uv run python3 mlpf/snakemake/produce_cms_validation_snakemake.py \
  --config validation_cms.yaml \
  --scenario cms_run3
snakemake --snakefile "$(cat .last_jobs_dir)/Snakefile" --dry-run
```

On a reviewed Pixi site configuration:

```bash
PROD=cms_run3 pixi run validation_cms --dry-run
PROD=cms_run3 pixi run validation_cms
```

The generator writes a workflow below `snakemake_validation/`. MC preparation produces PF and MLPF Parquet files; correction jobs write `jec_<jet type>_<sample>.npz`; plot jobs write the configured validation output tree and completion sentinels.

## Simulation validation

For MC, inspect particle and event content before corrected jets. Then compare PF and MLPF for:

- jet response, resolution, and matching efficiency in AK4 and AK8 collections;
- dependence on transverse momentum and detector region;
- missing transverse momentum response and resolution; and
- central summaries, tails, and low-statistics bins.

State the generator sample, pileup conditions, detector/CMSSW release, target definition, jet corrections, matching, fiducial cuts, and event counts with every result.

## Collision-data commissioning

Collision-data commissioning uses observed distributions because generator-level targets are available only in simulation. Apply the approved golden JSON and luminosity inputs, then compare stable observables between PF and MLPF across run periods and detector regions. Treat each difference as a diagnostic and trace it through object multiplicities, particle types, jets, MET, triggers, and selections before classifying it.

The full CMS workflow establishes claims involving reconstruction, calibrations, and luminosity accounting. Standalone TFDS inference efficiently establishes model-shape, checkpoint-load, and ONNX numerical behavior.

## Completion checks

Confirm that all Snakemake targets finished, PF and MLPF event counts and selections agree, correction files came from the configured sample, every requested fiducial region is populated, and plots carry enough metadata to identify the campaign. Preserve the resolved validation specification with the output.

Current follow-up work on CMS reconstruction includes [single-particle monitoring](https://github.com/jpata/particleflow/issues/357) and [outlier studies](https://github.com/jpata/particleflow/issues/327). Use these open issues as interpretative context; passing validation evidence comes from the checks and artifacts described above.
