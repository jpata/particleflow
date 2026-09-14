# CMS data production

This page covers the CMS-specific choices. Use the shared [generation procedure](generate.md) for site selection, dry runs, restart behavior, TFDS verification, and publication.

## Environment and representation

The `cms_run3` production recipe uses CMSSW 15.0.5 with `el8_amd64_gcc12` and a CMSSW RHEL 8 container. Its current TFDS recipe uses reconstructed tracks and calorimeter clusters as MLPF inputs.

The configured model is `pyg-cms-v1`, and the configured TFDS schema version is 3.2.0. Compatibility requires the CMSSW collections and schema expected by this site-dependent production path.

## Configured samples

| Dataset | Physics process | Pileup category |
|---|---|---|
| `cms_pf_ttbar` | top-quark pair production | 55--75 interactions |
| `cms_pf_qcd` | QCD multijet production | 55--75 interactions |
| `cms_pf_ztt` | hadronic tau pairs | 55--75 interactions |
| `cms_pf_ttbar_nopu` | top-quark pair production | no pileup |
| `cms_pf_qcd_nopu` | QCD multijet production | no pileup |
| `cms_pf_ztt_nopu` | hadronic tau pairs | no pileup |

The exact CMSSW process names, seed ranges, events per job, and output subdirectories live under `productions.cms_run3.samples` in `particleflow_spec.yaml`. Use those current values when configuring a campaign.

## CMS path through the common workflow

Generate and postprocess the configured CMS samples:

```bash
PROD=cms_run3 pixi run gen -- --dry-run --printshellcmds
PROD=cms_run3 pixi run gen
PROD=cms_run3 pixi run post
```

Generation writes PF ntuples below:

```text
<workspace>/gen/<pileup-subdirectory>/<CMSSW-process>/root/
```

`mlpf/data/cms/postprocessing2.py` converts each ntuple to `.pkl.bz2` below the corresponding `post/` hierarchy. The TFDS builder receives the pileup subdirectory containing the process directory and writes the six dataset families listed above.

## Checks before TFDS

CMS uses a compressed-pickle intermediate format and detector-specific integrity checks. Before committing resources to a production-scale campaign:

1. inspect generation and postprocessing logs for every selected seed;
2. verify that the expected `.pkl.bz2` output exists and opens;
3. run `scripts/local_test_cms.sh` in a disposable checkout to exercise CMS postprocessing, TFDS decoding, short training, checkpoint loading, and ONNX comparison.

These checks establish software and schema integrity. CMS physics performance requires the jet/MET validation, collision-data commissioning, calibrations, and luminosity selections described in the [CMS validation guide](../validation/cms.md).

After the checks pass, build the six configured TFDS families:

```bash
PROD=cms_run3 pixi run tfds
```

Open one event from each dataset/configuration using the procedure in [Download a dataset](download.md).

## CMS target information

The TFDS event contains detector elements (`X`), the reconstructable training target (`ytarget`), baseline PF candidates (`ycand`), generator/target jets, generator missing transverse momentum, and Pythia particles. The target features include pileup and generator/simulator status fields. Their definitions are implementation details of the current postprocessor and builder; pin the code commit together with dataset version 3.2.0 when publishing derived data.
