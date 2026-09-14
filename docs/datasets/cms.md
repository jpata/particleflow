# CMS data production

This page records only the CMS-specific choices. Use the shared [generation procedure](generate.md) for site selection, dry runs, restart behavior, TFDS verification, and publication.

## Environment and representation

The `cms_run3` production recipe uses CMSSW 15.0.5 with `el8_amd64_gcc12` and a CMSSW RHEL 8 container. Its MLPF inputs are reconstructed tracks and calorimeter clusters; there is no CMS hit-input TFDS recipe in the current specification.

The configured model is `pyg-cms-v1`, and the configured TFDS schema version is 3.2.0. CMS validation remains CMSSW- and site-dependent; the fact that a production recipe exists does not make arbitrary CMS inputs compatible.

## Configured samples

| Dataset | Physics process | Pileup category |
|---|---|---|
| `cms_pf_ttbar` | top-quark pair production | 55--75 interactions |
| `cms_pf_qcd` | QCD multijet production | 55--75 interactions |
| `cms_pf_ztt` | hadronic tau pairs | 55--75 interactions |
| `cms_pf_ttbar_nopu` | top-quark pair production | no pileup |
| `cms_pf_qcd_nopu` | QCD multijet production | no pileup |
| `cms_pf_ztt_nopu` | hadronic tau pairs | no pileup |

The exact CMSSW process names, seed ranges, events per job, and output subdirectories live under `productions.cms_run3.samples` in `particleflow_spec.yaml`. Review them rather than copying values from an older campaign.

## CMS path through the common workflow

```bash
PROD=cms_run3 pixi run gen -- --dry-run --printshellcmds
PROD=cms_run3 pixi run gen
PROD=cms_run3 pixi run post
PROD=cms_run3 pixi run tfds
```

Generation writes PF ntuples below:

```text
<workspace>/gen/<pileup-subdirectory>/<CMSSW-process>/root/
```

`mlpf/data/cms/postprocessing2.py` converts each ntuple to `.pkl.bz2` below the corresponding `post/` hierarchy. The TFDS builder receives the pileup subdirectory containing the process directory and writes the six dataset families listed above.

## Checks before TFDS

There is currently no CMS equivalent of the strict Key4HEP `validate_parquet.py` gate because the intermediate format and detector relationships differ. Before a production build:

1. inspect generation and postprocessing logs for every selected seed;
2. verify that the expected `.pkl.bz2` output exists and opens;
3. run `scripts/local_test_cms.sh` in a disposable checkout to exercise CMS postprocessing, TFDS decoding, short training, checkpoint loading, and ONNX comparison;
4. open one event from each produced TFDS dataset using the check in [Download a dataset](download.md).

These checks establish software and schema integrity, not CMS physics performance. Jet/MET validation, collision-data commissioning, calibrations, and luminosity selections belong to the future CMS validation guide.

## CMS target information

The TFDS event contains detector elements (`X`), the reconstructable training target (`ytarget`), baseline PF candidates (`ycand`), generator/target jets, generator missing transverse momentum, and Pythia particles. The target features include pileup and generator/simulator status fields. Their definitions are implementation details of the current postprocessor and builder; pin the code commit together with dataset version 3.2.0 when publishing derived data.
