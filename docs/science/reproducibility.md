# Reproducing published studies

Each MLPF publication uses a particular detector simulation, target definition, model, and software release. To reproduce a study, start with its archived code and data, then follow the installation and evaluation instructions from that release. The [publications page](publications.md) explains the scientific results; this guide connects them to the available artifacts.

Code, dataset, and checkpoint versions are independent. Keep their identifiers together with the model configuration, environment, and evaluation selection in the [experiment record](../reference/artifacts-and-directories.md). The [current dataset catalog](../datasets/catalog.md) and [training guide](../training/train.md) provide the starting point for new studies.

## Choosing a release

| Study | Detector and model | Archived code | Data and results |
|---|---|---|---|
| [2021 proof of concept](https://arxiv.org/html/2101.08578v3) | CMS-like Delphes simulation; GNN | [v1.1](https://zenodo.org/records/4559587) | [Simulated events and generation cards](https://zenodo.org/records/4559324) |
| [2021 workshop / 2023 CMS proceedings](https://arxiv.org/pdf/2203.00330) | CMS Run-3 simulation; TensorFlow GNN trained on PF candidates | [v1.4](https://zenodo.org/records/5520559) | Sample descriptions in the paper; CERN storage paths in the release instructions |
| [2022 CMS performance note](https://cds.cern.ch/record/2842375) | CMS simulation; GNN with simulation-derived targets | MLPF release remains to be identified | Sample and CMSSW identifiers in the [accompanying proceedings](https://arxiv.org/pdf/2303.17657) |
| [2024 CLIC study](https://arxiv.org/html/2309.06782v6) | Full CLICdet simulation; GNN and kernel transformer | [v1.6.2](https://zenodo.org/records/10893930) | [Raw data](https://zenodo.org/records/8260741), [TFDS](https://zenodo.org/records/8409592), [models and results](https://zenodo.org/records/10567397) |
| [2025 CLIC-to-CLD fine-tuning](https://arxiv.org/html/2503.00131v4) | Full CLICdet and CLD simulation; self-attention transformer | [v2.3.0](https://zenodo.org/records/14930299) | [CLIC TFDS](https://zenodo.org/records/15062717), [CLD ROOT](https://zenodo.org/records/14930758), [CLD TFDS](https://zenodo.org/records/14930610); partial samples as described below |
| [2026 CMS Run-3 paper](https://arxiv.org/html/2601.17554v2) | CMS simulation and collision data; self-attention transformer | [v2.5.0](https://zenodo.org/records/17519540) | Data availability follows CMS preservation policy |

## From Delphes to CMS

The 2021 proof of concept provides an openly archived simulation dataset and a matching `v1.1` code release. The dataset contains 50,000 top-pair and 5,000 QCD events at 14 TeV with average pileup 200, stored as compressed Python pickles. It includes Pythia and Delphes cards and a loading notebook. The paper reports training on 40,000 top-pair events.

Start with [`README_delphes.md` at `v1.1`](https://github.com/jpata/particleflow/blob/v1.1/README_delphes.md) and `parameters/delphes-gnn-skipconn.yaml`. These instructions cover conversion to TFRecord, training, evaluation, and timing.

The subsequent CMS implementation uses `v1.4`, explicitly cited in the proceedings. It trains a TensorFlow 2.6 GNN on reference PF candidates and exports the model to ONNX for validation within CMSSW. The paper describes a randomized 80:20 training/test split. The [`v1.4` CMS instructions](https://github.com/jpata/particleflow/blob/v1.4/README_cms.md) obtain training data from a personal EOS directory through `lxplus.cern.ch`, so this route requires CERN access. They also describe sample generation using local pileup files and a matching CMSSW release.

The 2022 CMS study introduces simulation-derived particle targets. Its [accompanying proceedings](https://arxiv.org/pdf/2303.17657) describe training samples, event-loss scans, and validation, and specify `CMSSW_12_3_0_pre6` together with minimum-bias and validation dataset names. A corresponding MLPF code tag and public event archive remain to be identified before assembling a complete reproduction setup.

## Full-simulation CLIC reconstruction

The 2024 CLIC study provides separate archives for simulation, processed training inputs, and trained models with evaluation results. Choose the artifact according to the stage you want to reproduce.

For simulation and preprocessing, the [raw-data record](https://zenodo.org/records/8260741), version `1.1`, contains small samples and a `dataset_full.txt` manifest pointing to bulk data at Jülich. For training, the [processed record](https://zenodo.org/records/8409592), version `v1.5.0`, contains four TFDS archives. The [README at `v2.3.0`](https://github.com/jpata/particleflow/blob/v2.3.0/README.md) links this processed dataset to the study.

For evaluation, start with the [results archive](https://zenodo.org/records/10567397), version `v2024.01`. Its `mlpf-clic-2023-results.tgz` contains trained weights, configurations, evaluation outputs, hyperparameter-search summaries, timing, and accelerator-scaling results.

The paper cites code archive [10893930](https://zenodo.org/records/10893930). The repository also links [10928968](https://zenodo.org/records/10928968). Both contain the same `v1.6.2` source: all 289 files match after removing the top-level archive directory, and both identify commit `cd2ed4100d928193b1fe71910c2819d8b4fa8d5d`. Use this version when following the paper; the related GitHub link to `v1.6` in record 10893930 is stale.

## Fine-tuning from CLIC to CLD

The 2025 transfer-learning study pairs code `v2.3.0` with CLIC pretraining data and CLD downstream data. The paper describes 24 million CLIC events and 4.2 million CLD events, each divided into training and validation with a 90:10 split.

The cited archives provide subsets of those samples. The [CLIC TFDS record](https://zenodo.org/records/15062717) contains two of ten subsets per sample. Its dataset label is `v2.5.0`, and its description explicitly declares compatibility with code `v2.3.0`. The [CLD TFDS record](https://zenodo.org/records/14930610) describes approximately 800,000 events. To work from simulation output, the [CLD ROOT record](https://zenodo.org/records/14930758) contains approximately 50,000 events together with generation and reconstruction settings.

These deposits support working with the study's data representation. Reproducing the full training-statistics scans also requires establishing access to the remaining samples and their split assignments.

## CMS Run-3 reconstruction

The published 2026 CMS paper cites `v2.5.0`. It describes the transformer training environment, including PyTorch 2.7.1 with ROCm 6.2.4, and the conditions used to measure inference performance. Use those settings together with the archived release when reconstructing the software environment.

The [journal availability statements](https://doi.org/10.1140/epjc/s10052-026-15754-5) refer to CMS data-preservation policy and provide no associated event-data release. The code statement links the public CMSSW repository. Access to the study's event samples therefore needs to be established separately from the MLPF code checkout.

An [earlier `v2.4.0` archive](https://zenodo.org/records/15573658) describes a preapproval freeze and links training and validation material. It records an earlier stage of the analysis; the published paper's code reference is `v2.5.0`.

## Verification and version identifiers

The links and release mappings were checked on **2026-10-30** against the cited papers, Zenodo metadata and file inventories, and historical repository files. Both CLIC code archives were downloaded, checksum-checked, and compared. Dataset payloads, trained models, external bulk-data hosts, and historical training runs were outside the verification scope.

The repository tags resolved to the following commits:

| Tag | Commit |
|---|---|
| `v1.1` | `d29b7e1c0b3d4879dced53bbdeaa0798c9e47a25` |
| `v1.4` | `4cf744f3f904829159a49db9b4a0b0efc974d975` |
| `v1.6.2` | `cd2ed4100d928193b1fe71910c2819d8b4fa8d5d` |
| `v2.3.0` | `03622d83d9e809f3b98db1e42134f02260d80d8d` |
| `v2.5.0` | `2cfab2615b968cfbf2ab028047d1d513e9101637` |
