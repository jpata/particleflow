[![CI](https://github.com/jpata/particleflow/workflows/CI/badge.svg)](https://github.com/jpata/particleflow/actions)

# Overview
MLPF focuses on developing full event reconstruction based on computationally scalable and flexible end-to-end ML models.

<p float="left">
  <img src="images/schematic.png" alt="High-level overview" width="600"/>
</p>

### Dataset compatibility table
The following table specifies which version of the jpata/particleflow software should be used with which version of the tensorflow datasets.

| Code  | CMS dataset | CLIC dataset |
| ----- | ----------- | ------------ |
| [1.9.0](https://github.com/jpata/particleflow/releases/v1.9.0) | 2.4.0    | 2.2.0    |
| [2.0.0](https://github.com/jpata/particleflow/releases/v2.0.0) | 2.4.0    | 2.3.0    |
| [2.1.0](https://github.com/jpata/particleflow/releases/v2.1.0) | 2.5.0    | 2.5.0    |
| [2.2.0](https://github.com/jpata/particleflow/releases/v2.2.0) | 2.5.0    | 2.5.0    |

## MLPF on open datasets

<p float="left">
  <img src="images/commphys_featured_image.png" alt="PF reconstruction" width="600"/>
</p>

  - paper: https://doi.org/10.1038/s42005-024-01599-5
  - code: https://doi.org/10.5281/zenodo.10893930
  - dataset: https://doi.org/10.5281/zenodo.8409592
  - results: https://doi.org/10.5281/zenodo.10567397
  - weights: https://huggingface.co/jpata/particleflow/tree/main/clic/clusters/v1.6

### Open datasets:
The following datasets are available to reproduce the studies. They include full Geant4 simulation and reconstruction based on the CLIC detector. We have no affiliation with the CLIC collaboration, therefore these datasets are to be used only for computational studies and come with no warranty.

- MLPF-CLIC, raw data: https://zenodo.org/records/8260741 or https://www.coe-raise.eu/od-pfr
- MLPF-CLIC, processed for ML, tracks and clusters: https://zenodo.org/records/8409592
- MLPF-CLIC, processed for ML, tracks and hits: https://zenodo.org/records/8414225

## MLPF development in CMS

<p float="left">
  <img src="images/cms/ev_pf.png" alt="PF reconstruction" width="300"/>
  <img src="images/cms/ev_mlpf.png" alt="MLPF reconstruction" width="300"/>
</p>

<p float="left">
  <img src="images/cms/ak4jet_puppi_pt_ttbar.png" alt="PUPPI jets in ttbar" width="300"/>
</p>

  - ACAT 2022:
    - CERN-CMS-DP-2022-061, http://cds.cern.ch/record/2842375
  - ACAT 2021:
    - J. Phys. Conf. Ser. 2438 012100, http://dx.doi.org/10.1088/1742-6596/2438/1/012100
    - CERN-CMS-DP-2021-030, https://cds.cern.ch/record/2792320

## Initial development with Delphes

<p float="left">
  <img src="images/delphes/num_particles.png" alt="Number of reconstructed particles" width="250"/>
  <img src="images/delphes/inference_time.png" alt="Scaling of the inference time" width="300"/>
</p>

  - paper: https://doi.org/10.1140/epjc/s10052-021-09158-w
  - code: https://doi.org/10.5281/zenodo.4559587
  - dataset: https://doi.org/10.5281/zenodo.4559324
