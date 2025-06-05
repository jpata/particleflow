[![CI](https://github.com/jpata/particleflow/workflows/CI/badge.svg)](https://github.com/jpata/particleflow/actions)

# Overview
MLPF focuses on developing full event reconstruction based on computationally scalable and flexible end-to-end ML models.

<p float="left">
  <img src="images/schematic.png" alt="High-level overview" width="600"/>
</p>

### Dataset compatibility table
The following table specifies which version of the jpata/particleflow software should be used with which version of the tensorflow datasets.

| Code  | CMS dataset | CLIC dataset | CLD dataset |
| ----- | ----------- | ------------ | ----------- |
| [1.9.0](https://github.com/jpata/particleflow/releases/v1.9.0) | 2.4.0    | 2.2.0    | NA |
| [2.0.0](https://github.com/jpata/particleflow/releases/v2.0.0) | 2.4.0    | 2.3.0    | NA |
| [2.1.0](https://github.com/jpata/particleflow/releases/v2.1.0) | 2.5.0    | 2.5.0    | NA |
| [2.2.0](https://github.com/jpata/particleflow/releases/v2.2.0) | 2.5.0    | 2.5.0    | 2.5.0 |
| [2.3.0](https://github.com/jpata/particleflow/releases/v2.3.0) | 2.5.0    | 2.5.0    | 2.5.0 |
| [2.4.0](https://github.com/jpata/particleflow/releases/v2.4.0) | 2.6.0    | 2.5.0    | 2.5.0 |

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

# Citations and reuse

You are welcome to reuse the code in your work in accordance with the [license](https://github.com/jpata/particleflow/blob/main/LICENSE).

For academic work, please consider citing the following papers:
- initial idea with scalable GNN, code [v1.1](https://zenodo.org/records/4559587): https://doi.org/10.1140/epjc/s10052-021-09158-w
- improved event-level performance in full simulation, code [v1.6.2](https://zenodo.org/records/10928968): https://doi.org/10.1038/s42005-024-01599-5
- studies in CMS: https://cds.cern.ch/record/2792320, http://dx.doi.org/10.1088/1742-6596/2438/1/012100, http://cds.cern.ch/record/2842375

If you use the code in a significant way for research purposes, please consider citing the [tagged version](https://zenodo.org/search?q=parent.id%3A4452541&f=allversions%3Atrue&l=list&p=1&s=10&sort=version) that you used, for example:
- Joosep Pata, Eric Wulff, Farouk Mokhtar, Javier Duarte, Aadi Tepper, Ka Wa Ho, & Lars Sørlie. (2025). jpata/particleflow: v2.2.0 (v2.2.0). Zenodo. https://doi.org/10.5281/zenodo.14650991

If you use the datasets prepared by the MLPF team for academic work, please cite the [appropriate dataset](https://zenodo.org/search?q=mlpf&f=allversions%3Atrue&f=resource_type%3Adataset&l=list&p=1&s=10&sort=version) via the zenodo link, as well as the corresponding paper.

At the moment, we are unable to release work-in-progress datasets before the corresponding academic publication is out.
If you have a collaboration idea that does not fit into the above categories, please [get in touch](https://github.com/jpata/particleflow/discussions/categories/general)!
