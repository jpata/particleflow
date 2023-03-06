[![CI](https://github.com/jpata/particleflow/workflows/CI/badge.svg)](https://github.com/jpata/particleflow/actions)

# MLPF overview
<p float="left">
  <img src="images/schematic.png" alt="High-level overview" width="600"/>
</p>

## MLPF with CMS


<p float="left">
  <img src="images/cms/ev_pf.png" alt="PF reconstruction" width="300"/>
  <img src="images/cms/ev_mlpf.png" alt="MLPF reconstruction" width="300"/>
</p>

<p float="left">
  <img src="images/cms/ak4jet_puppi_pt_ttbar.png" alt="PUPPI jets in ttbar" width="300"/>
</p>

  - ACAT 2022: http://cds.cern.ch/record/2842375
  - ACAT 2021: http://cds.cern.ch/record/2802826

## MLPF with Delphes

  - paper: https://doi.org/10.1140/epjc/s10052-021-09158-w
    - code: https://doi.org/10.5281/zenodo.4559587
    - dataset: https://doi.org/10.5281/zenodo.4559324

Short instructions with a single test file in [notebooks/delphes-tf-mlpf-quickstart.ipynb](notebooks/delphes-tf-mlpf-quickstart.ipynb).

Long instructions for reproducing the full training from scratch in [README_delphes.md](README_delphes.md).
The plots can be generated using the notebook [delphes/delphes_model_analysis.ipynb](delphes/delphes_model_analysis.ipynb).
