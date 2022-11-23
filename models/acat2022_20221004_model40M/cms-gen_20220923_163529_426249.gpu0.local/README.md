# CMS ACAT2022 MLPF model

- Parameters: 42573839
- Dashboard: https://www.comet.com/jpata/particleflow-tf/10404749e16c444585a08a2c97575c94?experiment-tab=code&file=769cca76276746bcb37b221b77d4cbcf&viewId=diAOFXE1zsuCTaLOqnlfpfUQL
- Poster: https://indico.cern.ch/event/1106990/contributions/4998026/
- Talk: https://indico.cern.ch/event/1159913/contributions/5101642/

Datasets:
  - TTbar Run3+PU, 100k
  - ZTauTau Run3+PU, 100k
  - QCD Run3+PU, 100k
  - QCD HighPt Run3+PU, 100k
  - SingleElectron, 10k
  - SingleGamma, 10k
  - SingleNeutron, 10k
  - SinglePi0, 10k

Script to generate datasets: https://github.com/jpata/particleflow/blob/main/mlpf/data_cms/prepare_args.py
Software for datasets and inference: `CMSSW_12_3_0_pre6 + jpata/cmssw:547a0fce7251bfaa6e855aef068f5a45c2d321ec`
