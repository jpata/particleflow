#!/bin/bash

data_path=/mnt/work/particleflow/CMSSW_15_0_5_mlpf_v2.6.0pre1_p05_ccd0c7/cuda_False/
max_files=200

mkdir out
papermill notebooks/cms/cmssw-validation.ipynb out/cmssw-validation-ttbar-pu.ipynb \
	--cwd notebooks/cms \
	-p path $data_path  \
	-p folder TTbar_PU_13p6 \
	-p physics_process cms_pf_ttbar \
	-p max_files $max_files \
	--log-output

papermill notebooks/cms/cmssw-validation.ipynb out/cmssw-validation-qcd-pu.ipynb \
	--cwd notebooks/cms \
	-p path $data_path  \
	-p folder QCD_PU_13p6 \
	-p physics_process cms_pf_qcd \
	-p max_files $max_files \
	--log-output

papermill notebooks/cms/cmssw-validation.ipynb out/cmssw-validation-ttbar-nopu.ipynb \
	--cwd notebooks/cms \
	-p path $data_path  \
	-p folder TTbar_noPU_13p6 \
	-p physics_process cms_pf_ttbar_nopu \
	-p max_files $max_files \
	--log-output

papermill notebooks/cms/cmssw-validation.ipynb out/cmssw-validation-qcd-nopu.ipynb \
	--cwd notebooks/cms \
	-p path $data_path  \
	-p folder QCD_noPU_13p6 \
	-p physics_process cms_pf_qcd_nopu \
	-p max_files $max_files \
	--log-output
