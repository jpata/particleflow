#!/bin/bash

export PYTHONPATH=`pwd`
IMG=~/HEP-KBFI/singularity/pytorch.simg\:2025-09-01
NANO_PATH=/mnt/work/particleflow/CMSSW_15_0_5_mlpf_v2.6.0pre1_puppi_2372e2/cuda_False

singularity exec -B /mnt/work/ $IMG  \
  python3 mlpf/plotting/data_preparation.py \
  --input-dir $NANO_PATH \
  --sample QCD_PU &

singularity exec -B /mnt/work/ $IMG \
  python3 mlpf/plotting/data_preparation.py \
  --input-dir $NANO_PATH \
  --sample QCD_PU_13p6 &

singularity exec -B /mnt/work/ $IMG \
  python3 mlpf/plotting/data_preparation.py \
  --input-dir $NANO_PATH \
  --sample TTbar_PU &

singularity exec -B /mnt/work/ $IMG \
  python3 mlpf/plotting/data_preparation.py \
  --input-dir $NANO_PATH \
  --sample TTbar_PU_13p6 &

singularity exec -B /mnt/work/ $IMG \
  python3 mlpf/plotting/data_preparation.py \
  --input-dir $NANO_PATH \
  --sample QCD_noPU_13p6 &

singularity exec -B /mnt/work/ $IMG \
  python3 mlpf/plotting/data_preparation.py \
  --input-dir $NANO_PATH \
  --sample TTbar_noPU_13p6 &

singularity exec -B /mnt/work/ $IMG \
  python3 mlpf/plotting/data_preparation.py \
  --input-dir $NANO_PATH \
  --sample PhotonJet_PU_13p6 &

singularity exec -B /mnt/work/ $IMG \
  python3 mlpf/plotting/data_preparation.py \
  --input-dir $NANO_PATH \
  --sample PhotonJet_noPU_13p6 &
wait


# # Exctract jet corrections
sample=QCD_PU_13p6
singularity exec -B /mnt/work/ $IMG \
  python3 mlpf/plotting/corrections.py \
  --input-pf-parquet ${sample}_pf.parquet \
  --input-mlpf-parquet ${sample}_mlpf.parquet \
  --corrections-file jec_ak4.npz \
  --output-dir ./plots/${sample}/ak4/jec \
  --jet-type ak4 \
  --sample-name ${sample} &

singularity exec -B /mnt/work/ $IMG \
  python3 mlpf/plotting/corrections.py \
  --input-pf-parquet ${sample}_pf.parquet \
  --input-mlpf-parquet ${sample}_mlpf.parquet \
  --corrections-file jec_ak8.npz \
  --output-dir ./plots/${sample}/ak8/jec \
  --jet-type ak8 \
  --sample-name ${sample} &
wait

# # # #Do jet-level plots
# for sample in QCD_noPU_13p6 QCD_PU_13p6 TTbar_PU_13p6 TTbar_noPU_13p6 PhotonJet_noPU_13p6 PhotonJet_PU_13p6; do
for sample in TTbar_PU_13p6 QCD_PU_13p6; do
  singularity exec -B /mnt/work/ $IMG \
    python3 mlpf/plotting/plot_validation.py \
    --input-pf-parquet ${sample}_pf.parquet \
    --input-mlpf-parquet ${sample}_mlpf.parquet \
    --corrections-file jec_ak4.npz \
    --output-dir ./plots \
    --jet-type ak4 \
    --sample-name ${sample} \
    --fiducial-cuts inclusive &

  singularity exec -B /mnt/work/ $IMG \
    python3 mlpf/plotting/plot_validation.py \
    --input-pf-parquet ${sample}_pf.parquet \
    --input-mlpf-parquet ${sample}_mlpf.parquet \
    --corrections-file jec_ak4.npz \
    --output-dir ./plots \
    --jet-type ak4 \
    --sample-name ${sample} \
    --fiducial-cuts eta_0_2p5 &

  singularity exec -B /mnt/work/ $IMG \
    python3 mlpf/plotting/plot_validation.py \
    --input-pf-parquet ${sample}_pf.parquet \
    --input-mlpf-parquet ${sample}_mlpf.parquet \
    --corrections-file jec_ak4.npz \
    --output-dir ./plots \
    --jet-type ak4 \
    --sample-name ${sample} \
    --fiducial-cuts eta_2p5_3 &

  singularity exec -B /mnt/work/ $IMG \
    python3 mlpf/plotting/plot_validation.py \
    --input-pf-parquet ${sample}_pf.parquet \
    --input-mlpf-parquet ${sample}_mlpf.parquet \
    --corrections-file jec_ak4.npz \
    --output-dir ./plots \
    --jet-type ak4 \
    --sample-name ${sample} \
    --fiducial-cuts eta_3_5 &

  singularity exec -B /mnt/work/ $IMG \
    python3 mlpf/plotting/plot_validation.py \
    --input-pf-parquet ${sample}_pf.parquet \
    --input-mlpf-parquet ${sample}_mlpf.parquet \
    --corrections-file jec_ak8.npz \
    --output-dir ./plots \
    --jet-type ak8 \
    --sample-name ${sample} \
    --fiducial-cuts inclusive &

    singularity exec -B /mnt/work/ $IMG \
      python3 mlpf/plotting/plot_met_validation.py \
      --input-pf-parquet ${sample}_pf.parquet \
      --input-mlpf-parquet ${sample}_mlpf.parquet \
      --sample-name ${sample} \
      --output-dir ./plots &
  wait
done

#Do the 14 TeV to 13.6 TeV comparison
for sample in QCD_PU_13p6 TTbar_PU_13p6; do
  singularity exec -B /mnt/work $IMG \
    python mlpf/plotting/plot_jet_response_comparison_v1.py \
      --input-pf-parquet ${sample}_pf.parquet \
      --input-mlpf-parquet ${sample}_mlpf.parquet \
      --output-dir ./plots \
      --sample-name $sample \
      --jet-type ak4 \
      --tev 13.6 &
done
wait

for sample in QCD_PU TTbar_PU; do
  singularity exec -B /mnt/work $IMG \
    python mlpf/plotting/plot_jet_response_comparison_v1.py \
      --input-pf-parquet ${sample}_pf.parquet \
      --input-mlpf-parquet ${sample}_mlpf.parquet \
      --output-dir ./plots \
      --sample-name $sample \
      --jet-type ak4 \
      --tev 14 &
done
wait

# #Do the 14 TeV to 13.6 TeV comparison
for sample in QCD_PU TTbar_PU; do
  singularity exec -B /mnt/work $IMG \
    python mlpf/plotting/plot_jet_response_comparison_v2.py \
      --input-13p6-tev-pf-parquet ${sample}_13p6_pf.parquet \
      --input-13p6-tev-mlpf-parquet ${sample}_13p6_mlpf.parquet \
      --input-14-tev-pf-parquet ${sample}_pf.parquet \
      --input-14-tev-mlpf-parquet ${sample}_mlpf.parquet \
      --output-dir ./plots \
      --sample-name $sample \
      --jet-type ak4 &

  singularity exec -B /mnt/work $IMG \
    python mlpf/plotting/plot_jet_response_comparison_v2.py \
      --input-13p6-tev-pf-parquet ${sample}_13p6_pf.parquet \
      --input-13p6-tev-mlpf-parquet ${sample}_13p6_mlpf.parquet \
      --input-14-tev-pf-parquet ${sample}_pf.parquet \
      --input-14-tev-mlpf-parquet ${sample}_mlpf.parquet \
      --output-dir ./plots \
      --sample-name $sample \
      --jet-type ak8 &
done
wait
