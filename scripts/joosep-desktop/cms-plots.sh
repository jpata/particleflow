#!/bin/bash

export PYTHONPATH=`pwd`
export QT_QPA_PLATFORM=offscreen
IMG=~/HEP-KBFI/singularity/pytorch.simg\:2025-09-01
NANO_PATH=/mnt/work/particleflow/CMSSW_15_0_5_mlpf_v2.6.0pre1_puppi_117d32/cuda_False

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
  --sample QCD_PU_13p6_v3 &

# singularity exec -B /mnt/work/ $IMG \
#   python3 mlpf/plotting/data_preparation.py \
#   --input-dir $NANO_PATH \
#   --sample TTbar_PU &

# singularity exec -B /mnt/work/ $IMG \
#   python3 mlpf/plotting/data_preparation.py \
#   --input-dir $NANO_PATH \
#   --sample TTbar_PU_13p6 &

# singularity exec -B /mnt/work/ $IMG \
#   python3 mlpf/plotting/data_preparation.py \
#   --input-dir $NANO_PATH \
#   --sample QCD_noPU_13p6 &

# singularity exec -B /mnt/work/ $IMG \
#   python3 mlpf/plotting/data_preparation.py \
#   --input-dir $NANO_PATH \
#   --sample TTbar_noPU_13p6 &

# singularity exec -B /mnt/work/ $IMG \
#   python3 mlpf/plotting/data_preparation.py \
#   --input-dir $NANO_PATH \
#   --sample PhotonJet_PU_13p6 &

# singularity exec -B /mnt/work/ $IMG \
#   python3 mlpf/plotting/data_preparation.py \
#   --input-dir $NANO_PATH \
#   --sample PhotonJet_noPU_13p6 &

wait


# Exctract jet corrections

sample=QCD_PU
singularity exec -B /mnt/work/ $IMG \
  python3 mlpf/plotting/corrections.py \
  --input-pf-parquet ${sample}_pf.parquet \
  --input-mlpf-parquet ${sample}_mlpf.parquet \
  --corrections-file jec_ak4_${sample}.npz \
  --output-dir ./plots/${sample}/ak4/jec \
  --jet-type ak4 \
  --sample-name ${sample} &

sample=QCD_PU_13p6
singularity exec -B /mnt/work/ $IMG \
  python3 mlpf/plotting/corrections.py \
  --input-pf-parquet ${sample}_pf.parquet \
  --input-mlpf-parquet ${sample}_mlpf.parquet \
  --corrections-file jec_ak4_${sample}.npz \
  --output-dir ./plots/${sample}/ak4/jec \
  --jet-type ak4 \
  --sample-name ${sample} &

sample=QCD_PU_13p6_v3
singularity exec -B /mnt/work/ $IMG \
  python3 mlpf/plotting/corrections.py \
  --input-pf-parquet ${sample}_pf.parquet \
  --input-mlpf-parquet ${sample}_mlpf.parquet \
  --corrections-file jec_ak4_${sample}.npz \
  --output-dir ./plots/${sample}/ak4/jec \
  --jet-type ak4 \
  --sample-name ${sample} &

# singularity exec -B /mnt/work/ $IMG \
#   python3 mlpf/plotting/corrections.py \
#   --input-pf-parquet ${sample}_pf.parquet \
#   --input-mlpf-parquet ${sample}_mlpf.parquet \
#   --corrections-file jec_ak8.npz \
#   --output-dir ./plots/${sample}/ak8/jec \
#   --jet-type ak8 \
#   --sample-name ${sample} &
wait

sample=QCD_PU
singularity exec -B /mnt/work/ $IMG \
  python3 mlpf/plotting/plot_validation.py \
  --input-pf-parquet ${sample}_pf.parquet \
  --input-mlpf-parquet ${sample}_mlpf.parquet \
  --corrections-file jec_ak4_${sample}.npz \
  --output-dir ./plots \
  --jet-type ak4 \
  --sample-name ${sample} \
  --fiducial-cuts eta_0_2p5 \
  --tev 14 &

singularity exec -B /mnt/work/ $IMG \
  python3 mlpf/plotting/plot_met_validation.py \
  --input-pf-parquet ${sample}_pf.parquet \
  --input-mlpf-parquet ${sample}_mlpf.parquet \
  --sample-name ${sample} \
  --output-dir ./plots \
  --tev 14 &
wait

#Do jet-level plots
# for sample in QCD_noPU_13p6 QCD_PU_13p6 TTbar_PU_13p6 TTbar_noPU_13p6 PhotonJet_noPU_13p6 PhotonJet_PU_13p6; do
for sample in QCD_PU_13p6 QCD_PU_13p6_v3; do
  # singularity exec -B /mnt/work/ $IMG \
  #   python3 mlpf/plotting/plot_validation.py \
  #   --input-pf-parquet ${sample}_pf.parquet \
  #   --input-mlpf-parquet ${sample}_mlpf.parquet \
  #   --corrections-file jec_ak4.npz \
  #   --output-dir ./plots \
  #   --jet-type ak4 \
  #   --sample-name ${sample} \
  #   --fiducial-cuts inclusive &

  singularity exec -B /mnt/work/ $IMG \
    python3 mlpf/plotting/plot_validation.py \
    --input-pf-parquet ${sample}_pf.parquet \
    --input-mlpf-parquet ${sample}_mlpf.parquet \
    --corrections-file jec_ak4_${sample}.npz \
    --output-dir ./plots \
    --jet-type ak4 \
    --sample-name ${sample} \
    --fiducial-cuts eta_0_2p5 \
    --tev 13.6 &

  # singularity exec -B /mnt/work/ $IMG \
  #   python3 mlpf/plotting/plot_validation.py \
  #   --input-pf-parquet ${sample}_pf.parquet \
  #   --input-mlpf-parquet ${sample}_mlpf.parquet \
  #   --corrections-file jec_ak4.npz \
  #   --output-dir ./plots \
  #   --jet-type ak4 \
  #   --sample-name ${sample} \
  #   --fiducial-cuts eta_2p5_3 &

  # singularity exec -B /mnt/work/ $IMG \
  #   python3 mlpf/plotting/plot_validation.py \
  #   --input-pf-parquet ${sample}_pf.parquet \
  #   --input-mlpf-parquet ${sample}_mlpf.parquet \
  #   --corrections-file jec_ak4.npz \
  #   --output-dir ./plots \
  #   --jet-type ak4 \
  #   --sample-name ${sample} \
  #   --fiducial-cuts eta_3_5 &

  # singularity exec -B /mnt/work/ $IMG \
  #   python3 mlpf/plotting/plot_validation.py \
  #   --input-pf-parquet ${sample}_pf.parquet \
  #   --input-mlpf-parquet ${sample}_mlpf.parquet \
  #   --corrections-file jec_ak8.npz \
  #   --output-dir ./plots \
  #   --jet-type ak8 \
  #   --sample-name ${sample} \
  #   --fiducial-cuts inclusive &

  singularity exec -B /mnt/work/ $IMG \
    python3 mlpf/plotting/plot_met_validation.py \
    --input-pf-parquet ${sample}_pf.parquet \
    --input-mlpf-parquet ${sample}_mlpf.parquet \
    --sample-name ${sample} \
    --output-dir ./plots \
    --tev 13.6 &

wait
done

# #Do the 14 TeV to 13.6 TeV comparison
# for sample in QCD_PU_13p6 TTbar_PU_13p6; do
#   singularity exec -B /mnt/work $IMG \
#     python mlpf/plotting/plot_jet_response_comparison_v1.py \
#       --input-pf-parquet ${sample}_pf.parquet \
#       --input-mlpf-parquet ${sample}_mlpf.parquet \
#       --output-dir ./plots \
#       --sample-name $sample \
#       --jet-type ak4 \
#       --tev 13.6 &
# done
# wait

# for sample in QCD_PU; do
#   singularity exec -B /mnt/work $IMG \
#     python mlpf/plotting/plot_jet_response_comparison_v1.py \
#       --input-pf-parquet ${sample}_pf.parquet \
#       --input-mlpf-parquet ${sample}_mlpf.parquet \
#       --output-dir ./plots \
#       --sample-name $sample \
#       --jet-type ak4 \
#       --tev 14 &

#   singularity exec -B /mnt/work $IMG \
#     python mlpf/plotting/plot_jet_response_comparison_v1.py \
#       --input-pf-parquet ${sample}_13p6_pf.parquet \
#       --input-mlpf-parquet ${sample}_13p6_mlpf.parquet \
#       --output-dir ./plots \
#       --sample-name ${sample}_13p6 \
#       --jet-type ak4 \
#       --tev 13.6 &

#   singularity exec -B /mnt/work $IMG \
#     python mlpf/plotting/plot_jet_response_comparison_v1.py \
#       --input-pf-parquet ${sample}_13p6_v2_pf.parquet \
#       --input-mlpf-parquet ${sample}_13p6_v2_mlpf.parquet \
#       --output-dir ./plots \
#       --sample-name ${sample}_13p6_v2 \
#       --jet-type ak4 \
#       --tev 13.6 &
# done
# wait

# for sample in QCD_PU; do
#   singularity exec -B /mnt/work $IMG \
#     python mlpf/plotting/plot_jet_response_comparison_v2.py \
#       --input-sample1-pf-parquet ${sample}_13p6_pf.parquet \
#       --input-sample1-mlpf-parquet ${sample}_13p6_mlpf.parquet \
#       --input-sample2-pf-parquet ${sample}_13p6_v2_pf.parquet \
#       --input-sample2-mlpf-parquet ${sample}_13p6_v2_mlpf.parquet \
#       --output-dir ./plots \
#       --sample-name $sample \
#       --sample1-label paper \
#       --sample2-label onlyCOM \
#       --jet-type ak4 &
# done
# wait
