#!/bin/bash
set -x
set -e

EXPDIR=experiments/clic_20230402_114621_120703.gpu0.local/evaluation/epoch_92/
OUTDIR=/home/joosep/Dropbox/Apps/Overleaf/MLPF-CLIC-2023/figures


for f in loss.pdf cls_loss.pdf pt_loss.pdf energy_loss.pdf eta_loss.pdf sin_phi_loss.pdf cos_phi_loss.pdf charge_loss.pdf; do
    cp $EXPDIR/../../$f $OUTDIR/
done

cp $EXPDIR/clic_edm_qq_pf/plots/jet_res_bins_0p5_1p5.pdf $OUTDIR/qq/
cp $EXPDIR/clic_edm_qq_pf/plots/met_res_bins_0_2.pdf $OUTDIR/qq/

cp $EXPDIR/clic_edm_ttbar_pf/plots/jet_res_bins_0p5_1p5.pdf $OUTDIR/ttbar/
cp $EXPDIR/clic_edm_ttbar_pf/plots/met_res_bins_0_2.pdf $OUTDIR/ttbar/
cp $EXPDIR/clic_edm_ttbar_pf/plots/jet_response_med_iqr.pdf $OUTDIR/ttbar/
cp $EXPDIR/clic_edm_ttbar_pf/plots/met_response_med_iqr.pdf $OUTDIR/ttbar/

cp $EXPDIR/clic_edm_zh_tautau_pf/plots/jet_res_bins_0p5_1p5.pdf $OUTDIR/zh/
cp $EXPDIR/clic_edm_zh_tautau_pf/plots/met_res_bins_0_2.pdf $OUTDIR/zh/

cp $EXPDIR/clic_edm_ww_fullhad_pf/plots/jet_res_bins_0p5_1p5.pdf $OUTDIR/ww_fullhad/
cp $EXPDIR/clic_edm_ww_fullhad_pf/plots/met_res_bins_0_5.pdf $OUTDIR/ww_fullhad/

for f in gen_particle_pid_pt.pdf gen_particle_pt.pdf loss.pdf num_clusters.pdf num_tracks.pdf pf_particle_pid_pt.pdf pf_particle_pt.pdf; do
    cp notebooks/plots/clic/$f $OUTDIR/inputs/
done
