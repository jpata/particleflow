#!/bin/bash


rm -f files_to_copy.txt
maxfiles=100
path=/local/joosep/mlpf/results/cms/pyg-cms_20241212_101648_120237/./preds_checkpoint-05-3.498507
targetpath=/scratch/persistent/joosep/huggingface/particleflow/cms/v2.2.0/pyg-cms_20241212_101648_120237/

samplestocopy=(
    "cms_pf_qcd"
    "cms_pf_qcd_nopu"
    "cms_pf_ttbar"
    "cms_pf_ttbar_nopu"
    "cms_pf_ztt"
    "cms_pf_ztt_nopu"
)

for sample in "${samplestocopy[@]}"; do
    find "$path/$sample" -type f | sort | head -n$maxfiles >> files_to_copy.txt
done

rsync --progress --relative --files-from files_to_copy.txt / $targetpath
