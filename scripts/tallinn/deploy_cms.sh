#!/bin/bash


rm -f files_to_copy.txt
maxfiles=100
path=/home/joosep/particleflow/experiments/pyg-cms_20250122_185427_365548/./preds_checkpoint-10-3.541986/
targetpath=/scratch/persistent/joosep/huggingface/particleflow/cms/v2.3.0/pyg-cms_20250122_185427_365548/

samplestocopy=(
    "cms_pf_qcd"
    "cms_pf_ttbar"
    "cms_pf_ztt"
    "cms_pf_single_ele"
)
#    "cms_pf_ttbar_nopu"
#    "cms_pf_qcd_nopu"
#    "cms_pf_ztt_nopu"

for sample in "${samplestocopy[@]}"; do
    find "$path/$sample" -type f | sort | head -n$maxfiles >> files_to_copy.txt
done

rsync --progress --relative --files-from files_to_copy.txt / $targetpath
