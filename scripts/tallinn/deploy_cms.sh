#!/bin/bash


rm -f files_to_copy.txt
maxfiles=100
path=experiments/pyg-cms_20241101_090645_682892/./preds_checkpoint-08-2.986092
targetpath=/home/joosep/huggingface/particleflow/cms/v2.1.0/pyg-cms_20241101_090645_682892/

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

rsync --progress --relative --files-from files_to_copy.txt `pwd` $targetpath
