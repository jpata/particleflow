#!/bin/bash


rm -f files_to_copy.txt
maxfiles=100
path=/home/joosep/particleflow/experiments/pyg-cms_20250410_144744_012331/./preds_checkpoint-10-3.539432/
targetpath=/scratch/persistent/joosep/huggingface/particleflow/cms/v2.4.0/pyg-cms_20250410_144744_012331/

mkdir -p $targetpath

samplestocopy=(
    "cms_pf_qcd"
    "cms_pf_ttbar"
    "cms_pf_ztt"
)

for sample in "${samplestocopy[@]}"; do
    find "$path/$sample" -type f | sort | head -n$maxfiles >> files_to_copy.txt
done

rsync --progress --relative --files-from files_to_copy.txt / $targetpath
