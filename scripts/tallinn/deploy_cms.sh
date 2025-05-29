#!/bin/bash

rm -f files_to_copy.txt
maxfiles=100
path=/home/joosep/particleflow/experiments/pyg-cms_20250517_232752_544969/./preds_checkpoint-08-3.863894
targetpath=/scratch/persistent/joosep/huggingface/particleflow/cms/v2.5.0/pyg-cms_20250517_232752_544969/

mkdir -p $targetpath
cp $path/../* $targetpath/
cp -R $path/../checkpoints $targetpath/
cp -R $path/../history $targetpath/
cp -R $path/../runs $targetpath/

samplestocopy=(
    "cms_pf_ttbar"
    "cms_pf_qcd"
    "cms_pf_ztt"
    "cms_pf_ttbar_nopu"
    "cms_pf_qcd_nopu"
    "cms_pf_ztt_nopu"
)

for sample in "${samplestocopy[@]}"; do
    find "$path/$sample" -type f | sort | head -n$maxfiles >> files_to_copy.txt
done

rsync --progress --relative --files-from files_to_copy.txt / $targetpath
