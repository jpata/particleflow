#!/bin/bash
set -e

rm -f files_to_copy.txt
maxfiles=100
path=/home/joosep/particleflow/experiments/pyg-cms_20250722_101813_274478/./preds_checkpoint-10-3.812332
targetpath=/scratch/persistent/joosep/huggingface/particleflow/cms/v2.6.0pre1/pyg-cms_20250722_101813_274478/

mkdir -p $targetpath
cp $path/../* $targetpath/
cp -R $path/../checkpoints $targetpath/
cp -R $path/../history $targetpath/
cp -R $path/../runs $targetpath/

samplestocopy=(
    "cms_pf_ttbar"
    "cms_pf_qcd"
    "cms_pf_ztt"
)

for sample in "${samplestocopy[@]}"; do
    find "$path/$sample" -type f | sort | head -n$maxfiles >> files_to_copy.txt
done

rsync --progress --relative --files-from files_to_copy.txt / $targetpath
