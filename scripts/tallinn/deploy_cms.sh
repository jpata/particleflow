#!/bin/bash

rm -f files_to_copy.txt
maxfiles=100
path=/home/joosep/particleflow/experiments/pyg-cms_20250510_121014_108409/./preds_checkpoint-20-3.352647
targetpath=/scratch/persistent/joosep/huggingface/particleflow/cms/v2.4.1/pyg-cms_20250510_121014_108409/

mkdir -p $targetpath
cp $path/../* $targetpath/
cp -R $path/../checkpoints $targetpath/
cp -R $path/../history $targetpath/
cp -R $path/../runs $targetpath/

samplestocopy=(
    "cms_pf_ttbar"
)

for sample in "${samplestocopy[@]}"; do
    find "$path/$sample" -type f | sort | head -n$maxfiles >> files_to_copy.txt
done

rsync --progress --relative --files-from files_to_copy.txt / $targetpath
