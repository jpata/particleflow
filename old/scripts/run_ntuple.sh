#!/bin/bash

usage() {
    echo "Usage: ./run_ntuple.sh /path/to/crab/output/DATASET ./data/DATASET"
    exit 0
}

INPUT_DIR=$1
if [ -z $INPUT_DIR ]; then
    usage
fi

OUTPUT_DIR=$2
if [ -z $OUTPUT_DIR ]; then
    usage
fi

mkdir -p $OUTPUT_DIR

INFILES=`find $INPUT_DIR -name "step3_AOD_*.root"`

echo $INFILES | xargs -n 1 -P 24 python test/ntuplizer.py $OUTPUT_DIR
