#!/bin/bash
export LD_LIBRARY_PATH=/opt/rocm-5.3.2/lib:$LD_LIBRARY_PATH
python3 mlpf/pipeline.py train -c $1 --plot-freq 5 --num-cpus 8 --batch-multiplier 30
