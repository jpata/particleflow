#!/bin/bash
python3 mlpf/pipeline.py train -c $1 --plot-freq 1 --num-cpus 16 --batch-multiplier 10
