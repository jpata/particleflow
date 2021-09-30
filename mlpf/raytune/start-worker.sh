#!/bin/bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

echo "starting ray worker node"
mkdir -p "$5$4_$2"
nvidia-smi --query-gpu=timestamp,name,pci.bus_id,pstate,power.draw,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 -f "$5$4_$2/worker_$3.csv" &
ray start --address $1
sleep infinity
