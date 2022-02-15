#!/bin/bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

echo "starting ray head node"
# Launch the head node
# mkdir -p "$4$3_$2"
# nvidia-smi --query-gpu=timestamp,name,pci.bus_id,pstate,power.draw,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 -f "$4$3_$2/head.csv" &
ray start --head --node-ip-address=$1 --dashboard-host 0.0.0.0 --port=6379 --block
sleep infinity
