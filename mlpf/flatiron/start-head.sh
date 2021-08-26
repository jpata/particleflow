#!/bin/bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

echo "starting ray head node"
# Launch the head node
mkdir -p "/mnt/ceph/users/ewulff/nvidia_smi_logs/$3_$2"
nvidia-smi --query-gpu=timestamp,name,pci.bus_id,pstate,power.draw,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 -f "/mnt/ceph/users/ewulff/nvidia_smi_logs/$3_$2/head.csv" &
ray start --head --node-ip-address=$1 --port=6379
sleep infinity
