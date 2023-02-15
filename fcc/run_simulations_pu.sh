#!/bin/bash

seq 1 2010 | parallel --gnu -j1 echo sbatch run_sim.sh {} p8_ee_tt_ecm365 p8_ee_gg_ecm365
seq 5000 7010 | parallel --gnu -j1 echo sbatch run_sim.sh {} p8_ee_qq_ecm365 p8_ee_gg_ecm365
