#!/bin/bash

#\ls -1 data/TTbar_run3/*ev*.npz | head -n500 | parallel --gnu -j5 -n5 \
#    singularity exec -B /storage ~/gpuservers/singularity/images/over_edge.simg python3 test/benchmark_solution.py {}
find ./data/NuGun_run3/ -name "*ev*.npz" | sort | head -n500 | parallel --gnu -j5 -n5 \
    singularity exec -B /storage ~/gpuservers/singularity/images/over_edge.simg python3 test/benchmark_solution.py {}
#find ./data/QCD_run3/ -name "*ev*.npz" | sort | head -n500 | parallel --gnu -j5 -n5 \
#    singularity exec -B /storage ~/gpuservers/singularity/images/over_edge.simg python3 test/benchmark_solution.py {}
