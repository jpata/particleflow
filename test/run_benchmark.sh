#!/bin/bash

\ls -1 data/TTbar/191009_155100/*ev*.npz | head -n500 | parallel --gnu -j24 -n5 \
    singularity exec -B /storage ~/gpuservers/singularity/images/over_edge.simg python3 test/benchmark_solution.py {}
