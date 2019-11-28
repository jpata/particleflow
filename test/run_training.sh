#!/bin/bash

SINGULARITY_IMAGE=/storage/user/jpata/gpuservers/singularity/images/over_edge.simg

#singularity exec --nv -B /storage $SINGULARITY_IMAGE python3 test/train_gnn.py
#singularity exec --nv -B /storage $SINGULARITY_IMAGE python3 test/draw_graphs.py
singularity exec --nv -B /storage $SINGULARITY_IMAGE python3 test/benchmark_solution.py /storage/user/jduarte/particleflow/graph_data/raw/step3_AOD_1_ev1*.npz
