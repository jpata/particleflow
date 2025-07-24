#!/bin/bash
export APPTAINER_BINDPATH=/afs,/cvmfs,/cvmfs/grid.cern.ch/etc/grid-security:/etc/grid-security,/cvmfs/grid.cern.ch/etc/grid-security/vomses:/etc/vomses,/eos,/etc/pki/ca-trust,/etc/tnsnames.ora,/run/user,/var/run/user

apptainer exec --nv --env PYTHONPATH=. /eos/user/j/jpata/www/pytorch.simg:2024-12-03 python3 mlpf/pipeline.py --config parameters/pytorch/pyg-cms-ttbar-nopu.yaml --gpus 1 --data-dir /eos/user/j/jpata/mlpf/tensorflow_datasets/cms --attention-type math --train --ntrain 1000 --ntest 1000 --nvalid 1000 --num-convs 1 --dtype float32
