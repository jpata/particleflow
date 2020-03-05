#!/bin/bash

source /opt/hepsim.sh
export LD_LIBRARY_PATH=/opt/hepsim/delphes:$LD_LIBRARY_PATH
export ROOT_INCLUDE_PATH=/opt/hepsim/delphes:/opt/hepsim/delphes/external

python ntuplizer.py 
