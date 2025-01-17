#!/bin/bash

if [ -n "$LXPLUSUSER" ]; then
    rsync --ignore-existing --progress --relative --files-from scripts/files_to_copy.txt / $LXPLUSUSER@lxplus.cern.ch:/eos/user/j/jpata/www/mlpf/clic_key4hep/
else
  echo "Please define LXPLUSUSER"
fi
