#!/bin/bash

if [ -n "$LXPLUSUSER" ]; then
    rsync --progress --relative --files-from scripts/files_to_copy.txt / $LXPLUSUSER@lxplus.cern.ch:/eos/user/j/jpata/www/mlpf/cms/
else
  echo "Please define LXPLUSUSER"
fi
