"""Check compute-node data access before invoking the generic scenario runner."""

import os
import sys
from pathlib import Path

from mlpf.training_scenarios import main

if __name__ == "__main__":
    data_dir = Path(os.environ["MLPF_CLD_DATA_DIR"])
    if not data_dir.is_dir():
        sys.exit(
            f"CLD TFDS directory is unavailable on this node: {data_dir}\n"
            "alphagh01 does not currently mount /mnt/lustre. Set MLPF_CLD_DATA_DIR "
            "to a compute-visible TFDS root (for example under /projects)."
        )
    main(sys.argv[1:])
