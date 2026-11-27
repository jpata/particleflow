#!/usr/bin/env python3
"""Download a TFDS data root. See tfds_webdav.py for inventory and upload commands."""

import sys

from tfds_webdav import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    selection = [] if any(arg == "--dataset" or arg.startswith("--dataset=") for arg in argv) else ["--all"]
    main(["download", *selection, *argv], empty_root=True)
