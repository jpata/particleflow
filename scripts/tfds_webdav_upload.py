#!/usr/bin/env python3
"""Upload TFDS datasets. See tfds_webdav.py for inventory and download commands."""

import sys

from tfds_webdav import main


if __name__ == "__main__":
    main(["upload", *sys.argv[1:]])
