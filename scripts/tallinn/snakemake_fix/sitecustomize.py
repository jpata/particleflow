import sys
import os

# The path to your wrapper script
wrapper = "/home/joosep/particleflow/scripts/tallinn/container-python"

# Force sys.executable to look like the wrapper.
# This happens BEFORE Snakemake is even imported.
if os.path.exists(wrapper):
    sys.executable = wrapper
