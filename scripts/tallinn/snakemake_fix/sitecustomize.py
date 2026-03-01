import sys
import os

# Get the absolute path of the directory containing this file (snakemake_fix/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up to the 'tallinn' directory and find the wrapper
wrapper = os.path.join(os.path.dirname(current_dir), "kbfi-slurm-container")

# Force sys.executable to look like the wrapper.
# This happens BEFORE Snakemake is even imported.
if os.path.exists(wrapper):
    sys.executable = wrapper
