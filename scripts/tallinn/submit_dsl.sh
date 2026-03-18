#!/bin/bash

# Configuration
CONFIG_FILE="mlpf/standalone/configs.txt"
DATA_DIR="/local/joosep/mlpf/cms/20260204_cmssw_15_0_5_117d32/tfds/"
LOG_DIR="logs/dsl_jobs"

mkdir -p $LOG_DIR

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: $CONFIG_FILE not found."
    exit 1
fi

# Line counter for naming
LINE_NUM=1

while IFS= read -r DSL || [ -n "$DSL" ]; do
    if [[ -z "$DSL" || "$DSL" == \#* ]]; then
        continue
    fi

    echo "Submitting job for DSL line $LINE_NUM: $DSL"

    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=mlpf_dsl_${LINE_NUM}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH -o ${LOG_DIR}/job_${LINE_NUM}.out
#SBATCH -e ${LOG_DIR}/job_${LINE_NUM}.err

export PF_SITE=tallinn
./scripts/tallinn/wrapper.sh python3 mlpf/standalone/eval.py --data-dir $DATA_DIR --dsl "$DSL"
EOT

    LINE_NUM=$((LINE_NUM + 1))
done < "$CONFIG_FILE"

echo "All $LINE_NUM jobs submitted."
