#!/usr/bin/env bash
# Real-data counterpart to the offline synthetic local_test_colliderml.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export KERAS_BACKEND=torch
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
PYTHON="${PYTHON:-$PWD/.venv/bin/python}"
NUM_EVENTS="${NUM_EVENTS:-1000}"
NUM_STEPS="${NUM_STEPS:-225}"
VAL_FREQ="${VAL_FREQ:-$NUM_STEPS}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-$NUM_STEPS}"
LOG_FREQ="${LOG_FREQ:-100}"
GPUS="${GPUS:-1}"
if [[ "$GPUS" == 0 ]]; then
    DTYPE="${DTYPE:-float32}"
    ATTENTION_TYPE="${ATTENTION_TYPE:-math}"
else
    DTYPE="${DTYPE:-bfloat16}"
    ATTENTION_TYPE="${ATTENTION_TYPE:-flash}"
fi
SOURCE="${COLLIDERML_SOURCE:-$PWD/local_test_data/detector_comparison/colliderml_source}"
# Include event limit and conversion/builder code in the cache identity.
fingerprint=$( { sha256sum mlpf/data/colliderml/*.py mlpf/heptfds/colliderml_utils/*.py mlpf/heptfds/colliderml_pf/ttbar.py scripts/build_colliderml_comparison_tfds.py; printf '%s\n' "$NUM_EVENTS" "$SOURCE"; } | sha256sum | cut -c1-12)
RUN_DIR="$PWD/local_test_data/colliderml_real/$fingerprint"
mkdir -p "$RUN_DIR"
if [[ ! -f "$SOURCE/provenance.json" ]]; then
    "$PYTHON" scripts/download_colliderml_comparison_sample.py --output-dir "$SOURCE"
fi
PARQUET="$RUN_DIR/parquet/train-00000-of-01000.parquet"
if [[ ! -f "$RUN_DIR/conversion.complete" ]]; then
    "$PYTHON" -m mlpf.data.colliderml.postprocessing --input "$SOURCE" --sample ttbar_pu0 \
        --outpath "$RUN_DIR/parquet" --shards 0:1 --num-events "$NUM_EVENTS" > "$RUN_DIR/conversion.log" 2>&1
    touch "$RUN_DIR/conversion.complete"
fi
if [[ ! -f "$RUN_DIR/tfds.complete" ]]; then
    "$PYTHON" scripts/build_colliderml_comparison_tfds.py --input "$PARQUET" \
        --manual-dir "$RUN_DIR/manual" --data-dir "$RUN_DIR/tensorflow_datasets" \
        --train-fraction 0.9 > "$RUN_DIR/tfds.log" 2>&1
    touch "$RUN_DIR/tfds.complete"
fi
echo "Real ColliderML data and logs: $RUN_DIR"
"$PYTHON" scripts/validate_colliderml_training_data.py --run-dir "$RUN_DIR" --num-events "$NUM_EVENTS"
if [[ "${PREPARE_ONLY:-0}" == 1 ]]; then
    exit 0
fi
# Use the production attention architecture. No --pipeline: its small-model
# and batch/split overrides are for CI rather than real training validation.
"$PYTHON" mlpf/pipeline.py --spec-file particleflow_spec.yaml --model-name pyg-colliderml-v1 \
    --production-name colliderml --data-dir "$RUN_DIR/tensorflow_datasets" \
    --experiments-dir "$RUN_DIR/experiments" --prefix real_ttbar_ \
    train --gpus "$GPUS" --dtype "$DTYPE" --num_steps "$NUM_STEPS" --checkpoint_freq "$CHECKPOINT_FREQ" \
    --val_freq "$VAL_FREQ" --tensorboard_step_freq "$LOG_FREQ" --gpu_batch_multiplier 4 --data_config 10 \
    --ntrain 900 --nvalid 100 --ntest 100 --num_workers 1 --prefetch_factor 1 \
    --model.attention.attention_type "$ATTENTION_TYPE" \
    2>&1 | tee "$RUN_DIR/training_${NUM_STEPS}_steps.log"
