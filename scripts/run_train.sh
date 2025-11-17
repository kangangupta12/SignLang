#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR=${1:-dataset/videos}
SEQ_LEN=${SEQ_LEN:-30}
BATCH_SIZE=${BATCH_SIZE:-8}
EPOCHS=${EPOCHS:-30}
USE_FRAMES=${USE_FRAMES:-false}

python -m src.train \
  --dataset_dir "${DATASET_DIR}" \
  --seq_len "${SEQ_LEN}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  $( [ "${USE_FRAMES}" = "true" ] && echo "--use_frames" )
