#!/bin/bash

set -u

: "${LUNA_DATA_ROOT:?Set LUNA_DATA_ROOT to the dataset root}"
: "${LUNA_TEACHER_CLIP_PATH:?Set LUNA_TEACHER_CLIP_PATH to the external teacher CLIP checkpoint}"
: "${LUNA_CROSS_DATASET_TEACHER_CHECKPOINT:?Set LUNA_CROSS_DATASET_TEACHER_CHECKPOINT to the external cross-dataset teacher checkpoint}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DATA="${LUNA_DATA_ROOT}"
TRAINER=LUNA
PYTHON="${LUNA_PYTHON:-python}"
OUTPUT_ROOT="${LUNA_OUTPUT_ROOT:-${WORKDIR}/output}"

DATASET=${1:?Usage: xd_train.sh DATASET SEED} # e.g. dtd
SEED=${2:?Usage: xd_train.sh DATASET SEED}

CFG=vit_b16_c2_ep20_batch8_4+4ctx_cross_datasets
SHOTS=0

DIR=${OUTPUT_ROOT}/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}" "${PYTHON}" "${WORKDIR}/train.py" \
    --root "${DATA}" \
    --seed "${SEED}" \
    --trainer "${TRAINER}" \
    --dataset-config-file "${WORKDIR}/configs/datasets/${DATASET}.yaml" \
    --config-file "${WORKDIR}/configs/trainers/${TRAINER}/${CFG}.yaml" \
    --output-dir "${DIR}" \
    DATASET.NUM_SHOTS "${SHOTS}" \
    DATASET.SUBSAMPLE_CLASSES all \
    TRAINER.LUNA.TEMPERATURE 1.0 \
    TRAINER.LUNA.KD_WEIGHT 1000.0 \
    TRAINER.LUNA.TEACHER_CLIP_PATH "${LUNA_TEACHER_CLIP_PATH}" \
    TRAINER.LUNA.CROSS_DATASET_TEACHER_CHECKPOINT "${LUNA_CROSS_DATASET_TEACHER_CHECKPOINT}" \
    TRAINER.MODAL cross
