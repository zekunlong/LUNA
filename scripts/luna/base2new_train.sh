#!/bin/bash

set -u

: "${LUNA_DATA_ROOT:?Set LUNA_DATA_ROOT to the dataset root}"
: "${LUNA_TEACHER_CLIP_PATH:?Set LUNA_TEACHER_CLIP_PATH to the external teacher CLIP checkpoint}"
: "${LUNA_TEACHER_CHECKPOINT_ROOT:?Set LUNA_TEACHER_CHECKPOINT_ROOT to the external teacher checkpoint root}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
TRAINER=LUNA
PYTHON="${LUNA_PYTHON:-python}"
DATA="${LUNA_DATA_ROOT}"
OUTPUT_ROOT="${LUNA_OUTPUT_ROOT:-${WORKDIR}/output}"

DATASET=${1:?Usage: base2new_train.sh DATASET SEED} # e.g. dtd
SEED=${2:?Usage: base2new_train.sh DATASET SEED}

CFG=vit_b16_c2_ep20_batch8_4+4ctx
# SHOTS=0
if [ "${DATASET}" = "imagenet" ]; then
    SHOTS=16
else
    SHOTS=0
fi

DIR=${OUTPUT_ROOT}/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed_${SEED}

# fgvc_aircraft, oxford_flowers, dtd: KD_WEIGHT:200
# imagenet, caltech101, eurosat, food101, oxford_pets, stanford_cars, sun397, ucf101, KD_WEIGHT:1000
# CUDA_VISIBLE_DEVICES=7
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" "${PYTHON}" "${WORKDIR}/train.py" \
    --root "${DATA}" \
    --seed "${SEED}" \
    --trainer "${TRAINER}" \
    --dataset-config-file "${WORKDIR}/configs/datasets/${DATASET}.yaml" \
    --config-file "${WORKDIR}/configs/trainers/${TRAINER}/${CFG}.yaml" \
    --output-dir "${DIR}" \
    DATASET.NUM_SHOTS "${SHOTS}" \
    TRAINER.MODAL base2novel \
    TRAINER.LUNA.TEMPERATURE 1.0 \
    TRAINER.LUNA.KD_WEIGHT 1000.0 \
    TRAINER.LUNA.TEACHER_CLIP_PATH "${LUNA_TEACHER_CLIP_PATH}" \
    TRAINER.LUNA.TEACHER_CHECKPOINT_ROOT "${LUNA_TEACHER_CHECKPOINT_ROOT}" \
    TEST.SPLIT val

TRAIN_STATUS=$?
if [ "${TRAIN_STATUS}" -ne 0 ]; then
    exit "${TRAIN_STATUS}"
fi

"${PYTHON}" "${WORKDIR}/tools/plot_loss_curves.py" \
    --csv "${DIR}/loss_trends.csv" \
    --output "${DIR}/loss_curves.png" \
    --title "${DATASET} (seed ${SEED})"
    
