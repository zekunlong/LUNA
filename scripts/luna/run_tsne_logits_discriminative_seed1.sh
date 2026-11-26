#!/usr/bin/env bash

set -u

PYTHON="${LUNA_PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_WORKDIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORKDIR="${LUNA_WORKDIR:-${DEFAULT_WORKDIR}}"
LOGDIR=${WORKDIR}/experiment_logs
SCRIPT=${WORKDIR}/tools/visualize_logits_tsne.py
PREFIX=tsne_logits_discriminative

mkdir -p "${LOGDIR}"

run_dataset() {
    local dataset=$1
    local gpu=$2
    local log=${LOGDIR}/${PREFIX}_${dataset}_seed1.log
    local exit_file=${LOGDIR}/${PREFIX}_${dataset}_seed1.exitcode

    printf 'dataset=%s gpu=%s started=%s\n' "${dataset}" "${gpu}" "$(date -Iseconds)" > "${log}"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PYTHON}" "${SCRIPT}" "${dataset}" \
        --seed 1 \
        --min-classes 15 \
        --max-classes 20 \
        --samples-per-class 30 \
        --batch-size 128 \
        --iterations 1500 >> "${log}" 2>&1
    local code=$?
    printf '%s\n' "${code}" > "${exit_file}"
    printf 'dataset=%s gpu=%s finished=%s exitcode=%s\n' \
        "${dataset}" "${gpu}" "$(date -Iseconds)" "${code}" >> "${log}"
    return 0
}

(
    run_dataset caltech101 0
    run_dataset stanford_cars 0
) &
worker0=$!

(
    run_dataset eurosat 1
    run_dataset sun397 1
) &
worker1=$!

run_dataset fgvc_aircraft 2 &
worker2=$!

run_dataset food101 4 &
worker4=$!

run_dataset oxford_flowers 5 &
worker5=$!

run_dataset oxford_pets 6 &
worker6=$!

run_dataset ucf101 7 &
worker7=$!

wait "${worker0}"
wait "${worker1}"
wait "${worker2}"
wait "${worker4}"
wait "${worker5}"
wait "${worker6}"
wait "${worker7}"

overall=0
for dataset in caltech101 eurosat fgvc_aircraft food101 oxford_flowers \
    oxford_pets stanford_cars sun397 ucf101; do
    exit_file=${LOGDIR}/${PREFIX}_${dataset}_seed1.exitcode
    if [[ ! -s "${exit_file}" ]] || [[ "$(<"${exit_file}")" != "0" ]]; then
        overall=1
    fi
done

printf '%s\n' "${overall}" > "${LOGDIR}/${PREFIX}_remaining_seed1.exitcode"
exit "${overall}"
