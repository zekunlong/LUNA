#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
WORKDIR="${LUNA_WORKDIR:-$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${LUNA_PYTHON:-python}"
LOGDIR="${WORKDIR}/experiment_logs"
CFG_NAME="vit_b16_c2_ep20_batch8_4+4ctx"
GPU_ID="${1:-5}"
CANDIDATE_LIMIT=1000
STATUS_FILE="${LOGDIR}/luna_attention_queue_status.tsv"
DONE_FILE="${LOGDIR}/luna_attention_queue.done"
DATASETS=(
    dtd
    caltech101
    eurosat
    fgvc_aircraft
    food101
    oxford_flowers
    oxford_pets
    stanford_cars
    sun397
    ucf101
)

mkdir -p "${LOGDIR}"
rm -f "${DONE_FILE}"
printf 'dataset\tstatus\tdetail\n' > "${STATUS_FILE}"

log_message() {
    printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${LOGDIR}/luna_attention_queue.log"
}

validate_loss_csv() {
    local csv="$1"
    [ -s "${csv}" ] || return 1
    [ "$(wc -l < "${csv}")" -eq 6 ] || return 1
    awk -F, '
        { sub(/\r$/, "", $NF) }
        NR == 1 { ok = ($1 == "epoch" && $2 == "loss_kd" && $3 == "loss_cc" && $4 == "loss_bc"); next }
        NR > 1 { ok = ok && ($1 == NR - 1) && NF == 4 }
        END { exit(ok ? 0 : 1) }
    ' "${csv}"
}

training_is_complete() {
    local dataset="$1"
    local base csv checkpoint exit_file
    base="${WORKDIR}/output/base2new/train_base/${dataset}/shots_0/LUNA/${CFG_NAME}/seed_1"
    csv="${base}/loss_trends.csv"
    checkpoint="${base}/VLPromptLearner/model-best.pth.tar"
    exit_file="${LOGDIR}/${dataset}_seed1.exitcode"
    [ -s "${exit_file}" ] || return 1
    [ "$(tr -d '[:space:]' < "${exit_file}")" = "0" ] || return 1
    [ -s "${checkpoint}" ] || return 1
    validate_loss_csv "${csv}"
}

attention_is_complete() {
    local dataset="$1"
    local output_dir sample_count
    output_dir="${WORKDIR}/output/luna_attention/seed_1/${dataset}"
    [ -s "${output_dir}/attention_comparison.png" ] || return 1
    [ -s "${output_dir}/selection_manifest.csv" ] || return 1
    [ -s "${output_dir}/selected_heatmaps.npz" ] || return 1
    [ -s "${output_dir}/candidate_limit.txt" ] || return 1
    [ "$(tr -d '[:space:]' < "${output_dir}/candidate_limit.txt")" = "${CANDIDATE_LIMIT}" ] || return 1
    sample_count="$(find "${output_dir}" -maxdepth 1 -type f -name 'sample_*.png' | wc -l)"
    [ "${sample_count}" -eq 3 ]
}

log_message "Waiting for the seed-1 loss-study recovery gate"
while [ ! -f "${LOGDIR}/loss_study_recovery.done" ]; do
    sleep 60
done

overall_status=0
for dataset in "${DATASETS[@]}"; do
    if attention_is_complete "${dataset}"; then
        printf '%s\talready_complete\t3 validated comparisons\n' "${dataset}" >> "${STATUS_FILE}"
        continue
    fi

    if ! training_is_complete "${dataset}"; then
        log_message "Skipping ${dataset}: complete seed-1 training weight is unavailable"
        printf '%s\tblocked\tmissing complete seed-1 training result\n' "${dataset}" >> "${STATUS_FILE}"
        overall_status=1
        continue
    fi

    log_message "Visualizing ${dataset} on GPU ${GPU_ID}"
    (
        cd "${WORKDIR}" || exit 125
        "${PYTHON_BIN}" tools/visualize_luna_attention.py \
            "${dataset}" \
            --seed 1 \
            --gpu "${GPU_ID}" \
            --num-samples 3 \
            --candidate-limit "${CANDIDATE_LIMIT}" \
            --batch-size 16
    ) > "${LOGDIR}/visualize_${dataset}_seed1.log" 2>&1
    run_status=$?
    printf '%s\n' "${run_status}" > "${LOGDIR}/visualize_${dataset}_seed1.exitcode"
    if [ "${run_status}" -eq 0 ]; then
        printf '%s\tsuccess\t3 selected comparisons\n' "${dataset}" >> "${STATUS_FILE}"
    else
        log_message "Visualization failed for ${dataset} with exit ${run_status}; no retry or cleanup performed"
        printf '%s\tfailed\texit %s\n' "${dataset}" "${run_status}" >> "${STATUS_FILE}"
        overall_status=1
    fi
done

printf '%s\n' "${overall_status}" > "${DONE_FILE}"
log_message "Attention visualization queue finished with status ${overall_status}"
exit "${overall_status}"
