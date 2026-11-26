#!/usr/bin/env bash

set -u

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 GPU_ID EXISTING_DATASET DATASET [DATASET ...]" >&2
    exit 2
fi

GPU_ID="$1"
EXISTING_DATASET="$2"
shift 2

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
WORKDIR="${LUNA_WORKDIR:-$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)}"
CFG_NAME="vit_b16_c2_ep20_batch8_4+4ctx"
LOGDIR="${WORKDIR}/experiment_logs"
QUEUE_LOG="${LOGDIR}/queue_gpu${GPU_ID}.log"
QUEUE_STATUS="${LOGDIR}/queue_gpu${GPU_ID}_status.tsv"
QUEUE_DONE="${LOGDIR}/queue_gpu${GPU_ID}.done"
MAX_ATTEMPTS="${LUNA_MAX_ATTEMPTS:-3}"

mkdir -p "${LOGDIR}"
rm -f "${QUEUE_DONE}"
printf 'dataset\tstatus\tattempts\n' > "${QUEUE_STATUS}"

log_message() {
    printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${QUEUE_LOG}"
}

output_dir() {
    printf '%s/output/base2new/train_base/%s/shots_0/LUNA/%s/seed_1' \
        "${WORKDIR}" "$1" "${CFG_NAME}"
}

validate_result() {
    local dataset="$1"
    local out csv plot checkpoint
    out="$(output_dir "${dataset}")"
    csv="${out}/loss_trends.csv"
    plot="${out}/loss_curves.png"
    checkpoint="${out}/VLPromptLearner/model-best.pth.tar"

    [ -s "${csv}" ] || return 1
    [ -s "${plot}" ] || return 1
    [ -s "${checkpoint}" ] || return 1
    [ "$(wc -l < "${csv}")" -eq 6 ] || return 1
    awk -F, '
        { sub(/\r$/, "", $NF) }
        NR == 1 { ok = ($1 == "epoch" && $2 == "loss_kd" && $3 == "loss_cc" && $4 == "loss_bc"); next }
        NR > 1 { ok = ok && ($1 == NR - 1) && NF == 4 }
        END { exit(ok ? 0 : 1) }
    ' "${csv}"
}

cleanup_failed_result() {
    local dataset="$1"
    local out resolved
    out="$(output_dir "${dataset}")"
    resolved="$(realpath -m "${out}")"

    case "${resolved}" in
        "${WORKDIR}"/output/base2new/train_base/*/shots_0/LUNA/"${CFG_NAME}"/seed_1)
            ;;
        *)
            log_message "Refusing unsafe cleanup path: ${resolved}"
            return 97
            ;;
    esac

    if [ -e "${resolved}" ]; then
        find "${resolved}" -depth -delete
    fi
    rm -f \
        "${LOGDIR}/${dataset}_seed1.log" \
        "${LOGDIR}/${dataset}_seed1.started" \
        "${LOGDIR}/${dataset}_seed1.exitcode"
}

run_one() {
    local dataset="$1"
    local attempt run_status

    if validate_result "${dataset}"; then
        log_message "Keeping already-complete ${dataset} result"
        printf '%s\talready_complete\t0\n' "${dataset}" >> "${QUEUE_STATUS}"
        return 0
    fi

    attempt=1
    while [ "${attempt}" -le "${MAX_ATTEMPTS}" ]; do
        cleanup_failed_result "${dataset}" || return $?
        log_message "Starting ${dataset}, seed=1, gpu=${GPU_ID}, attempt=${attempt}, NUM_WORKERS=config(8)"
        date -Is > "${LOGDIR}/${dataset}_seed1.started"

        (
            cd "${WORKDIR}" || exit 125
            CUDA_VISIBLE_DEVICES="${GPU_ID}" sh scripts/luna/base2new_train.sh "${dataset}" 1
        ) > "${LOGDIR}/${dataset}_seed1.log" 2>&1
        run_status=$?
        printf '%s\n' "${run_status}" > "${LOGDIR}/${dataset}_seed1.exitcode"

        if [ "${run_status}" -eq 0 ] && validate_result "${dataset}"; then
            log_message "Completed ${dataset} on attempt ${attempt}"
            printf '%s\tsuccess\t%s\n' "${dataset}" "${attempt}" >> "${QUEUE_STATUS}"
            return 0
        fi

        log_message "Failed validation for ${dataset} on attempt ${attempt} (exit=${run_status}); deleting only this failed run"
        cleanup_failed_result "${dataset}" || return $?
        attempt=$((attempt + 1))
    done

    log_message "Gave up ${dataset} after ${MAX_ATTEMPTS} unchanged attempts"
    printf '%s\tfailed\t%s\n' "${dataset}" "${MAX_ATTEMPTS}" >> "${QUEUE_STATUS}"
    return 1
}

wait_for_existing() {
    local dataset="$1"
    local session_name="luna_${dataset}_s1"
    local exit_file="${LOGDIR}/${dataset}_seed1.exitcode"
    local run_status

    log_message "Waiting for already-running ${dataset} on GPU ${GPU_ID}"
    while tmux has-session -t "${session_name}" 2>/dev/null; do
        sleep 30
    done

    for _ in 1 2 3 4; do
        [ -f "${exit_file}" ] && break
        sleep 5
    done

    run_status=missing
    [ -f "${exit_file}" ] && run_status="$(tr -d '[:space:]' < "${exit_file}")"
    if [ "${run_status}" = "0" ] && validate_result "${dataset}"; then
        log_message "Completed existing ${dataset} run"
        printf '%s\tsuccess\t1\n' "${dataset}" >> "${QUEUE_STATUS}"
        return 0
    fi

    log_message "Existing ${dataset} run failed or was incomplete (exit=${run_status}); deleting only this failed run before retry"
    cleanup_failed_result "${dataset}" || return $?
    run_one "${dataset}"
}

overall_status=0

if [ "${EXISTING_DATASET}" != "-" ]; then
    wait_for_existing "${EXISTING_DATASET}" || overall_status=1
fi

for dataset in "$@"; do
    run_one "${dataset}" || overall_status=1
done

printf '%s\n' "${overall_status}" > "${QUEUE_DONE}"
log_message "GPU ${GPU_ID} queue finished with status ${overall_status}"
exit "${overall_status}"
