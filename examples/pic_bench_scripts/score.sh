#!/bin/bash
set -e
# ------------ Args ------------ #
CONFIG_NAME="${1:-default_scoring}" # fallback to 'default_scoring' if is unset or empty
if [[ -n "${2:-}" ]]; then
	SHARD_INDEX="$2"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
	SHARD_INDEX="${SLURM_ARRAY_TASK_ID}"
else
	echo "Error: SHARD_INDEX (arg 2) or SLURM_ARRAY_TASK_ID must be provided." >&2
	exit 1
fi

NUM_CPUS="${3:-$SLURM_CPUS_PER_TASK}"
MODEL_NAME="${4:-}"

if [[ -z "${MODEL_NAME}" ]]; then
	echo "Error: MODEL_NAME (argument 4) is required." >&2
	exit 1
fi
# ---------- End Args ---------- #

TASKS=('entity_bios' 'pop_bio_param' 'pop_bio_cf' 'askhistorians' 'eli5' 'expertqa' 'facts' 'xsum')

NUM_TASKS=${#TASKS[@]}
TASK=${TASKS[$SHARD_INDEX]}

echo "Index $SHARD_INDEX → MODEL=$MODEL_NAME, TASK=$TASK"
echo "Using $NUM_CPUS workers"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(realpath "${SCRIPT_DIR}/../../")"
export PYTHONPATH="$REPO_ROOT"
MODEL_TAG="${MODEL_NAME//\//_}"
FP_DIR="${REPO_ROOT}/out/pic_bench/${MODEL_TAG}"
FP=${FP_DIR}/${TASK}.jsonl

[ -f "${FP}" ] || {
	echo "Error: File ${FP} does not exist." >&2
	exit 1
}
echo "FP: ${FP}"

python -m pic_bench.score_eval_data \
	--config-name ${CONFIG_NAME} \
	filepath=${FP} \
	max_workers=${NUM_CPUS}
