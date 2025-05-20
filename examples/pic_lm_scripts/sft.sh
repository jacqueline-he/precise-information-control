#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(realpath "${SCRIPT_DIR}/../../")"
RUN_NAME="${REPO_ROOT}/out/pic_lm/pic-lm-8b-sft"
export PYTHONPATH="$REPO_ROOT"

# ------------ Args ------------ #
NUM_GPUS=2
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=32
GRAD_ACC=$(($TOTAL_BATCH_SIZE / ($NUM_GPUS * $BATCH_SIZE_PER_GPU)))
# ---------- End Args ---------- #

if [ $(($GRAD_ACC * $NUM_GPUS * $BATCH_SIZE_PER_GPU)) -ne $TOTAL_BATCH_SIZE ]; then
	echo "⚠️  Skipping: incompatible batch config for total=$TOTAL_BATCH_SIZE"
	exit 1
fi

export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))

echo "Starting run: $RUN_NAME"
start_time=$(date +%s)

accelerate launch \
	--mixed_precision bf16 \
	--num_processes ${NUM_GPUS} \
	--use_deepspeed \
	--deepspeed_config_file ${REPO_ROOT}/pic_lm/ds_configs/stage2_accelerate.conf \
	pic_lm/finetune.py \
	--config-name default_sft \
	run.output_dir="${RUN_NAME}" \
	run.num_gpus=${NUM_GPUS} \
	run.gradient_accumulation_steps=${GRAD_ACC} \
	run.per_device_train_batch_size=${BATCH_SIZE_PER_GPU} \
	run.total_batch_size=${TOTAL_BATCH_SIZE}

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))

hours=$((elapsed_time / 3600))
minutes=$(((elapsed_time % 3600) / 60))
seconds=$((elapsed_time % 60))
echo "Total runtime: ${hours}h ${minutes}m ${seconds}s"
echo "Finished training run: $RUN_NAME"
