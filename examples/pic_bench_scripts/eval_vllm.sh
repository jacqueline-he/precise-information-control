#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(realpath "${SCRIPT_DIR}/../../")"
export PYTHONPATH="$REPO_ROOT"

MODEL_PATH="jacquelinehe/Llama-3.1-PIC-LM-8B"
MODEL_TAG="${MODEL_PATH//\//_}"
DIR="${REPO_ROOT}/out/pic_bench/${MODEL_TAG}"
mkdir -p "${DIR}"

python -m pic_bench.generate_eval_data \
	--config-name default_gen_vllm \
	model_name_or_path=${MODEL_PATH} \
	save_dir=${DIR} \
	tensor_parallel_size=1
