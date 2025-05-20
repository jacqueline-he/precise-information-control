#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(realpath "${SCRIPT_DIR}/../../")"
export PYTHONPATH="$REPO_ROOT"

MODEL_NAME="jacquelinehe/Llama-3.1-PIC-LM-8B"
MODEL_TAG="${MODEL_NAME//\//_}"
FP="${REPO_ROOT}/out/${MODEL_TAG}_asqa.jsonl"
MODE="all"

python -m use_cases.rag \
	--config-name default_rag_asqa \
	model_name_or_path=${MODEL_NAME} \
	download_dir=/gscratch/zlab/jyyh \
	save_fp=${FP} \
	mode=${MODE}
