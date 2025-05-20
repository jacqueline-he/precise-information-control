#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(realpath "${SCRIPT_DIR}/../../")"
export PYTHONPATH="$REPO_ROOT"

BASE_MODEL_PATH="meta-llama/Llama-3.3-70B-Instruct"
FINAL_MODEL_PATH="jacquelinehe/Llama-3.1-PIC-LM-8B"

DIR="${REPO_ROOT}/out/qampari_pipeline"
mkdir -p "${DIR}"

DRAFT_FP="${DIR}/qampari_draft.jsonl"
VERIFY_FP="${DIR}/qampari_verify.jsonl"
FINAL_MODEL_TAG="${FINAL_MODEL_PATH//\//_}"
FINAL_FP="${DIR}/${FINAL_MODEL_TAG}_qampari_final.jsonl"

python -m use_cases.pipeline \
	--config-name default_pipeline_qampari \
	base_llm.model_name_or_path=${BASE_MODEL_PATH} \
	final_lm.model_name_or_path=${FINAL_MODEL_PATH} \
	draft_fp=${DRAFT_FP} \
	verify_fp=${VERIFY_FP} \
	final_fp=${FINAL_FP}

python -m use_cases.evaluate_qampari \
	--config-name default_pipeline_qampari \
	final_fp=${FINAL_FP}
