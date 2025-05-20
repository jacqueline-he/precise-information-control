#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(realpath "${SCRIPT_DIR}/../../")"
export PYTHONPATH="$REPO_ROOT"

DIR="${REPO_ROOT}/out/pic_bench/gpt-4o"
python -m pic_bench.generate_eval_data \
	--config-name default_gen_openai \
	save_dir=${DIR}
