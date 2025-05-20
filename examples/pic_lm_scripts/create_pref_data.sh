#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(realpath "${SCRIPT_DIR}/../../")"
export PYTHONPATH="$REPO_ROOT"

python -m pic_lm.create_preference_data \
	--config-name default_pref_data \
	run.output_dir="out/pic_lm/pref_data"
