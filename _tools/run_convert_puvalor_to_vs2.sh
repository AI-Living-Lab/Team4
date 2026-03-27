#!/bin/bash
SCRIPT_DIR=$(cd $(dirname $0); pwd)
if [ -f "$SCRIPT_DIR/../paths.env" ]; then
    source "$SCRIPT_DIR/../paths.env"
else
    echo "[WARNING] paths.env not found. Copy paths.env.example to paths.env and fill in the paths."
fi

python ${BASE_DIR}/_tools/convert_puvalor_to_vs2.py \
    --stage3_json ${BASE_DIR}/data/stage3.json \
    --video_dir   ${PUVALOR_DIR}/videos \
    --audio_dir   ${PUVALOR_DIR}/audios \
    --output_json ${BASE_DIR}/data/pu_valor_train.json
