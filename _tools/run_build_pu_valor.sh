#!/bin/bash
SCRIPT_DIR=$(cd $(dirname $0); pwd)
if [ -f "$SCRIPT_DIR/../paths.env" ]; then
    source "$SCRIPT_DIR/../paths.env"
else
    echo "[WARNING] paths.env not found. Copy paths.env.example to paths.env and fill in the paths."
fi

python ${BASE_DIR}/_tools/build_pu_valor.py \
    --stage3_json ${BASE_DIR}/data/stage3.json \
    --valor_dir   ${VALOR_DIR} \
    --output_dir  ${PUVALOR_DIR} \
    --workers     32
