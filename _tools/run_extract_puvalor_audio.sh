#!/bin/bash
SCRIPT_DIR=$(cd $(dirname $0); pwd)
if [ -f "$SCRIPT_DIR/../paths.env" ]; then
    source "$SCRIPT_DIR/../paths.env"
else
    echo "[WARNING] paths.env not found. Copy paths.env.example to paths.env and fill in the paths."
fi

python ${BASE_DIR}/_tools/extract_puvalor_audio.py \
    --video_dir ${PUVALOR_DIR}/videos \
    --audio_dir ${PUVALOR_DIR}/audios \
    --workers   16
