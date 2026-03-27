#!/bin/bash
SCRIPT_DIR=$(cd $(dirname $0); pwd)
if [ -f "$SCRIPT_DIR/../paths.env" ]; then
    source "$SCRIPT_DIR/../paths.env"
else
    echo "[WARNING] paths.env not found. Copy paths.env.example to paths.env and fill in the paths."
fi

# test set 만들 때
python ${BASE_DIR}/_tools/unav_to_vs2_sft.py \
  --unav_json ${UNAV_ANNO} \
  --video_dir ${UNAV_VIDEO_DIR} \
  --audio_dir ${UNAV_AUDIO_DIR} \
  --split test \
  --out ${BASE_DIR}/data/unav100_test_dense.json \
  --skip_missing_files \
  --mode dense

# train SFT 만들 때 (GT 포함)
python ${BASE_DIR}/_tools/unav_to_vs2_sft.py \
  --unav_json ${UNAV_ANNO} \
  --video_dir ${UNAV_VIDEO_DIR} \
  --audio_dir ${UNAV_AUDIO_DIR} \
  --split train \
  --out ${BASE_DIR}/data/unav100_train_dense.json \
  --skip_missing_files \
  --mode dense

#   # single mode (기존 방식 - annotation 하나당 샘플 1개)
# python ${BASE_DIR}/_tools/unav_to_vs2_sft.py \
#   --unav_json ${UNAV_ANNO} \
#   --video_dir ${UNAV_VIDEO_DIR} \
#   --audio_dir ${UNAV_AUDIO_DIR} \
#   --split train \
#   --out ${BASE_DIR}/data/unav100_train_single.json \
#   --skip_missing_files \
#   --mode single
