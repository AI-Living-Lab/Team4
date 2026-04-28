#!/bin/bash
# Extract CLIP (video) + CLAP (audio) features for all LongVALE videos.
set -u
set -e

AVICUNA_DIR=/workspace/jsy/Team4/AVicuna-main
PY=/workspace/miniconda3/envs/avicuna2/bin/python

OUT_DIR=/workspace/jsy/output/avicuna_longvale
VIDEO_FEAT_DIR=${OUT_DIR}/features/video_clip
AUDIO_FEAT_DIR=${OUT_DIR}/features/audio_clap
LOG=${OUT_DIR}/extract_features.log

mkdir -p "${VIDEO_FEAT_DIR}" "${AUDIO_FEAT_DIR}"

cd "${AVICUNA_DIR}"
: > "${LOG}"
echo "[start] $(date)" | tee -a "${LOG}"

${PY} extract_features.py \
    --video_dir /workspace/datasets/LongVALE/videos \
    --wav_dir   /workspace/datasets/LongVALE/audios \
    --output_video_dir "${VIDEO_FEAT_DIR}" \
    --output_audio_dir "${AUDIO_FEAT_DIR}" \
    --annotation "${OUT_DIR}/longvale_annotations.json" \
    --clip_path "checkpoints/clip/ViT-L-14.pt" \
    --clap_path "checkpoints/clap/630k-fusion-best.pt" \
    --subset test \
    --skip_existing 2>&1 | tee -a "${LOG}"

echo "[done] $(date)" | tee -a "${LOG}"
