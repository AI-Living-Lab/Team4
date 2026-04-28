#!/bin/bash
# ===========================================================================
#  run_pipeline.sh — AVicuna UnAV-100 AVEDL Reproduction Pipeline (RunPod)
#
#  Step 0: mp4/wav → CLIP video features + CLAP audio features
#  Step 1: AVicuna inference → raw text outputs
#  Step 2: Parse raw text → structured events
#  Step 3: CLAP text mapping → UnAV-100 class labels
#  Step 4: Evaluate mAP @ tIoU thresholds
#
#  Usage:
#    bash run_pipeline.sh              # Run all steps (0-4)
#    bash run_pipeline.sh --step 1     # Step 1 only
#    bash run_pipeline.sh --from 2     # Steps 2-4
# ===========================================================================

set -e

# ========================== CONFIG =========================================

# Raw data (already on RunPod)
VIDEO_DIR="/workspace/datasets/unav_100/videos"
WAV_DIR="/workspace/datasets/unav_100/audio"
ANNOTATION="data/annotations/unav100_annotations.json"
LABEL_FILE="data/labels/unav100_class_labels.txt"

# Extracted features (output of Step 0)
VIDEO_FEAT_DIR="data/unav100/features/video_clip"
AUDIO_FEAT_DIR="data/unav100/features/audio_clap"

# Model checkpoints
CLIP_PATH="checkpoints/clip/ViT-L-14.pt"
CLAP_PATH="checkpoints/clap/630k-fusion-best.pt"
STAGE1="checkpoints/avicuna-vicuna-v1-5-7b-stage1/mm_projector.bin"
STAGE2="checkpoints/avicuna-vicuna-v1-5-7b-stage2/mm_projector_a.bin"
STAGE3="checkpoints/avicuna-vicuna-v1-5-7b-stage3"
STAGE4="checkpoints/avicuna-vicuna-v1-5-7b-stage4"

# Output
OUTPUT_DIR="output"
RAW_PRED="${OUTPUT_DIR}/raw_predictions.json"
PARSED_PRED="${OUTPUT_DIR}/parsed_predictions.json"
MAPPED_PRED="${OUTPUT_DIR}/mapped_predictions.json"

# ========================== ARG PARSING ====================================
STEP=""
FROM_STEP=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --step) STEP="$2"; shift 2 ;;
        --from) FROM_STEP="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

should_run() {
    local step_num=$1
    if [ -n "$STEP" ]; then
        [ "$STEP" -eq "$step_num" ]
    else
        [ "$step_num" -ge "$FROM_STEP" ]
    fi
}

mkdir -p "$OUTPUT_DIR"

# ========================== STEP 0: FEATURE EXTRACTION =====================
if should_run 0; then
    echo ""
    echo "=============================================="
    echo "  Step 0/4: Extract CLIP + CLAP Features"
    echo "=============================================="
    python extract_features.py \
        --video_dir "$VIDEO_DIR" \
        --wav_dir "$WAV_DIR" \
        --output_video_dir "$VIDEO_FEAT_DIR" \
        --output_audio_dir "$AUDIO_FEAT_DIR" \
        --annotation "$ANNOTATION" \
        --clip_path "$CLIP_PATH" \
        --clap_path "$CLAP_PATH" \
        --subset test \
        --skip_existing
    echo "Step 0 done."
fi

# ========================== STEP 1: INFERENCE ==============================
if should_run 1; then
    echo ""
    echo "=============================================="
    echo "  Step 1/4: AVicuna Inference"
    echo "=============================================="
    python inference_unav_batch.py \
        --video_feat_dir "$VIDEO_FEAT_DIR" \
        --audio_feat_dir "$AUDIO_FEAT_DIR" \
        --annotation "$ANNOTATION" \
        --output "$RAW_PRED" \
        --clip_path "$CLIP_PATH" \
        --pretrain_mm_mlp_adapter "$STAGE1" \
        --pretrain_mm_mlp_adapter_a "$STAGE2" \
        --stage3 "$STAGE3" \
        --stage4 "$STAGE4" \
        --av_ratio 0.25
    echo "Step 1 done: $RAW_PRED"
fi

# ========================== STEP 2: PARSE ==================================
if should_run 2; then
    echo ""
    echo "=============================================="
    echo "  Step 2/4: Parse Raw Predictions"
    echo "=============================================="
    python parse_raw_predictions.py \
        --input "$RAW_PRED" \
        --output "$PARSED_PRED"
    echo "Step 2 done: $PARSED_PRED"
fi

# ========================== STEP 3: CLAP MAPPING ===========================
if should_run 3; then
    echo ""
    echo "=============================================="
    echo "  Step 3/4: CLAP Label Mapping"
    echo "=============================================="
    python map_with_clap.py \
        --input "$PARSED_PRED" \
        --labels "$LABEL_FILE" \
        --output "$MAPPED_PRED"
    echo "Step 3 done: $MAPPED_PRED"
fi

# ========================== STEP 4: EVALUATE ===============================
if should_run 4; then
    echo ""
    echo "=============================================="
    echo "  Step 4/4: mAP Evaluation"
    echo "=============================================="
    python eval_unav100_map.py \
        --gt "$ANNOTATION" \
        --pred "$MAPPED_PRED"
fi

echo ""
echo "Pipeline complete!"
