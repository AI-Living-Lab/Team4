"""
Step 1: AVicuna Batch Inference on UnAV-100 (test set)

Loads pre-extracted features:
  - Video: CLIP ViT-L-14 features (100, 768)  from extract_features.py
  - Audio: CLAP 630k-fusion features (M, 512) from extract_features.py

Then runs AVicuna inference, saves raw text outputs.

Usage:
    python inference_unav_batch.py \
        --video_feat_dir data/unav100/features/video_clip \
        --audio_feat_dir data/unav100/features/audio_clap \
        --annotation data/unav100_annotations.json \
        --output output/raw_predictions.json
"""

import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm

from avicuna.model.builder import load_pretrained_model
from avicuna.inference import inference
from avicuna.mm_utils import get_model_name_from_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run AVicuna inference on UnAV-100 test set"
    )

    # ----- data paths -----
    parser.add_argument("--video_feat_dir", type=str, required=True,
                        help="Dir with CLIP video .npy features (100, 768)")
    parser.add_argument("--audio_feat_dir", type=str, required=True,
                        help="Dir with CLAP audio .npy features (M, 512)")
    parser.add_argument("--annotation", type=str, required=True,
                        help="unav100_annotations.json (to filter test set)")
    parser.add_argument("--output", type=str, default="output/raw_predictions.json")

    # ----- model checkpoints -----
    parser.add_argument("--clip_path", type=str,
                        default="checkpoints/clip/ViT-L-14.pt")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str,
                        default="checkpoints/avicuna-vicuna-v1-5-7b-stage1/mm_projector.bin")
    parser.add_argument("--pretrain_mm_mlp_adapter_a", type=str,
                        default="checkpoints/avicuna-vicuna-v1-5-7b-stage2/mm_projector_a.bin")
    parser.add_argument("--stage3", type=str,
                        default="checkpoints/avicuna-vicuna-v1-5-7b-stage3")
    parser.add_argument("--stage4", type=str,
                        default="checkpoints/avicuna-vicuna-v1-5-7b-stage4")
    parser.add_argument("--model_base", type=str,
                        default="lmsys/vicuna-7b-v1.5")

    # ----- inference settings -----
    parser.add_argument("--query", type=str,
                        default="Describe the events in the video with timestamps.")
    parser.add_argument("--av_ratio", type=float, default=0.25,
                        help="Audio-visual ratio. 0.25 → 25 audio + 75 video tokens")
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


# ------------------------------------------------------------------ #
#  Model loading
# ------------------------------------------------------------------ #

def load_model(args):
    print("Loading AVicuna model...")
    tokenizer, model, context_len = load_pretrained_model(
        args, args.stage3, args.stage4
    )
    model = model.to(args.device)
    model.to(torch.bfloat16)
    model.eval()
    return tokenizer, model


# ------------------------------------------------------------------ #
#  Test set video IDs
# ------------------------------------------------------------------ #

def get_test_video_ids(annotation_path: str):
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    db = data["database"]
    return [vid for vid, info in db.items() if info.get("subset") == "test"]


# ------------------------------------------------------------------ #
#  Inference from pre-extracted features
# ------------------------------------------------------------------ #

def run_inference_from_features(
    tokenizer, model, video_feat_path, audio_feat_path, args
):
    """
    Load CLIP video (100, 768) + CLAP audio (M, 512) features,
    adjust counts to match av_ratio, then run AVicuna inference.
    """
    n_audio_feats = int(100 * args.av_ratio)       # 25
    n_video_feats = int(100 - n_audio_feats)        # 75

    # --- Load features ---
    v_features = torch.tensor(
        np.load(video_feat_path), dtype=torch.bfloat16
    ).to(args.device)   # (100, 768)

    a_features = torch.tensor(
        np.load(audio_feat_path), dtype=torch.bfloat16
    ).to(args.device)   # (M, 512)

    # --- Adjust video feature count (100 → 75) ---
    # get_clip.py extracts 100 frames; demo uses VideoExtractor(N=75) directly.
    # Linear index selection to match.
    if v_features.shape[0] != n_video_feats:
        indices = torch.linspace(0, v_features.shape[0] - 1, n_video_feats).long()
        v_features = v_features[indices]

    # --- Adjust audio feature count → 25 ---
    # Same repeat logic as demo inference.py
    tmp_len = a_features.shape[0]
    if tmp_len != n_audio_feats:
        repeat_factor = n_audio_feats // tmp_len
        remainder = n_audio_feats % tmp_len
        a_features = torch.cat([
            a_features[i].unsqueeze(0).repeat(
                repeat_factor + (1 if i < remainder else 0), 1
            )
            for i in range(tmp_len)
        ], dim=0)

    # --- Build feature list (same as demo) ---
    features = [v_features.unsqueeze(0), a_features.unsqueeze(0)]

    # --- Inference ---
    query = "<video>\n " + args.query
    # Reset aud_mask for each video (different audio lengths)
    model.aud_mask = None
    output = inference(model, features, query, tokenizer)

    return output


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
    args = parse_args()

    tokenizer, model = load_model(args)

    test_ids = get_test_video_ids(args.annotation)
    print(f"Test videos: {len(test_ids)}")

    results = []
    skipped = 0

    for video_id in tqdm(test_ids, desc="Inference"):
        video_feat_path = os.path.join(args.video_feat_dir, f"{video_id}.npy")
        audio_feat_path = os.path.join(args.audio_feat_dir, f"{video_id}.npy")

        if not os.path.exists(video_feat_path) or not os.path.exists(audio_feat_path):
            skipped += 1
            continue

        try:
            output_text = run_inference_from_features(
                tokenizer, model, video_feat_path, audio_feat_path, args
            )
            results.append({
                "video_id": video_id,
                "query": args.query,
                "raw_output": output_text,
            })
        except Exception as e:
            print(f"Error on {video_id}: {e}")
            continue

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} results to {args.output}")
    print(f"Skipped (missing features): {skipped}")


if __name__ == "__main__":
    main()
