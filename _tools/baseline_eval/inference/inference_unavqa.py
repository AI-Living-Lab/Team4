"""
Avicuna inference for UnAV-100 QA multiseg (unav100_test_full.json, 3455 rows).

Reuses helpers from inference_longvale.py (resample_video, resample_audio,
generate_one settings). Input is a flat list of samples; prompt is the
human conversation value verbatim (already has "<video>\\n" prefix, which
matches Avicuna's DEFAULT_IMAGE_TOKEN).

Output: JSONL per row with raw model output + GT + duration.
Parsing + multi-segment IoU live in eval_unavqa_tvg.py so we can iterate
on regex without re-running inference.

Usage:
  python inference_unavqa.py \
      --test_json /workspace/jsy/output/avicuna_unavqa/unav100_test_full.json \
      --durations /workspace/jsy/output/avicuna_unavqa/durations.json \
      --video_feat_dir /workspace/jsy/Team4/AVicuna-main/data/unav100/features/video_clip \
      --audio_feat_dir /workspace/jsy/Team4/AVicuna-main/data/unav100/features/audio_clap \
      --output /workspace/jsy/output/avicuna_unavqa/predictions.jsonl \
      [--limit N] [--resume]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict as edict
from tqdm import tqdm

AVICUNA_DIR = "/workspace/jsy/Team4/AVicuna-main"
if AVICUNA_DIR not in sys.path:
    sys.path.insert(0, AVICUNA_DIR)
sys.path.insert(0, "/workspace/jsy/scripts")

from inference_longvale import (  # noqa: E402
    resample_video, resample_audio, generate_one,
)
from avicuna.model.builder import load_pretrained_model  # noqa: E402
from avicuna.utils import disable_torch_init  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_json", required=True)
    p.add_argument("--durations", required=True)
    p.add_argument("--video_feat_dir", required=True)
    p.add_argument("--audio_feat_dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--model_base", default="lmsys/vicuna-7b-v1.5")
    p.add_argument("--stage3", default=f"{AVICUNA_DIR}/checkpoints/avicuna-vicuna-v1-5-7b-stage3")
    p.add_argument("--stage4", default=f"{AVICUNA_DIR}/checkpoints/avicuna-vicuna-v1-5-7b-stage4")
    p.add_argument("--clip_path", default=f"{AVICUNA_DIR}/checkpoints/clip/ViT-L-14.pt")
    p.add_argument("--stage1_adapter", default=f"{AVICUNA_DIR}/checkpoints/avicuna-vicuna-v1-5-7b-stage1/mm_projector.bin")
    p.add_argument("--stage2_adapter", default=f"{AVICUNA_DIR}/checkpoints/avicuna-vicuna-v1-5-7b-stage2/mm_projector_a.bin")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    Path(os.path.dirname(args.output)).mkdir(parents=True, exist_ok=True)

    with open(args.test_json) as f:
        samples = json.load(f)
    durations = json.load(open(args.durations))

    end = args.end if args.end is not None else len(samples)
    if args.limit is not None:
        end = min(end, args.start + args.limit)
    samples = samples[args.start:end]
    print(f"Samples to run: {len(samples)} (indices {args.start}..{end-1})", flush=True)

    done = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add(r["row_idx"])
                except Exception:
                    pass
        print(f"Resume: {len(done)} already done", flush=True)

    disable_torch_init()
    model_args = edict({
        "model_base": args.model_base,
        "clip_path": args.clip_path,
        "pretrain_mm_mlp_adapter": args.stage1_adapter,
        "pretrain_mm_mlp_adapter_a": args.stage2_adapter,
    })
    print("Loading Avicuna...", flush=True)
    tokenizer, model, _ = load_pretrained_model(model_args, args.stage3, args.stage4)
    model = model.cuda().to(torch.bfloat16)
    model.eval()

    feat_cache = {"vid": None, "v": None, "a": None}
    out_f = open(args.output, "a", encoding="utf-8")

    for i, s in enumerate(tqdm(samples, desc="inference")):
        row_idx = args.start + i
        if row_idx in done:
            continue

        vid = os.path.splitext(os.path.basename(s["video"]))[0]
        duration = durations.get(vid)
        if duration is None:
            rec = {"row_idx": row_idx, "vid": vid, "error": "no duration",
                   "gt_label": s.get("gt_label"), "gt_segments": s.get("gt_segments", []),
                   "raw": ""}
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n"); out_f.flush()
            continue

        if feat_cache["vid"] != vid:
            v_path = os.path.join(args.video_feat_dir, f"{vid}.npy")
            a_path = os.path.join(args.audio_feat_dir, f"{vid}.npy")
            if not (os.path.exists(v_path) and os.path.exists(a_path)):
                rec = {"row_idx": row_idx, "vid": vid, "duration": duration,
                       "error": "feature missing",
                       "gt_label": s.get("gt_label"), "gt_segments": s.get("gt_segments", []),
                       "raw": ""}
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n"); out_f.flush()
                continue
            v_raw = torch.from_numpy(np.load(v_path)).float()
            a_raw = torch.from_numpy(np.load(a_path)).float()
            v = resample_video(v_raw).to(torch.bfloat16).cuda()
            a = resample_audio(a_raw).to(torch.bfloat16).cuda()
            feat_cache = {"vid": vid, "v": v, "a": a}

        features = [feat_cache["v"].unsqueeze(0), feat_cache["a"].unsqueeze(0)]
        query = s["conversations"][0]["value"]  # already "<video>\n..."
        try:
            raw = generate_one(model, tokenizer, features, query)
            err = None
        except Exception as e:
            raw = ""
            err = f"gen_error: {type(e).__name__}: {e}"

        rec = {
            "row_idx": row_idx, "vid": vid, "duration": duration,
            "gt_label": s.get("gt_label"),
            "gt_segments": s.get("gt_segments", []),
            "question": query,
            "raw": raw,
        }
        if err:
            rec["error"] = err
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n"); out_f.flush()

    out_f.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
