"""
Avicuna inference for LongVALE TVG.

Loads pre-extracted CLIP (video) + CLAP (audio) features, builds the same
training-time prompt for a single-event TVG query, and writes raw model
outputs plus coarse parsed (start_pct, end_pct) per query to JSONL.

Key design choices (from AVicuna training, avicuna/train/dataset.py):
  - Video: if N >= 75, sample via `v[(j*N)//75]` for j in [0,75); else pad zeros.
  - Audio: repeat-pattern to reach 25 features (truncates-first-25 when N>25).
  - av_ratio = 0.25 (n_image_feats=75, n_audio_feats=25). Matches stage4.sh.
  - Greedy decoding (do_sample=False). Demo used temp=0.05 which is near-greedy.
  - Prompt follows stage4 single-event TVG pattern:
      "<video>\nAt what timestamps can we observe {event} occurring,
       either through sight or sound in the video?"

Usage:
  python inference_longvale.py \
      --annotation /workspace/jsy/output/avicuna_longvale/longvale_annotations.json \
      --video_feat_dir /workspace/jsy/output/avicuna_longvale/features/video_clip \
      --audio_feat_dir /workspace/jsy/output/avicuna_longvale/features/audio_clap \
      --output /workspace/jsy/output/avicuna_longvale/predictions.jsonl \
      [--limit 10] [--start 0] [--end 13867]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict as edict
from tqdm import tqdm

AVICUNA_DIR = "/workspace/jsy/Team4/AVicuna-main"
if AVICUNA_DIR not in sys.path:
    sys.path.insert(0, AVICUNA_DIR)

from avicuna.constants import IMAGE_TOKEN_INDEX
from avicuna.conversation import conv_templates, SeparatorStyle
from avicuna.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from avicuna.model.builder import load_pretrained_model
from avicuna.utils import disable_torch_init


N_IMAGE_FEATS = 75
N_AUDIO_FEATS = 25


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--annotation", required=True)
    p.add_argument("--video_feat_dir", required=True)
    p.add_argument("--audio_feat_dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--start", type=int, default=0, help="query index start (inclusive)")
    p.add_argument("--end", type=int, default=None, help="query index end (exclusive)")
    p.add_argument("--limit", type=int, default=None, help="max queries (overrides end)")
    p.add_argument("--model_base", default="lmsys/vicuna-7b-v1.5")
    p.add_argument("--stage3", default=f"{AVICUNA_DIR}/checkpoints/avicuna-vicuna-v1-5-7b-stage3")
    p.add_argument("--stage4", default=f"{AVICUNA_DIR}/checkpoints/avicuna-vicuna-v1-5-7b-stage4")
    p.add_argument("--clip_path", default=f"{AVICUNA_DIR}/checkpoints/clip/ViT-L-14.pt")
    p.add_argument("--stage1_adapter",
                   default=f"{AVICUNA_DIR}/checkpoints/avicuna-vicuna-v1-5-7b-stage1/mm_projector.bin")
    p.add_argument("--stage2_adapter",
                   default=f"{AVICUNA_DIR}/checkpoints/avicuna-vicuna-v1-5-7b-stage2/mm_projector_a.bin")
    p.add_argument("--resume", action="store_true",
                   help="Skip queries already present in output JSONL")
    return p.parse_args()


def build_queries(ann_path: str):
    """Flatten {database: {vid: {duration, annotations: [{segment, label}]}}} into query list."""
    with open(ann_path) as f:
        data = json.load(f)
    queries = []
    for vid, info in data["database"].items():
        if info.get("subset") and info["subset"] != "test":
            continue
        duration = float(info["duration"])
        for ann_idx, ann in enumerate(info["annotations"]):
            s, e = ann["segment"]
            queries.append({
                "vid": vid,
                "ann_idx": ann_idx,
                "label": ann["label"],
                "gt_start": float(s),
                "gt_end": float(e),
                "duration": duration,
            })
    return queries


def resample_video(v_feat: torch.Tensor) -> torch.Tensor:
    """Match train-time: stride sample if N>=75, else zero-pad. v_feat: (N,768)."""
    N = v_feat.shape[0]
    if N >= N_IMAGE_FEATS:
        idx = torch.tensor([(j * N) // N_IMAGE_FEATS for j in range(N_IMAGE_FEATS)])
        return v_feat[idx]
    pad = torch.zeros(N_IMAGE_FEATS - N, v_feat.shape[1], dtype=v_feat.dtype)
    return torch.cat([v_feat, pad], 0)


def resample_audio(a_feat: torch.Tensor) -> torch.Tensor:
    """Match train-time: repeat-pattern. N>25 truncates to first 25. a_feat: (N,512)."""
    N = a_feat.shape[0]
    if N == N_AUDIO_FEATS:
        return a_feat
    repeat_factor = N_AUDIO_FEATS // N
    remainder = N_AUDIO_FEATS % N
    parts = []
    for i in range(N):
        rep = repeat_factor + (1 if i < remainder else 0)
        if rep > 0:
            parts.append(a_feat[i].unsqueeze(0).repeat(rep, 1))
    return torch.cat(parts, 0)


def build_prompt_query(event_label: str) -> str:
    # Official LongVALE TVG protocol prompt (longvalellm/eval/eval.py).
    # No space after "<video>\n" matches Avicuna stage4 training data.
    return (f"<video>\nAt which time interval can we find {event_label} "
            "taking place in the video? Give the timestamps in the fromat: From xx to xx.")


PRED_RE = re.compile(r"(\d{1,3})\s*(?:to|-|and|~)\s*(\d{1,3})", re.IGNORECASE)


def parse_prediction(text: str):
    m = PRED_RE.search(text)
    if not m:
        return None, None
    a, b = int(m.group(1)), int(m.group(2))
    if a > 99: a = 99
    if b > 99: b = 99
    return a, b


def generate_one(model, tokenizer, features, query: str) -> str:
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with torch.inference_mode():
        out_ids = model.generate(
            input_ids,
            images=features,
            do_sample=False,
            num_beams=1,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping],
        )
    n_in = input_ids.shape[1]
    text = tokenizer.batch_decode(out_ids[:, n_in:], skip_special_tokens=True)[0].strip()
    if text.endswith(stop_str):
        text = text[:-len(stop_str)].strip()
    return text


def main():
    args = parse_args()
    Path(os.path.dirname(args.output)).mkdir(parents=True, exist_ok=True)

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

    queries = build_queries(args.annotation)
    end = args.end if args.end is not None else len(queries)
    if args.limit is not None:
        end = min(end, args.start + args.limit)
    queries = queries[args.start:end]
    print(f"Queries to run: {len(queries)} (indices {args.start}..{end-1})", flush=True)

    done = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["vid"], r["ann_idx"]))
                except Exception:
                    pass
        print(f"Resume: {len(done)} already done", flush=True)

    feat_cache = {"vid": None, "v": None, "a": None}
    out_f = open(args.output, "a", encoding="utf-8")

    for q in tqdm(queries, desc="inference"):
        key = (q["vid"], q["ann_idx"])
        if key in done:
            continue

        if feat_cache["vid"] != q["vid"]:
            v_path = os.path.join(args.video_feat_dir, f"{q['vid']}.npy")
            a_path = os.path.join(args.audio_feat_dir, f"{q['vid']}.npy")
            if not (os.path.exists(v_path) and os.path.exists(a_path)):
                rec = {**q, "error": "feature missing", "raw": ""}
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                continue
            v_raw = torch.from_numpy(np.load(v_path)).float()  # (100, 768)
            a_raw = torch.from_numpy(np.load(a_path)).float()  # (M, 512)
            v = resample_video(v_raw).to(torch.bfloat16).cuda()
            a = resample_audio(a_raw).to(torch.bfloat16).cuda()
            feat_cache = {"vid": q["vid"], "v": v, "a": a}

        features = [feat_cache["v"].unsqueeze(0), feat_cache["a"].unsqueeze(0)]
        query_str = build_prompt_query(q["label"])
        try:
            raw = generate_one(model, tokenizer, features, query_str)
            err = None
        except Exception as e:
            raw = ""
            err = f"gen_error: {type(e).__name__}: {e}"

        s_pct, e_pct = parse_prediction(raw)
        rec = {
            "vid": q["vid"],
            "ann_idx": q["ann_idx"],
            "label": q["label"],
            "duration": q["duration"],
            "gt_start": q["gt_start"],
            "gt_end": q["gt_end"],
            "raw": raw,
            "pred_start_pct": s_pct,
            "pred_end_pct": e_pct,
        }
        if err:
            rec["error"] = err
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()

    out_f.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
