"""
AVicuna Charades-STA Evaluation
Metrics: R@1 IoU={0.3, 0.5, 0.7}, mIoU

Usage:
    cd /workspace/jsy/Team4/AVicuna-main
    python eval_charades_sta.py
"""
import os, json, torch, numpy as np, re, sys, time
sys.path.insert(0, '.')

from tqdm import tqdm
from avicuna.constants import IMAGE_TOKEN_INDEX
from avicuna.conversation import conv_templates, SeparatorStyle
from avicuna.model.builder import load_pretrained_model
from avicuna.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from easydict import EasyDict as edict

# ================================================================== #
#  Config — 경로만 확인하고 수정하세요
# ================================================================== #
VIDEO_FEAT_DIR  = "data/charades/features/video_clip"   # extract_features.py 출력 경로
AUDIO_FEAT_DIR  = "data/charades/features/audio_clap"
ANN_JSON        = "data/charades_sta_annotations.json"  # make_charades_annotation.py 출력
OUTPUT_PATH     = "output/charades_sta_eval_results.json"

ARGS = edict({
    'clip_path':                   'checkpoints/clip/ViT-L-14.pt',
    'pretrain_mm_mlp_adapter':     'checkpoints/avicuna-vicuna-v1-5-7b-stage1/mm_projector.bin',
    'pretrain_mm_mlp_adapter_a':   'checkpoints/avicuna-vicuna-v1-5-7b-stage2/mm_projector_a.bin',
    'stage3':                      'checkpoints/avicuna-vicuna-v1-5-7b-stage3',
    'stage4':                      'checkpoints/avicuna-vicuna-v1-5-7b-stage4',
    'model_base':                  '/workspace/models/vicuna-7b-v1.5',
})

# ================================================================== #
#  Feature loader  (inference_v3_10.py 의 load_feat 동일 방식)
# ================================================================== #
def load_feat(vid):
    v = torch.tensor(
        np.load(os.path.join(VIDEO_FEAT_DIR, f"{vid}.npy")), dtype=torch.bfloat16
    ).cuda()
    a = torch.tensor(
        np.load(os.path.join(AUDIO_FEAT_DIR, f"{vid}.npy")), dtype=torch.bfloat16
    ).cuda()

    # video: 75 frames 균등 샘플
    idx = torch.linspace(0, v.shape[0] - 1, 75).long()
    v = v[idx]

    # audio: 25 슬롯으로 리샘플
    tmp = a.shape[0]
    rf, rm = 25 // tmp, 25 % tmp
    a = torch.cat(
        [a[i].unsqueeze(0).repeat(rf + (1 if i < rm else 0), 1) for i in range(tmp)],
        dim=0
    )
    return [v.unsqueeze(0), a.unsqueeze(0)]

# ================================================================== #
#  Inference
# ================================================================== #
def inference(model, features, query, tokenizer):
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=features,
            do_sample=False,
            temperature=1.0,
            num_beams=1,
            max_new_tokens=128,
            use_cache=True,
        )

    input_token_len = input_ids.shape[1]
    text = tokenizer.batch_decode(
        [output_ids[0, input_token_len:]], skip_special_tokens=True
    )[0].strip()
    if text.endswith(stop_str):
        text = text[:-len(stop_str)]
    return text.strip()

# ================================================================== #
#  Timestamp parser
#  AVicuna 출력: "from 30 to 70" → 0~100 퍼센트 → 실제 초로 변환
# ================================================================== #
def parse_timestamps(text, duration):
    """
    Returns (start_sec, end_sec) or (None, None).
    AVicuna outputs 0~100 percentage values.
    """
    # "from 30 to 70" / "from 30.5 to 70.2"
    m = re.search(
        r'from\s+([\d]+(?:\.[\d]+)?)\s+to\s+([\d]+(?:\.[\d]+)?)',
        text, re.IGNORECASE
    )
    if m:
        s = float(m.group(1)) / 100.0 * duration
        e = float(m.group(2)) / 100.0 * duration
        return s, e

    # "[30, 70]" 형식 fallback
    m = re.search(r'\[([\d.]+),\s*([\d.]+)\]', text)
    if m:
        s = float(m.group(1)) / 100.0 * duration
        e = float(m.group(2)) / 100.0 * duration
        return s, e

    # 숫자 2개 fallback
    nums = re.findall(r'[\d]+\.?[\d]*', text)
    if len(nums) >= 2:
        s = float(nums[0]) / 100.0 * duration
        e = float(nums[1]) / 100.0 * duration
        return s, e

    return None, None

# ================================================================== #
#  Metrics
# ================================================================== #
def compute_iou(ps, pe, gs, ge):
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0.0

def compute_metrics(results):
    thresholds = [0.3, 0.5, 0.7]
    recall = {t: 0 for t in thresholds}
    ious   = []

    for r in results:
        ps, pe = r["pred_start"], r["pred_end"]
        gs, ge = r["gt_start"],   r["gt_end"]

        if ps is None or pe is None or ps > pe:
            ious.append(0.0)
            continue

        v = compute_iou(ps, pe, gs, ge)
        ious.append(v)
        for t in thresholds:
            if v >= t:
                recall[t] += 1

    n = len(results)
    metrics = {f"R@1_IoU={t}": round(recall[t] / n * 100, 2) for t in thresholds}
    metrics["mIoU"]       = round(float(np.mean(ious)) * 100, 2)
    metrics["parse_fail"] = sum(1 for r in results if r["pred_start"] is None)
    metrics["total"]      = n
    return metrics

# ================================================================== #
#  Prompt
# ================================================================== #
PROMPT = (
    '<video>\n'
    'Given the video and audio, find the time interval (as percentage 0-100) '
    'where the following event occurs.\n'
    'Event: "{query}"\n'
    'Answer in the format: from START to END. '
    'If not present, answer NO.'
)

# ================================================================== #
#  Main
# ================================================================== #
def main():
    print("Loading model...")
    tokenizer, model, _ = load_pretrained_model(ARGS, ARGS.stage3, ARGS.stage4)
    model = model.cuda().to(torch.bfloat16).eval()

    with open(ANN_JSON) as f:
        ann = json.load(f)

    # test set 샘플 수집
    samples = []
    for vid_id, info in ann["database"].items():
        if info.get("subset") != "test":
            continue
        duration = info.get("duration", 0.0)
        for seg in info["annotations"]:
            samples.append({
                "video_id": vid_id,
                "query":    seg["label"],
                "gt_start": seg["segment"][0],
                "gt_end":   seg["segment"][1],
                "duration": duration,
            })

    print(f"Test samples: {len(samples)}")

    results = []
    parse_fail = 0
    t0 = time.time()

    for s in tqdm(samples, desc="Evaluating"):
        vid  = s["video_id"]
        vfeat = os.path.join(VIDEO_FEAT_DIR, f"{vid}.npy")
        afeat = os.path.join(AUDIO_FEAT_DIR, f"{vid}.npy")

        if not os.path.exists(vfeat) or not os.path.exists(afeat):
            results.append({**s, "pred_start": None, "pred_end": None, "raw_output": "FEAT_MISSING"})
            parse_fail += 1
            continue

        features = load_feat(vid)
        query    = PROMPT.format(query=s["query"])

        try:
            raw = inference(model, features, query, tokenizer)
        except Exception as e:
            raw = f"ERROR: {e}"
            results.append({**s, "pred_start": None, "pred_end": None, "raw_output": raw})
            parse_fail += 1
            continue

        # NO 응답 처리
        if raw.strip().upper().startswith("NO") or "NOT PRESENT" in raw.upper():
            results.append({**s, "pred_start": None, "pred_end": None, "raw_output": raw})
            parse_fail += 1
            continue

        ps, pe = parse_timestamps(raw, s["duration"])
        if ps is None:
            parse_fail += 1

        results.append({**s, "pred_start": ps, "pred_end": pe, "raw_output": raw})

    # ---- metrics ----
    metrics = compute_metrics(results)
    elapsed = time.time() - t0

    print("\n" + "="*50)
    print("  Charades-STA  |  AVicuna Evaluation")
    print("="*50)
    for k, v in metrics.items():
        print(f"  {k:<20}: {v}")
    print(f"  Elapsed        : {elapsed/60:.1f} min")
    print("="*50)

    # ---- save ----
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

