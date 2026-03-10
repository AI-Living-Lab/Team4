#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import numpy as np

UNAV_JSON = "/home/aix23102/audiolm/CCNet/data/unav_100/annotations/unav100_annotations.json"
VS2_RESULTS = "/home/aix23102/audiolm/video-SALMONN-2/output/test/2_unav_single/test_results.json"  # 실제 경로로 수정
SPLIT = "test"
IOU_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def temporal_iou(ps, pe, gs, ge):
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0.0

def parse_event_from_prompt(prompt_value: str) -> str:
    # "Event: xxxx" 추출
    m = re.search(r"Event:\s*(.+)\s*$", prompt_value.strip())
    return m.group(1).strip() if m else ""

def parse_pred_segment(pred_text: str):
    # pred: "{\"start\": 0, \"end\": 19.852}<|im_end|>"
    if not isinstance(pred_text, str):
        return None
    pred_text = pred_text.replace("<|im_end|>", "").strip()
    # JSON object 추출
    m = re.search(r"\{.*\}", pred_text, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        s = float(obj["start"])
        e = float(obj["end"])
        if e > s:
            return s, e
    except Exception:
        return None
    return None

def vid_from_id_field(id_field):
    # id: [video_path, audio_path]
    if isinstance(id_field, list) and len(id_field) > 0:
        video_path = id_field[0]
        return os.path.splitext(os.path.basename(video_path))[0]
    if isinstance(id_field, str):
        return os.path.splitext(os.path.basename(id_field))[0]
    return ""

# ---- load GT from unav json ----
with open(UNAV_JSON, "r") as f:
    db = json.load(f)["database"]

# gt_by_video_label[vid][label] = list of (start,end)
gt_by_video_label = {}
for vid, v in db.items():
    if v.get("subset", "").lower() != SPLIT:
        continue
    for ann in v.get("annotations", []):
        lab = ann["label"]
        s, e = ann["segment"]
        gt_by_video_label.setdefault(vid, {}).setdefault(lab, []).append((float(s), float(e)))

# ---- load vs2 results ----
with open(VS2_RESULTS, "r") as f:
    res = json.load(f)

# ✅ 이제 total은 "샘플 수"로 셉니다.
total = 0
hit = {t: 0 for t in IOU_THRESHOLDS}
missing_gt_pair = 0
bad_pred = 0
multi_gt_pair = 0  # ✅ 전제 위반(=GT가 2개 이상인 경우) 카운트

for item in res:
    vid = vid_from_id_field(item.get("id"))
    if not vid:
        continue

    prompt_list = item.get("prompt", [])
    prompt_value = ""
    if isinstance(prompt_list, list) and len(prompt_list) > 0 and isinstance(prompt_list[0], dict):
        prompt_value = prompt_list[0].get("value", "")
    label = parse_event_from_prompt(prompt_value)
    if not label:
        continue

    gt_segments = gt_by_video_label.get(vid, {}).get(label, None)
    if not gt_segments:
        missing_gt_pair += 1
        continue

    # ✅ "샘플당 GT 1개" 전제 강제
    if len(gt_segments) != 1:
        multi_gt_pair += 1
        # 권장: 전제 위반이면 스킵 (평가 정의를 깨끗하게 유지)
        continue
        # 만약 '그냥 첫 번째만 사용'하고 싶으면 위 continue를 주석 처리하고 아래 사용:
        # gt_seg = gt_segments[0]

    gt_seg = gt_segments[0]
    gs, ge = gt_seg

    pred_seg = parse_pred_segment(item.get("pred", ""))
    if pred_seg is None:
        bad_pred += 1
        # ✅ GT는 있는데 pred 파싱 실패면 해당 샘플은 miss 처리
        total += 1
        continue

    ps, pe = pred_seg
    total += 1  # ✅ 샘플 1개 평가

    # ✅ 샘플당 hit는 최대 1회 (tau별로는 각각 1회 가능)
    iou = temporal_iou(ps, pe, gs, ge)
    for tau in IOU_THRESHOLDS:
        if iou >= tau:
            hit[tau] += 1

print(f"Total test samples evaluated: {total}")
print(f"Pairs with no GT found (skipped): {missing_gt_pair}")
print(f"Pairs with multiple GT segments (skipped): {multi_gt_pair}")
print(f"Pred JSON parse failures: {bad_pred}")

recalls = []
for tau in IOU_THRESHOLDS:
    r = hit[tau] / total if total > 0 else 0.0
    recalls.append(r)
    print(f"Recall@{tau}: {r:.4f}")

mean_recall = float(np.mean(recalls)) if recalls else 0.0
print(f"Mean Recall@0.1:0.9: {mean_recall:.4f}")
