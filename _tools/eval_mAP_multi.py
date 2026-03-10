#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate Video-SALMONN-2 (VS2) multi-mode outputs on UnAV-100 using ActivityNet-style
temporal action detection mAP, aligned with CCNet's ANETdetection evaluation paradigm.

- GT: unav100_annotations.json (database -> subset -> annotations[label, segment])
- Pred: VS2 test_results.json (list of items with id/prompt/pred)
        pred is expected to be a JSON array string:
        [{"label": str, "start": float, "end": float, "score": float}, ...]

Outputs:
- mAP at tIoU thresholds (default: 0.1..0.9)
- average mAP across thresholds
"""

import argparse
import json
import os
import re
import string
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from datetime import datetime

import numpy as np


# --------------------------
# Utilities
# --------------------------

def temporal_iou(ps: float, pe: float, gs: float, ge: float) -> float:
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = (pe - ps) + (ge - gs) - inter
    return inter / union if union > 0 else 0.0


def vid_from_id_field(id_field) -> str:
    """
    VS2 test_results.json typically has:
      "id": [video_path, audio_path]
    """
    if isinstance(id_field, list) and len(id_field) > 0:
        video_path = id_field[0]
        return os.path.splitext(os.path.basename(video_path))[0]
    if isinstance(id_field, str):
        return os.path.splitext(os.path.basename(id_field))[0]
    return ""


def extract_json_array(text: str):
    """
    Robustly extract the first JSON array from a string.
    Handles cases like: '[{...}, {...}]<|im_end|>' or extra whitespace.
    """
    if not isinstance(text, str):
        return None
    s = text.replace("<|im_end|>", "").strip()

    # Find first [...] block (non-greedy) that looks like a JSON array
    m = re.search(r"\[[\s\S]*\]", s)
    if not m:
        return None
    cand = m.group(0).strip()

    try:
        arr = json.loads(cand)
        if isinstance(arr, list):
            return arr
    except Exception:
        return None
    return None

def extract_pred_events(text: str):
    """
    VS2 pred 문자열에서 이벤트 리스트를 최대한 복구합니다.
    지원:
      1) JSON array: [ {...}, {...} ]
      2) JSON object: { ... }
      3) JSON objects stream: { ... },\n{ ... }\n ... (배열 괄호 없이)
    또한 <|im_end|> 및 뒤따르는 설명 텍스트가 있어도 최대한 앞부분 JSON만 파싱합니다.
    """
    if not isinstance(text, str):
        return None

    s = text.replace("<|im_end|>", "").strip()

    # 1) 우선 "[...]" 배열이 있으면 그걸 최우선으로 파싱
    m = re.search(r"\[[\s\S]*\]", s)
    if m:
        cand = m.group(0).strip()
        try:
            arr = json.loads(cand)
            if isinstance(arr, list):
                return arr
        except Exception:
            pass  # fallthrough

    # 2) 배열이 없거나 배열 파싱 실패면, JSON 값을 연속으로 읽어보기
    dec = json.JSONDecoder()
    i = 0
    n = len(s)
    out = []

    # NOTE 같은 텍스트가 나오면 그 이후는 중단(모델이 설명을 붙인 경우)
    stop_markers = ["Note:", "NOTE:", "note:"]

    while i < n:
        # stop marker가 시작되면 종료
        for mk in stop_markers:
            if s.startswith(mk, i):
                i = n
                break
        if i >= n:
            break

        # 공백/쉼표/개행 스킵
        while i < n and s[i] in " \t\r\n,":
            i += 1
        if i >= n:
            break

        # JSON object 또는 array가 시작하는 경우만 시도
        if s[i] not in "[{":
            # JSON이 아닌 글자가 나오면 중단 (뒤 설명일 가능성 큼)
            break

        try:
            obj, j = dec.raw_decode(s, i)
            i = j
            if isinstance(obj, list):
                out.extend([x for x in obj if isinstance(x, dict)])
            elif isinstance(obj, dict):
                out.append(obj)
            else:
                # 숫자/문자열 같은 건 무시
                pass
        except Exception:
            # 여기서 실패하면 더 이상 복구 어렵다고 보고 종료
            break

    return out if len(out) > 0 else None


def voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """
    VOC-style AP (integral) with precision envelope.
    """
    if rec.size == 0 or prec.size == 0:
        return 0.0

    # Append sentinel points
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    # Precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # Integrate area under PR curve where recall changes
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap

# --------------------------
# Label normalization / filtering
# --------------------------

_END_PUNCT = set(list(string.punctuation))  # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
_SEP_RE = re.compile(r"[_\-/]+")            # common separators -> space
_WS_RE = re.compile(r"\s+")

def normalize_label(raw: str) -> str:
    """
    Safe normalization (format-only):
    - strip leading/trailing whitespace
    - lowercase
    - replace common separators with space
    - collapse whitespace
    - strip punctuation only at BOTH ends (not in the middle)
    """
    if not isinstance(raw, str):
        return ""
    s = raw.strip()
    if not s:
        return ""
    s = s.lower()
    s = _SEP_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()

    # strip punct at ends only
    while s and s[0] in _END_PUNCT:
        s = s[1:].lstrip()
    while s and s[-1] in _END_PUNCT:
        s = s[:-1].rstrip()

    s = _WS_RE.sub(" ", s).strip()
    return s

def classify_ooc_type(raw: str) -> str:
    """
    For reporting only. NOTE: evaluation drops all non-GT labels anyway.
    """
    if not isinstance(raw, str):
        return "other"
    s = raw

    if s != s.strip():
        return "leading_or_trailing_space"
    if any(c.isupper() for c in s):
        return "has_uppercase"
    if any(c in string.punctuation for c in s):
        # separator char handled separately
        if any(c in "_-/" for c in s):
            return "has_separator_char"
        return "has_punct"
    # crude sentence-like heuristic: many words or looks like a sentence
    if len(s.split()) >= 4 or (("." in s) or ("," in s) or ("note:" in s.lower())):
        return "sentence_like_label"
    return "other"


# --------------------------
# Load GT
# --------------------------

def load_unav_gt(unav_json: str, split: str):
    """
    Returns:
      gt_by_label_vid[label][vid] = list of (start,end)
      labels_with_gt = sorted list
      vids_in_split = sorted list
      num_gt_by_label[label] = total instances in split
    """
    with open(unav_json, "r", encoding="utf-8") as f:
        db = json.load(f)["database"]

    gt_by_label_vid: Dict[str, Dict[str, List[Tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    vids = []
    for vid, v in db.items():
        if v.get("subset", "").lower() != split.lower():
            continue
        vids.append(vid)

        raw = v.get("annotations", [])
        ants = remove_duplicate_annotations_unav(raw)
        if len(raw) != len(ants):
            print(f"[DBG DEDUP] {vid}: {len(raw)} -> {len(ants)}")
        
        for ann in ants:
            lab = normalize_label(ann["label"])
            if not lab:
                continue
            s, e = ann["segment"]
            gt_by_label_vid[lab][vid].append((float(s), float(e)))

    num_gt_by_label = {lab: sum(len(segs) for segs in vidmap.values()) for lab, vidmap in gt_by_label_vid.items()}
    labels_with_gt = sorted(list(gt_by_label_vid.keys()))

    # Build normalized GT set for safe matching
    gt_label_set_norm = set(labels_with_gt)
    # Defensive: remove empty
    gt_label_set_norm.discard("")

    print("[DBG] GT raw label examples:", labels_with_gt[:5])
    print("[DBG] GT norm label examples:", list(gt_label_set_norm)[:5])
    print("[DBG] Any raw!=norm in GT?:", any(normalize_label(l)!=l for l in labels_with_gt))

    return gt_by_label_vid, labels_with_gt, sorted(vids), num_gt_by_label, gt_label_set_norm


def remove_duplicate_annotations_unav(ants, tol=1e-3):
    valid = []
    for ann in ants:
        s, e, l = float(ann["segment"][0]), float(ann["segment"][1]), ann["label"]
        ok = True
        for p in valid:
            if abs(s - float(p["segment"][0])) <= tol and abs(e - float(p["segment"][1])) <= tol and l == p["label"]:
                ok = False
                break
        if ok:
            valid.append(ann)
    return valid

# --------------------------
# Load Predictions (VS2 multi)
# --------------------------

def load_vs2_preds_multi(vs2_results_json: str, gt_label_set_norm: set):

    """
    Returns:
      preds_by_label: dict[label] = list of (vid, score, start, end)
    """
    bad_field_reason_counter = defaultdict(int)
    bad_field_examples = []  # keep small
    MAX_BAD_EX = 30

    with open(vs2_results_json, "r", encoding="utf-8") as f:
        res = json.load(f)

    # (vid, score, start, end, is_ooc)
    preds_by_label: Dict[str, List[Tuple[str, float, float, float, bool]]] = defaultdict(list)
    ooc_best_by_vid = {}  # vid -> (vid, sc, s, e, True)

    bad_items = 0
    dropped_ooc = 0
    ooc_type_counter = defaultdict(int)
    bad_pred_json = 0
    bad_fields = 0

    for item in res:
        vid = vid_from_id_field(item.get("id"))
        if not vid:
            bad_items += 1
            continue

        arr = extract_pred_events(item.get("pred", ""))
        if arr is None:
            # It's okay to have empty/invalid outputs; they contribute no preds.
            bad_pred_json += 1
            continue

        for obj in arr:
            if not isinstance(obj, dict):
                bad_fields += 1
                continue

            raw_lab = obj.get("label", None)
            s = obj.get("start", None)
            e = obj.get("end", None)
            sc = obj.get("score", None)

            if sc is None:
                sc = 0.5

            try:
                if not isinstance(raw_lab, str) or raw_lab.strip() == "":
                    bad_field_reason_counter["label_missing_or_not_str"] += 1
                    raise ValueError("bad label")

                s = float(s)
                e = float(e)
                sc = float(sc)

                if e <= s:
                    bad_field_reason_counter["end_le_start"] += 1
                    raise ValueError("end<=start")

                if sc < 0.0:
                    sc = 0.0
                if sc > 1.0:
                    sc = 1.0

            except Exception:
                bad_fields += 1
                if len(bad_field_examples) < MAX_BAD_EX:
                    bad_field_examples.append({
                        "vid": vid,
                        "raw_obj": obj,
                        "raw_label": raw_lab,
                        "raw_start": s,
                        "raw_end": e,
                        "raw_score": sc,
                    })
                continue

            norm_lab = normalize_label(raw_lab)
            if not norm_lab or (norm_lab not in gt_label_set_norm):
                ooc_type = classify_ooc_type(raw_lab)
                ooc_type_counter[ooc_type] += 1
                dropped_ooc += 1

                # ✅ 비디오당 OOC 1개만 유지: 가장 높은 score만 남김
                prev = ooc_best_by_vid.get(vid, None)
                if (prev is None) or (sc > prev[1]):
                    ooc_best_by_vid[vid] = (vid, sc, s, e, True)
                continue

            preds_by_label[norm_lab].append((vid, sc, s, e, False))

    num_videos_in_results = len({vid_from_id_field(it.get("id")) for it in res if vid_from_id_field(it.get("id"))})
    ooc_rate = (len(ooc_best_by_vid) / num_videos_in_results) if num_videos_in_results > 0 else 0.0

    stats = {
        "비디오 ID를 추출하지 못한 샘플 수": bad_items,
        "예측 결과에서 JSON 이벤트 배열을 파싱하지 못한 샘플 수": bad_pred_json,
        "JSON은 파싱됐지만 필드 오류로 버려진 예측 이벤트 수": bad_fields,
        "최종적으로 예측이 존재한 클래스 개수": len(preds_by_label),
        "평가에 사용된 전체 예측 이벤트 개수": sum(len(v) for v in preds_by_label.values()),
        "정규화 후에도 GT에 없어 OOC로 판정된 예측 이벤트 수": dropped_ooc,
        "OOC로 판정된 이유별 이벤트 개수": dict(ooc_type_counter),
        "bad_pred_fields_by_reason": dict(bad_field_reason_counter),
        "bad_pred_fields_examples": bad_field_examples,
        "ooc_videos_with_any_ooc": len(ooc_best_by_vid),
        "num_videos_in_vs2_results": num_videos_in_results,
        "ooc_video_rate": ooc_rate,

    }

    ooc_preds = list(ooc_best_by_vid.values())

    return preds_by_label, ooc_preds, stats

def apply_topk_per_video(preds_by_label, ooc_preds, topk):
    if topk is None or topk <= 0:
        return preds_by_label, ooc_preds

    # gather all preds into by_video
    by_video = defaultdict(list)  # vid -> list of (label_or_none, score, start, end, is_ooc)
    for lab, items in preds_by_label.items():
        for (vid, sc, s, e, is_ooc) in items:
            by_video[vid].append((lab, sc, s, e, is_ooc))

    for (vid, sc, s, e, is_ooc) in ooc_preds:
        by_video[vid].append((None, sc, s, e, True))

    # prune per video
    new_preds_by_label = defaultdict(list)
    new_ooc_preds = []

    for vid, items in by_video.items():
        items.sort(key=lambda x: x[1], reverse=True)
        keep = items[:topk]

        for lab, sc, s, e, is_ooc in keep:
            if is_ooc:
                new_ooc_preds.append((vid, sc, s, e, True))
            else:
                new_preds_by_label[lab].append((vid, sc, s, e, False))

    return new_preds_by_label, new_ooc_preds

# --------------------------
# mAP Evaluation
# --------------------------

def evaluate_map(
    gt_by_label_vid: Dict[str, Dict[str, List[Tuple[float, float]]]],
    num_gt_by_label: Dict[str, int],
    preds_by_label: Dict[str, List[Tuple[str, float, float, float, bool]]],
    ooc_preds: List[Tuple[str, float, float, float, bool]],
    tiou_thresholds: List[float],
):
    """
    For each label and each tIoU threshold:
      - Sort predictions by score desc
      - Match each prediction to an unmatched GT instance in the same video with max IoU
      - TP if IoU >= threshold, else FP
      - Compute PR and AP
    Returns:
      ap_by_tiou[label][tiou] = ap
      map_by_tiou[tiou] = mean over labels with GT
    """
    labels = sorted(list(gt_by_label_vid.keys()))
    ap_by_tiou: Dict[str, Dict[float, float]] = {lab: {} for lab in labels}

    # Pre-build GT arrays per label per vid for faster access
    for lab in labels:
        for tau in tiou_thresholds:
            preds = preds_by_label.get(lab, [])
            npos = num_gt_by_label.get(lab, 0)
            if npos == 0:
                ap_by_tiou[lab][tau] = 0.0
                continue

            # Sort preds by score descending
            preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)

            tp = np.zeros(len(preds_sorted), dtype=np.float32)
            fp = np.zeros(len(preds_sorted), dtype=np.float32)

            # For each video, maintain matched flags for each GT instance
            matched: Dict[str, np.ndarray] = {}
            for vid, segs in gt_by_label_vid[lab].items():
                matched[vid] = np.zeros(len(segs), dtype=bool)

            # Evaluate each prediction
            for i, (vid, score, ps, pe, is_ooc) in enumerate(preds_sorted):
                if is_ooc:
                    fp[i] = 1.0
                    continue
                gt_segs = gt_by_label_vid[lab].get(vid, [])
                if not gt_segs:
                    fp[i] = 1.0
                    continue

                # Find best unmatched GT by IoU
                ious = []
                for (gs, ge) in gt_segs:
                    ious.append(temporal_iou(ps, pe, gs, ge))
                ious = np.asarray(ious, dtype=np.float32)

                # Among GT instances, choose highest IoU
                best_idx = int(np.argmax(ious))
                best_iou = float(ious[best_idx])

                # If best GT already matched, try next best unmatched (standard)
                # We'll select unmatched GT with highest IoU.
                unmatched_mask = ~matched[vid]
                if not np.any(unmatched_mask):
                    fp[i] = 1.0
                    continue

                # pick unmatched with max IoU
                masked_ious = ious.copy()
                masked_ious[~unmatched_mask] = -1.0
                best_idx = int(np.argmax(masked_ious))
                best_iou = float(masked_ious[best_idx])

                if best_iou >= tau and best_iou >= 0.0:
                    tp[i] = 1.0
                    matched[vid][best_idx] = True
                else:
                    fp[i] = 1.0

            # PR
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)

            rec = tp_cum / float(npos)
            prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

            ap = voc_ap(rec, prec)
            ap_by_tiou[lab][tau] = ap

    # mAP over labels (labels with GT)
    map_by_tiou = {}
    for tau in tiou_thresholds:
        aps = [ap_by_tiou[lab][tau] for lab in labels if num_gt_by_label.get(lab, 0) > 0]
        map_by_tiou[tau] = float(np.mean(aps)) if len(aps) > 0 else 0.0

    avg_map = float(np.mean([map_by_tiou[t] for t in tiou_thresholds])) if tiou_thresholds else 0.0
    return ap_by_tiou, map_by_tiou, avg_map

def save_eval_results(
    out_path: str,
    split: str,
    tiou_thresholds: List[float],
    labels_with_gt: List[str],
    num_gt_by_label: Dict[str, int],
    pred_stats: Dict[str, Any],
    ap_by_tiou: Dict[str, Dict[float, float]],
    map_by_tiou: Dict[float, float],
    avg_map: float,
):
    # float key(0.1 등)를 json key로 쓰기 위해 문자열로 변환
    payload = {
        "meta": {
            "split": split,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tiou_thresholds": [float(t) for t in tiou_thresholds],
            "num_classes_with_gt": int(len(labels_with_gt)),
            "total_gt_instances": int(sum(num_gt_by_label.values())),
            "prediction_parse_stats": pred_stats,
        },
        "mAP_by_tiou": {f"{t:.2f}": float(map_by_tiou[t]) for t in tiou_thresholds},
        "average_mAP": float(avg_map),
        "AP_by_class": {
            lab: {
                "num_gt": int(num_gt_by_label.get(lab, 0)),
                "ap_by_tiou": {f"{t:.2f}": float(ap_by_tiou[lab][t]) for t in tiou_thresholds},
            }
            for lab in labels_with_gt
        },
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVED] Per-class AP + mAP saved to: {out_path}")


# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unav_json", required=True, help="Path to unav100_annotations.json")
    ap.add_argument("--vs2_results", required=True, help="Path to VS2 test_results.json")
    ap.add_argument("--split", default="test", help="Split name in UnAV annotations (default: test)")
    ap.add_argument("--tiou", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
                    help="Comma-separated tIoU thresholds")
    ap.add_argument("--print_per_class", action="store_true",
                    help="Print per-class AP at each tIoU (can be long)")
    ap.add_argument("--topk", type=int, default=100,
                help="Per-video topK predictions kept before mAP eval (-1 = keep all)")
    args = ap.parse_args()

    tiou_thresholds = [float(x.strip()) for x in args.tiou.split(",") if x.strip() != ""]
    tiou_thresholds = [t for t in tiou_thresholds if 0.0 < t < 1.0]

    # Load GT
    gt_by_label_vid, labels_with_gt, vids_in_split, num_gt_by_label, gt_label_set_norm = load_unav_gt(args.unav_json, args.split)

    # Load preds
    preds_by_label, ooc_preds, pred_stats = load_vs2_preds_multi(args.vs2_results, gt_label_set_norm)

    preds_by_label, ooc_preds = apply_topk_per_video(preds_by_label, ooc_preds, args.topk)
    pred_stats["applied_video_topk"] = args.topk

    # Evaluate
    ap_by_tiou, map_by_tiou, avg_map = evaluate_map(
        gt_by_label_vid=gt_by_label_vid,
        num_gt_by_label=num_gt_by_label,
        preds_by_label=preds_by_label,
        ooc_preds=ooc_preds,
        tiou_thresholds=tiou_thresholds
    )

    # Report
    print("========== VS2 (multi) UnAV mAP Evaluation ==========")
    print(f"Split: {args.split}")
    print(f"GT videos in split: {len(vids_in_split)}")
    print(f"GT labels in split: {len(labels_with_gt)}")
    print(f"Total GT instances: {sum(num_gt_by_label.values())}")

    print("\n[Prediction parse stats]")
    for k, v in pred_stats.items():
        print(f"- {k}: {v}")

    if "ooc_events_excluded_from_fp" in pred_stats:
        print("\n[Non-GT labels dropped from eval (treated as FN / not FP)]")
        print(f"- ooc_events_excluded_from_fp: {pred_stats['ooc_events_excluded_from_fp']}")
        by_type = pred_stats.get("dropped_non_gt_by_type", {})
        if isinstance(by_type, dict) and len(by_type) > 0:
            for k, v in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {k}: {v}")

    print("\n[mAP @ tIoU]")
    for tau in tiou_thresholds:
        print(f"|tIoU = {tau:.2f}: mAP = {map_by_tiou[tau]*100:.2f} (%)")
    print(f"Avearge mAP: {avg_map*100:.2f} (%)")

    # --- ALWAYS SAVE (no argparse option) ---
    out_path = os.path.join(os.path.dirname(args.vs2_results), f"vs2_unav_map_{args.split}.json")
    save_eval_results(
        out_path=out_path,
        split=args.split,
        tiou_thresholds=tiou_thresholds,
        labels_with_gt=labels_with_gt,
        num_gt_by_label=num_gt_by_label,
        pred_stats=pred_stats,
        ap_by_tiou=ap_by_tiou,
        map_by_tiou=map_by_tiou,
        avg_map=avg_map,
    )

    if args.print_per_class:
        print("\n[Per-class AP]")
        for lab in labels_with_gt:
            line = [f"{lab}"]
            for tau in tiou_thresholds:
                line.append(f"{ap_by_tiou[lab][tau]*100:.2f}")
            print(" | ".join(line))


if __name__ == "__main__":
    main()
