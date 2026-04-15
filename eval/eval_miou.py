#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_miou.py
  - vS2 inference 결과(test_results.json) → mIoU + R@1 계산
  - 라벨 매칭: AVicuna 스타일 word-overlap fuzzy match (CLAP 사용 안 함)

Usage:
  python eval/eval_miou.py \
    --results   eval/results/unav100_exp1/test_results.json \
    --unav_json /home/aix23102/audiolm/CCNet/data/unav_100/annotations/unav100_annotations.json \
    --split     test \
    --max_time  261.0 \
    --out_dir   eval/results/unav100_exp1
"""
import argparse
import ast
import json
import os
import re
import numpy as np
from collections import defaultdict


# ──────────────────────────────────────────────────────────────
# 1. video_id 추출
# ──────────────────────────────────────────────────────────────

def extract_video_id(item: dict):
    raw_id = item.get("id", None)
    if isinstance(raw_id, list) and len(raw_id) > 0:
        raw_id = raw_id[0]
    if not isinstance(raw_id, str):
        return None
    raw_id = raw_id.strip()
    if raw_id.startswith("["):
        try:
            parsed = ast.literal_eval(raw_id)
            if isinstance(parsed, list) and len(parsed) > 0:
                raw_id = parsed[0]
        except Exception:
            pass
    m = re.search(r"([^/\\]+)\.(mp4|avi|mkv|mov|wav|flac)", raw_id, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


# ──────────────────────────────────────────────────────────────
# 2. VTG-LLM time token 역변환
# ──────────────────────────────────────────────────────────────

def decode_vtg_time(token_str: str, max_time: float = 60.0):
    has_dot = "<tdot>" in token_str
    if has_dot:
        parts = token_str.split("<tdot>")
        int_part = re.findall(r"<t(\d)>", parts[0])
        dec_part = re.findall(r"<t(\d)>", parts[1]) if len(parts) > 1 else []
    else:
        int_part = re.findall(r"<t(\d)>", token_str)
        dec_part = []
    if not int_part:
        return None
    integer_part = int("".join(int_part))
    decimal_part = int(dec_part[0]) if dec_part else 0
    t = integer_part + decimal_part / 10.0
    return min(t, max_time)


# ──────────────────────────────────────────────────────────────
# 3. 라벨 매칭: word-overlap fuzzy match (AVicuna 방식)
# ──────────────────────────────────────────────────────────────

def normalize_label(s: str) -> str:
    return re.sub(r"[\s_\-]+", " ", s.strip().lower())


def fuzzy_match_label(pred_label: str, valid_labels: list[str]) -> str:
    """Word-overlap 기반 fuzzy matching (AVicuna eval_unav100.py 방식)."""
    pred_words = set(normalize_label(pred_label).split())
    best_label = valid_labels[0]
    best_score = 0
    for vl in valid_labels:
        vl_words = set(normalize_label(vl).split())
        overlap = len(pred_words & vl_words)
        if overlap > best_score:
            best_score = overlap
            best_label = vl
    if best_score == 0:
        # substring fallback
        pred_lower = normalize_label(pred_label)
        for vl in valid_labels:
            vl_lower = normalize_label(vl)
            if vl_lower in pred_lower or pred_lower in vl_lower:
                return vl
    return best_label


# ──────────────────────────────────────────────────────────────
# 4. pred 텍스트 파싱
# ──────────────────────────────────────────────────────────────

def _extract_event_dicts(raw: str) -> list[dict]:
    raw = re.sub(r"[a-zA-Z가-힣]{4,}(\{)", r"\1", raw)
    raw = re.sub(r"(\})[a-zA-Z가-힣]{4,}", r"\1", raw)

    def to_double_quote(s: str) -> str:
        s = re.sub(r"'([^']*)'(\s*:)", r'"\1"\2', s)
        s = re.sub(r":\s*'([^']*)'", r': "\1"', s)
        return s

    raw_dq = to_double_quote(raw)

    # JSON 배열
    m = re.search(r"(\[.*?\])", raw_dq, re.DOTALL)
    if m:
        try:
            items = json.loads(m.group(1))
            if isinstance(items, list):
                return [x for x in items if isinstance(x, dict)]
        except json.JSONDecodeError:
            pass

    m = re.search(r"(\[.*?\])", raw, re.DOTALL)
    if m:
        try:
            items = json.loads(m.group(1))
            if isinstance(items, list):
                return [x for x in items if isinstance(x, dict)]
        except json.JSONDecodeError:
            pass

    # 단일 객체
    m = re.search(r"(\{[^{}]*\})", raw_dq, re.DOTALL)
    if m:
        try:
            item = json.loads(m.group(1))
            if isinstance(item, dict):
                return [item]
        except json.JSONDecodeError:
            pass

    # 정규식 fallback
    items = []
    blocks = re.split(r"(?=\{[^{}]*['\"]event['\"])", raw)
    for block in blocks:
        event_m = re.search(r'["\']event["\']\s*["\']?\s*:\s*["\']([^"\']+)["\']', block)
        start_m = re.search(r'["\']start["\']\s*:\s*["\']([^"\']*)["\']', block)
        end_m = re.search(r'["\']end["\']\s*:\s*["\']([^"\']*)["\']', block)
        if event_m and (start_m or end_m):
            items.append({
                "event": event_m.group(1),
                "start": start_m.group(1) if start_m else "",
                "end": end_m.group(1) if end_m else "",
            })
    return items


def parse_single_output(raw: str, valid_labels: list[str],
                        max_time: float = 60.0) -> list[dict]:
    event_dicts = _extract_event_dicts(raw)
    predictions = []
    for item in event_dicts:
        event_raw = item.get("event", item.get("label", "")) or ""
        event_raw = event_raw.strip()
        start_tok = item.get("start", "") or ""
        end_tok = item.get("end", "") or ""

        start = decode_vtg_time(start_tok, max_time)
        end = decode_vtg_time(end_tok, max_time)

        if start is None and end is None:
            continue
        if start is None:
            start = 0.0
        if end is None:
            end = max_time
        if end <= start:
            end = min(start + 1.0, max_time)

        label = fuzzy_match_label(event_raw, valid_labels)
        predictions.append({
            "label": label,
            "segment": [start, end],
        })
    return predictions


# ──────────────────────────────────────────────────────────────
# 5. GT 로드
# ──────────────────────────────────────────────────────────────

def load_gt(unav_json: str, split: str):
    with open(unav_json, "r") as f:
        db = json.load(f)["database"]

    gt_dict = {}
    label_set = set()
    for vid, item in db.items():
        if item.get("subset", "").lower() != split.lower():
            continue
        anns = []
        for ann in item.get("annotations", []):
            lbl = ann["label"]
            seg = [float(ann["segment"][0]), float(ann["segment"][1])]
            anns.append({"label": lbl, "segment": seg})
            label_set.add(lbl)
        if anns:
            gt_dict[vid] = anns
    return gt_dict, sorted(label_set)


# ──────────────────────────────────────────────────────────────
# 6. tIoU 계산
# ──────────────────────────────────────────────────────────────

def compute_tiou(seg1, seg2):
    inter_s = max(seg1[0], seg2[0])
    inter_e = min(seg1[1], seg2[1])
    inter = max(0.0, inter_e - inter_s)
    union = (seg1[1] - seg1[0]) + (seg2[1] - seg2[0]) - inter
    return inter / (union + 1e-8) if union > 0 else 0.0


# ──────────────────────────────────────────────────────────────
# 7. mIoU + R@1 계산
# ──────────────────────────────────────────────────────────────

def compute_miou_and_recall(gt_dict: dict, pred_dict: dict,
                            iou_thresholds=(0.3, 0.5, 0.7)):
    """
    각 GT segment에 대해 같은 label의 prediction 중 가장 높은 IoU를 찾아서:
      - mIoU: 모든 GT segment에 대한 best IoU 평균
      - R@1 @ threshold: best IoU >= threshold인 GT segment 비율
    """
    all_best_ious = []

    for vid, gt_anns in gt_dict.items():
        preds = pred_dict.get(vid, [])

        for gt_ann in gt_anns:
            gt_label = gt_ann["label"]
            gt_seg = gt_ann["segment"]

            # 같은 라벨의 prediction 필터
            matched_preds = [p for p in preds if p["label"] == gt_label]

            if not matched_preds:
                all_best_ious.append(0.0)
                continue

            # 가장 높은 IoU
            best_iou = max(compute_tiou(p["segment"], gt_seg) for p in matched_preds)
            all_best_ious.append(best_iou)

    all_best_ious = np.array(all_best_ious)
    n_gt = len(all_best_ious)

    miou = float(np.mean(all_best_ious)) if n_gt > 0 else 0.0

    recall_at = {}
    for th in iou_thresholds:
        recall_at[th] = float(np.mean(all_best_ious >= th)) if n_gt > 0 else 0.0

    return {
        "n_gt_segments": n_gt,
        "mIoU": miou,
        "R@1": recall_at,
    }


# ──────────────────────────────────────────────────────────────
# 8. 메인
# ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--unav_json", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_time", type=float, default=261.0)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--subset_videos", default=None,
                    help="JSON file with subset test data; only evaluate GT for videos in this file")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.results))
    os.makedirs(out_dir, exist_ok=True)

    # 1. GT 로드
    print("[1/4] Loading GT annotations ...")
    gt_dict, all_labels = load_gt(args.unav_json, args.split)

    # subset 필터링
    if args.subset_videos:
        with open(args.subset_videos, "r") as f:
            subset_data = json.load(f)
        subset_vids = set()
        for item in subset_data:
            vpath = item.get("video", "")
            m = re.search(r"([^/\\]+)\.(mp4|avi|mkv|mov)", vpath, re.IGNORECASE)
            if m:
                subset_vids.add(m.group(1))
        gt_dict = {k: v for k, v in gt_dict.items() if k in subset_vids}
        print(f"      [SUBSET] filtered to {len(gt_dict)} videos")
    n_gt_segs = sum(len(v) for v in gt_dict.values())
    print(f"      {len(gt_dict)} videos, {len(all_labels)} classes, {n_gt_segs} GT segments")

    # 2. Inference 결과 로드
    print("[2/4] Loading inference results ...")
    with open(args.results, "r") as f:
        raw_results = json.load(f)
    print(f"      {len(raw_results)} items")

    # 3. 파싱
    print("[3/4] Parsing predictions (fuzzy word-overlap matching) ...")
    pred_dict = {}
    parse_stats = {"ok": 0, "empty": 0, "no_vid": 0}
    failed_preds = []

    for item in raw_results:
        vid = extract_video_id(item)
        if vid is None:
            parse_stats["no_vid"] += 1
            continue

        raw_pred = item.get("pred", "")
        preds = parse_single_output(raw_pred, all_labels, max_time=args.max_time)

        if not preds:
            parse_stats["empty"] += 1
            failed_preds.append({"video_id": vid, "pred": raw_pred[:300]})
        else:
            parse_stats["ok"] += 1

        if vid not in pred_dict:
            pred_dict[vid] = []
        pred_dict[vid].extend(preds)

    total = sum(parse_stats.values())
    print(f"      total={total}  ok={parse_stats['ok']}  "
          f"empty={parse_stats['empty']}  no_vid={parse_stats['no_vid']}")

    # 샘플 출력
    print("\n  [SAMPLE PREDICTIONS]")
    for i, (vid, dets) in enumerate(list(pred_dict.items())[:3]):
        print(f"    {vid}: {dets[:2]}")
    if failed_preds:
        print(f"\n  [FAILED PARSE SAMPLES] (first 3)")
        for fp in failed_preds[:3]:
            print(f"    {fp['video_id']}: {repr(fp['pred'][:100])}")

    # 4. mIoU + R@1 계산
    print("\n[4/4] Computing mIoU + R@1 ...")
    iou_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    result = compute_miou_and_recall(gt_dict, pred_dict, iou_thresholds)

    # 결과 출력
    SEP = "=" * 52
    print(f"\n{SEP}")
    print("  UnAV-100 mIoU + R@1 Evaluation")
    print(SEP)
    print(f"  GT segments: {result['n_gt_segments']}")
    print(f"  mIoU:        {result['mIoU'] * 100:.2f}%")
    print()
    for th, val in sorted(result["R@1"].items()):
        print(f"  R@1 @ IoU={th:.1f}:  {val * 100:.2f}%")
    print(f"\n{SEP}\n")

    # 저장
    summary = {
        "mIoU_%": round(result["mIoU"] * 100, 4),
        "R@1": {
            str(k): round(v * 100, 4)
            for k, v in result["R@1"].items()
        },
        "n_gt_segments": result["n_gt_segments"],
        "parse_stats": parse_stats,
    }

    summary_path = os.path.join(out_dir, "eval_miou_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    pred_path = os.path.join(out_dir, "predictions_miou.json")
    with open(pred_path, "w") as f:
        json.dump(pred_dict, f, indent=2, ensure_ascii=False)

    failed_path = os.path.join(out_dir, "failed_parses_miou.json")
    with open(failed_path, "w") as f:
        json.dump(failed_preds, f, indent=2, ensure_ascii=False)

    print(f"[SAVED] {summary_path}")
    print(f"[SAVED] {pred_path}")
    print(f"[SAVED] {failed_path}  ({len(failed_preds)} items)")


if __name__ == "__main__":
    main()
