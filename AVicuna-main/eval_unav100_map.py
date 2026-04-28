"""
Step 4: Evaluate mapped predictions against UnAV-100 ground truth.

Computes mAP @ tIoU thresholds [0.5, 0.6, 0.7, 0.8, 0.9] + Average.
Matches the protocol used in Table 3 of the AVicuna paper.

Input:
    --gt   : unav100_annotations.json  (original format: {database: {vid: {annotations, subset}}})
    --pred : mapped_predictions.json   (from Step 3)

Usage:
    python eval_unav100_map.py \
        --gt data/unav100_annotations.json \
        --pred output/mapped_predictions.json
"""

import argparse
import json
import numpy as np
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str, required=True,
                        help="Ground truth annotation json")
    parser.add_argument("--pred", type=str, required=True,
                        help="Mapped predictions json")
    return parser.parse_args()


# ------------------------------------------------------------------ #
#  tIoU
# ------------------------------------------------------------------ #

def tiou(seg1, seg2):
    s1, e1 = seg1
    s2, e2 = seg2
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    if union <= 0:
        return 0.0
    return inter / union


# ------------------------------------------------------------------ #
#  GT loading  — 원본 annotation format 지원
# ------------------------------------------------------------------ #

def load_gt(gt_path: str):
    """
    Load GT from original UnAV-100 annotation format:
    {
        "database": {
            "<video_id>": {
                "duration": ...,
                "annotations": [{"segment": [s, e], "label": "...", "label_id": ...}],
                "subset": "test"
            }
        }
    }

    Returns:
        gt_by_label[label][video_id] = list of {"start": ..., "end": ...}
    """
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_by_label = defaultdict(lambda: defaultdict(list))

    database = data["database"]
    n_test_videos = 0
    n_test_events = 0

    for video_id, info in database.items():
        if info.get("subset") != "test":
            continue
        n_test_videos += 1

        for ann in info.get("annotations", []):
            label = ann["label"]
            seg = ann["segment"]
            gt_by_label[label][video_id].append({
                "start": float(seg[0]),
                "end": float(seg[1]),
            })
            n_test_events += 1

    print(f"GT loaded: {n_test_videos} test videos, {n_test_events} events, "
          f"{len(gt_by_label)} classes")

    return gt_by_label


# ------------------------------------------------------------------ #
#  Prediction loading
# ------------------------------------------------------------------ #

def load_pred(pred_path: str):
    """
    Returns:
        pred_by_label[label] = list of {"video_id", "start", "end", "score"}
    """
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pred_by_label = defaultdict(list)
    n_events = 0

    for item in data:
        video_id = item["video_id"]
        for ev in item["events"]:
            pred_by_label[ev["mapped_label"]].append({
                "video_id": video_id,
                "start": float(ev["start"]),
                "end": float(ev["end"]),
                "score": float(ev.get("score", 1.0)),
            })
            n_events += 1

    print(f"Predictions loaded: {len(data)} videos, {n_events} events, "
          f"{len(pred_by_label)} classes")

    return pred_by_label


# ------------------------------------------------------------------ #
#  VOC-style AP
# ------------------------------------------------------------------ #

def compute_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap


# ------------------------------------------------------------------ #
#  Per-label evaluation
# ------------------------------------------------------------------ #

def evaluate_label(preds, gt_by_video, tiou_threshold):
    npos = sum(len(v) for v in gt_by_video.values())
    if npos == 0:
        return None

    preds = sorted(preds, key=lambda x: x["score"], reverse=True)

    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))

    # fresh GT usage state per threshold
    gt_used = {}
    for video_id, gts in gt_by_video.items():
        gt_used[video_id] = [
            {"start": g["start"], "end": g["end"], "used": False}
            for g in gts
        ]

    for i, pred in enumerate(preds):
        video_id = pred["video_id"]
        pred_seg = [pred["start"], pred["end"]]

        if video_id not in gt_used:
            fp[i] = 1
            continue

        gts = gt_used[video_id]
        best_iou = -1
        best_j = -1

        for j, gt in enumerate(gts):
            iou = tiou(pred_seg, [gt["start"], gt["end"]])
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= tiou_threshold and best_j >= 0 and not gts[best_j]["used"]:
            tp[i] = 1
            gts[best_j]["used"] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    rec = tp_cum / max(npos, np.finfo(np.float64).eps)
    prec = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)

    return compute_ap(rec, prec)


# ------------------------------------------------------------------ #
#  Full evaluation
# ------------------------------------------------------------------ #

def evaluate_map(gt_by_label, pred_by_label, tiou_thresholds):
    # evaluate only over GT labels (standard protocol)
    all_labels = sorted(gt_by_label.keys())
    results = {}

    for thr in tiou_thresholds:
        aps = []
        for label in all_labels:
            gt_for_label = gt_by_label.get(label, {})
            pred_for_label = pred_by_label.get(label, [])
            ap = evaluate_label(pred_for_label, gt_for_label, thr)
            if ap is not None:
                aps.append(ap)
        results[thr] = float(np.mean(aps)) if aps else 0.0

    return results


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
    args = parse_args()

    gt_by_label = load_gt(args.gt)
    pred_by_label = load_pred(args.pred)

    tiou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    results = evaluate_map(gt_by_label, pred_by_label, tiou_thresholds)

    print("\n" + "=" * 50)
    print("  UnAV-100 AVEDL Evaluation Results")
    print("=" * 50)

    # Paper reference (AVicuna)
    paper_ref = {0.5: 60.0, 0.6: 50.4, 0.7: 49.6, 0.8: 43.5, 0.9: 36.5}

    print(f"{'tIoU':>8}  {'Ours':>8}  {'Paper':>8}  {'Diff':>8}")
    print("-" * 40)

    for thr in tiou_thresholds:
        ours = results[thr] * 100
        paper = paper_ref.get(thr, 0)
        diff = ours - paper
        print(f"  {thr:.1f}    {ours:>7.2f}   {paper:>7.1f}   {diff:>+7.2f}")

    avg_ours = np.mean([results[t] * 100 for t in tiou_thresholds])
    avg_paper = 60.3
    print("-" * 40)
    print(f"  Avg     {avg_ours:>7.2f}   {avg_paper:>7.1f}   {avg_ours - avg_paper:>+7.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
