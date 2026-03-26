import argparse
import json
import numpy as np
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str, required=True, help="Ground truth json path")
    parser.add_argument("--pred", type=str, required=True, help="Prediction json path")
    return parser.parse_args()


def tiou(seg1, seg2):
    s1, e1 = seg1
    s2, e2 = seg2
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    if union <= 0:
        return 0.0
    return inter / union


def flatten_gt(gt_data):
    """
    Returns:
        gt_by_label[label][video_id] = list of dicts
    """
    gt_by_label = defaultdict(lambda: defaultdict(list))

    for item in gt_data:
        video_id = item["video_id"]
        for ev in item["events"]:
            label = ev["label"]
            gt_by_label[label][video_id].append({
                "start": float(ev["start"]),
                "end": float(ev["end"]),
                "used": False
            })

    return gt_by_label


def flatten_pred(pred_data):
    """
    Returns:
        pred_by_label[label] = list of dicts
    """
    pred_by_label = defaultdict(list)

    for item in pred_data:
        video_id = item["video_id"]
        for ev in item["events"]:
            label = ev["mapped_label"]
            pred_by_label[label].append({
                "video_id": video_id,
                "start": float(ev["start"]),
                "end": float(ev["end"]),
                "score": float(ev.get("score", 1.0)),
                "pred_text": ev.get("pred_text", "")
            })

    return pred_by_label


def compute_ap(rec, prec):
    """
    VOC-style AP by interpolation
    """
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap


def evaluate_label(preds, gt_by_video, tiou_threshold):
    """
    preds: list of predictions for one label
    gt_by_video: dict[video_id] -> list of gt events for one label
    """
    npos = sum(len(v) for v in gt_by_video.values())
    if npos == 0:
        return None

    preds = sorted(preds, key=lambda x: x["score"], reverse=True)

    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))

    # copy GT usage state fresh for this threshold
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

    ap = compute_ap(rec, prec)
    return ap


def evaluate_map(gt_data, pred_data, tiou_thresholds):
    gt_by_label = flatten_gt(gt_data)
    pred_by_label = flatten_pred(pred_data)

    all_labels = sorted(set(list(gt_by_label.keys()) + list(pred_by_label.keys())))
    results = {}

    for thr in tiou_thresholds:
        aps = []

        for label in all_labels:
            gt_for_label = gt_by_label.get(label, {})
            pred_for_label = pred_by_label.get(label, [])

            ap = evaluate_label(pred_for_label, gt_for_label, thr)
            if ap is not None:
                aps.append(ap)

        results[thr] = float(np.mean(aps)) if len(aps) > 0 else 0.0

    return results


def main():
    args = parse_args()

    with open(args.gt, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    with open(args.pred, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    tiou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    results = evaluate_map(gt_data, pred_data, tiou_thresholds)

    print("==== UnAV-100 Evaluation ====")
    for thr in tiou_thresholds:
        print(f"mAP@{thr:.1f}: {results[thr]:.4f}")

    mean_map = float(np.mean([results[t] for t in tiou_thresholds]))
    print(f"Average mAP: {mean_map:.4f}")


if __name__ == "__main__":
    main()
