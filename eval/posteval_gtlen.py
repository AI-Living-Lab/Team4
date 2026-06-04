#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
posteval_gtlen.py — eval 결과에 FINDINGS 1.2/1.3 분해를 그대로 적용 (2026-06-02)

★ 목적: 다음 reward 설계의 핵심 데이터 한 개 — "새 cold-start base 에서
   짧은 GT 탐지(R@0.3)가 0 을 벗어났나" (= short_bonus 항 설계 가능 여부).
   덤으로 lumping(단일 통짜) 최종 상태도 같이.

   IoU/파싱 로직은 lib_tvg.py(평가 원본 재현) 재사용 → eval_miou_summary 와 일치.
   GPU 불필요(test_results json 만 읽음). GT 절대길이 버킷은 duration 불필요.

사용:
  python posteval_gtlen.py --results <test_results_rank0.json> \
      --test_json <.../unav100_v0_500/_full.json> --out <gtlen_analysis.json> [--label NAME]
"""
import json, os, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib_tvg as L

# FINDINGS 1.2 와 동일 버킷
GTLEN_BUCKETS = [(0, 2, "<2s"), (2, 5, "2-5s"), (5, 10, "5-10s"),
                 (10, 20, "10-20s"), (20, 1e9, "20+s")]


def bucket_of(gtlen):
    for lo, hi, name in GTLEN_BUCKETS:
        if lo <= gtlen < hi:
            return name
    return "20+s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--test_json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="newrun")
    a = ap.parse_args()

    results = json.load(open(a.results))
    test = json.load(open(a.test_json))
    by_vid = {t["video"]: t for t in test}          # id(=video 경로) -> GT 샘플

    buck = {name: {"n": 0, "iou_sum": 0.0, "ge3": 0, "ge5": 0} for *_, name in GTLEN_BUCKETS}
    n_single = n_multi = 0
    iou_single = iou_multi = 0.0
    pred_nseg_hist = {}
    pred_nseg_sum = gt_nseg_sum = n_sample = n_decimal = n_matched = 0

    for r in results:
        # cdh train_qwen.py 산출 test_results 는 'id' 없이 'video' 키 사용.
        # hyj 러너(id=video 경로)와 양쪽 호환되도록 id→video fallback.
        t = by_vid.get(r.get("id") or r.get("video"))
        if t is None:
            continue
        n_matched += 1
        gt_segs = [(float(s), float(e)) for s, e in t["gt_segments"]]
        pred_segs = L.clean_segments(L.parse_pred_segments(r.get("pred", "")))

        n_sample += 1
        n_gt = len(gt_segs)
        gt_nseg_sum += n_gt
        np_ = len(pred_segs)
        pred_nseg_sum += np_
        pred_nseg_hist[np_] = pred_nseg_hist.get(np_, 0) + 1
        if any(abs((x * 10) % 10) > 0.01 for s, e in pred_segs for x in (s, e)):
            n_decimal += 1

        bests = L.best_ious_for_sample(gt_segs, pred_segs)
        sample_miou = sum(bests) / len(bests) if bests else 0.0
        if n_gt >= 2:
            n_multi += 1; iou_multi += sample_miou
        else:
            n_single += 1; iou_single += sample_miou
        for (s, e), iou in zip(gt_segs, bests):
            b = buck[bucket_of(e - s)]
            b["n"] += 1; b["iou_sum"] += iou
            b["ge3"] += (iou >= 0.3); b["ge5"] += (iou >= 0.5)

    def pct(x, n): return round(100.0 * x / n, 1) if n else 0.0
    summary = {
        "label": a.label, "n_results": len(results), "n_matched": n_matched,
        "gtlen_buckets": {name: {
            "n": buck[name]["n"],
            "mIoU": round(100 * buck[name]["iou_sum"] / buck[name]["n"], 1) if buck[name]["n"] else 0.0,
            "R@0.3": pct(buck[name]["ge3"], buck[name]["n"]),
            "R@0.5": pct(buck[name]["ge5"], buck[name]["n"]),
        } for *_, name in GTLEN_BUCKETS},
        "single": {"n": n_single, "mIoU": round(100 * iou_single / n_single, 1) if n_single else 0.0},
        "multi": {"n": n_multi, "mIoU": round(100 * iou_multi / n_multi, 1) if n_multi else 0.0},
        "pred_nseg_mean": round(pred_nseg_sum / n_sample, 2) if n_sample else 0.0,
        "gt_nseg_mean": round(gt_nseg_sum / n_sample, 2) if n_sample else 0.0,
        "pred_nseg_hist": dict(sorted(pred_nseg_hist.items())),
        "decimal_usage_pct": pct(n_decimal, n_sample),
    }
    json.dump(summary, open(a.out, "w"), ensure_ascii=False, indent=2)

    print("=" * 64)
    print(f"[GT길이/멀티세그 분해]  label={a.label}  matched={n_matched}/{len(results)}")
    print("=" * 64)
    print("\n### ★ GT 절대길이 버킷 — 짧은 GT 탐지(R@0.3)가 0 벗어났나?")
    print(f"{'bucket':>8} {'n':>5} {'mIoU':>7} {'R@0.3':>7} {'R@0.5':>7}")
    for *_, name in GTLEN_BUCKETS:
        b = summary["gtlen_buckets"][name]
        print(f"{name:>8} {b['n']:>5} {b['mIoU']:>7} {b['R@0.3']:>7} {b['R@0.5']:>7}")
    s, m = summary["single"], summary["multi"]
    print(f"\n### 단일 vs 멀티: single n={s['n']} mIoU={s['mIoU']} | multi n={m['n']} mIoU={m['mIoU']}")
    print(f"### lumping: pred 평균세그={summary['pred_nseg_mean']} (GT 평균={summary['gt_nseg_mean']}), "
          f"분포={summary['pred_nseg_hist']}")
    print(f"### 소수점 사용={summary['decimal_usage_pct']}%")
    print(f"\n✓ 저장: {a.out}")


if __name__ == "__main__":
    main()
