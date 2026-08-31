#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step1_category_stats.py — 테스트 split 기준 카테고리 통계 -> categories_stats.csv

⚠ 이 단계는 예측 파일을 일절 열지 않는다(순환논증 방지). GT 와 영상 길이만 본다.

컬럼: category, n_samples, n_segments, mean_N_gt, multiseg_ratio,
      mean_seg_duration, median_seg_duration, mean_video_duration
  - multiseg_ratio    = N_gt>=2 인 샘플 비율
  - *_seg_duration    = 개별 GT 세그먼트 길이(초)의 평균/중앙값
  - mean_video_duration = 카테고리에 등장하는 샘플들의 영상 길이 평균(초).
                          같은 영상이 여러 샘플에 걸치면 샘플 단위로 중복 반영한다
                          (샘플 가중 평균 = 평가 표본의 실제 분포).
"""
import contextlib
import csv
import json
import os
import statistics
import wave

from paths import TEST_SPLIT, CLASS_LABELS, AUDIO_DIR, OUT_CATEGORIES, log


def video_duration(vid, _cache={}):
    """wav 헤더에서 길이(초). 원본 영상에서 추출한 오디오라 영상 길이와 같다."""
    if vid in _cache:
        return _cache[vid]
    p = os.path.join(AUDIO_DIR, f"{vid}.wav")
    if not os.path.exists(p):
        _cache[vid] = None
        return None
    with contextlib.closing(wave.open(p)) as w:
        d = w.getnframes() / float(w.getframerate())
    _cache[vid] = d
    return d


def main():
    log("STEP1", f"start — test split = {TEST_SPLIT}  (예측 파일 미참조)")
    rows = json.load(open(TEST_SPLIT))
    official = [l.strip() for l in open(CLASS_LABELS) if l.strip()]
    log("STEP1", f"samples={len(rows)}  official categories={len(official)}")

    per_cat = {}
    for x in rows:
        cat = x["gt_label"]
        segs = [list(s) for s in x.get("gt_segments") or []]
        d = per_cat.setdefault(cat, {"n": 0, "nseg": 0, "ngt": [], "seglen": [], "vdur": []})
        d["n"] += 1
        d["nseg"] += len(segs)
        d["ngt"].append(len(segs))
        d["seglen"].extend(max(0.0, e - s) for s, e in segs)
        vd = video_duration(x["vid"])
        if vd is not None:
            d["vdur"].append(vd)

    seen = set(per_cat)
    missing = [c for c in official if c not in seen]
    extra = [c for c in seen if c not in official]
    if missing:
        log("STEP1", f"⚠ 공식목록에 있으나 테스트 split 에 없는 카테고리 {len(missing)}: {missing}")
    if extra:
        log("STEP1", f"⚠ 공식목록에 없는 카테고리 {len(extra)}: {extra}")

    with open(OUT_CATEGORIES, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "n_samples", "n_segments", "mean_N_gt", "multiseg_ratio",
                    "mean_seg_duration", "median_seg_duration", "mean_video_duration"])
        for cat in sorted(per_cat):
            d = per_cat[cat]
            w.writerow([
                cat, d["n"], d["nseg"],
                round(statistics.mean(d["ngt"]), 4),
                round(sum(1 for g in d["ngt"] if g >= 2) / d["n"], 4),
                round(statistics.mean(d["seglen"]), 4) if d["seglen"] else "",
                round(statistics.median(d["seglen"]), 4) if d["seglen"] else "",
                round(statistics.mean(d["vdur"]), 4) if d["vdur"] else "",
            ])
    log("STEP1", f"saved {OUT_CATEGORIES}  (categories={len(per_cat)}, "
                 f"total_segments={sum(d['nseg'] for d in per_cat.values())})")


if __name__ == "__main__":
    main()
