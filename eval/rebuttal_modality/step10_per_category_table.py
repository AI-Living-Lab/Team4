#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step10_per_category_table.py — TiTok(audio on) 과 w/o audio 를 카테고리 100개
각각에 대해 따로 집계한 표.

Step 9 는 모달리티 그룹(A/AV/V/H) 단위였다. 여기서는 그룹으로 묶지 않고
카테고리별로 metric 을 낸다. 카테고리당 n 이 평균 35 밖에 안 되므로 개별 수치는
노이즈가 크다 — 정렬/필터용으로 쓰고 단독 주장 근거로는 쓰지 말 것.

metric 은 재구현하지 않고 Team4/eval/eval_miou.py 를 그대로 import 한다.
짝맞춤은 Step 9 와 같이 (video, gt_label, ref) 키로 한다 (파일 순서가 다르다).

출력: per_category_<model>.csv  (모델별로 따로)
      per_category_compare.csv  (두 모델 + delta 를 한 행에 놓은 wide 표)
"""
import csv
import os
import sys

from paths import PREDS, HERE, log

sys.path.insert(0, HERE)
from step9_audio_delta_table import load, metrics, MODALITY_CSV               # noqa

ON, OFF = "titok", "titok_noaudio_trained"
MODELS = [(ON, "TiTok (audio on, ckpt-2000)"),
          (OFF, "TiTok w/o audio (noaudio ckpt-200)")]

# 모달리티 그룹은 정렬/그룹핑용으로만 쓴다 — 집계 단위는 어디까지나 카테고리다.
GROUP_ORDER = {"A": 0, "AV": 1, "V": 2, "H": 3}
COLS = ["mIoU", "F1@0.5", "F1@0.7", "USA", "OSA", "CountF1", "mean_N_pred"]


def main():
    log("STEP10", f"start — per-category, models={[m for m, _ in MODELS]}")
    label = {r["category"]: r["final_label"]
             for r in csv.DictReader(open(MODALITY_CSV, encoding="utf-8"))}

    data = {m: load(m) for m, _ in MODELS}
    keys = set(data[ON]) & set(data[OFF])
    assert len(keys) == len(data[ON]) == len(data[OFF]), "샘플 집합 불일치"
    log("STEP10", f"paired n={len(keys)}  categories={len(label)}")

    # category -> model -> [(gt, pred)]
    per = {}
    for k in keys:
        cat = data[ON][k][0]
        d = per.setdefault(cat, {m: [] for m, _ in MODELS})
        for m, _ in MODELS:
            _, gt, pred = data[m][k]
            d[m].append((gt, pred))

    missing = set(label) - set(per)
    if missing:
        log("STEP10", f"⚠ split 에 샘플이 없는 카테고리 {len(missing)}개: {sorted(missing)}")

    stats = {}   # (model, cat) -> metric dict
    for cat, d in per.items():
        for m, _ in MODELS:
            stats[(m, cat)] = metrics(d[m])

    cats = sorted(per, key=lambda c: (GROUP_ORDER.get(label.get(c), 9), c))

    # ---------------- 모델별 표 (따로따로) ----------------
    for m, title in MODELS:
        print("\n" + "=" * 122)
        print(title)
        print("=" * 122)
        hdr = (f"{'grp':>4}  {'category':<30}{'n':>6}" +
               "".join(f"{c:>13}" for c in COLS))
        print(hdr); print("-" * len(hdr))
        prev = None
        for cat in cats:
            g = label.get(cat, "?")
            if prev is not None and g != prev:
                print("-" * len(hdr))
            prev = g
            r = stats[(m, cat)]
            print(f"{g:>4}  {cat:<30}{r['n']:>6}" +
                  "".join(f"{r[c]:>13.4f}" for c in COLS))

        path = f"{HERE}/per_category_{m}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["category", "group", "n"] + COLS)
            for cat in cats:
                r = stats[(m, cat)]
                w.writerow([cat, label.get(cat, ""), r["n"]] + [r[c] for c in COLS])
        log("STEP10", f"saved {path}  rows={len(cats)}")

    # ---------------- wide 비교 CSV ----------------
    path = f"{HERE}/per_category_compare.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "group", "n"] +
                   [f"on_{c}" for c in COLS] +
                   [f"off_{c}" for c in COLS] +
                   [f"d_{c}" for c in COLS])
        for cat in cats:
            a, b = stats[(ON, cat)], stats[(OFF, cat)]
            w.writerow([cat, label.get(cat, ""), a["n"]] +
                       [a[c] for c in COLS] + [b[c] for c in COLS] +
                       [round(a[c] - b[c], 4) for c in COLS])
    log("STEP10", f"saved {path}  rows={len(cats)}")

    # ΔmIoU 양극단 — 어느 카테고리가 그룹 평균을 끌고 가는지 확인용.
    dm = sorted(((stats[(ON, c)]["mIoU"] - stats[(OFF, c)]["mIoU"], c) for c in cats),
                reverse=True)
    print("\nΔmIoU top 10 (audio 이득 큰 순)")
    for d, c in dm[:10]:
        print(f"  {d:>+8.2f}  [{label.get(c,'?'):>2}] {c}  (n={stats[(ON,c)]['n']})")
    print("\nΔmIoU bottom 10 (audio 가 손해인 순)")
    for d, c in dm[-10:]:
        print(f"  {d:>+8.2f}  [{label.get(c,'?'):>2}] {c}  (n={stats[(ON,c)]['n']})")


if __name__ == "__main__":
    main()
