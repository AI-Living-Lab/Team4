#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2_modality_split.py — 3-run 라벨 -> 다수결/신뢰도/Fleiss' kappa -> unav100_modality_split.csv

⚠ 이 단계도 예측 파일을 열지 않는다. modality_labels.py 의 라벨은 카테고리 이름만 보고 매겼다.

confidence: 3/3 일치=high, 2/3=medium, 셋 다 다름=needs_review(final_label 은 run1 을 잠정 표기).
Fleiss' kappa: N=100 items, n=3 raters, k=4 categories(A/V/AV/H).
"""
import csv
from collections import Counter

from paths import OUT_MODALITY, OUT_CATEGORIES, log
from modality_labels import LABELS

CATS = ["A", "V", "AV", "H"]


def fleiss_kappa(rows_of_labels):
    """rows_of_labels: [[l1,l2,l3], ...] -> (kappa, P_bar, P_e, p_j)"""
    N = len(rows_of_labels)
    n = len(rows_of_labels[0])
    counts = [[r.count(c) for c in CATS] for r in rows_of_labels]
    P_i = [(sum(x * x for x in row) - n) / float(n * (n - 1)) for row in counts]
    P_bar = sum(P_i) / N
    p_j = [sum(row[j] for row in counts) / float(N * n) for j in range(len(CATS))]
    P_e = sum(p * p for p in p_j)
    kappa = (P_bar - P_e) / (1 - P_e) if (1 - P_e) > 0 else float("nan")
    return kappa, P_bar, P_e, p_j


def main():
    log("STEP2", "start — 카테고리 이름만 보고 3-run 라벨링 (예측 파일 미참조)")
    cats = [r[0] for r in list(csv.reader(open(OUT_CATEGORIES)))[1:]]

    out, runs_matrix, needs, medium = [], [], [], []
    for c in cats:
        r1, r2, r3, off, interm, contvis, why = LABELS[c]
        runs = [r1, r2, r3]
        runs_matrix.append(runs)
        cnt = Counter(runs)
        top, ntop = cnt.most_common(1)[0]
        if ntop == 3:
            conf, final = "high", top
        elif ntop == 2:
            conf, final = "medium", top
        else:
            conf, final = "needs_review", r1
        row = [c, r1, r2, r3, final, conf, off, interm, contvis, why]
        out.append(row)
        if conf == "needs_review":
            needs.append(row)
        elif conf == "medium":
            medium.append(row)

    with open(OUT_MODALITY, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "run1", "run2", "run3", "final_label", "confidence",
                    "offscreen_possible", "intermittent_sound", "continuous_visual",
                    "rationale"])
        w.writerows(out)
    log("STEP2", f"saved {OUT_MODALITY}")

    kappa, P_bar, P_e, p_j = fleiss_kappa(runs_matrix)
    pair = 0
    for r in runs_matrix:
        pair += sum(1 for a in range(3) for b in range(a + 1, 3) if r[a] == r[b])
    pair_agree = pair / (len(runs_matrix) * 3.0)

    dist = Counter(r[4] for r in out)
    conf_dist = Counter(r[5] for r in out)

    print("\n" + "=" * 78)
    print("Step 2 — 3-run 라벨링 결과")
    print("=" * 78)
    print(f"  final_label 분포 : " + "  ".join(f"{k}={dist.get(k,0)}" for k in CATS))
    print(f"  confidence 분포  : high={conf_dist['high']}  medium={conf_dist['medium']}"
          f"  needs_review={conf_dist['needs_review']}")
    print(f"\n  Fleiss' kappa    : {kappa:.4f}")
    print(f"    P_bar (관측일치) = {P_bar:.4f}   P_e (우연일치) = {P_e:.4f}")
    print(f"    카테고리 주변확률 p_j: " +
          "  ".join(f"{c}={p:.3f}" for c, p in zip(CATS, p_j)))
    print(f"    쌍별 단순일치율   = {pair_agree:.4f}")

    print("\n" + "-" * 78)
    print(f"[needs_review] {len(needs)}건 — 3-run 이 전부 갈림 (수동 검수 필요)")
    print("-" * 78)
    for r in needs:
        print(f"  {r[0]:<32} run1={r[1]:<2} run2={r[2]:<2} run3={r[3]:<2} -> 잠정 {r[4]}")
        print(f"      {r[9]}")
    print("\n" + "-" * 78)
    print(f"[confidence=medium] {len(medium)}건 — 2/3 다수결 (수동 검수 권장)")
    print("-" * 78)
    for r in medium:
        odd = [x for x in (r[1], r[2], r[3]) if x != r[4]]
        print(f"  {r[0]:<32} {r[1]}/{r[2]}/{r[3]} -> {r[4]:<2} (소수의견 {odd[0]})")
        print(f"      {r[9]}")
    log("STEP2", f"kappa={kappa:.4f}  high={conf_dist['high']} "
                 f"medium={conf_dist['medium']} needs_review={conf_dist['needs_review']}")


if __name__ == "__main__":
    main()
