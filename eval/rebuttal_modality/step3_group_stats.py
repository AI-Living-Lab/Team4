#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step3_group_stats.py — final_label 그룹(A/V/AV/H)별 난이도 통계 -> group_stats.csv

그룹 간 난이도가 다르면 audio 효과와 교란되므로, 그 크기를 먼저 재고 경고한다.
CountF1 의 baseline b 는 그룹마다 GT 분포가 달라서 반드시 그룹별로 재계산해야 한다:
    b_group = mean_{N_gt>=2 인 샘플} 1/N_gt
"""
import csv
import json
import statistics

from paths import TEST_SPLIT, OUT_MODALITY, OUT_GROUPS, log

GROUPS = ["A", "V", "AV", "H"]


def main():
    log("STEP3", "start — 그룹별 난이도/교란 변수 집계 (예측 파일 미참조)")
    label = {r["category"]: r["final_label"]
             for r in csv.DictReader(open(OUT_MODALITY, encoding="utf-8"))}
    rows = json.load(open(TEST_SPLIT))

    g = {k: {"cats": set(), "ngt": [], "seglen": [], "inv": []} for k in GROUPS}
    for x in rows:
        cat = x["gt_label"]
        k = label[cat]
        segs = x.get("gt_segments") or []
        d = g[k]
        d["cats"].add(cat)
        d["ngt"].append(len(segs))
        d["seglen"].extend(max(0.0, e - s) for s, e in segs)
        if len(segs) >= 2:
            d["inv"].append(1.0 / len(segs))

    out = []
    for k in GROUPS:
        d = g[k]
        n = len(d["ngt"])
        if n == 0:
            continue
        out.append({
            "group": k,
            "n_categories": len(d["cats"]),
            "n_samples": n,
            "n_multiseg": len(d["inv"]),
            "mean_N_gt": round(statistics.mean(d["ngt"]), 4),
            "multiseg_ratio": round(len(d["inv"]) / n, 4),
            "mean_seg_duration": round(statistics.mean(d["seglen"]), 4) if d["seglen"] else "",
            "b_countf1": round(statistics.mean(d["inv"]), 6) if d["inv"] else "",
        })

    with open(OUT_GROUPS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    log("STEP3", f"saved {OUT_GROUPS}")

    hdr = (f"{'group':>6}{'n_cat':>7}{'n_samp':>8}{'n_multi':>9}{'mean_N_gt':>11}"
           f"{'multiseg%':>11}{'seg_dur':>9}{'b':>9}")
    print("\n" + "=" * 78)
    print("Step 3 — 그룹별 통계 (교란 변수 점검)")
    print("=" * 78)
    print(hdr)
    print("-" * len(hdr))
    for r in out:
        print(f"{r['group']:>6}{r['n_categories']:>7}{r['n_samples']:>8}{r['n_multiseg']:>9}"
              f"{r['mean_N_gt']:>11.4f}{100*r['multiseg_ratio']:>10.2f}%"
              f"{r['mean_seg_duration']:>9.2f}{r['b_countf1']:>9.4f}")

    # ---------- 교란 경고 ----------
    print("\n" + "!" * 78)
    warn = []
    big = [r for r in out if r["n_samples"] >= 100]     # 표본이 너무 작은 그룹은 비교 대상에서 뺀다
    small = [r for r in out if r["n_samples"] < 100]
    if small:
        warn.append("표본 과소 그룹: " + ", ".join(
            f"{r['group']}(n_cat={r['n_categories']}, n={r['n_samples']})" for r in small)
            + " -> 그룹 단위 지표가 불안정하다. 리뷰탈 표에서는 수치를 싣되 "
              "'단일 카테고리 기반'임을 각주로 밝히거나 A/AV 대비만 주장할 것.")
    if len(big) >= 2:
        mr = {r["group"]: r["multiseg_ratio"] for r in big}
        ng = {r["group"]: r["mean_N_gt"] for r in big}
        d_mr = max(mr.values()) - min(mr.values())
        d_ng = max(ng.values()) - min(ng.values())
        print(f"  multiseg_ratio 격차 = {d_mr:.4f}  ({', '.join(f'{k}={v:.3f}' for k,v in mr.items())})")
        print(f"  mean_N_gt      격차 = {d_ng:.4f}  ({', '.join(f'{k}={v:.3f}' for k,v in ng.items())})")
        if d_mr >= 0.05:
            warn.append(f"multiseg_ratio 가 그룹 간 {d_mr:.3f} 차이난다 -> 다중세그 난이도가 "
                        "audio 효과와 교란된다. 그룹 간 절대값 비교 금지, "
                        "모델 내 delta(=w/ audio - w/o audio) 로만 주장할 것.")
        if d_ng >= 0.10:
            warn.append(f"mean_N_gt 가 그룹 간 {d_ng:.3f} 차이난다 -> CountF1 의 b 도 달라지므로 "
                        "b 를 그룹별로 재계산해 쓴다(Step 4 에서 적용).")
    if warn:
        for i, w_ in enumerate(warn, 1):
            print(f"\n  [경고 {i}] {w_}")
    else:
        print("  그룹 간 난이도 격차가 임계 미만 — 특이사항 없음.")
    print("!" * 78)
    log("STEP3", f"warnings={len(warn)}")


if __name__ == "__main__":
    main()
