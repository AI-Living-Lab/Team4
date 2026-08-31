#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step8_apply_review.py — 사람이 검수한 CSV 를 검증하고 자동 라벨 대비 변경분을 요약한다.

검증 항목
  1) 카테고리 집합이 categories_stats.csv 와 정확히 일치하는가(누락/오타/중복)
  2) final_label 이 A/V/AV/H 안에 있는가
  3) changed 플래그와 실제 (auto_label != final_label) 이 어긋나지 않는가
통과하면 Step 3~6 을 이 라벨로 재실행할 수 있다(MODALITY_CSV 환경변수).
"""
import csv
import sys
from collections import Counter

from paths import OUT_CATEGORIES, OUT_MODALITY, log

VALID = {"A", "V", "AV", "H"}


def main(path):
    cats = [r[0] for r in list(csv.reader(open(OUT_CATEGORIES, encoding="utf-8-sig")))[1:]]
    auto = {r["category"]: r["final_label"]
            for r in csv.DictReader(open(OUT_MODALITY, encoding="utf-8-sig"))}
    conf = {r["category"]: r["confidence"]
            for r in csv.DictReader(open(OUT_MODALITY, encoding="utf-8-sig"))}
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))

    errs = []
    seen = Counter(r["category"] for r in rows)
    dup = [c for c, n in seen.items() if n > 1]
    missing = [c for c in cats if c not in seen]
    extra = [c for c in seen if c not in set(cats)]
    if dup:
        errs.append(f"중복 카테고리 {len(dup)}: {dup[:5]}")
    if missing:
        errs.append(f"누락 카테고리 {len(missing)}: {missing[:5]}")
    if extra:
        errs.append(f"알 수 없는 카테고리 {len(extra)}: {extra[:5]}")
    bad = [(r["category"], r["final_label"]) for r in rows
           if r["final_label"] not in VALID]
    if bad:
        errs.append(f"허용되지 않는 라벨 {len(bad)}: {bad[:5]}")

    if errs:
        print("검증 실패:")
        for e in errs:
            print("  ✗ " + e)
        log("STEP8", "검증 실패 — " + "; ".join(errs))
        return 1

    new = {r["category"]: r["final_label"] for r in rows}
    flag_mismatch = [r["category"] for r in rows
                     if (r.get("changed") == "1") != (r["auto_label"] != r["final_label"])]

    changes = [(c, auto[c], new[c], conf[c]) for c in cats if auto[c] != new[c]]
    dist_old = Counter(auto[c] for c in cats)
    dist_new = Counter(new[c] for c in cats)

    print("=" * 76)
    print(f"검수본 검증 통과 — {len(rows)} 카테고리")
    print("=" * 76)
    print("  라벨 분포   " + "   ".join(
        f"{g}: {dist_old.get(g,0)} → {dist_new.get(g,0)} ({dist_new.get(g,0)-dist_old.get(g,0):+d})"
        for g in ["A", "V", "AV", "H"]))
    print(f"  변경 건수   {len(changes)} / {len(cats)}")
    if flag_mismatch:
        print(f"  ⚠ changed 플래그 불일치 {len(flag_mismatch)}건 (final_label 기준으로 진행): "
              f"{flag_mismatch[:5]}")

    by_conf = Counter(c[3] for c in changes)
    print(f"  변경분의 원래 confidence  " +
          "  ".join(f"{k}={v}" for k, v in by_conf.most_common()))

    if changes:
        print("\n" + "-" * 76)
        print("변경 내역")
        print("-" * 76)
        for c, a, n, cf in changes:
            tag = {"high": "  ", "medium": "2/3", "needs_review": "!!!"}.get(cf, "")
            print(f"  {tag:>3} {c:<34} {a:>2} → {n:<2}")
        print("\n  전이 행렬 (auto → reviewed)")
        tr = Counter((a, n) for _, a, n, _ in changes)
        for (a, n), k in sorted(tr.items(), key=lambda x: -x[1]):
            print(f"    {a:>2} → {n:<2}  {k}")
    log("STEP8", f"검수본 검증 통과 — 변경 {len(changes)}건, "
                 f"분포 {dict(dist_new)}")
    return 0


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "unav100_modality_split.reviewed.csv"
    sys.exit(main(p))
