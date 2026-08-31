#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step6_confound_control.py — (보조) N_gt 를 통제한 뒤에도 A vs AV delta 차이가 남는가.

Step 3 경고: A 그룹이 AV 보다 multiseg 비율이 높다(43.3% vs 32.7%).
다중세그 샘플은 양 조건 모두 IoU 가 낮아 delta 여지 자체가 작으므로,
Step 5 의 "AV 에서 이득이 더 크다"가 난이도 교란의 산물일 수 있다.

두 가지로 통제한다:
  (1) N_gt 층별(1 / 2 / 3+) 로 delta_A, delta_AV 를 따로 내고 CI 를 붙인다.
  (2) 층 가중 표준화(direct standardization): AV 그룹의 N_gt 분포를 A 그룹의 분포에
      맞춰 재가중한 delta_AV* 를 만들어 delta_A 와 비교한다.
"""
import csv
import json
import random
import statistics
import sys

from paths import EVAL_DIR, OUT_MODALITY, PREDS, MAX_TIME, HERE, log

sys.path.insert(0, EVAL_DIR)
from eval_miou import (extract_answer_scope, parse_natural, parse_tokens,  # noqa
                       sample_iou_parts)
from count_f1 import _normalize_plain                                      # noqa

N_BOOT = 5000
SEED = 20260831
STRATA = ["1", "2", "3+"]


def stratum(n):
    return "1" if n == 1 else ("2" if n == 2 else "3+")


def parse_pred(raw, fmt):
    sc = extract_answer_scope(raw or "")
    return parse_tokens(sc, MAX_TIME) if fmt == "token" else parse_natural(_normalize_plain(sc), MAX_TIME)


def rows(model):
    path, fmt, gt_src = PREDS[model]
    out = []
    for x in json.load(open(path)):
        gt = ([list(s) for s in x.get("gt_segments") or []] if gt_src == "embedded"
              else parse_tokens(x.get("ref", ""), MAX_TIME))
        pd = parse_pred(x.get("pred", ""), fmt)
        out.append((x["gt_label"], len(gt), sample_iou_parts(gt, pd)[0] if gt else None))
    return out


def mean_delta(pairs):
    return 100.0 * (statistics.mean(a for a, _ in pairs) -
                    statistics.mean(b for _, b in pairs))


def main():
    log("STEP6", "start — N_gt 통제 후 A vs AV delta 재검정")
    label = {r["category"]: r["final_label"]
             for r in csv.DictReader(open(OUT_MODALITY, encoding="utf-8"))}
    w1, w0 = rows("titok"), rows("titok_wo_audio")
    assert [c for c, _, _ in w1] == [c for c, _, _ in w0]

    # cell[(group, stratum)] = [(iou_with, iou_without), ...]
    cell = {}
    for (cat, ng, i1), (_, _, i0) in zip(w1, w0):
        if i1 is None or i0 is None:
            continue
        cell.setdefault((label[cat], stratum(ng)), []).append((i1, i0))

    rng = random.Random(SEED)

    def ci(pairs, B=N_BOOT):
        bs = []
        for _ in range(B):
            s = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
            bs.append(mean_delta(s))
        bs.sort()
        return bs[int(0.025 * B)], bs[int(0.975 * B)]

    print("\n" + "=" * 92)
    print("Step 6 (보조) — N_gt 층별 audio delta (mIoU pt)")
    print("=" * 92)
    print(f"{'N_gt':>5}{'n(A)':>7}{'delta_A':>10}{'CI_A':>20}"
          f"{'n(AV)':>7}{'delta_AV':>10}{'CI_AV':>20}{'A-AV':>9}")
    print("-" * 92)
    out_rows = []
    for s in STRATA:
        pa, pav = cell.get(("A", s), []), cell.get(("AV", s), [])
        if not pa or not pav:
            continue
        da, dav = mean_delta(pa), mean_delta(pav)
        la, ha = ci(pa)
        lv, hv = ci(pav)
        print(f"{s:>5}{len(pa):>7}{da:>+10.2f}{f'[{la:+.2f}, {ha:+.2f}]':>20}"
              f"{len(pav):>7}{dav:>+10.2f}{f'[{lv:+.2f}, {hv:+.2f}]':>20}{da-dav:>+9.2f}")
        out_rows.append([s, len(pa), round(da, 3), round(la, 3), round(ha, 3),
                         len(pav), round(dav, 3), round(lv, 3), round(hv, 3),
                         round(da - dav, 3)])

    # ---------- (2) 층 가중 표준화 ----------
    #  AV 를 A 의 N_gt 분포로 재가중 -> 난이도 구성이 같아진 상태의 delta_AV*
    nA = {s: len(cell.get(("A", s), [])) for s in STRATA}
    totA = sum(nA.values())
    wts = {s: nA[s] / totA for s in STRATA}

    def std_delta(get):
        """get(s) -> pairs. A 의 층 비중으로 가중평균."""
        num = 0.0
        for s in STRATA:
            p = get(s)
            if p:
                num += wts[s] * mean_delta(p)
        return num

    d_A = std_delta(lambda s: cell.get(("A", s), []))
    d_AVs = std_delta(lambda s: cell.get(("AV", s), []))
    point = d_A - d_AVs
    diffs = []
    for _ in range(N_BOOT):
        def resamp(g):
            return lambda s: ([cell[(g, s)][rng.randrange(len(cell[(g, s)]))]
                               for _ in range(len(cell[(g, s)]))]
                              if cell.get((g, s)) else [])
        diffs.append(std_delta(resamp("A")) - std_delta(resamp("AV")))
    diffs.sort()
    lo, hi = diffs[int(0.025 * N_BOOT)], diffs[int(0.975 * N_BOOT)]
    p_gt = sum(1 for x in diffs if x > 0) / len(diffs)

    print("\n" + "-" * 92)
    print("N_gt 분포를 A 그룹 기준으로 맞춘 뒤(direct standardization) 비교")
    print("-" * 92)
    print(f"  층 가중치 (A 의 N_gt 분포): " + "  ".join(f"{s}={wts[s]:.3f}" for s in STRATA))
    print(f"  delta_A  (표준화)      = {d_A:+.2f}")
    print(f"  delta_AV (A 분포로 표준화) = {d_AVs:+.2f}")
    print(f"  차이 = {point:+.2f}   95% CI = [{lo:+.2f}, {hi:+.2f}]   P(>0) = {p_gt:.3f}")
    verdict = ("A 에서 더 크다" if lo > 0 else
               "AV 에서 더 크다" if hi < 0 else
               "★ 0 을 포함 — 난이도 통제 후에는 그룹 간 차이를 주장할 수 없다 ★")
    print(f"  판정: {verdict}")

    with open(f"{HERE}/confound_control.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["N_gt", "n_A", "delta_A", "ciA_lo", "ciA_hi",
                    "n_AV", "delta_AV", "ciAV_lo", "ciAV_hi", "delta_A_minus_AV"])
        w.writerows(out_rows)
        w.writerow([])
        w.writerow(["standardized(delta_A - delta_AV)", round(point, 4),
                    round(lo, 4), round(hi, 4), round(p_gt, 4), N_BOOT, SEED])
    log("STEP6", f"standardized delta_A-delta_AV = {point:+.2f} "
                 f"CI[{lo:+.2f},{hi:+.2f}] p={p_gt:.3f}")
    log("STEP6", f"saved {HERE}/confound_control.csv")


if __name__ == "__main__":
    main()
