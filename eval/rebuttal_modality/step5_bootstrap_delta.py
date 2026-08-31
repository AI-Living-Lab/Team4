#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step5_bootstrap_delta.py — (보조) audio 효과의 그룹 간 차이에 신뢰구간을 붙인다.

리뷰어의 요구는 사실상 "A(sound-delimited) 에서 audio 이득이 더 큰가?" 이다.
Step 4 의 점추정만으로는 그 주장을 못 한다. 여기서는 샘플 단위 IoU 를 paired 로
부트스트랩해서 delta_A - delta_AV 의 95% CI 를 낸다.

  delta_G = mean_{i in G} IoU_i(w/ audio) - mean_{i in G} IoU_i(w/o audio)
  관심량   = delta_A - delta_AV        (0 을 포함하면 "A 에서 더 크다"고 못 박는다)

IoU 는 eval_miou.sample_iou_parts (sample mIoU 와 같은 정의) 를 그대로 쓴다.
"""
import csv
import json
import random
import statistics
import sys

from paths import EVAL_DIR, OUT_MODALITY, PREDS, MAX_TIME, HERE, log

sys.path.insert(0, EVAL_DIR)
from eval_miou import (extract_answer_scope, parse_natural, parse_tokens,   # noqa
                       sample_iou_parts)
from count_f1 import _normalize_plain                                       # noqa

N_BOOT = 5000
SEED = 20260831


def parse_pred(raw, fmt):
    sc = extract_answer_scope(raw or "")
    return parse_tokens(sc, MAX_TIME) if fmt == "token" else parse_natural(_normalize_plain(sc), MAX_TIME)


def per_sample_iou(model):
    path, fmt, gt_src = PREDS[model]
    out = []
    for x in json.load(open(path)):
        gt = ([list(s) for s in x.get("gt_segments") or []] if gt_src == "embedded"
              else parse_tokens(x.get("ref", ""), MAX_TIME))
        pd = parse_pred(x.get("pred", ""), fmt)
        out.append((x["gt_label"], sample_iou_parts(gt, pd)[0] if gt else None))
    return out


def main():
    log("STEP5", f"start — paired bootstrap (B={N_BOOT}, seed={SEED})")
    label = {r["category"]: r["final_label"]
             for r in csv.DictReader(open(OUT_MODALITY, encoding="utf-8"))}
    with_a = per_sample_iou("titok")
    wo_a = per_sample_iou("titok_wo_audio")
    assert len(with_a) == len(wo_a)
    # 두 파일은 같은 split 을 같은 순서로 돈다 — 카테고리 시퀀스로 확인
    assert [c for c, _ in with_a] == [c for c, _ in wo_a], "샘플 정렬 불일치"

    by = {}
    for (cat, i1), (_, i0) in zip(with_a, wo_a):
        if i1 is None or i0 is None:
            continue
        by.setdefault(label[cat], []).append((i1, i0))

    rng = random.Random(SEED)

    def delta(pairs):
        return 100.0 * (statistics.mean(a for a, _ in pairs) -
                        statistics.mean(b for _, b in pairs))

    print("\n" + "=" * 84)
    print(f"Step 5 (보조) — audio 이득의 paired bootstrap  B={N_BOOT}")
    print("=" * 84)
    print(f"{'group':>6}{'n':>7}{'delta mIoU':>13}{'95% CI':>26}")
    print("-" * 84)
    boots = {}
    for g in ["A", "AV", "V", "H"]:
        pairs = by.get(g, [])
        if not pairs:
            continue
        d = delta(pairs)
        bs = []
        for _ in range(N_BOOT):
            samp = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
            bs.append(delta(samp))
        bs.sort()
        lo, hi = bs[int(0.025 * N_BOOT)], bs[int(0.975 * N_BOOT)]
        boots[g] = (pairs, bs)
        print(f"{g:>6}{len(pairs):>7}{d:>+13.2f}{f'[{lo:+.2f}, {hi:+.2f}]':>26}")

    # ---- 관심량: delta_A - delta_AV ----
    if "A" in boots and "AV" in boots:
        pa, pav = boots["A"][0], boots["AV"][0]
        point = delta(pa) - delta(pav)
        diffs = []
        for _ in range(N_BOOT):
            sa = [pa[rng.randrange(len(pa))] for _ in range(len(pa))]
            sav = [pav[rng.randrange(len(pav))] for _ in range(len(pav))]
            diffs.append(delta(sa) - delta(sav))
        diffs.sort()
        lo, hi = diffs[int(0.025 * N_BOOT)], diffs[int(0.975 * N_BOOT)]
        p_gt = sum(1 for x in diffs if x > 0) / len(diffs)
        print("\n" + "-" * 84)
        print("관심량: delta_A - delta_AV   (>0 이면 sound-delimited 에서 audio 이득이 더 큼)")
        print("-" * 84)
        print(f"  점추정 = {point:+.2f} mIoU pt")
        print(f"  95% CI = [{lo:+.2f}, {hi:+.2f}]")
        print(f"  P(delta_A > delta_AV) = {p_gt:.3f}")
        verdict = ("A 에서 이득이 더 크다고 말할 수 있다" if lo > 0 else
                   "AV 에서 이득이 더 크다고 말할 수 있다" if hi < 0 else
                   "★ CI 가 0 을 포함 — 그룹 간 차이를 주장할 수 없다 ★")
        print(f"  판정: {verdict}")
        log("STEP5", f"delta_A-delta_AV = {point:+.2f} CI[{lo:+.2f},{hi:+.2f}] p={p_gt:.3f}")

        with open(f"{HERE}/bootstrap_delta.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["quantity", "point_estimate", "ci95_low", "ci95_high",
                        "P(>0)", "n_boot", "seed"])
            for g in boots:
                bs = boots[g][1]
                w.writerow([f"delta_mIoU[{g}]", round(delta(boots[g][0]), 4),
                            round(bs[int(0.025*N_BOOT)], 4), round(bs[int(0.975*N_BOOT)], 4),
                            "", N_BOOT, SEED])
            w.writerow(["delta_A - delta_AV", round(point, 4), round(lo, 4), round(hi, 4),
                        round(p_gt, 4), N_BOOT, SEED])
        log("STEP5", f"saved {HERE}/bootstrap_delta.csv")


if __name__ == "__main__":
    main()
