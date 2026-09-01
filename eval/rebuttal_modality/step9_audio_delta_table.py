#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9_audio_delta_table.py — audio ON(ckpt-2000) vs audio OFF(noaudio ckpt-200) 을
unav100_modality_split_reviewed.csv 의 final_label 그룹별로 비교한 delta 표.

Step 4 와 다른 점: 비교 대상이 titok_wo_audio(같은 체크포인트 audio-drop) 가 아니라
audio 없이 학습된 별도 런(titok_noaudio_trained) 이다.

metric 은 재구현하지 않고 Team4/eval/eval_miou.py 를 그대로 import 한다.
ΔmIoU 는 샘플이 짝지어지므로 paired bootstrap 으로 95% CI 를 같이 낸다.

출력: audio_delta_by_modality.csv / .tex  (기존 .reviewed 산출물은 건드리지 않는다)
"""
import csv
import json
import os
import random
import sys

from paths import EVAL_DIR, TEST_SPLIT, PREDS, MAX_TIME, HERE, log

sys.path.insert(0, EVAL_DIR)
from eval_miou import (extract_answer_scope, parse_natural, parse_tokens,      # noqa
                       compute_method_block, compute_count_metrics,
                       sample_iou_parts)
from count_f1 import _normalize_plain                                          # noqa

MODALITY_CSV = os.environ.get(
    "MODALITY_CSV", f"{HERE}/unav100_modality_split_reviewed.csv")
ON, OFF = "titok", "titok_noaudio_trained"

# 표에 쓸 순서와 사람이 읽는 이름.
GROUPS = [("A",  "Sound-delimited"),
          ("AV", "Audio-visual co-occurring"),
          ("V",  "Visually-dominant (control)"),
          ("H",  "Ambiguous")]
N_BOOT = 5000
SEED = 20260901

OUT_CSV = f"{HERE}/audio_delta_by_modality.csv"
OUT_TEX = f"{HERE}/audio_delta_by_modality.tex"


def parse_pred(raw, fmt):
    sc = extract_answer_scope(raw or "")
    return parse_tokens(sc, MAX_TIME) if fmt == "token" else parse_natural(
        _normalize_plain(sc), MAX_TIME)


def load(model):
    """-> {key: (category, gt_segs, pred_segs)}

    두 런은 같은 split 을 돌지만 파일 내 순서가 다르다(rank 분할 순서 차이).
    그래서 위치가 아니라 (video, gt_label, ref) 키로 짝을 맞춘다 — 이 3-튜플은
    두 파일 모두에서 3,455개가 전부 유일하다.
    """
    path, fmt, gt_src = PREDS[model]
    out = {}
    for x in json.load(open(path)):
        gt = ([list(s) for s in x.get("gt_segments") or []] if gt_src == "embedded"
              else parse_tokens(x.get("ref", ""), MAX_TIME))
        k = (x["video"], x["gt_label"], x["ref"])
        assert k not in out, f"{model}: 키 중복 {k}"
        out[k] = (x["gt_label"], gt, parse_pred(x.get("pred", ""), fmt))
    return out


def metrics(samples):
    smp = compute_method_block(samples, "sample")
    pw = compute_method_block(samples, "pairwise")
    cnt = compute_count_metrics(samples)
    return {
        "n": len(samples),
        "mIoU": smp["mIoU_%"],
        "F1@0.5": pw["F1"]["0.5"],
        "F1@0.7": pw["F1"]["0.7"],
        "USA": cnt["CR_star"],
        "OSA": cnt["SingleAcc"],
        "CountF1": cnt["CountF1"],
        "mean_N_pred": round(sum(len(p) for _, p in samples) / len(samples), 4),
    }


def boot_ci(pairs, rng):
    """pairs: [(iou_on, iou_off)] -> ΔmIoU 의 95% percentile CI (percent point)."""
    if len(pairs) < 2:
        return (None, None)
    n = len(pairs)
    ds = []
    for _ in range(N_BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        on = sum(pairs[i][0] for i in idx) / n
        off = sum(pairs[i][1] for i in idx) / n
        ds.append((on - off) * 100.0)
    ds.sort()
    return (ds[int(0.025 * N_BOOT)], ds[int(0.975 * N_BOOT) - 1])


def main():
    log("STEP9", f"start — {ON} vs {OFF}, labels={os.path.basename(MODALITY_CSV)}")
    label = {r["category"]: r["final_label"]
             for r in csv.DictReader(open(MODALITY_CSV, encoding="utf-8"))}
    ncat = {}
    for cat, g in label.items():
        ncat[g] = ncat.get(g, 0) + 1

    on, off = load(ON), load(OFF)
    assert set(on) == set(off), (
        f"샘플 집합 불일치 (on-only={len(set(on)-set(off))}, "
        f"off-only={len(set(off)-set(on))})")
    keys = list(on)
    log("STEP9", f"paired n={len(keys)} (키 기준 정렬, 파일 순서 무시)")

    # split 의 카테고리별 표본 수와도 대조 (라벨 CSV 가 split 과 어긋나면 잡힌다)
    split_cats = {x["gt_label"] for x in json.load(open(TEST_SPLIT))}
    missing = split_cats - set(label)
    if missing:
        log("STEP9", f"⚠ 라벨 없는 카테고리 {len(missing)}개: {sorted(missing)[:5]}")

    by = {g: {"on": [], "off": [], "iou": []} for g, _ in GROUPS}
    for k in keys:
        cat, gt, p_on = on[k]
        _, _, p_off = off[k]
        g = label.get(cat)
        if g not in by:
            continue
        by[g]["on"].append((gt, p_on))
        by[g]["off"].append((gt, p_off))
        if gt:
            by[g]["iou"].append((sample_iou_parts(gt, p_on)[0],
                                 sample_iou_parts(gt, p_off)[0]))

    rng = random.Random(SEED)
    rows = []
    for g, name in GROUPS + [("ALL", "합계")]:
        if g == "ALL":
            m_on = metrics([s for gg, _ in GROUPS for s in by[gg]["on"]])
            m_off = metrics([s for gg, _ in GROUPS for s in by[gg]["off"]])
            pairs = [p for gg, _ in GROUPS for p in by[gg]["iou"]]
            nc = sum(ncat.get(gg, 0) for gg, _ in GROUPS)
        else:
            if not by[g]["on"]:
                continue
            m_on, m_off = metrics(by[g]["on"]), metrics(by[g]["off"])
            pairs = by[g]["iou"]
            nc = ncat.get(g, 0)
        lo, hi = boot_ci(pairs, rng)
        rows.append({
            "group": g, "group_name": name, "n_categories": nc, "n": m_on["n"],
            "d_mIoU": round(m_on["mIoU"] - m_off["mIoU"], 4),
            "d_mIoU_ci_lo": None if lo is None else round(lo, 4),
            "d_mIoU_ci_hi": None if hi is None else round(hi, 4),
            "d_F1@0.5": round(m_on["F1@0.5"] - m_off["F1@0.5"], 4),
            "d_F1@0.7": round(m_on["F1@0.7"] - m_off["F1@0.7"], 4),
            "d_USA": round(m_on["USA"] - m_off["USA"], 4),
            "d_OSA": round(m_on["OSA"] - m_off["OSA"], 4),
            "d_CountF1": round(m_on["CountF1"] - m_off["CountF1"], 4),
            **{f"on_{k}": v for k, v in m_on.items() if k != "n"},
            **{f"off_{k}": v for k, v in m_off.items() if k != "n"},
        })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log("STEP9", f"saved {OUT_CSV}  rows={len(rows)}")

    # ---------------- 요청받은 표 ----------------
    print()
    print(f"Δ = audio ON ({ON}, ckpt-2000)  −  audio OFF ({OFF}, ckpt-200)")
    print("=" * 118)
    hdr = (f"{'Category group':<30}{'category':>9}{'n':>7}{'ΔmIoU':>10}"
           f"{'ΔF1@0.5':>10}{'ΔF1@0.7':>10}{'ΔUSA':>9}{'ΔOSA':>9}"
           f"{'ΔCountF1':>11}{'ΔmIoU 95% CI':>22}")
    print(hdr)
    print("-" * 118)
    for r in rows:
        ci = ("—" if r["d_mIoU_ci_lo"] is None
              else f"[{r['d_mIoU_ci_lo']:+.2f}, {r['d_mIoU_ci_hi']:+.2f}]")
        print(f"{r['group_name']:<30}{r['n_categories']:>9}{r['n']:>7}"
              f"{r['d_mIoU']:>+10.2f}{r['d_F1@0.5']:>+10.2f}{r['d_F1@0.7']:>+10.2f}"
              f"{r['d_USA']:>+9.3f}{r['d_OSA']:>+9.3f}"
              f"{r['d_CountF1']:>+11.3f}{ci:>22}")
    print("=" * 118)

    # 절대값도 같이 — delta 만 보면 어느 쪽이 낮아서 생긴 차이인지 안 보인다.
    print("\n절대값 (ON / OFF)")
    cols = ["mIoU", "F1@0.5", "F1@0.7", "USA", "OSA", "CountF1", "mean_N_pred"]
    hdr = f"{'Category group':<30}{'n':>7}" + "".join(f"{c:>18}" for c in cols)
    print(hdr); print("-" * len(hdr))
    for r in rows:
        cells = "".join(f"{r['on_'+c]:>8.2f} /{r['off_'+c]:>8.2f}" for c in cols)
        print(f"{r['group_name']:<30}{r['n']:>7}{cells}")

    write_latex(rows)
    log("STEP9", f"saved {OUT_TEX}")


def write_latex(rows):
    L = ["% audio_delta_by_modality.tex — Step 9 산출. \\usepackage{booktabs} 필요.",
         "\\begin{table}[t]", "\\centering\\small",
         "\\caption{Effect of audio on UnAV-100, stratified by boundary-defining "
         "modality. $\\Delta$ = audio-on model $-$ audio-off-trained model. "
         "Categories are labelled from the class name alone, before any model output "
         "is inspected. CI is a paired percentile bootstrap ($B=5000$) over per-sample "
         "IoU.}",
         "\\label{tab:audio_delta_modality}",
         "\\begin{tabular}{lrrrrrrrr}", "\\toprule",
         "Category group & \\#cat & $n$ & $\\Delta$mIoU & $\\Delta$F1@0.5 & "
         "$\\Delta$F1@0.7 & $\\Delta$USA & $\\Delta$OSA & $\\Delta$CountF1 \\\\",
         "\\midrule"]
    for r in rows:
        if r["group"] == "ALL":
            L.append("\\midrule")
        L.append(f"{r['group_name']} & {r['n_categories']} & {r['n']} & "
                 f"{r['d_mIoU']:+.2f} & {r['d_F1@0.5']:+.2f} & {r['d_F1@0.7']:+.2f} & "
                 f"{r['d_USA']:+.3f} & {r['d_OSA']:+.3f} & "
                 f"{r['d_CountF1']:+.3f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(OUT_TEX, "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
