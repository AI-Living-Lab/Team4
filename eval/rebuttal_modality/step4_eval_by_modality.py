#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step4_eval_by_modality.py — 예측 4종을 모달리티 그룹으로 stratify 해서 재평가.

metric 정의는 재구현하지 않고 Team4/eval/eval_miou.py 를 import 해서 그대로 쓴다:
  compute_method_block(samples, "sample"|"pairwise")  -> mIoU / F1@{0.1,0.3,0.5,0.7}
  compute_count_metrics(samples)                      -> CR*/SingleAcc/CountF1
CountF1 의 b(chance_floor)는 그룹 부분집합에서 계산되므로 자동으로 '그룹별 b'가 된다.
Step 3 의 b 와 일치하는지 대조해서 로그에 남긴다.

출력: results_by_modality.csv  (model x group x metric, TiTok delta 포함)
      results_by_modality.tex  (논문 표 조각)
"""
import csv
import json
import os
import sys

from paths import (EVAL_DIR, TEST_SPLIT, OUT_MODALITY, OUT_GROUPS, OUT_RESULTS,
                   OUT_LATEX, PREDS, MAX_TIME, log)

sys.path.insert(0, EVAL_DIR)
from eval_miou import (extract_answer_scope, parse_natural, parse_tokens,      # noqa
                       compute_method_block, compute_count_metrics)
from count_f1 import _normalize_plain                                          # noqa

GROUPS = ["A", "AV", "V", "H"]
THRS = ["0.1", "0.3", "0.5", "0.7"]


def parse_pred(raw, fmt):
    sc = extract_answer_scope(raw or "")
    return parse_tokens(sc, MAX_TIME) if fmt == "token" else parse_natural(_normalize_plain(sc), MAX_TIME)


def load_preds(path, fmt, gt_src):
    """-> [(gt_segs, pred_segs, category)]"""
    out = []
    for x in json.load(open(path)):
        gt = ([list(s) for s in x.get("gt_segments") or []] if gt_src == "embedded"
              else parse_tokens(x.get("ref", ""), MAX_TIME))
        out.append((gt, parse_pred(x.get("pred", ""), fmt), x["gt_label"]))
    return out


def metrics(samples):
    """samples: [(gt, pred)] -> 지표 dict (기존 정의 그대로)."""
    smp = compute_method_block(samples, "sample")
    pw = compute_method_block(samples, "pairwise")
    cnt = compute_count_metrics(samples)
    return {
        "n": len(samples),
        "mIoU": smp["mIoU_%"],
        "pwIoU": pw["mIoU_%"],
        **{f"F1@{t}": pw["F1"][t] for t in THRS},
        "USA": cnt["CR_star"], "OSA": cnt["SingleAcc"],
        "CountF1": cnt["CountF1"], "b": cnt["chance_floor"],
        "mean_N_pred": round(sum(len(p) for _, p in samples) / len(samples), 4),
    }


def main():
    log("STEP4", "start — 예측 파일 최초 참조 (Step 2 라벨링은 이미 확정·커밋됨)")
    label = {r["category"]: r["final_label"]
             for r in csv.DictReader(open(OUT_MODALITY, encoding="utf-8"))}
    step3_b = {r["group"]: float(r["b_countf1"]) if r["b_countf1"] else None
               for r in csv.DictReader(open(OUT_GROUPS, encoding="utf-8"))}
    split_ngt = {}
    for x in json.load(open(TEST_SPLIT)):
        split_ngt.setdefault(x["gt_label"], []).append(len(x.get("gt_segments") or []))

    rows = []
    for model, (path, fmt, gt_src) in PREDS.items():
        if not os.path.exists(path):
            log("STEP4", f"⚠ 없음, 건너뜀: {model} -> {path}")
            continue
        data = load_preds(path, fmt, gt_src)
        # --- 정합성 확인: 그룹별 표본 수가 Step 3 과 같아야 한다 ---
        by = {g: [] for g in GROUPS}
        unknown = 0
        for gt, pd, cat in data:
            g = label.get(cat)
            if g is None:
                unknown += 1
                continue
            by[g].append((gt, pd))
        if unknown:
            log("STEP4", f"⚠ {model}: 라벨 없는 카테고리 샘플 {unknown}건")
        log("STEP4", f"{model}: n={len(data)}  " +
            "  ".join(f"{g}={len(by[g])}" for g in GROUPS))

        for g in GROUPS + ["ALL"]:
            s = [x for gg in GROUPS for x in by[gg]] if g == "ALL" else by[g]
            if not s:
                continue
            m = metrics(s)
            if g != "ALL" and step3_b.get(g) is not None and m["b"] is not None:
                if abs(m["b"] - step3_b[g]) > 1e-4:
                    log("STEP4", f"⚠ {model}/{g}: b 불일치 step3={step3_b[g]} step4={m['b']}")
            rows.append({"model": model, "group": g, **m})

    with open(OUT_RESULTS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log("STEP4", f"saved {OUT_RESULTS}  rows={len(rows)}")

    # ---------------- 콘솔 표 ----------------
    idx = {(r["model"], r["group"]): r for r in rows}
    cols = ["n", "mIoU", "pwIoU", "F1@0.1", "F1@0.3", "F1@0.5", "F1@0.7",
            "USA", "OSA", "CountF1", "b", "mean_N_pred"]
    for model in PREDS:
        if (model, "ALL") not in idx:
            continue
        print("\n" + "=" * 108)
        print(f"{model}")
        print("=" * 108)
        hdr = f"{'group':>6}" + "".join(f"{c:>10}" for c in cols)
        print(hdr); print("-" * len(hdr))
        for g in GROUPS + ["ALL"]:
            r = idx.get((model, g))
            if not r:
                continue
            cells = []
            for c in cols:
                v = r[c]
                cells.append("-" if v is None else
                             (f"{v:>10d}" if c == "n" else f"{v:>10.4f}"))
            print(f"{g:>6}" + "".join(cells))

    # ---------------- TiTok audio ablation delta ----------------
    a, b_ = "titok", "titok_wo_audio"
    if (a, "ALL") in idx and (b_, "ALL") in idx:
        print("\n" + "=" * 108)
        print("TiTok audio ablation — delta = TiTok(w/ audio) - TiTok(w/o audio)")
        print("=" * 108)
        dcols = ["mIoU", "pwIoU", "F1@0.3", "F1@0.5", "F1@0.7", "USA", "OSA", "CountF1"]
        hdr = f"{'group':>6}{'n':>7}" + "".join(f"{c:>11}" for c in dcols)
        print(hdr); print("-" * len(hdr))
        for g in GROUPS + ["ALL"]:
            ra, rb = idx.get((a, g)), idx.get((b_, g))
            if not ra or not rb:
                continue
            cells = []
            for c in dcols:
                if ra[c] is None or rb[c] is None:
                    cells.append(f"{'-':>11}")
                else:
                    cells.append(f"{ra[c]-rb[c]:>+11.4f}")
            print(f"{g:>6}{ra['n']:>7}" + "".join(cells))

    write_latex(idx)
    log("STEP4", f"saved {OUT_LATEX}")


def write_latex(idx):
    def cell(r, c, nd=2):
        if r is None or r.get(c) is None:
            return "--"
        v = r[c]
        return f"{v:.{nd}f}" if c not in ("USA", "OSA", "CountF1") else f"{v:.3f}"

    L = []
    L.append("% results_by_modality.tex — Step 4 산출. \\usepackage{booktabs} 필요.")
    L.append("\\begin{table}[t]")
    L.append("\\centering\\small")
    L.append("\\caption{UnAV-100 test set stratified by boundary-defining modality. "
             "Categories are labelled from the class name alone, before any model "
             "output is inspected (A: sound-delimited, AV: both, V: vision-delimited, "
             "H: ambiguous). $b$ is the CountF1 chance floor, recomputed per group.}")
    L.append("\\label{tab:modality_breakdown}")
    L.append("\\begin{tabular}{llrrrrrrr}")
    L.append("\\toprule")
    L.append("Model & Group & $n$ & mIoU & F1@0.3 & F1@0.5 & F1@0.7 & CountF1 & $b$ \\\\")
    L.append("\\midrule")
    for model in PREDS:
        first = True
        for g in GROUPS + ["ALL"]:
            r = idx.get((model, g))
            if not r:
                continue
            name = model.replace("_", "\\_") if first else ""
            first = False
            L.append(f"{name} & {g} & {r['n']} & {cell(r,'mIoU')} & {cell(r,'F1@0.3')} & "
                     f"{cell(r,'F1@0.5')} & {cell(r,'F1@0.7')} & {cell(r,'CountF1')} & "
                     f"{cell(r,'b')} \\\\")
        L.append("\\midrule")
    L[-1] = "\\bottomrule"
    L.append("\\end{tabular}")
    L.append("\\end{table}")

    # audio ablation delta 표
    L.append("")
    L.append("\\begin{table}[t]")
    L.append("\\centering\\small")
    L.append("\\caption{Audio ablation on TiTok, stratified by boundary-defining "
             "modality. $\\Delta$ = with audio $-$ without audio (same checkpoint). "
             "Group-wise absolute values are not comparable across groups because the "
             "multi-segment ratio differs (see Sec.~X); only within-model $\\Delta$ is.}")
    L.append("\\label{tab:audio_ablation_modality}")
    L.append("\\begin{tabular}{lrrrrrr}")
    L.append("\\toprule")
    L.append("Group & $n$ & $\\Delta$mIoU & $\\Delta$F1@0.3 & $\\Delta$F1@0.5 & "
             "$\\Delta$F1@0.7 & $\\Delta$CountF1 \\\\")
    L.append("\\midrule")
    for g in GROUPS + ["ALL"]:
        ra, rb = idx.get(("titok", g)), idx.get(("titok_wo_audio", g))
        if not ra or not rb:
            continue
        def d(c, nd=2):
            if ra[c] is None or rb[c] is None:
                return "--"
            return f"{ra[c]-rb[c]:+.{nd}f}"
        L.append(f"{g} & {ra['n']} & {d('mIoU')} & {d('F1@0.3')} & {d('F1@0.5')} & "
                 f"{d('F1@0.7')} & {d('CountF1', 3)} \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\end{table}")
    open(OUT_LATEX, "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
