#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step11_group_abs_table.py — 모달리티 그룹별 '절대값' 표를 모델마다 따로.

Step 9 는 두 모델의 delta 였다. 여기서는 delta 를 빼지 않고 audio on / w/o audio
각각의 원값을 같은 형식으로 낸다. F1 은 4개 threshold 를 모두 싣는다.

metric 은 재구현하지 않고 Team4/eval/eval_miou.py 를 그대로 import 한다.
CountF1 의 chance floor b 는 그룹 부분집합에서 계산되므로 자동으로 그룹별 b 다.

출력: group_abs_<model>.csv, group_abs.tex
"""
import csv
import sys

from paths import EVAL_DIR, HERE, log

sys.path.insert(0, EVAL_DIR)
sys.path.insert(0, HERE)
from eval_miou import compute_method_block, compute_count_metrics             # noqa
from step9_audio_delta_table import load, MODALITY_CSV                        # noqa

ON, OFF = "titok", "titok_noaudio_trained"
MODELS = [(ON, "TiTok (audio on, ckpt-2000)"),
          (OFF, "TiTok w/o audio (noaudio ckpt-200)")]

GROUPS = [("A",  "Sound-delimited"),
          ("AV", "Audio-visual co-occurring"),
          ("V",  "Visually-dominant (control)"),
          ("H",  "Ambiguous")]
THRS = ["0.1", "0.3", "0.5", "0.7"]
COLS = ["mIoU"] + [f"F1@{t}" for t in THRS] + ["USA", "OSA", "CountF1"]


def metrics(samples):
    smp = compute_method_block(samples, "sample")
    pw = compute_method_block(samples, "pairwise")
    cnt = compute_count_metrics(samples)
    return {
        "n": len(samples),
        "mIoU": smp["mIoU_%"],
        **{f"F1@{t}": pw["F1"][t] for t in THRS},
        "USA": cnt["CR_star"], "OSA": cnt["SingleAcc"],
        "CountF1": cnt["CountF1"], "b": cnt["chance_floor"],
        "mean_N_pred": round(sum(len(p) for _, p in samples) / len(samples), 4),
    }


def main():
    log("STEP11", f"start — group absolute values, models={[m for m, _ in MODELS]}")
    label = {r["category"]: r["final_label"]
             for r in csv.DictReader(open(MODALITY_CSV, encoding="utf-8"))}
    ncat = {}
    for g in label.values():
        ncat[g] = ncat.get(g, 0) + 1

    data = {m: load(m) for m, _ in MODELS}
    assert set(data[ON]) == set(data[OFF]), "샘플 집합 불일치"
    log("STEP11", f"paired n={len(data[ON])}")

    out = {}   # model -> [row]
    for m, title in MODELS:
        by = {g: [] for g, _ in GROUPS}
        for cat, gt, pred in data[m].values():
            g = label.get(cat)
            if g in by:
                by[g].append((gt, pred))
        rows = []
        for g, name in GROUPS + [("ALL", "합계")]:
            s = ([x for gg, _ in GROUPS for x in by[gg]] if g == "ALL" else by[g])
            if not s:
                continue
            nc = (sum(ncat.get(gg, 0) for gg, _ in GROUPS) if g == "ALL"
                  else ncat.get(g, 0))
            rows.append({"group": g, "group_name": name,
                         "n_categories": nc, **metrics(s)})
        out[m] = rows

        print("\n" + "=" * 128)
        print(title)
        print("=" * 128)
        hdr = (f"{'Category group':<30}{'category':>9}{'n':>7}" +
               "".join(f"{c:>10}" for c in COLS) + f"{'b':>9}{'meanNp':>9}")
        print(hdr); print("-" * len(hdr))
        for r in rows:
            if r["group"] == "ALL":
                print("-" * len(hdr))
            print(f"{r['group_name']:<30}{r['n_categories']:>9}{r['n']:>7}" +
                  "".join(f"{r[c]:>10.4f}" for c in COLS) +
                  f"{r['b']:>9.4f}{r['mean_N_pred']:>9.4f}")

        path = f"{HERE}/group_abs_{m}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        log("STEP11", f"saved {path}  rows={len(rows)}")

    write_latex(out)
    log("STEP11", f"saved {HERE}/group_abs.tex")


def write_latex(out):
    L = ["% group_abs.tex — Step 11 산출. \\usepackage{booktabs} 필요."]
    for m, title in MODELS:
        L += ["\\begin{table}[t]", "\\centering\\small",
              f"\\caption{{{title.replace('_', chr(92)+'_')} on UnAV-100, stratified "
              "by boundary-defining modality. Categories are labelled from the class "
              "name alone, before any model output is inspected. Absolute values are "
              "not comparable across groups because the multi-segment ratio differs.}",
              f"\\label{{tab:group_abs_{m}}}",
              "\\begin{tabular}{lrrrrrrrrr}", "\\toprule",
              "Category group & \\#cat & $n$ & mIoU & F1@0.1 & F1@0.3 & F1@0.5 & "
              "F1@0.7 & USA & OSA \\\\", "\\midrule"]
        for r in out[m]:
            if r["group"] == "ALL":
                L.append("\\midrule")
            L.append(f"{r['group_name']} & {r['n_categories']} & {r['n']} & " +
                     " & ".join(f"{r[c]:.2f}" for c in COLS[:6]) +
                     f" & {r['USA']:.3f} & {r['OSA']:.3f} \\\\")
        L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    open(f"{HERE}/group_abs.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
