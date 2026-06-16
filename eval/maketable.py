#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
maketable.py — 폴더 경로를 주면 그 하위의 모든 평가 summary 를 탐색해
<경로>/table.txt 에 단일 표(탭 구분)를 생성한다. 재실행 시 최신 내용으로 덮어쓴다.

결과 폴더(=pairwise_miou_summary.json 이 있는 폴더)당 1행. n_samples<500 은 제외.

열(탭 구분, 25개):
  ID  ckpt  sample_mIoU
  AVG_357  AVG_3579  AVG_13579           # = mean(sample_mIoU, F1_avg, R_avg, P_avg)
  AVGf1_357  AVGf1_3579  AVGf1_13579     # = mean(sample_mIoU, F1_avg)
  gt(mean)  pred(mean)  n_samples  testset
  F1@0.1/0.3/0.5/0.7/0.9  R@0.1/0.3/0.5/0.7/0.9  P@0.1/0.3/0.5/0.7/0.9
  F1_avg357 R_avg357 P_avg357  F1_avg3579 R_avg3579 P_avg3579  F1_avg13579 R_avg13579 P_avg13579
  - sample_mIoU : sample_miou_summary.json 의 mIoU_%
  - 그 외 F1/R/P : pairwise_miou_summary.json 기준
  - *_avgNNN    : threshold 부분집합 평균 (357=.3.5.7 / 3579=.3.5.7.9 / 13579=전부)
  - gt/pred(mean): gt_segments.mean_per_sample, pred_segments.mean_per_sample (각 열)

사용:  python3 maketable.py /home/aix23102/audiolm/Team404/outputs/gdpo
"""
import argparse
import json
import os
import re

THS = ["0.1", "0.3", "0.5", "0.7", "0.9"]
AVG_SETS = [("357", ["0.3", "0.5", "0.7"]),
            ("3579", ["0.3", "0.5", "0.7", "0.9"]),
            ("13579", ["0.1", "0.3", "0.5", "0.7", "0.9"])]
SUFS = [suf for suf, _ in AVG_SETS]

HEADER = (["ID", "ckpt", "sample_mIoU"]
          + [f"AVG_{s}" for s in SUFS]
          + [f"AVGf1_{s}" for s in SUFS]
          + ["gt(mean)", "pred(mean)", "n_samples", "testset",
             "F1@" + "/".join(THS), "R@" + "/".join(THS), "P@" + "/".join(THS)]
          + [c for s in SUFS for c in (f"F1_avg{s}", f"R_avg{s}", f"P_avg{s}")])


def _num(x):
    try:
        return f"{float(x):.2f}"
    except (TypeError, ValueError):
        return "-"


def _isnum(x):
    try:
        float(x); return True
    except (TypeError, ValueError):
        return False


def _avg(d, keys):
    if not isinstance(d, dict):
        return "-"
    vals = [float(d[k]) for k in keys if k in d and _isnum(d[k])]
    return f"{sum(vals)/len(vals):.2f}" if vals else "-"


def _mean_str(strvals):
    vals = [float(v) for v in strvals if _isnum(v)]
    return f"{sum(vals)/len(vals):.2f}" if vals else "-"


def _series(d, keys=THS):
    if not isinstance(d, dict):
        return "/".join("-" for _ in keys)
    return "/".join(_num(d.get(k)) for k in keys)


def parse_path(dirpath):
    """결과 폴더 절대경로 → (ID, ckpt, testset). root 위치와 무관하게 동작
    (leaf 폴더를 직접 가리켜도 OK). checkpoint-* 앞 컴포넌트를 ID 로 본다."""
    parts = [p for p in os.path.abspath(dirpath).split(os.sep) if p]
    testset = parts[-1] if parts else "-"
    ckpt, ck_idx = "-", None
    for i, p in enumerate(parts):
        m = re.fullmatch(r"checkpoint-(\w+)", p)
        if m:
            ckpt, ck_idx = m.group(1), i
            break
    if ck_idx is not None and ck_idx > 0:
        ID = parts[ck_idx - 1]                      # checkpoint-* 바로 앞 = run 이름
    else:
        ID = "-"
        for stage in ("gdpo", "base", "sft", "merged"):   # checkpoint 없는 경우
            if stage in parts and parts.index(stage) + 1 < len(parts):
                ID = parts[parts.index(stage) + 1]
                break
    return ID, ckpt, testset


def _load(path):
    try:
        return json.load(open(path))
    except (json.JSONDecodeError, OSError):
        return {}


def main():
    ap = argparse.ArgumentParser(description="폴더 하위 평가 summary → table.txt (단일 표, 탭 구분)")
    ap.add_argument("path", help="탐색 루트 (예: outputs/gdpo). 이 안에 table.txt 생성")
    ap.add_argument("--out", default="table.txt", help="출력 파일명 (기본 table.txt)")
    args = ap.parse_args()
    root = os.path.abspath(args.path)
    if not os.path.isdir(root):
        raise SystemExit(f"[에러] 폴더가 아님: {root}")

    rows = []
    for dirpath, _d, files in os.walk(root):
        if "pairwise_miou_summary.json" not in files:
            continue
        p = _load(os.path.join(dirpath, "pairwise_miou_summary.json"))
        s = _load(os.path.join(dirpath, "sample_miou_summary.json"))
        if (p.get("n_samples") or 0) < 500:
            continue
        ID, ckpt, testset = parse_path(dirpath)
        sample = _num(s.get("mIoU_%"))

        # set 별 F1/R/P avg
        f1a = [_avg(p.get("F1"), ks) for _s, ks in AVG_SETS]
        ra  = [_avg(p.get("Recall"), ks) for _s, ks in AVG_SETS]
        pa  = [_avg(p.get("Precision"), ks) for _s, ks in AVG_SETS]

        # 복합 평균
        comp   = [_mean_str([sample, f1a[i], ra[i], pa[i]]) for i in range(len(SUFS))]  # AVG_NNN
        compf1 = [_mean_str([sample, f1a[i]]) for i in range(len(SUFS))]                # AVGf1_NNN

        gmean = _num(p.get("gt_segments", {}).get("mean_per_sample"))
        pmean = _num(p.get("pred_segments", {}).get("mean_per_sample"))

        set_avgs = []
        for i in range(len(SUFS)):
            set_avgs += [f1a[i], ra[i], pa[i]]

        rows.append([ID, ckpt, sample]
                    + comp + compf1
                    + [gmean, pmean, str(p.get("n_samples", "-")), testset,
                       _series(p.get("F1")), _series(p.get("Recall")), _series(p.get("Precision"))]
                    + set_avgs)

    # AVG_13579 (열 인덱스 5) 내림차순
    rows.sort(key=lambda r: (float(r[5]) if _isnum(r[5]) else -1.0), reverse=True)

    lines = ["\t".join(HEADER)] + ["\t".join(r) for r in rows]
    out_path = os.path.join(root, args.out)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, out_path)
    print(f"[SAVED] {out_path}  ({len(rows)} 행, {len(HEADER)} 열)")


if __name__ == "__main__":
    main()
