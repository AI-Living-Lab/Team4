"""Boosted Natural Language parser — cdh 정책 (no tail / no format hint) baseline 평가.

지원 패턴 (순차 시도, 중복 허용, 마지막에 merge):
  A. "from X to Y (seconds)"               — basic
  B. "X - Y (seconds)"                     — dash
  C. "starts at X ... (continues|lasts|runs|ends|finishes) (until|at|to) Y"
  D. "first N seconds"                     → [0, N]
  E. "throughout (the entire|the whole)? video" → [0, max_time]
  F. "HH:MM:SS - HH:MM:SS" / "MM:SS - MM:SS"
  G. "HH:MM:SS ... (until|to|and) HH:MM:SS"

단독으로 start 만 명시된 경우는 skip (끝 없이 IoU 계산 불가).
Seconds / sec / s 단위 선택.

cdh 표준 인터페이스:
  --results --test_json --max_time --out_dir
출력:
  {out_dir}/eval_miou_summary.json (Union-IoU + FP_rate + FN_rate)
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_utils import score_sample, summarize, print_report

# ---------- 정규식 패턴 ----------

# HH:MM:SS 또는 MM:SS
HMS_STR = r"(?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?"

# 숫자 (소수점 허용) — 초 단위
NUM = r"\d+(?:\.\d+)?"

# unit — seconds, second, sec, s (단, 혼란 방지로 단위 명시한 것만 잡음)
UNIT_OPT = r"(?:\s*(?:seconds?|secs?|s\b))?"

# 기본 "from X (seconds?) to Y seconds" — 단위가 X 뒤에도 있을 수 있음 허용
PAT_FROM_TO = re.compile(
    rf"(?:from\s+)?({NUM})(?:\s*(?:seconds?|secs?|s\b))?\s*(?:to|till|until|\-|–|~|and)\s*({NUM})\s*(?:seconds?|secs?|s\b)(?:\s+mark)?",
    re.IGNORECASE,
)

# "starts at/around/approximately X ... ends at/until/to Y"
PAT_STARTS_UNTIL = re.compile(
    rf"(?:starts?|begins?|commences?|occurs?|happens?)\s+(?:at|from|around|approximately|about|in)?\s*({NUM})\s*(?:seconds?|secs?)?"
    rf"[\s\S]{{0,120}}?"
    rf"(?:continues?|lasts?|runs?|ends?|finishes?|goes?|stops?|concludes?)\s+(?:until|at|to|till|around|approximately|about)?\s*({NUM})\s*(?:seconds?|secs?)",
    re.IGNORECASE,
)

# "first N seconds" → [0, N]
PAT_FIRST_N = re.compile(
    rf"(?:in the\s+|for the\s+|the\s+)?first\s+({NUM})\s*(?:seconds?|secs?)",
    re.IGNORECASE,
)

# "throughout (the entire|the whole|the)? video" → [0, max_time]
PAT_THROUGHOUT = re.compile(
    r"throughout\s+(?:the\s+(?:entire|whole|full)\s+)?video",
    re.IGNORECASE,
)

# "HH:MM:SS ... HH:MM:SS" (both ends as HH:MM:SS)
PAT_HMS_RANGE = re.compile(
    rf"({HMS_STR})\s*(?:\-|–|to|till|until|and)\s*({HMS_STR})",
    re.IGNORECASE,
)

# "between X and Y seconds"
PAT_BETWEEN = re.compile(
    rf"between\s+({NUM})\s*(?:seconds?|secs?)?\s+and\s+({NUM})\s*(?:seconds?|secs?)",
    re.IGNORECASE,
)


def hms_to_sec(s: str) -> float:
    parts = s.split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return float(s)


def _add(out, s, e, max_time):
    s = max(0.0, min(float(s), max_time))
    e = max(0.0, min(float(e), max_time))
    if e < s:
        s, e = e, s
    if e <= s:
        return
    out.append([s, e])


def parse_segments(raw: str, max_time: float = 60.0, duration_hint=None):
    """raw text → list of [s,e] in seconds. Duration hint 가 있으면 'throughout' 용."""
    out = []

    # F. HH:MM:SS range 먼저 (숫자 추출 전에)
    for m in PAT_HMS_RANGE.finditer(raw):
        try:
            s = hms_to_sec(m.group(1))
            e = hms_to_sec(m.group(2))
            _add(out, s, e, max_time)
        except Exception:
            pass

    # A. from X to Y seconds
    for m in PAT_FROM_TO.finditer(raw):
        _add(out, m.group(1), m.group(2), max_time)

    # C. starts at X ... until Y
    for m in PAT_STARTS_UNTIL.finditer(raw):
        _add(out, m.group(1), m.group(2), max_time)

    # G. between X and Y seconds
    for m in PAT_BETWEEN.finditer(raw):
        _add(out, m.group(1), m.group(2), max_time)

    # D. first N seconds → [0, N]
    for m in PAT_FIRST_N.finditer(raw):
        _add(out, 0.0, m.group(1), max_time)

    # E. throughout → [0, duration_hint or max_time]
    if PAT_THROUGHOUT.search(raw):
        end = float(duration_hint) if duration_hint else max_time
        _add(out, 0.0, end, max_time)

    # merge near-duplicate / overlapping segments (sorted + deduped)
    if not out:
        return []
    out.sort()
    merged = [list(out[0])]
    for s, e in out[1:]:
        # 동일 or overlap 이면 병합
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(x) for x in merged]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--test_json", required=True)
    ap.add_argument("--max_time", type=float, default=60.0)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.results))
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    with open(args.test_json) as f:
        test_data = json.load(f)
    with open(args.results) as f:
        raw_results = json.load(f)

    print(f"[1/3] Test: {len(test_data)}, Results: {len(raw_results)}")

    all_ious = []
    total_preds = fp_preds = total_gts = fn_gts = 0
    parse_ok = parse_fail = 0
    fail_samples = []  # 파싱 실패한 pred 일부 저장 (보강 용)

    for result, gt_item in zip(raw_results, test_data):
        gt_segments = gt_item.get("gt_segments", [])
        if not gt_segments:
            continue
        if result.get("error"):
            parse_fail += 1
            for _ in gt_segments:
                all_ious.append(0.0)
            total_gts += len(gt_segments)
            fn_gts += len(gt_segments)
            continue
        # duration hint: gt_segments 의 max end 또는 max_time
        dur_hint = max(float(s[1]) for s in gt_segments)
        pred_segs = parse_segments(result.get("pred", ""), max_time=args.max_time,
                                   duration_hint=dur_hint)
        if not pred_segs:
            parse_fail += 1
            for _ in gt_segments:
                all_ious.append(0.0)
            total_gts += len(gt_segments)
            fn_gts += len(gt_segments)
            if len(fail_samples) < 20:
                fail_samples.append({"id": result.get("id"), "pred": result.get("pred", "")[:200]})
            continue
        parse_ok += 1
        per_gt, n_fp, n_pred, n_fn, n_gt = score_sample(gt_segments, pred_segs)
        all_ious.extend(per_gt)
        total_preds += n_pred
        fp_preds += n_fp
        total_gts += n_gt
        fn_gts += n_fn

    summary = summarize(all_ious, total_preds, fp_preds, total_gts, fn_gts,
                        parse_ok, parse_fail, len(raw_results))
    summary["parser"] = "NL boosted (from-to | dash | starts-until | first-N | throughout | HMS | between)"

    print_report("NL-Boosted Baseline Eval — Union-IoU", summary)

    out_path = os.path.join(out_dir, "eval_miou_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] {out_path}")

    # 파싱 실패 샘플 저장 (보강 포인트 분석용)
    if fail_samples:
        fail_path = os.path.join(out_dir, "parse_fail_samples.json")
        with open(fail_path, "w") as f:
            json.dump(fail_samples, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] {fail_path} (first 20 parse-fail preds)")


if __name__ == "__main__":
    main()
