#!/usr/bin/env python3
"""
eval_all_miou_multiseg.py
  - 출력폴더 경로를 입력하면 하위의 모든 test_results_rank0.json 을 찾아
    같은 위치에 eval_all_miou_summary.json 을 생성한다.
  - 멀티세그 QA 형식의 inference 결과에 대해 세 가지 mIoU 를 함께 계산:
      * All-IoU   : 샘플 안의 모든 GT/모든 pred 를 각각 하나의 타임라인으로 merge 한 뒤
                    IoU(=|G∩P|/|G∪P|) 를 구함. 평균 단위 = 샘플.
      * Best-IoU  : GT 세그먼트마다 IoU 가 가장 높은 pred 1개와 매칭 → 샘플 내 평균.
      * Union-IoU : GT 세그먼트마다 그와 겹치는 pred 들의 합집합과의 IoU → 샘플 내 평균.
    (Best/Union 도 비교를 위해 All 과 같은 '샘플 단위'로 평균낸 뒤 전체 평균)
  - 모델 응답: "From <t...> to <t...>. From <t...> to <t...>."
  - GT(gt_segments) 소스 해석 순서:
      1) --test_json 으로 명시한 경우 그 파일
      2) results 항목에 gt_segments 가 박혀 있으면 그대로 사용
      3) test_results 가 있는 리프 폴더명으로 <test_dir>/<리프명>.json (또는 <리프명>/_full.json)
    매칭은 (basename(video/id), gt_label) 키 → 실패 시 동일 길이일 때 positional fallback.
"""
import argparse
import datetime as _dt
import json
import os
import re


# ----------------------------- 시간 토큰 파싱 -----------------------------
def decode_vtg_time(token_str, max_time=9999.9):
    has_dot = "<tdot>" in token_str
    if has_dot:
        parts = token_str.split("<tdot>")
        int_part = re.findall(r"<t(\d)>", parts[0])
        dec_part = re.findall(r"<t(\d)>", parts[1]) if len(parts) > 1 else []
    else:
        int_part = re.findall(r"<t(\d)>", token_str)
        dec_part = []
    if not int_part:
        return None
    integer_part = int("".join(int_part))
    decimal_part = int(dec_part[0]) if dec_part else 0
    t = integer_part + decimal_part / 10.0
    return min(t, max_time)


def parse_multi_segments(raw, max_time=9999.9):
    """Parse 'From X to Y. From X to Y.' format into list of segments."""
    segments = []
    pattern = r"[Ff]rom\s+((?:<t\d>)+(?:<tdot>(?:<t\d>)+)?)\s+to\s+((?:<t\d>)+(?:<tdot>(?:<t\d>)+)?)"
    for m in re.finditer(pattern, raw):
        start = decode_vtg_time(m.group(1), max_time)
        end = decode_vtg_time(m.group(2), max_time)
        if start is not None and end is not None:
            if end <= start:
                # 비정상 예측은 보수적으로 최소 단위(0.1초)만 부여 — 점수 이득 최소화
                end = min(start + 0.1, max_time)
            segments.append([start, end])
    return segments


# ----------------------------- 구간 연산 헬퍼 -----------------------------
def compute_tiou(seg1, seg2):
    inter_s = max(seg1[0], seg2[0])
    inter_e = min(seg1[1], seg2[1])
    inter = max(0.0, inter_e - inter_s)
    union = (seg1[1] - seg1[0]) + (seg2[1] - seg2[0]) - inter
    return inter / union if union > 0 else 0.0


def merge_intervals(intervals):
    """Sort & merge overlapping 1D intervals → list of [start, end]."""
    if not intervals:
        return []
    s = sorted([list(x) for x in intervals])
    out = [list(s[0])]
    for a, b in s[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def intervals_total_len(intervals):
    return sum(max(0.0, b - a) for a, b in intervals)


def intervals_intersect(a_list, b_list):
    """두 병합된 구간 리스트의 교집합."""
    out, i, j = [], 0, 0
    while i < len(a_list) and j < len(b_list):
        s = max(a_list[i][0], b_list[j][0])
        e = min(a_list[i][1], b_list[j][1])
        if e > s:
            out.append([s, e])
        if a_list[i][1] < b_list[j][1]:
            i += 1
        else:
            j += 1
    return out


# ----------------------------- 세 가지 IoU -----------------------------
def compute_all_iou(gt_segs, pred_segs):
    """All-IoU: 샘플 내 GT 전체 merge vs pred 전체 merge 의 집합 IoU."""
    G = merge_intervals(gt_segs)
    P = merge_intervals(pred_segs)
    union = intervals_total_len(merge_intervals(G + P))
    if union <= 0:
        return 0.0
    inter = intervals_total_len(intervals_intersect(G, P))
    return inter / union


def compute_best_iou_per_gt(gt_seg, pred_segs):
    """Best-IoU: GT 1개 vs 가장 IoU 가 높은 pred 1개."""
    if not pred_segs:
        return 0.0
    return max(compute_tiou(gt_seg, p) for p in pred_segs)


def compute_union_iou_per_gt(gt_seg, pred_segs):
    """Union-IoU: GT 1개 vs 그와 겹치는 pred 들의 합집합 사이의 IoU."""
    overlapping = [
        p for p in pred_segs
        if min(p[1], gt_seg[1]) > max(p[0], gt_seg[0])
    ]
    if not overlapping:
        return 0.0
    U = merge_intervals(overlapping)
    G = [list(gt_seg)]
    inter_len = intervals_total_len(intervals_intersect(G, U))
    union_len = intervals_total_len(merge_intervals(G + U))
    return inter_len / union_len if union_len > 0 else 0.0


# ----------------------------- GT 소스 해석 -----------------------------
def _basename(p):
    return os.path.basename(str(p)) if p else ""


def _coerce_segments(val):
    """gt_segments 값을 [[s,e],...] 로 정규화. list 또는 '(s,e), (s,e)' 문자열 모두 허용."""
    if not val:
        return []
    if isinstance(val, str):
        out = []
        for m in re.finditer(r"\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)", val):
            out.append([float(m.group(1)), float(m.group(2))])
        return out
    out = []
    for s in val:
        try:
            out.append([float(s[0]), float(s[1])])
        except (TypeError, ValueError, IndexError):
            continue
    return out


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _default_test_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    # <WORKSPACE>/master/Team4/eval -> <WORKSPACE>/data/test
    return os.path.normpath(os.path.join(here, "..", "..", "..", "data", "test"))


def resolve_gt_json_path(results_file, test_dir):
    """리프 폴더명으로 GT json 경로 추정. flat(<leaf>.json) 우선, 없으면 <leaf>/_full.json."""
    leaf = os.path.basename(os.path.dirname(os.path.abspath(results_file)))
    flat = os.path.join(test_dir, f"{leaf}.json")
    if os.path.isfile(flat):
        return flat
    nested = os.path.join(test_dir, leaf, "_full.json")
    if os.path.isfile(nested):
        return nested
    return None


def build_gt_for_results(results, results_file, args):
    """
    results 와 길이가 같은 gt_segments 리스트 + 메타 반환.
    return: (gt_list, source_str, matched, unmatched)
    """
    n = len(results)

    # (1) 결과에 이미 박혀 있으면 그대로
    if all("gt_segments" in r for r in results) and not args.test_json:
        gt_list = [_coerce_segments(r.get("gt_segments")) for r in results]
        matched = sum(1 for g in gt_list if g)
        return gt_list, "embedded", matched, n - matched

    # (2) test_json 경로 결정
    gt_path = args.test_json or resolve_gt_json_path(results_file, args.test_dir)
    if not gt_path or not os.path.isfile(gt_path):
        raise FileNotFoundError(
            f"GT json 을 찾을 수 없음 (results={results_file}). "
            f"--test_json 으로 직접 지정하세요. (탐색 test_dir={args.test_dir})"
        )
    gt_data = _load_json(gt_path)

    # (3) (basename(video/id), gt_label) 키 매핑 구성
    key_map = {}
    ambiguous = set()
    for item in gt_data:
        vid = item.get("video") or item.get("id") or item.get("audio")
        key = (_basename(vid), item.get("gt_label", ""))
        segs = _coerce_segments(item.get("gt_segments"))
        if key in key_map and key_map[key] != segs:
            ambiguous.add(key)
        key_map[key] = segs

    gt_list, matched, unmatched = [], 0, 0
    use_positional = len(gt_data) == n
    for i, r in enumerate(results):
        key = (_basename(r.get("id")), r.get("gt_label", ""))
        if key in key_map and key not in ambiguous:
            gt_list.append(key_map[key])
            matched += 1
        elif use_positional:
            gt_list.append(_coerce_segments(gt_data[i].get("gt_segments")))
            matched += 1
        else:
            gt_list.append([])
            unmatched += 1
    return gt_list, gt_path, matched, unmatched


# ----------------------------- 한 폴더 평가 -----------------------------
def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


# 5개 mIoU 블록 제목 (출력/콘솔에서 동일하게 사용)
BLOCK_TITLES = [
    "All_IoU (샘플 단위)",
    "Best_IoU (샘플 단위)",
    "Union_IoU (샘플 단위)",
    "Best_IoU (GT세그먼트 단위)",
    "Union_IoU (GT세그먼트 단위)",
]


def compute_core_summary(samples):
    """samples: list of (gt_segments, pred_segments) → 5개 블록 + 진단 dict.

    pred 파싱 방식과 무관하게 동일한 All/Best/Union 집계를 보장한다
    (eval_all_miou_multiseg_natural.py 가 base 모델 파서로 이 함수를 재사용).
    """
    all_sample, best_sample, union_sample = [], [], []   # 샘플 단위 (샘플당 1개)
    best_perseg, union_perseg = [], []                   # 세그먼트 단위 (GT세그먼트당 1개)
    parse_ok = parse_fail = 0
    total_preds = fp_preds = 0
    n_gt_segments = 0

    for gt_segments, pred_segments in samples:
        n_gt_segments += len(gt_segments)
        if pred_segments:
            parse_ok += 1
        else:
            parse_fail += 1

        # FP 세그먼트 카운트: 어떤 GT 와도 안 겹치는 pred
        for p in pred_segments:
            total_preds += 1
            if not any(min(p[1], g[1]) > max(p[0], g[0]) for g in gt_segments):
                fp_preds += 1

        # All-IoU: 샘플 단위 (본질적으로 샘플당 1개 → 세그먼트 단위 버전 없음)
        all_sample.append(compute_all_iou(gt_segments, pred_segments))

        # Best/Union: GT 세그먼트마다 IoU 1개씩
        best_ious = [compute_best_iou_per_gt(g, pred_segments) for g in gt_segments]
        union_ious = [compute_union_iou_per_gt(g, pred_segments) for g in gt_segments]
        best_perseg.extend(best_ious)
        union_perseg.extend(union_ious)
        best_sample.append(_mean(best_ious))
        union_sample.append(_mean(union_ious))

    n = len(all_sample)
    thresholds = [0.3, 0.5, 0.7]

    def recall(arr):
        m = len(arr)
        return {str(th): round(100.0 * (sum(1 for x in arr if x >= th) / m if m else 0.0), 4)
                for th in thresholds}

    def block(arr):
        return {"mIoU_%": round(100.0 * _mean(arr), 4), "R@1": recall(arr)}

    fp_rate = fp_preds / total_preds if total_preds > 0 else 0.0
    fn_samples = sum(1 for x in all_sample if x == 0.0)
    fn_rate = fn_samples / n if n > 0 else 0.0

    return {
        # --- mIoU 총 5개 (각 블록: mIoU_% + R@1) ---
        BLOCK_TITLES[0]: block(all_sample),
        BLOCK_TITLES[1]: block(best_sample),
        BLOCK_TITLES[2]: block(union_sample),
        BLOCK_TITLES[3]: block(best_perseg),
        BLOCK_TITLES[4]: block(union_perseg),
        # --- 진단 ---
        "FP_rate_%": round(100.0 * fp_rate, 4),
        "FN_rate_%": round(100.0 * fn_rate, 4),
        "n_samples": n,
        "n_gt_segments": n_gt_segments,
        "n_pred_segments": total_preds,
        "n_fp_segments": fp_preds,
        "n_fn_samples": fn_samples,
        "parse_ok": parse_ok,
        "parse_fail": parse_fail,
    }


def write_and_report(results_file, summary, args, out_name="eval_all_miou_summary.json"):
    """summary 를 results_file 옆에 저장하고 콘솔 리포트 출력."""
    out_path = os.path.join(os.path.dirname(os.path.abspath(results_file)), out_name)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    os.replace(tmp, out_path)

    if not args.quiet:
        SEP = "=" * 56
        n = summary["n_samples"]
        print(f"\n{SEP}")
        print(f"  {results_file}")
        print(SEP)
        print(f"  GT source:   {summary.get('gt_source')}  "
              f"(matched={summary.get('gt_matched')}, unmatched={summary.get('gt_unmatched')})")
        if summary.get("pred_format"):
            print(f"  pred_format: {summary['pred_format']}")
        print(f"  Samples:     {n}   GT segs: {summary['n_gt_segments']}   "
              f"Pred segs: {summary['n_pred_segments']}")
        print(f"  Parse OK:    {summary['parse_ok']} ({summary['parse_ok']*100/max(n,1):.1f}%)   "
              f"fail: {summary['parse_fail']} ({summary['parse_fail']*100/max(n,1):.1f}%)")
        print(f"  -- mIoU 5개 (mIoU% | R@.3 / R@.5 / R@.7) --")
        for title in BLOCK_TITLES:
            b = summary[title]
            r = b["R@1"]
            print(f"  {title:24s}: {b['mIoU_%']:6.2f}%  |  "
                  f"{r['0.3']:.2f}% / {r['0.5']:.2f}% / {r['0.7']:.2f}%")
        print(f"  FP_rate:     {summary['FP_rate_%']:.2f}%  "
              f"({summary['n_fp_segments']}/{summary['n_pred_segments']})")
        print(f"  FN_rate:     {summary['FN_rate_%']:.2f}%  ({summary['n_fn_samples']}/{n})")
        print(f"  [SAVED] {out_path}")
        print(SEP)

    if args.progress_log:
        snapshot = {"timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
                    "results_file": results_file, **summary}
        with open(args.progress_log, "a") as f:
            f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")


def evaluate_one(results_file, args):
    results = _load_json(results_file)
    gt_list, gt_source, matched, unmatched = build_gt_for_results(results, results_file, args)

    samples = [
        (gt_segments, parse_multi_segments(r.get("pred", ""), max_time=args.max_time))
        for r, gt_segments in zip(results, gt_list)
    ]
    summary = compute_core_summary(samples)
    summary["gt_source"] = gt_source
    summary["gt_matched"] = matched
    summary["gt_unmatched"] = unmatched

    write_and_report(results_file, summary, args)
    return summary


# ----------------------------- main -----------------------------
def find_results_files(root):
    if os.path.isfile(root):
        return [root]
    found = []
    for dirpath, _dirs, files in os.walk(root):
        if "test_results_rank0.json" in files:
            found.append(os.path.join(dirpath, "test_results_rank0.json"))
    return sorted(found)


def main():
    ap = argparse.ArgumentParser(
        description="출력폴더 하위의 test_results_rank0.json 들에 대해 All/Best/Union mIoU 계산")
    ap.add_argument("results_dir",
                    help="출력 폴더 경로 (하위에서 test_results_rank0.json 을 재귀 탐색) "
                         "또는 단일 test_results_rank0.json 경로")
    ap.add_argument("--test_json", default=None,
                    help="GT json 직접 지정 (생략 시 리프폴더명/embedded 로 자동 탐색)")
    ap.add_argument("--test_dir", default=None,
                    help="GT json 들을 찾을 디렉토리 (기본: <WORKSPACE>/data/test)")
    ap.add_argument("--max_time", type=float, default=9999.9)
    ap.add_argument("--progress_log", default=None,
                    help="설정 시 metric 스냅샷을 JSONL 로 append")
    ap.add_argument("--quiet", action="store_true",
                    help="콘솔 리포트 억제. summary JSON 은 그대로 생성됨")
    args = ap.parse_args()

    if args.test_dir is None:
        args.test_dir = _default_test_dir()

    results_files = find_results_files(args.results_dir)
    if not results_files:
        raise SystemExit(f"[에러] test_results_rank0.json 을 찾지 못함: {args.results_dir}")

    if not args.quiet:
        print(f"[INFO] {len(results_files)} 개의 test_results_rank0.json 발견")

    for rf in results_files:
        try:
            evaluate_one(rf, args)
        except Exception as e:  # noqa: BLE001
            print(f"[SKIP] {rf}: {e}")


if __name__ == "__main__":
    main()
