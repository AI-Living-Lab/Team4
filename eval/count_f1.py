#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
count_f1.py — UnAV-100 멀티세그먼트 grounding 의 **개수(count)** 전용 평가기.

경계값(IoU)은 일절 쓰지 않고 샘플별 (N_gt, N_pred) 개수만으로 채점한다.
"몇 개를 내야 하는지"를 모델이 아는가를 위치 정확도와 분리해서 본다.

  S_multi  = {i : N_gt >= 2}          S_single = {i : N_gt == 1}
  CR       = mean_{S_multi} min(N_gt,N_pred)/max(N_gt,N_pred)
  b        = mean_{S_multi} 1/N_gt    # "항상 1개만 예측"의 기대 CR (GT 에만 의존)
  USA      = max(0, (CR - b) / (1 - b))     # under-segmentation 해소력, chance 보정
  OSA      = |{i in S_single : N_pred <= 1}| / |S_single|   # 단일을 안 쪼개는 능력
  CountF1  = 2*USA*OSA/(USA+OSA)      # 둘 다 0 이면 0

USA/OSA 는 기존 eval_miou.py 의 CR_star/SingleAcc 와 정의가 동일하므로
같은 입력에 대해 같은 값이 나온다(회귀 대조용으로 --compare-eval-miou 참고).

입력 (JSONL, 한 줄에 객체 하나)
  GT   : {"video_id": ..., "query": ..., "segments": [[s,e], ...]}
  예측 : {"video_id": ..., "query": ..., "raw_output": "<모델 원문>"}
  (video_id, query) 로 조인한다.

사용 예
  python3 count_f1.py --gt gt.jsonl \
      --pred ours=pred_ours.jsonl --pred museg=pred_museg.jsonl \
      --format auto --csv countf1.csv

  python3 count_f1.py --selftest                 # 합성 데이터 단위테스트
  python3 count_f1.py --selftest --gt gt.jsonl   # + 실제 GT 로 b 재현 확인
"""

import argparse
import csv
import json
import os
import re
import sys

# ------------------------------------------------------------------
# 파서는 기존 eval_miou.py 것을 그대로 재사용한다.
#   - 이미 실전 검증된 구현이고,
#   - 같은 raw_output 에 대해 기존 파이프라인과 N_pred 가 어긋나지 않는다.
# eval_miou.py 는 import-safe (모든 실행 코드가 __main__ 가드 아래).
# ------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_miou import (  # noqa: E402
    extract_answer_scope,
    parse_natural,
    parse_tokens,
)

DEFAULT_MAX_TIME = 999.9
GT_BUCKETS = ["1", "2", "3", "4", ">=5"]


# ============================== 파싱 ==============================
# token 포맷: "From <t0><t0><t2><tdot><t5> to <t0><t2><t9><tdot><t4>."
#   정수부 자릿수는 고정으로 가정하지 않는다 — eval_miou 의 정규식이
#   (?:<t\d>)+(?:<tdot>(?:<t\d>)+)? 라 3자리든 4자리든 그대로 회수된다.
#   소수부는 첫 자리만 사용(0.1s 해상도).

_BARE_SEC_UNIT = re.compile(r"(\d+(?:\.\d+)?)\s*s\b", re.IGNORECASE)


def _normalize_plain(text):
    """plain 파서 전처리: 숫자에 바로 붙은 "s" 단위를 뗀다. "45.6s" -> "45.6 ".

    eval_miou.parse_natural 은 second{}/HH:MM:SS/"X to Y seconds"/맨숫자 구간은
    잡지만 "12.3s to 45.6s" 처럼 단위가 붙으면 "숫자 (to) 숫자" 패턴이 안 맞아
    놓친다. 단위를 늘리는(->"seconds") 방향은 오히려 그 패턴을 더 깨뜨리므로
    떼는 쪽이 맞다. 여기서만 정규화하고 eval_miou.py 는 건드리지 않는다
    (기존 파이프라인 동작 불변).

    "secs"/"seconds" 는 s 뒤에 단어경계가 없어 영향받지 않고,
    "second{2.5}" 처럼 숫자 뒤가 s 가 아닌 경우도 매칭되지 않는다.
    """
    return _BARE_SEC_UNIT.sub(r"\1 ", text or "")


def parse_segments(raw, fmt, max_time):
    """raw_output -> [[s,e], ...]. CoT 는 <answer>...</answer> 안만 본다."""
    scope = extract_answer_scope(raw or "")
    if fmt == "token":
        return parse_tokens(scope, max_time)
    if fmt == "plain":
        return parse_natural(_normalize_plain(scope), max_time)
    # auto: 시간토큰이 하나라도 잡히면 token, 아니면 plain 으로 폴백.
    segs = parse_tokens(scope, max_time)
    if segs:
        return segs
    return parse_natural(_normalize_plain(scope), max_time)


def merge_overlapping(segs):
    """겹치거나 맞닿은 구간을 합친다. count 가 바뀌므로 기본은 끄고(flag) 쓴다.

    맞닿음(s == 직전 e)도 합친다 — 모델이 한 이벤트를 두 조각으로 쪼갠 경우를
    한 개로 세기 위함. 떨어져 있으면 별개로 둔다.
    """
    if not segs:
        return []
    ordered = sorted([list(x) for x in segs])
    out = [ordered[0]]
    for s, e in ordered[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


# ============================== 입출력 ==============================
def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"[에러] {path}:{ln} JSON 파싱 실패: {e}")
    return rows


def _key(obj):
    return (str(obj.get("video_id", "")), str(obj.get("query", "")))


def load_gt(path):
    gt = {}
    for r in _read_jsonl(path):
        gt[_key(r)] = [list(x) for x in (r.get("segments") or [])]
    return gt


def load_pred(path, fmt, max_time, merge_overlap):
    """-> {key: (n_pred, parsed_ok)}  parsed_ok=False 는 세그먼트 0개."""
    out = {}
    for r in _read_jsonl(path):
        segs = parse_segments(r.get("raw_output"), fmt, max_time)
        if merge_overlap:
            segs = merge_overlapping(segs)
        out[_key(r)] = (len(segs), len(segs) > 0)
    return out


# ============================== 지표 ==============================
def compute_count_f1(pairs):
    """pairs: [(n_gt, n_pred), ...] -> 지표 dict.

    N_gt == 0 샘플은 S_multi/S_single 어디에도 안 들어간다(개수 과제가 정의 안 됨).
    서브셋이 비면 해당 지표는 None 이고 CountF1 도 None (0 이 아니라 '측정 불가').
    """
    cr_terms, b_terms, single_ok = [], [], []
    for n_gt, n_pred in pairs:
        if n_gt >= 2:
            hi = max(n_gt, n_pred)
            # hi 는 n_gt>=2 라 항상 >0. n_pred=0 이면 이 항은 0 → CR 을 끌어내린다.
            cr_terms.append(min(n_gt, n_pred) / hi if hi > 0 else 0.0)
            b_terms.append(1.0 / n_gt)
        elif n_gt == 1:
            single_ok.append(1.0 if n_pred <= 1 else 0.0)

    res = {
        "n_multi": len(cr_terms),
        "n_single": len(single_ok),
        "CR": None,
        "b": None,
        "USA": None,
        "OSA": None,
        "CountF1": None,
    }

    if cr_terms:
        cr = sum(cr_terms) / len(cr_terms)
        b = sum(b_terms) / len(b_terms)
        res["CR"] = cr
        res["b"] = b
        # b 는 S_multi 가 전부 N_gt==1 일 때만 1 이 되는데 정의상 N_gt>=2 라
        # b <= 0.5 다. 그래도 방어적으로 0 division 을 막는다.
        res["USA"] = max(0.0, (cr - b) / (1.0 - b)) if (1.0 - b) > 0 else 0.0
    if single_ok:
        res["OSA"] = sum(single_ok) / len(single_ok)

    u, o = res["USA"], res["OSA"]
    if u is not None and o is not None:
        res["CountF1"] = (2 * u * o / (u + o)) if (u + o) > 0 else 0.0
    return res


def compute_breakdown(pairs):
    """N_gt 별(1,2,3,4,>=5) 세부 지표."""
    buckets = {k: [] for k in GT_BUCKETS}
    for n_gt, n_pred in pairs:
        if n_gt <= 0:
            continue
        k = str(n_gt) if n_gt <= 4 else ">=5"
        buckets[k].append((n_gt, n_pred))

    rows = {}
    for k, items in buckets.items():
        if not items:
            rows[k] = {"n": 0, "mean_n_pred": None, "CR": None,
                       "exact_acc": None, "count_mae": None}
            continue
        n = len(items)
        rows[k] = {
            "n": n,
            "mean_n_pred": sum(p for _, p in items) / n,
            "CR": sum((min(g, p) / max(g, p)) if max(g, p) > 0 else 0.0
                      for g, p in items) / n,
            "exact_acc": sum(1 for g, p in items if g == p) / n,
            "count_mae": sum(abs(g - p) for g, p in items) / n,
        }
    return rows


def build_pairs(gt, pred, unparseable):
    """GT 와 예측을 조인해 [(n_gt, n_pred)] 생성.

    엣지 케이스 — 파싱 실패/세그먼트 0개 (N_pred = 0):
      --unparseable zero (기본)
          N_pred=0 으로 그대로 채점한다. 이때 S_multi 에서는 CR 항이 0 이 되어
          제대로 페널티를 받지만, **S_single 에서는 N_pred<=1 이라 OSA 에서
          '성공'으로 잡힌다.** 아무 답도 못 낸 모델이 OSA 를 벌어가는 셈이라
          의도적으로 관대한 동작이다. 개수 지표는 "과분할하지 않았다"만 보고
          "맞췄다"를 보지 않기 때문에 정의상 이렇게 된다.
          baseline 이 파싱 실패를 많이 내면 OSA/CountF1 이 부풀 수 있으니
          반드시 parse_fail_rate 를 같이 읽어야 한다.
      --unparseable skip
          해당 샘플을 S_multi/S_single 양쪽에서 제외한다. 파싱 가능한 샘플만으로
          채점하므로 위 관대함은 사라지지만, 모델 간 평가 표본이 달라진다.

    GT 에는 있는데 예측에 없는 키도 '예측 못 함'으로 보고 같은 규칙을 적용한다.
    """
    pairs, n_missing, n_unparseable = [], 0, 0
    for k, segs in gt.items():
        if k in pred:
            n_pred, ok = pred[k]
        else:
            n_pred, ok = 0, False
            n_missing += 1
        if not ok:
            n_unparseable += 1
            if unparseable == "skip":
                continue
        pairs.append((len(segs), n_pred))
    return pairs, n_missing, n_unparseable


# ============================== 리포트 ==============================
def _f(x, nd=4):
    return "-" if x is None else f"{x:.{nd}f}"


def print_table(results, gt_total):
    hdr = ["model", "USA", "OSA", "CountF1", "CR", "b",
           "|S_multi|", "|S_single|", "parse_fail%"]
    rows = []
    for name, r in results:
        m = r["metrics"]
        rows.append([
            name, _f(m["USA"]), _f(m["OSA"]), _f(m["CountF1"]),
            _f(m["CR"]), _f(m["b"]),
            str(m["n_multi"]), str(m["n_single"]),
            f"{100.0 * r['n_unparseable'] / gt_total:.2f}" if gt_total else "-",
        ])
    widths = [max(len(hdr[i]), *(len(r[i]) for r in rows)) for i in range(len(hdr))]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(hdr))
    print("\n" + line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(c.ljust(widths[i]) for i, c in enumerate(r)))


def print_breakdown(name, bd):
    print(f"\n[{name}] N_gt 별 breakdown")
    hdr = ["N_gt", "n", "mean_N_pred", "CR", "exact_acc", "count_MAE"]
    print("  ".join(h.ljust(11) for h in hdr))
    print("-" * 72)
    for k in GT_BUCKETS:
        r = bd[k]
        cells = [k, str(r["n"]), _f(r["mean_n_pred"], 3), _f(r["CR"]),
                 _f(r["exact_acc"]), _f(r["count_mae"], 3)]
        print("  ".join(c.ljust(11) for c in cells))


def save_csv(path, results, gt_total):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "USA", "OSA", "CountF1", "CR", "b",
                    "n_multi", "n_single", "n_unparseable", "parse_fail_rate",
                    "n_missing_in_pred"])
        for name, r in results:
            m = r["metrics"]
            w.writerow([name, m["USA"], m["OSA"], m["CountF1"], m["CR"], m["b"],
                        m["n_multi"], m["n_single"], r["n_unparseable"],
                        (r["n_unparseable"] / gt_total) if gt_total else "",
                        r["n_missing"]])
        w.writerow([])
        w.writerow(["model", "N_gt", "n", "mean_N_pred", "CR",
                    "exact_acc", "count_MAE"])
        for name, r in results:
            for k in GT_BUCKETS:
                b = r["breakdown"][k]
                w.writerow([name, k, b["n"], b["mean_n_pred"], b["CR"],
                            b["exact_acc"], b["count_mae"]])
    print(f"\n[SAVED] {path}")


# ============================== 셀프테스트 ==============================
def _synthetic_gt():
    """N_gt 분포가 섞인 합성 GT."""
    counts = [1] * 20 + [2] * 10 + [3] * 6 + [4] * 3 + [5] * 2 + [7] * 1
    gt = {}
    for i, c in enumerate(counts):
        gt[(f"v{i}", "q")] = [[float(2 * j), float(2 * j + 1)] for j in range(c)]
    return gt


def selftest(gt_path=None):
    ok = True

    def check(label, cond, got):
        nonlocal ok
        mark = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
        print(f"  [{mark}] {label}: {got}")

    gt = _synthetic_gt()
    pairs_all = [(len(v), None) for v in gt.values()]
    print("합성 GT: "
          f"n={len(gt)}, S_multi={sum(1 for g, _ in pairs_all if g >= 2)}, "
          f"S_single={sum(1 for g, _ in pairs_all if g == 1)}")

    # 1) 항상 1개만 예측하는 더미 -> CR == b 이므로 USA == 0
    print("\n[테스트 1] 항상 1개만 예측하는 더미 모델")
    m1 = compute_count_f1([(len(v), 1) for v in gt.values()])
    check("USA == 0", abs(m1["USA"] - 0.0) < 1e-12, f"USA={m1['USA']:.6f}")
    check("CR == b", abs(m1["CR"] - m1["b"]) < 1e-12,
          f"CR={m1['CR']:.6f} b={m1['b']:.6f}")
    check("OSA == 1 (단일은 1개 예측이 정답)", abs(m1["OSA"] - 1.0) < 1e-12,
          f"OSA={m1['OSA']:.6f}")
    check("CountF1 == 0", abs(m1["CountF1"] - 0.0) < 1e-12,
          f"CountF1={m1['CountF1']:.6f}")

    # 2) GT 를 그대로 예측 -> 완벽
    print("\n[테스트 2] GT 를 그대로 예측")
    m2 = compute_count_f1([(len(v), len(v)) for v in gt.values()])
    check("USA == 1", abs(m2["USA"] - 1.0) < 1e-12, f"USA={m2['USA']:.6f}")
    check("OSA == 1", abs(m2["OSA"] - 1.0) < 1e-12, f"OSA={m2['OSA']:.6f}")
    check("CountF1 == 1", abs(m2["CountF1"] - 1.0) < 1e-12,
          f"CountF1={m2['CountF1']:.6f}")

    # 3) 파서 왕복
    print("\n[테스트 3] 파서")
    tok = ("From <t0><t0><t2><tdot><t5> to <t0><t0><t2><tdot><t9>. "
           "From <t0><t1><t2><tdot><t3> to <t0><t4><t5><tdot><t6>.")
    check("token 2세그", len(parse_segments(tok, "token", DEFAULT_MAX_TIME)) == 2,
          str(parse_segments(tok, "token", DEFAULT_MAX_TIME)))
    pl = "second{2.5}-second{2.9} and 12.3s to 45.6s"
    check("plain 2세그", len(parse_segments(pl, "plain", DEFAULT_MAX_TIME)) == 2,
          str(parse_segments(pl, "plain", DEFAULT_MAX_TIME)))
    check("auto=token 폴백 없음",
          len(parse_segments(tok, "auto", DEFAULT_MAX_TIME)) == 2,
          str(parse_segments(tok, "auto", DEFAULT_MAX_TIME)))
    check("auto=plain 폴백",
          len(parse_segments(pl, "auto", DEFAULT_MAX_TIME)) == 2,
          str(parse_segments(pl, "auto", DEFAULT_MAX_TIME)))
    check("파싱 실패 -> 0세그",
          len(parse_segments("I don't know.", "auto", DEFAULT_MAX_TIME)) == 0,
          str(parse_segments("I don't know.", "auto", DEFAULT_MAX_TIME)))
    mo = [[0.0, 5.0], [4.0, 9.0], [20.0, 22.0]]
    check("merge-overlap 3->2", len(merge_overlapping(mo)) == 2,
          str(merge_overlapping(mo)))

    # 4) 빈 서브셋 zero division
    print("\n[테스트 4] 빈 서브셋")
    m_no_single = compute_count_f1([(2, 2), (3, 3)])
    check("S_single 비면 OSA/CountF1 = None",
          m_no_single["OSA"] is None and m_no_single["CountF1"] is None,
          f"OSA={m_no_single['OSA']} CountF1={m_no_single['CountF1']}")
    m_no_multi = compute_count_f1([(1, 1), (1, 2)])
    check("S_multi 비면 USA/CountF1 = None",
          m_no_multi["USA"] is None and m_no_multi["CountF1"] is None,
          f"USA={m_no_multi['USA']} CountF1={m_no_multi['CountF1']}")

    # 5) 실제 GT 로 b 재현 (UnAV-100 test split 기대값 0.397)
    if gt_path:
        print(f"\n[테스트 5] 실제 GT 의 chance floor b — {gt_path}")
        real = load_gt(gt_path)
        mr = compute_count_f1([(len(v), 1) for v in real.values()])
        print(f"  n={len(real)}  |S_multi|={mr['n_multi']}  "
              f"|S_single|={mr['n_single']}")
        got = mr["b"]
        check("b ≈ 0.397 (허용 ±0.001)", abs(got - 0.397) <= 0.001,
              f"b={got:.4f}")
    else:
        print("\n[테스트 5] --gt 를 같이 주면 실제 GT 의 b 를 대조합니다 (기대 0.397)")

    print("\n=== " + ("ALL PASS" if ok else "FAIL 있음") + " ===")
    return 0 if ok else 1


# ============================== main ==============================
def main():
    ap = argparse.ArgumentParser(
        description="UnAV-100 멀티세그먼트 grounding 의 개수 전용 지표(CountF1).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", help="GT JSONL {video_id, query, segments}")
    ap.add_argument("--pred", action="append", default=[], metavar="NAME=PATH",
                    help="예측 JSONL. 여러 번 주면 모델별 행으로 비교표를 낸다.")
    ap.add_argument("--format", choices=["token", "plain", "auto"], default="auto",
                    help="raw_output 파서 (기본 auto: 토큰 우선, 없으면 plain)")
    ap.add_argument("--unparseable", choices=["zero", "skip"], default="zero",
                    help="파싱 실패 처리 (기본 zero=N_pred 0 으로 채점)")
    ap.add_argument("--merge-overlap", action="store_true",
                    help="예측의 겹치는/맞닿은 구간을 합친 뒤 센다 (기본 off)")
    ap.add_argument("--max-time", type=float, default=DEFAULT_MAX_TIME,
                    help=f"초 clamp 상한 (기본 {DEFAULT_MAX_TIME})")
    ap.add_argument("--csv", help="비교표+breakdown 을 CSV 로 저장")
    ap.add_argument("--selftest", action="store_true",
                    help="단위테스트 실행 (--gt 를 주면 b 재현도 확인)")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest(args.gt))

    if not args.gt or not args.pred:
        ap.error("--gt 와 --pred 는 필수입니다 (또는 --selftest).")

    gt = load_gt(args.gt)
    if not gt:
        raise SystemExit(f"[에러] GT 가 비었습니다: {args.gt}")

    n_gt_zero = sum(1 for v in gt.values() if not v)
    print(f"GT: {len(gt)} 샘플  ({args.gt})")
    if n_gt_zero:
        print(f"  ⚠ N_gt==0 인 {n_gt_zero} 샘플은 S_multi/S_single 양쪽에서 제외")
    print(f"설정: format={args.format}  unparseable={args.unparseable}  "
          f"merge_overlap={args.merge_overlap}  max_time={args.max_time}")

    results = []
    for spec in args.pred:
        if "=" not in spec:
            ap.error(f"--pred 는 NAME=PATH 형식이어야 합니다: {spec}")
        name, path = spec.split("=", 1)
        pred = load_pred(path, args.format, args.max_time, args.merge_overlap)
        pairs, n_missing, n_unparse = build_pairs(gt, pred, args.unparseable)
        results.append((name, {
            "metrics": compute_count_f1(pairs),
            "breakdown": compute_breakdown(pairs),
            "n_missing": n_missing,
            "n_unparseable": n_unparse,
            "n_pred_rows": len(pred),
        }))
        if n_missing:
            print(f"  ⚠ [{name}] GT 에 있는데 예측에 없는 키 {n_missing}개 "
                  f"→ N_pred=0 으로 처리")

    print_table(results, len(gt))
    for name, r in results:
        print_breakdown(name, r["breakdown"])
    if args.csv:
        save_csv(args.csv, results, len(gt))


if __name__ == "__main__":
    main()
