#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""seed_compare.py — 시드 복제 런과 원본 런을 같은 샘플 위에서 비교한다.

멀티시드 리버틀용. 두 가지를 낸다:

  1) 고정 부분집합(기본 chunk 0~2 = 1500샘플) 위의 sample mIoU — paired 비교
  2) paired bootstrap 으로 "두 런의 차이"에 95% CI  (0 을 포함하면 구분 불가)

IoU 정의는 eval_miou.sample_iou_parts 를 그대로 쓴다 (sample_miou_summary.json 과 동일).

샘플 join 은 (video basename, gt_label) 키로 한다 — 예측 파일마다 video 경로
접두사가 다르기 때문이다 (랩 서버 /data0/aix23102/... vs 로컬 /home/team404/...).
위치(인덱스) 기준으로 자르면 청크 순서가 어긋나 조용히 틀린 비교가 된다.

사용:
  python3 seed_compare.py \
      --a  <s2025 test_results_rank0.json>  --a_label "seed 2025 ckpt-1400" \
      --b  <s2024 test_results_rank0.json>  --b_label "seed 2024 ckpt-1400 (July)" \
      [--testdir /home/team404/workspace/data/test/unav100_titok] [--chunks 0 1 2] \
      [--fmt token|natural]
"""
import argparse
import json
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_miou import (extract_answer_scope, parse_natural, parse_tokens,  # noqa: E402
                       sample_iou_parts)

MAX_TIME = 1e9
N_BOOT = 5000
BOOT_SEED = 20260901


def key(x):
    return (os.path.basename(x.get("video") or ""), x.get("gt_label"))


def parse_pred(raw, fmt):
    scope = extract_answer_scope(raw or "")
    return parse_natural(scope, MAX_TIME) if fmt == "natural" else parse_tokens(scope, MAX_TIME)


def gt_of(x):
    """GT 는 embedded gt_segments 우선, 없으면 ref 파싱 (예측 파일에 따라 다름)."""
    segs = x.get("gt_segments") or []
    return [list(s) for s in segs] if segs else parse_tokens(x.get("ref", ""), MAX_TIME)


def load_subset(testdir, chunks):
    sub = []
    for c in chunks:
        path = os.path.join(testdir, f"chunk_{c:04d}.json")
        for x in json.load(open(path)):
            sub.append(key(x))
    return sub


def per_sample(path, fmt):
    idx = {}
    for x in json.load(open(path)):
        idx.setdefault(key(x), x)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="비교 대상 A 의 test_results_rank0.json")
    ap.add_argument("--b", required=True, help="비교 대상 B 의 test_results_rank0.json")
    ap.add_argument("--a_label", default="A")
    ap.add_argument("--b_label", default="B")
    ap.add_argument("--testdir", default="/home/team404/workspace/data/test/unav100_titok")
    ap.add_argument("--chunks", type=int, nargs="*", default=[0, 1, 2],
                    help="고정 부분집합으로 쓸 청크 번호. 비우면 _full.json 전체.")
    ap.add_argument("--fmt", choices=["token", "natural"], default="token")
    args = ap.parse_args()

    if args.chunks:
        sub = load_subset(args.testdir, args.chunks)
    else:
        sub = [key(x) for x in json.load(open(os.path.join(args.testdir, "_full.json")))]
    sub_u = list(dict.fromkeys(sub))
    print(f"[subset] chunks={args.chunks or 'full'}  n={len(sub_u)} (중복 제거 전 {len(sub)})")

    ia, ib = per_sample(args.a, args.fmt), per_sample(args.b, args.fmt)

    # 양쪽 모두에 존재하는 샘플만 — paired 비교의 전제
    paired, miss_a, miss_b = [], 0, 0
    for k in sub_u:
        xa, xb = ia.get(k), ib.get(k)
        if xa is None:
            miss_a += 1
            continue
        if xb is None:
            miss_b += 1
            continue
        gt = gt_of(xa) or gt_of(xb)
        if not gt:
            continue
        paired.append((sample_iou_parts(gt, parse_pred(xa.get("pred", ""), args.fmt))[0],
                       sample_iou_parts(gt, parse_pred(xb.get("pred", ""), args.fmt))[0]))

    if miss_a or miss_b:
        print(f"[warn] 부분집합 중 A 에 없음={miss_a}, B 에 없음={miss_b} → paired 에서 제외")
    n = len(paired)
    if n == 0:
        sys.exit("paired 샘플이 0개 — join 키(video basename, gt_label)를 확인하세요.")

    a_m = 100 * statistics.mean(p[0] for p in paired)
    b_m = 100 * statistics.mean(p[1] for p in paired)
    print(f"[paired] n={n}")
    print(f"  {args.a_label:38s} sample mIoU = {a_m:.2f}")
    print(f"  {args.b_label:38s} sample mIoU = {b_m:.2f}")
    print(f"  delta (A - B)                          = {a_m - b_m:+.2f} pp")

    # paired bootstrap: 샘플을 쌍째로 재추출해 delta 의 분포를 만든다.
    rng = random.Random(BOOT_SEED)
    deltas, a_means = [], []
    for _ in range(N_BOOT):
        pick = [paired[rng.randrange(n)] for _ in range(n)]
        ma = sum(p[0] for p in pick) / n
        mb = sum(p[1] for p in pick) / n
        deltas.append(100 * (ma - mb))
        a_means.append(100 * ma)
    deltas.sort()
    a_means.sort()
    lo, hi = deltas[int(0.025 * N_BOOT)], deltas[int(0.975 * N_BOOT)]
    alo, ahi = a_means[int(0.025 * N_BOOT)], a_means[int(0.975 * N_BOOT)]
    print(f"[bootstrap B={N_BOOT}]")
    print(f"  delta 95% CI = [{lo:+.2f}, {hi:+.2f}] pp"
          f"   → {'구분 불가 (0 포함)' if lo <= 0 <= hi else '유의한 차이 (0 미포함)'}")
    print(f"  {args.a_label} 자체 95% CI = [{alo:.2f}, {ahi:.2f}]")


if __name__ == "__main__":
    main()
