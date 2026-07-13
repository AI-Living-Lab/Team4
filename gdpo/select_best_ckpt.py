#!/usr/bin/env python3
"""val_metrics.jsonl 을 읽어 best checkpoint 를 선택 (checkpoint selection).

RL 보상곡선은 non-monotonic·noisy 라서 단일 step 의 max 만 보면 노이즈 peak 을 고를 수 있다.
→ 옵션으로 EMA/이동평균 스무딩 후 best 를 고른다(기본 raw). 지연-점프(700↓→1000↑)를
   놓치지 않도록 학습은 끝까지 돌린 전제(early-stop 안 함)에서 사후 선택한다.

사용:
  python3 select_best_ckpt.py <OUTPUT_DIR 또는 val_metrics.jsonl> [--metric combined] [--smooth raw|ema|ma3]
"""
import argparse, json, os, sys


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, "val_metrics.jsonl")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    # step 중복 시 마지막 것만(재개 등)
    by_step = {}
    for r in rows:
        by_step[int(r["step"])] = r
    return [by_step[s] for s in sorted(by_step)], path


def smooth(vals, mode):
    if mode == "raw":
        return list(vals)
    if mode == "ema":
        out, a, prev = [], 0.5, None
        for v in vals:
            prev = v if prev is None else a * v + (1 - a) * prev
            out.append(prev)
        return out
    if mode == "ma3":
        out = []
        for i in range(len(vals)):
            w = vals[max(0, i - 1):i + 2]
            out.append(sum(w) / len(w))
        return out
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="OUTPUT_DIR 또는 val_metrics.jsonl")
    ap.add_argument("--metric", default="combined", choices=["combined", "seg_miou", "f1_avg"])
    ap.add_argument("--smooth", default="raw", choices=["raw", "ema", "ma3"])
    args = ap.parse_args()

    rows, used = load(args.path)
    if not rows:
        print(f"[ERR] 데이터 없음: {used}", file=sys.stderr)
        sys.exit(1)

    steps = [r["step"] for r in rows]
    raw = [r[args.metric] for r in rows]
    sm = smooth(raw, args.smooth)
    best_i = max(range(len(rows)), key=lambda i: sm[i])
    best = rows[best_i]

    print(f"# val_metrics: {used}")
    print(f"# metric={args.metric}  smooth={args.smooth}  (eval points={len(rows)})")
    print(f"{'step':>7} {'combined':>9} {'seg_miou':>9} {'f1_avg':>8} {'parse_ok':>9}"
          f" {'(' + args.smooth + ')':>9}")
    for i, r in enumerate(rows):
        mark = "  <== BEST" if i == best_i else ""
        print(f"{r['step']:>7} {r['combined']:>9.2f} {r['seg_miou']:>9.2f} {r['f1_avg']:>8.2f}"
              f" {r.get('n_parse_ok', '?'):>9} {sm[i]:>9.2f}{mark}")
    print("-" * 60)
    print(f"BEST step = {best['step']}  "
          f"(combined={best['combined']:.2f}, seg_miou={best['seg_miou']:.2f}, f1_avg={best['f1_avg']:.2f})")
    print(f"→ 최종 test 평가는 checkpoint-{best['step']} 하나만 돌리면 됨:")
    print(f"   bash eval/eval.sh STAGE=gdpo CKPT_MODEL_ID=<MODEL_ID> CKPT_STEP={best['step']} \\")
    print(f"        BASE_MODEL_ID=base/salmonn2p_7b_unav_v8 TEST_JSON=<TEST.json> GPUS=<g>")


if __name__ == "__main__":
    main()
