#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_cot_data.py — P2/P4 CoT think 데이터 생성 (CoT 개선 STEP 3)

route (a): 기존 unav100_v2_cot.json 의 <answer> 토큰을 decode_vtg_time 으로 초 변환 →
           그 초로 <think> 를 채운다. <answer> 토큰은 원본 그대로 유지(채점 일관).
           video/audio/use_audio/event 그대로 보존.

think 포맷:
  P2 (timestep 태그): <think>The event occurs at <timestep>3.3 to 5.6</timestep>, ... seconds.</think>
  P4 (자유 자연어):   <think>The relevant event spans 3.3 to 5.6, 9.1 to 10.0 seconds.</think>
소수 1자리(토큰 정밀도 XXX.Y 와 통일).

사용:
  python _tools/gen_cot_data.py --src data/train/unav100_v2_cot.json --mode p2 --out data/train/unav100_v2_p2.json
  python _tools/gen_cot_data.py --src ... --mode p4 --out ...      [--limit 5 = 미니검증]
"""
import argparse, json, os, re, sys

# decode_vtg_time = single source of truth (reward_functions)
_GDPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GDPO")
sys.path.insert(0, _GDPO)
from reward_functions import decode_vtg_time, _SEG_CAPTURE_RE  # noqa: E402

# 프롬프트 (질문부는 유지, 지시부만 P1→P2/P4 로 교체)
_Q = "At what point in the video does {event} occur in terms of both video and audio?"
_FMT = ('Then give the joint segments as the answer. Use the format '
        '"From <tX><tX><tX><tdot><tX> to <tX><tX><tX><tdot><tX>." for every segment '
        '(multiple segments separated by a period and a space).\n'
        'Output like "<think>...</think><answer>From ... to ...</answer>".')

P2_PROMPT = ("<video>\n" + _Q + " Output your thought process in <think> </think> tags, "
             "including specific time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.\n" + _FMT)
P4_PROMPT = ("<video>\n" + _Q + " Reason inside <think> </think> tags.\n" + _FMT)

# P5 (다중발생 열거): "몇 번 일어나는지 세고 각각 짚어라" — under-prediction(blanket) 직격. joint 잔재 제거.
_P5_INSTR = ("\nThe event may occur MULTIPLE times. Output your thought process in <think> </think> tags. "
             "Then give the relevant segments as the answer. For example, if it occurs 2 times:\n"
             "<think>The event occurs 2 times, ...</think>"
             "<answer>From <tX><tX><tX><tdot><tX> to <tX><tX><tX><tdot><tX>. "
             "From <tX><tX><tX><tdot><tX> to <tX><tX><tX><tdot><tX>.</answer>")
P5_PROMPT = "<video>\n" + _Q + _P5_INSTR


def parse_answer(ans_value):
    """원본 gpt value 에서 <answer>..</answer> 추출 + 토큰→초 세그먼트.
    return (segs[(s,e)..], answer_full_str or None)."""
    m = re.search(r"<answer>(.*?)</answer>", ans_value, re.S)
    block = m.group(1) if m else ans_value
    segs = []
    for s_str, e_str in _SEG_CAPTURE_RE.findall(block):
        s, e = decode_vtg_time(s_str), decode_vtg_time(e_str)
        if s is not None and e is not None and e > s:
            segs.append((s, e))
    return segs, (m.group(0) if m else None)


def make_think(segs, mode):
    ranges = [f"{s:.1f} to {e:.1f}" for s, e in segs]
    if mode == "p2":
        body = ", ".join(f"<timestep>{r}</timestep>" for r in ranges)
        return f"<think>The event occurs at {body} seconds.</think>"
    if mode == "p5":
        n = len(ranges)
        items = "; ".join(f"{i+1}) {r}" for i, r in enumerate(ranges))
        return f"<think>The event occurs {n} time(s): {items} seconds.</think>"
    return f"<think>The relevant event spans {', '.join(ranges)} seconds.</think>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--mode", choices=["p2", "p4", "p5"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=-1, help="미니검증용 앞 N개만")
    args = ap.parse_args()

    data = json.load(open(args.src))
    if args.limit > 0:
        data = data[: args.limit]
    prompt = {"p2": P2_PROMPT, "p4": P4_PROMPT, "p5": P5_PROMPT}[args.mode]

    out, skipped = [], 0
    for it in data:
        try:
            ans_value = it["conversations"][1]["value"]
        except (KeyError, IndexError):
            skipped += 1; continue
        segs, answer_full = parse_answer(ans_value)
        if not segs or answer_full is None:
            skipped += 1; continue
        ev = it.get("event") or "the event"
        new_it = dict(it)  # video/audio/use_audio/event 보존
        new_it["conversations"] = [
            {"from": "human", "value": prompt.format(event=ev)},
            {"from": "gpt", "value": make_think(segs, args.mode) + answer_full},
        ]
        out.append(new_it)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[done] mode={args.mode}  written={len(out)}  skipped={skipped}  src={len(data)}  → {args.out}")


if __name__ == "__main__":
    main()
