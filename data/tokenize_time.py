#!/usr/bin/env python3
"""
Stage3-style JSON (<s0>/<e0> placeholders + meta.token) → tokenized JSON.

각 conversation turn 의 placeholder 를 time-digit token 문자열로 치환.

토큰화 스킴: ndig_int 자리 정수부 + ndig_dec 자리 소수부 + (소수부 있으면) <tdot>.
  4+1: 0000.0 → <t><t><t><t><tdot><t>            (max 9999.9, res 0.1)
  3+1: 000.0  → <t><t><t><tdot><t>               (max  999.9, res 0.1)
  3+2: 000.00 → <t><t><t><tdot><t><t>            (max  999.99, res 0.01)
  2+2: 00.00  → <t><t><tdot><t><t>               (max   99.99, res 0.01)

입력 placeholder pattern: <s\d+>, <e\d+>
입력 meta.token: {"<s0>": 0.0, "<e0>": 11.0, ...}

Out-of-range timestamp 는 clip (max value) + warning count.
"""
import argparse
import json
import os
import re
from collections import Counter


PLACEHOLDER_PAT = re.compile(r"<[se]\d+>")


def time_to_tokens(sec, ndig_int, ndig_dec):
    """초 → time-digit token 문자열.
    예) sec=11.0, 3+2 → "<t0><t1><t1><tdot><t0><t0>"
        sec=6.30087, 3+2 → "<t0><t0><t6><tdot><t3><t0>" (round)
    """
    max_val = 10**ndig_int - 10**(-ndig_dec) if ndig_dec > 0 else 10**ndig_int - 1
    clipped = False
    if sec < 0:
        sec = 0.0
        clipped = True
    if sec > max_val:
        sec = max_val
        clipped = True

    # round to ndig_dec
    scale = 10**ndig_dec
    iv = round(sec * scale)
    int_part = iv // scale
    dec_part = iv % scale

    int_str = f"{int_part:0{ndig_int}d}"
    tokens = "".join(f"<t{d}>" for d in int_str)
    if ndig_dec > 0:
        dec_str = f"{dec_part:0{ndig_dec}d}"
        tokens += "<tdot>"
        tokens += "".join(f"<t{d}>" for d in dec_str)
    return tokens, clipped


def tokenize_sample(s, ndig_int, ndig_dec, stats):
    """sample 의 모든 conversation turn 에서 placeholder 치환."""
    token_map = s.get("meta", {}).get("token", {})
    if not token_map:
        stats["no_token_map"] += 1
        return s

    # placeholder → token string 미리 계산 (각 placeholder 한 번씩)
    placeholder_to_str = {}
    for ph, sec in token_map.items():
        toks, clipped = time_to_tokens(float(sec), ndig_int, ndig_dec)
        placeholder_to_str[ph] = toks
        if clipped:
            stats["clipped"] += 1

    def repl(m):
        ph = m.group(0)
        if ph not in placeholder_to_str:
            stats["unknown_placeholder"] += 1
            return ph
        return placeholder_to_str[ph]

    out = dict(s)
    new_convs = []
    for c in s.get("conversations", []):
        new_text = PLACEHOLDER_PAT.sub(repl, c.get("value", ""))
        new_convs.append({**c, "value": new_text})
    out["conversations"] = new_convs

    # 메타에 토큰화 정보 추가
    out["meta"] = dict(s["meta"])
    out["meta"]["tokenization"] = f"{ndig_int}+{ndig_dec}"

    stats["tokenized"] += 1
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Stage3-style JSON")
    parser.add_argument("--output", required=True, help="Tokenized JSON")
    parser.add_argument("--ndig_int", type=int, required=True, help="정수부 자리수")
    parser.add_argument("--ndig_dec", type=int, required=True, help="소수부 자리수")
    args = parser.parse_args()

    assert args.ndig_int >= 1, "ndig_int >= 1"
    assert args.ndig_dec >= 0, "ndig_dec >= 0"

    print(f"Tokenization: {args.ndig_int}+{args.ndig_dec}")
    print(f"  max representable: {10**args.ndig_int - (10**(-args.ndig_dec) if args.ndig_dec>0 else 1)}")
    print(f"  resolution: {10**(-args.ndig_dec)}")

    print(f"\nLoading {args.input} ...")
    with open(args.input) as f:
        data = json.load(f)
    print(f"  {len(data)} samples")

    stats = Counter()
    out = [tokenize_sample(s, args.ndig_int, args.ndig_dec, stats) for s in data]

    print(f"\nStats: {dict(stats)}")

    print(f"\nSaving → {args.output}")
    with open(args.output, "w") as f:
        json.dump(out, f)
    sz = os.path.getsize(args.output) / 1024 / 1024
    print(f"  {sz:.1f} MB")


if __name__ == "__main__":
    main()
