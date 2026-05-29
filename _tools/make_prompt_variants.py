#!/usr/bin/env python3
"""
Generate prompt-variant test JSONs by appending a format hint to every human turn.
Hint placement: after the original question (and after `<video>\\n` for the first turn).
"""
import argparse
import copy
import json


def add_hint_to_human_turn(value: str, hint: str) -> str:
    return value.rstrip() + "\n\n" + hint


def make_variant(samples, hint):
    out = []
    for s in samples:
        ns = copy.deepcopy(s)
        for c in ns["conversations"]:
            if c["from"] == "human":
                c["value"] = add_hint_to_human_turn(c["value"], hint)
        out.append(ns)
    return out


VARIANT_HINTS = {
    "v1": "Answer format: 'From <ts> to <te>.'",
    "v3_3_1": "Example: 'From <t0><t0><t1><tdot><t0> to <t0><t0><t5><tdot><t0>.'",
    "v3_4_1": "Example: 'From <t0><t0><t0><t1><tdot><t0> to <t0><t0><t0><t5><tdot><t0>.'",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--variant", required=True, choices=list(VARIANT_HINTS.keys()))
    args = ap.parse_args()

    hint = VARIANT_HINTS[args.variant]
    data = json.load(open(args.input))
    print(f"In: {args.input} ({len(data)} samples)")
    print(f"Variant: {args.variant}")
    print(f"Hint: {hint}")
    out = make_variant(data, hint)
    json.dump(out, open(args.output, "w"), ensure_ascii=False)
    print(f"Out: {args.output}")

    # sanity: show first human turn after modification
    print("\nFirst human turn (modified):")
    print("  ", out[0]["conversations"][0]["value"][:300])


if __name__ == "__main__":
    main()
