"""Build a small balanced subset of PU-VALOR test (single 100 + multi 100 = 200)
in three prompt-hint variants for quick A/B testing of v4 checkpoints.

Outputs (same indices, only human-turn text differs):
  - puvalor_test_quick_full.json : current full hint (5-line format spec + Example)
  - puvalor_test_quick_none.json : no hint (original puvalor_test_gout.json text)
  - puvalor_test_quick_min.json  : minimal one-line answer-format example
"""

import json
import os
import random

SRC = "/home/aix23102/audiolm/vS2_eunji/data/puvalor_test_gout.json"
DST_DIR = "/home/aix23102/audiolm/vS2_eunji/data"

N_SINGLE = 100
N_MULTI = 100
SEED = 42

HINT_FULL = (
    "\n\nAnswer using time tokens of the form <t0><t0><tD><tD><tdot><tD> "
    "(four digit tokens for the integer seconds, then <tdot>, then one digit "
    "token for the decimal). Format each event segment as "
    "\"From <t...> to <t...>\". "
    "Example: From <t0><t0><t0><t5><tdot><t0> to <t0><t0><t1><t2><tdot><t3>."
)
HINT_MIN = "\n\nAnswer format: \"From <t...> to <t...>\"."


def add_hint(text, hint):
    if hint and hint.strip() in text:
        return text
    return text + hint if hint else text


def build_variant(samples, hint, dst_path):
    out = []
    for s in samples:
        e = json.loads(json.dumps(s))
        for t in e["conversations"]:
            if t.get("from") == "human":
                t["value"] = add_hint(t["value"], hint)
        out.append(e)
    with open(dst_path, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"  -> {dst_path}  ({len(out)} samples)")


def main():
    data = json.load(open(SRC))
    singles = [d for d in data if d.get("n_gt_segments", len(d.get("gt_segments", []))) == 1]
    multis = [d for d in data if d.get("n_gt_segments", len(d.get("gt_segments", []))) > 1]
    print(f"src: {len(data)}  (single={len(singles)} multi={len(multis)})")

    rng = random.Random(SEED)
    sub_single = rng.sample(singles, N_SINGLE)
    sub_multi = rng.sample(multis, N_MULTI)
    subset = sub_single + sub_multi
    rng.shuffle(subset)
    print(f"subset: {len(subset)}  (single={N_SINGLE} multi={N_MULTI})")

    build_variant(subset, HINT_FULL, os.path.join(DST_DIR, "puvalor_test_quick_full.json"))
    build_variant(subset, "", os.path.join(DST_DIR, "puvalor_test_quick_none.json"))
    build_variant(subset, HINT_MIN, os.path.join(DST_DIR, "puvalor_test_quick_min.json"))

    print("\n--- preview of first sample human turn (each variant) ---")
    for tag, h in [("full", HINT_FULL), ("none", ""), ("min", HINT_MIN)]:
        sample0 = json.loads(json.dumps(subset[0]))
        for t in sample0["conversations"]:
            if t["from"] == "human":
                v = add_hint(t["value"], h)
                print(f"[{tag}]\n{v[:400]}{'...' if len(v) > 400 else ''}\n")
                break


if __name__ == "__main__":
    main()
