"""Build puvalor_test_gout_hinted.json by appending a generic answer-format
example to every human turn. This nudges the model toward correct time-token
formatting at inference without changing the underlying questions."""

import json
import os

SRC = "/home/aix23102/audiolm/vS2_eunji/data/puvalor_test_gout.json"
DST = "/home/aix23102/audiolm/vS2_eunji/data/puvalor_test_gout_hinted.json"

HINT = (
    "\n\nAnswer using time tokens of the form <t0><t0><tD><tD><tdot><tD> "
    "(four digit tokens for the integer seconds, then <tdot>, then one digit "
    "token for the decimal). Format each event segment as "
    "\"From <t...> to <t...>\". "
    "Example: From <t0><t0><t0><t5><tdot><t0> to <t0><t0><t1><t2><tdot><t3>."
)


def add_hint(text: str) -> str:
    if HINT.strip() in text:
        return text
    return text + HINT


def main() -> None:
    data = json.load(open(SRC))
    n_modified = 0
    for entry in data:
        for turn in entry["conversations"]:
            if turn.get("from") == "human":
                new_v = add_hint(turn["value"])
                if new_v != turn["value"]:
                    n_modified += 1
                turn["value"] = new_v
    with open(DST, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"src: {SRC}  ({len(data)} samples)")
    print(f"dst: {DST}  (modified {n_modified} human turns)")
    print("--- preview of first 3 prompts ---")
    for i in range(3):
        for t in data[i]["conversations"]:
            if t["from"] == "human":
                print(f"[{i}] {t['value'][:300]}{'...' if len(t['value'])>300 else ''}")
                print()
                break


if __name__ == "__main__":
    main()
