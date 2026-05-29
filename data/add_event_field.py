#!/usr/bin/env python3
"""
Stage3-style placeholder JSON 의 sample 에 `event` 필드 추가.

규칙 (task_type 별):
  T1 (DAVC):  list per segment  — narrative 의 segment 별 desc 추출
  T2 (TSG):   single string     — GPT turn-3 description echo
  T3 (SC):    list per segment  — GPT turn-1, turn-3 description

source 별 narrative 패턴이 다르므로 두 regex 모두 시도:
  (a) "From <sN> to <eN>, [desc]."  (UnAV / PU-VALOR 일부)
  (b) "[desc], from <sN> to <eN>."  (PU-VALOR 일부)
"""
import argparse
import json
import os
import re


_PAT_A = re.compile(r"[Ff]rom\s+<s(\d+)>\s+to\s+<e\d+>,\s*([^.]+?)\.")
_PAT_B = re.compile(r"([^.]+?),\s*[Ff]rom\s+<s(\d+)>\s+to\s+<e\d+>\.")


def extract_event_t1(sample):
    """Multi-segment narrative 에서 segment 별 desc 추출. 두 패턴 모두 시도."""
    gpt_text = sample["conversations"][1]["value"]
    events = {}
    for m in _PAT_A.finditer(gpt_text):
        events[int(m.group(1))] = m.group(2).strip()
    for m in _PAT_B.finditer(gpt_text):
        idx = int(m.group(2))
        if idx not in events:
            events[idx] = m.group(1).strip()
    if not events:
        return None
    return [events[i] for i in sorted(events.keys())]


_GROUNDING_ONLY = re.compile(r"^\s*[Ff]rom\s+<s\d+>\s+to\s+<e\d+>\.\s*$")


def extract_event_t2(sample):
    """PU-VALOR T2 는 두 가지 sub-structure:

      (a) single grounding + description echo (T2a):
            turn1 (gpt): "From <s0> to <e0>."
            turn3 (gpt): "<event description echo>"
          → event = single string (turn3)

      (b) double grounding (T2b):
            turn1 (gpt): "From <s0> to <e0>."
            turn3 (gpt): "From <s1> to <e1>."
          → event = list of 2 human prompts (caption 부분 포함된 풀 텍스트)
    """
    convs = sample["conversations"]
    if len(convs) < 4:
        return None
    gpt_turns = [c["value"] for c in convs if c["from"] == "gpt"]
    if not gpt_turns:
        return None
    all_grounding = all(_GROUNDING_ONLY.match(t) for t in gpt_turns)
    if all_grounding and len(gpt_turns) >= 2:
        # T2b — multiple grounding queries, event 는 각 human prompt 자체
        human_turns = [
            c["value"].replace("<video>", "").replace("<image>", "").strip()
            for c in convs if c["from"] == "human"
        ]
        return human_turns
    # T2a — single grounding + description echo
    return convs[-1]["value"].strip()


def extract_event_t3(sample):
    """모든 GPT turn 의 응답을 list 로 (segment 별 desc)."""
    return [c["value"].strip() for c in sample["conversations"] if c["from"] == "gpt"]


def add_event(sample, force=False):
    """sample 에 event 필드 추가. force=True 면 기존 event 덮어쓰기.

    UnAV 의 raw event (단순 label) 는 보존하고, PU-VALOR 의 자체 추출만 덮어씀.
    """
    if not force and "event" in sample and sample["event"]:
        return sample, "kept"
    # UnAV 의 기존 짧은 label event 는 보존 (강제 모드여도)
    if force and sample.get("source_id") == "unav" and "event" in sample and sample["event"]:
        return sample, "kept_unav"
    task = sample.get("task_type", "")
    if task == "T1":
        ev = extract_event_t1(sample)
    elif task == "T2":
        ev = extract_event_t2(sample)
    elif task == "T3":
        ev = extract_event_t3(sample)
    else:
        return sample, "skip_unknown_task"
    if ev is None:
        return sample, "extraction_failed"
    sample = dict(sample)
    sample["event"] = ev
    return sample, "added"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None,
                    help="default = inplace overwrite")
    ap.add_argument("--force", action="store_true",
                    help="기존 event 덮어쓰기 (UnAV 원본 label 은 보존)")
    args = ap.parse_args()

    out_path = args.output or args.input
    with open(args.input) as f:
        data = json.load(f)
    print(f"[add_event] {args.input}: N={len(data)}")

    from collections import Counter
    stats = Counter()
    out = []
    for s in data:
        s2, status = add_event(s, force=args.force)
        out.append(s2)
        stats[(s.get("task_type", "?"), status)] += 1

    print(f"[add_event] stats: {dict(stats)}")
    with open(out_path, "w") as f:
        json.dump(out, f)
    sz = os.path.getsize(out_path) / 1024 / 1024
    print(f"[add_event] saved → {out_path} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
