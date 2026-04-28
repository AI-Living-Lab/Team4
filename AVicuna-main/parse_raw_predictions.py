"""
Step 2: Parse AVicuna raw text outputs into structured events.
"""

import argparse
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def safe_float(x):
    try:
        return float(x.strip())
    except:
        return None


def clean_text(text):
    text = text.strip()
    text = text.strip("-\u2022* \t.,")
    text = re.sub(r"\s+", " ", text)
    return text


def parse_raw_output(raw_output):
    if not raw_output:
        return []
    events = []

    # Pattern 1: "From XX to YY, label" (label after comma)
    p1 = re.findall(
        r'[Ff]rom\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*,\s*(.+?)(?=\s+[Ff]rom\s+\d|$)',
        raw_output
    )
    for start_str, end_str, label in p1:
        start = safe_float(start_str)
        end = safe_float(end_str)
        label = clean_text(label)
        if label and start is not None and end is not None and end >= start:
            events.append({"text": label, "start": start, "end": end})

    # Pattern 2: "Label, from XX to YY" (label before comma)
    p2 = re.findall(
        r'([A-Z][^,\n]+?),?\s+from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)',
        raw_output
    )
    for label, start_str, end_str in p2:
        start = safe_float(start_str)
        end = safe_float(end_str)
        label = clean_text(label)
        if label and start is not None and end is not None and end >= start:
            # Avoid duplicates
            dup = False
            for ev in events:
                if ev["text"] == label and ev["start"] == start and ev["end"] == end:
                    dup = True
                    break
            if not dup:
                events.append({"text": label, "start": start, "end": end})

    # Fallback: pipe format "label | start | end"
    if not events:
        for raw_line in raw_output.splitlines():
            line = clean_text(raw_line)
            if not line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 3:
                label, s, e = parts
                s, e = safe_float(s), safe_float(e)
                if label and s is not None and e is not None and e >= s:
                    events.append({"text": clean_text(label), "start": s, "end": e})

    return events


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Load annotations for duration scaling
    ann_path = "data/annotations/unav100_annotations.json"
    duration_map = {}
    try:
        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)
        for vid, info in ann["database"].items():
            duration_map[vid] = info.get("duration", 0)
    except:
        print("Warning: could not load annotations for duration scaling")

    parsed_results = []
    total_events = 0

    for item in raw_data:
        video_id = item.get("video_id")
        raw_output = item.get("raw_output", "")
        events = parse_raw_output(raw_output)

        # Scale 0-99 timestamps to actual duration
        duration = duration_map.get(video_id, 0)
        if duration > 0:
            for ev in events:
                ev["start"] = ev["start"] / 100.0 * duration
                ev["end"] = ev["end"] / 100.0 * duration

        total_events += len(events)
        parsed_results.append({
            "video_id": video_id,
            "raw_output": raw_output,
            "events": events,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed_results, f, ensure_ascii=False, indent=2)

    n_empty = sum(1 for r in parsed_results if len(r["events"]) == 0)
    print(f"Saved parsed predictions to: {output_path}")
    print(f"  Samples: {len(parsed_results)}")
    print(f"  Total events extracted: {total_events}")
    print(f"  Samples with 0 events: {n_empty}")


if __name__ == "__main__":
    main()
