import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args():
    parser = argparse.ArgumentParser(description="Parse AVicuna raw predictions into structured events.")
    parser.add_argument("--input", type=str, required=True, help="Path to raw predictions JSON")
    parser.add_argument("--output", type=str, required=True, help="Path to save parsed JSON")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="If set, lines that do not exactly match expected format will be skipped."
    )
    return parser.parse_args()


def safe_float(x: str) -> Optional[float]:
    try:
        return float(x.strip())
    except Exception:
        return None


def clean_text(text: str) -> str:
    text = text.strip()
    text = text.strip("-•* \t")
    text = re.sub(r"\s+", " ", text)
    return text


def parse_line_pipe(line: str) -> Optional[Dict[str, Any]]:
    """
    Expected format:
    [event label] | [start time] | [end time]
    """
    parts = [p.strip() for p in line.split("|")]
    if len(parts) != 3:
        return None

    label, start_str, end_str = parts
    start = safe_float(start_str)
    end = safe_float(end_str)

    if not label or start is None or end is None:
        return None

    return {
        "text": clean_text(label),
        "start": start,
        "end": end,
    }


def parse_line_fallback(line: str) -> Optional[Dict[str, Any]]:
    """
    Fallback parser for lines like:
    - playing guitar from 12.3 to 15.8
    - clapping: 18.0 - 20.2
    """
    line = clean_text(line)

    patterns = [
        r"^(?P<label>.+?)\s+from\s+(?P<start>\d+(?:\.\d+)?)\s+to\s+(?P<end>\d+(?:\.\d+)?)$",
        r"^(?P<label>.+?)\s*[:\-]\s*(?P<start>\d+(?:\.\d+)?)\s*[-~]\s*(?P<end>\d+(?:\.\d+)?)$",
        r"^(?P<label>.+?)\s*\|\s*(?P<start>\d+(?:\.\d+)?)\s*\|\s*(?P<end>\d+(?:\.\d+)?)$",
    ]

    for pat in patterns:
        m = re.match(pat, line, flags=re.IGNORECASE)
        if m:
            label = clean_text(m.group("label"))
            start = safe_float(m.group("start"))
            end = safe_float(m.group("end"))
            if label and start is not None and end is not None:
                return {
                    "text": label,
                    "start": start,
                    "end": end,
                }

    return None


def parse_raw_output(raw_output: str, strict: bool = False) -> List[Dict[str, Any]]:
    if raw_output is None:
        return []

    events: List[Dict[str, Any]] = []
    lines = raw_output.splitlines()

    for raw_line in lines:
        line = clean_text(raw_line)
        if not line:
            continue

        parsed = parse_line_pipe(line)
        if parsed is None and not strict:
            parsed = parse_line_fallback(line)

        if parsed is not None:
            if parsed["end"] >= parsed["start"]:
                events.append(parsed)

    return events


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    parsed_results = []

    for item in raw_data:
        video_id = item.get("video_id")
        video_path = item.get("video_path")
        audio_path = item.get("audio_path")
        query = item.get("query")
        raw_output = item.get("raw_output", "")

        events = parse_raw_output(raw_output, strict=args.strict)

        parsed_results.append(
            {
                "video_id": video_id,
                "video_path": video_path,
                "audio_path": audio_path,
                "query": query,
                "raw_output": raw_output,
                "events": events,
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed_results, f, ensure_ascii=False, indent=2)

    print(f"Saved parsed predictions to: {output_path}")
    print(f"Num samples: {len(parsed_results)}")


if __name__ == "__main__":
    main()
