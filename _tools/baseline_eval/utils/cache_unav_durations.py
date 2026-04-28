"""Compute and cache UnAV-100 video durations via decord (avicuna2 env)."""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import decord

TEST_JSON = "/workspace/jsy/output/avicuna_unavqa/unav100_test_full.json"
VIDEO_DIR = "/workspace/datasets/unav_100/videos"
OUT = "/workspace/jsy/output/avicuna_unavqa/durations.json"


def video_duration(mp4: str):
    """Return duration in seconds via decord. Reads container metadata only via VideoReader."""
    try:
        vr = decord.VideoReader(mp4, num_threads=1)
        n = len(vr)
        fps = vr.get_avg_fps()
        if fps <= 0:
            return None
        return n / fps
    except Exception:
        return None


def main():
    with open(TEST_JSON) as f:
        rows = json.load(f)
    vids = {os.path.splitext(os.path.basename(r["video"]))[0] for r in rows}
    print(f"unique videos: {len(vids)}", flush=True)

    Path(os.path.dirname(OUT)).mkdir(parents=True, exist_ok=True)

    cache = {}
    if os.path.exists(OUT):
        cache = json.load(open(OUT))
        print(f"resume: {len(cache)} cached", flush=True)

    todo = [v for v in vids if v not in cache]
    print(f"to compute: {len(todo)}", flush=True)

    missing = []
    fails = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        existing = [v for v in todo if os.path.exists(os.path.join(VIDEO_DIR, f"{v}.mp4"))]
        missing = [v for v in todo if v not in set(existing)]
        futs = {ex.submit(video_duration, os.path.join(VIDEO_DIR, f"{v}.mp4")): v
                for v in existing}
        for i, fut in enumerate(as_completed(futs), 1):
            v = futs[fut]
            d = fut.result()
            if d is None:
                fails.append(v)
            else:
                cache[v] = d
            if i % 200 == 0:
                print(f"  done {i}/{len(futs)}", flush=True)

    with open(OUT, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"saved {OUT}: {len(cache)} entries")
    if missing:
        print(f"missing mp4 ({len(missing)}): {missing[:5]}...")
    if fails:
        print(f"decord failed ({len(fails)}): {fails[:5]}...")


if __name__ == "__main__":
    main()
