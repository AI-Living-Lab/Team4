#!/usr/bin/env python3
"""
unav100_test_sub80.json 에서 8개 샘플을 선별해 debug_interleave_samples.json 생성.

선정 기준:
  1. 짧은 영상         — duration 최소 근처
  2. 중간 영상         — median 근처
  3. ~60s 영상         — duration 55~65s 구간
  4. 최장 영상         — duration 최대
  5. multi-segment GT  — segments ≥ 3
  6. multi-seg 짧음    — segments ≥ 2 & duration ≤ 20s
  7. audio 없음        — #3(또는 대체) 복사본에서 audio=null, use_audio=false
  8. video 없음        — #3(또는 대체) 복사본에서 video=null + <video> 태그 제거

실행:
  python _tools/debug/build_debug_samples.py \
    --src  data/unav100_test_sub80.json \
    --out  data/debug_interleave_samples.json
"""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path


def probe_duration(path):
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            stderr=subprocess.DEVNULL, timeout=15,
        )
        return float(out.strip())
    except Exception:
        return None


def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] * (hi - k) + xs[hi] * (k - lo) if lo != hi else xs[lo]


def select_sample(pool, used, predicate, label, prefer_mid=True):
    """pool 에서 predicate 만족 & used 에 없는 것 선택. prefer_mid=True 면 중앙값."""
    cands = [i for i in range(len(pool)) if i not in used and predicate(pool[i])]
    if not cands:
        return None
    if prefer_mid:
        cands.sort(key=lambda i: pool[i]["_duration"])
        return cands[len(cands) // 2]
    return cands[0]


def make_entry(e, tag, note):
    out = {k: v for k, v in e.items() if not k.startswith("_")}
    out["_debug_tag"] = tag
    out["_debug_note"] = note
    out["_duration_sec"] = round(e["_duration"], 2)
    return out


def strip_video_tag(conversations):
    for conv in conversations:
        if "value" in conv:
            conv["value"] = (
                conv["value"]
                .replace("<video>\n", "")
                .replace("<video>", "")
            )
    return conversations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    entries = json.load(open(args.src))
    print(f"[src] {len(entries)} entries from {args.src}")

    # ---- duration 탐색 ----
    print("[probe] ffprobe durations...")
    for i, e in enumerate(entries):
        e["_duration"] = probe_duration(e["video"])
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(entries)}]")
    pool = [e for e in entries if e["_duration"] is not None]
    print(f"[probe] {len(pool)}/{len(entries)} have duration")

    ds = [e["_duration"] for e in pool]
    print(f"[dist] min={min(ds):.1f}  p25={pct(ds,25):.1f}  "
          f"median={pct(ds,50):.1f}  p75={pct(ds,75):.1f}  max={max(ds):.1f}")

    used = set()
    picked = []

    # ---- 1) 짧은 ----
    i_short = select_sample(
        pool, used,
        predicate=lambda e: e["_duration"] <= 10.0,
        label="short ≤10s", prefer_mid=False,
    )
    if i_short is None:
        # fallback: 최단
        i_short = min(range(len(pool)), key=lambda i: pool[i]["_duration"])
    used.add(i_short)
    picked.append(make_entry(pool[i_short], "short",
                             f"duration={pool[i_short]['_duration']:.1f}s (최단 계열)"))

    # ---- 2) 중간 15~25s ----
    i_med = select_sample(
        pool, used,
        predicate=lambda e: 15.0 <= e["_duration"] <= 25.0,
        label="medium 15~25s",
    )
    if i_med is None:
        # fallback: median 근처
        target = pct(ds, 50)
        i_med = min((i for i in range(len(pool)) if i not in used),
                    key=lambda i: abs(pool[i]["_duration"] - target))
    used.add(i_med)
    picked.append(make_entry(pool[i_med], "medium",
                             f"duration={pool[i_med]['_duration']:.1f}s (중간)"))

    # ---- 3) ~60s ----
    i_60 = select_sample(
        pool, used,
        predicate=lambda e: 55.0 <= e["_duration"] <= 65.0,
        label="~60s",
    )
    if i_60 is None:
        target = 60.0
        i_60 = min((i for i in range(len(pool)) if i not in used),
                   key=lambda i: abs(pool[i]["_duration"] - target))
    used.add(i_60)
    ref_60 = pool[i_60]  # no_audio / no_video 복제 원본
    picked.append(make_entry(ref_60, "long_60s",
                             f"duration={ref_60['_duration']:.1f}s (~60s)"))

    # ---- 4) 최장 ----
    remaining = [i for i in range(len(pool)) if i not in used]
    i_longest = max(remaining, key=lambda i: pool[i]["_duration"])
    used.add(i_longest)
    picked.append(make_entry(pool[i_longest], "longest",
                             f"duration={pool[i_longest]['_duration']:.1f}s (최장)"))

    # ---- 5) multi-seg ≥3 ----
    i_multi3 = select_sample(
        pool, used,
        predicate=lambda e: len(e.get("gt_segments", [])) >= 3,
        label="multi-seg ≥3",
    )
    if i_multi3 is None:
        # ≥2 로 완화
        i_multi3 = select_sample(
            pool, used,
            predicate=lambda e: len(e.get("gt_segments", [])) >= 2,
            label="multi-seg ≥2 (fallback)",
        )
    if i_multi3 is not None:
        used.add(i_multi3)
        e = pool[i_multi3]
        picked.append(make_entry(e, "multiseg_3plus",
                                 f"segments={len(e['gt_segments'])}, duration={e['_duration']:.1f}s"))
    else:
        print("[warn] multiseg_3plus 선정 실패 — 스킵")

    # ---- 6) multi-seg & short ----
    i_multi_s = select_sample(
        pool, used,
        predicate=lambda e: len(e.get("gt_segments", [])) >= 2 and e["_duration"] <= 20.0,
        label="multi-seg & short",
    )
    if i_multi_s is None:
        # 완화: <=30s
        i_multi_s = select_sample(
            pool, used,
            predicate=lambda e: len(e.get("gt_segments", [])) >= 2 and e["_duration"] <= 30.0,
            label="multi-seg & short (≤30s)",
        )
    if i_multi_s is not None:
        used.add(i_multi_s)
        e = pool[i_multi_s]
        picked.append(make_entry(e, "multiseg_short",
                                 f"segments={len(e['gt_segments'])}, duration={e['_duration']:.1f}s"))
    else:
        print("[warn] multiseg_short 선정 실패 — 스킵")

    # ---- 7) audio 없음 (ref_60 복제) ----
    #   dataset.py 는 "audio" key 존재 여부로 분기하므로 키 자체를 pop.
    no_audio = copy.deepcopy(ref_60)
    no_audio.pop("audio", None)
    no_audio["use_audio"] = False
    picked.append(make_entry(
        no_audio, "no_audio",
        f"{ref_60.get('video','?').split('/')[-1]} 복제 — audio key 제거, use_audio=false "
        f"(duration={ref_60['_duration']:.1f}s)"))

    # ---- 8) video 없음 (ref_60 복제) ----
    #   "video" key pop + <video> 태그 제거 — dataset.py 에서 video 브랜치가 스킵되고
    #   vision span 이 없는 순수 text/audio 경로 검증용.
    no_video = copy.deepcopy(ref_60)
    no_video.pop("video", None)
    no_video["conversations"] = strip_video_tag(no_video.get("conversations", []))
    picked.append(make_entry(
        no_video, "no_video",
        f"ref(long_60s) 복제 — video key 제거, <video> 태그 제거 "
        f"(duration≈{ref_60['_duration']:.1f}s)"))

    # ---- summary ----
    print(f"\n[selected] {len(picked)} samples:")
    for p in picked:
        segs = len(p.get("gt_segments") or [])
        has_a = "Y" if p.get("use_audio") and p.get("audio") else "N"
        has_v = "Y" if p.get("video") else "N"
        print(f"  [{p['_debug_tag']:16}] dur={p['_duration_sec']:>6}s  "
              f"segs={segs}  a={has_a}  v={has_v}  | {p['_debug_note']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(picked, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out_path} ({len(picked)} samples)")


if __name__ == "__main__":
    sys.exit(main())
