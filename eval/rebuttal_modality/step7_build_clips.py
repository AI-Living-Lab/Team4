#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step7_build_clips.py — 카테고리별 대표 클립 1개씩 인코딩 (검수 페이지 embed 용).

아티팩트는 외부 미디어를 못 불러오고 이 계정엔 assets 기능이 없어서
클립을 base64 data URI 로 페이지에 직접 넣어야 한다(페이지 전체 16MB 상한).
그래서 아주 작게 만든다: 4초, 256x144@12fps, AAC mono 48k.

대표 샘플 선택 규칙 (결정적 — 같은 입력이면 항상 같은 결과):
  1) 카테고리의 mean_N_gt 를 반올림한 값과 N_gt 가 같은 샘플 우선
  2) 그중 첫 GT 세그먼트 시작이 1.5s 이후인 것 우선 (onset 앞 여유가 있어야 경계 판단 가능)
  3) 남은 후보를 id 사전순으로 정렬해 첫 번째
클립 구간: [seg0_start - 1.5, +4.0s] — 이벤트 onset 이 클립 t=1.5s 지점에 오게 잘라
경계를 실제로 짚어볼 수 있게 한다. onset 위치는 페이지에서 '정답 보기' 로만 노출한다.
"""
import base64
import contextlib
import json
import os
import subprocess
import sys
import wave

import imageio_ffmpeg

from paths import TEST_SPLIT, HERE, log

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
VID = "/home/team404/workspace/datasets/unav_100/videos"
AUD = "/home/team404/workspace/datasets/unav_100/audio"
OUT = f"{HERE}/clips"
PRE, DUR = 1.5, 4.0


def vdur(vid):
    p = os.path.join(AUD, f"{vid}.wav")
    if not os.path.exists(p):
        return None
    with contextlib.closing(wave.open(p)) as w:
        return w.getnframes() / float(w.getframerate())


def pick(rows):
    """카테고리 샘플 목록 -> 대표 1개."""
    import statistics
    mean_ngt = statistics.mean(len(r.get("gt_segments") or []) for r in rows)
    target = max(1, round(mean_ngt))
    def key(r):
        segs = sorted(r.get("gt_segments") or [])
        n = len(segs)
        s0 = segs[0][0] if segs else 0.0
        return (abs(n - target), 0 if s0 >= PRE else 1, r["id"])
    cands = [r for r in rows if (r.get("gt_segments") and
                                 os.path.exists(os.path.join(VID, r["vid"] + ".mp4")))]
    return sorted(cands, key=key)[0] if cands else None


def encode(vid, start, out):
    dur = vdur(vid) or 1e9
    ss = max(0.0, min(start - PRE, max(0.0, dur - DUR)))
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-ss", f"{ss:.3f}",
           "-i", os.path.join(VID, vid + ".mp4"), "-t", f"{DUR:.3f}",
           "-vf", "scale=256:-2:flags=bicubic,fps=12",
           "-c:v", "libx264", "-profile:v", "main", "-pix_fmt", "yuv420p",
           "-b:v", "64k", "-maxrate", "80k", "-bufsize", "160k", "-preset", "slow",
           "-c:a", "aac", "-ac", "1", "-ar", "32000", "-b:a", "48k",
           "-movflags", "+faststart", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return (r.returncode == 0 and os.path.exists(out)), (ss, r.stderr.strip()[:200])


def main():
    only = sys.argv[1:] or None
    os.makedirs(OUT, exist_ok=True)
    rows = json.load(open(TEST_SPLIT))
    by = {}
    for r in rows:
        by.setdefault(r["gt_label"], []).append(r)

    log("STEP7", f"start — ffmpeg={os.path.basename(FFMPEG)}  categories={len(by)}")
    manifest, total, fail = {}, 0, []
    cats = sorted(by)
    if only:
        cats = [c for c in cats if c in only]
    for i, cat in enumerate(cats, 1):
        r = pick(by[cat])
        if r is None:
            fail.append((cat, "후보 없음(영상 파일 부재)"))
            continue
        segs = sorted(r["gt_segments"])
        s0 = float(segs[0][0])
        safe = "".join(ch if ch.isalnum() else "_" for ch in cat)
        path = os.path.join(OUT, safe + ".mp4")
        ok, (ss, err) = encode(r["vid"], s0, path)
        if not ok:
            fail.append((cat, err))
            continue
        sz = os.path.getsize(path)
        total += sz
        manifest[cat] = {
            "id": r["id"], "vid": r["vid"], "file": os.path.basename(path),
            "clip_start": round(ss, 3), "dur": DUR,
            "onset": round(s0 - ss, 3),           # 클립 내 onset 위치(초)
            "segs": [[round(a, 2), round(b, 2)] for a, b in segs],
            "n_gt": len(segs), "bytes": sz,
        }
        if i % 20 == 0:
            log("STEP7", f"  {i}/{len(cats)}  누적 {total/1e6:.2f} MB")

    json.dump(manifest, open(f"{HERE}/clips_manifest.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    b64 = int(total * 4 / 3)
    log("STEP7", f"done — {len(manifest)} clips, raw {total/1e6:.2f} MB, "
                 f"base64 약 {b64/1e6:.2f} MB, 실패 {len(fail)}")
    for c, e in fail:
        log("STEP7", f"  ⚠ 실패 {c}: {e}")
    print(f"\nclips={len(manifest)}  raw={total/1e6:.2f}MB  base64≈{b64/1e6:.2f}MB  fail={len(fail)}")


if __name__ == "__main__":
    main()
