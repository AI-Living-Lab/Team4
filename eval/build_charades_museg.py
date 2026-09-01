#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_charades_museg.py — MUSEG 추론/채점용 Charades-STA 테스트셋 생성.

datasets/charades_sta/annotations/charades_sta_test.txt (공식 3720 문장-구간 쌍) →
  data/test/charades_sta_museg.json          (flat, 3720)
  data/test/charades_sta_museg_sanity30.json (스모크용 30)
  data/test/charades_sta_museg/chunk_XXXX.json (500씩, 8청크)

스키마는 data/test/thumos_tail_museg.json 과 동일(video/audio/use_audio/conversations/
gt_label/gt_segments) + MUSEG 파이프라인이 요구하는 id/vid/question/groundtruth 를 추가.
 - id     : "{vid}_{글로벌index:05d}"  (ChronusOmni charades 런과 동일 규칙, eval_museg.py 매칭 키)
 - question: format 블록 없는 stem 만. MUSEG 네이티브 TEMPLATE 이 <think>/<answer> 를
             이미 강제하므로 우리 format 블록은 중복 → 넣지 않음(build_museg_inputs.py 와 동일 방침).
 - groundtruth: eval_grounding.py 가 요구하는 필드(gt_segments 사본).

Charades-STA 는 문장당 세그먼트 1개(멀티세그 아님) → gpt 답도 단일 "X.XX-X.XX".
멱등(재실행 안전).
"""
import argparse, json, os, re

WS = "/home/team404/workspace"
STEM = ("You are given a video about human daily activities. Watch the video carefully "
        "and find the visual event described by the sentence: '{sent}'. "
        'Output in the format of "X.XX-X.XX".')
VID_RE = re.compile(r"([A-Z0-9]{5})$")


def parse_ann(path, video_root):
    """'{vid} {start} {end}##{sentence}' 파싱. 깨진 vid 토큰(경로 접두어 오염)은 뒤 5자로 복구."""
    rows, repaired = [], []
    for ln in open(path):
        ln = ln.strip()
        if not ln:
            continue
        head, sent = ln.split("##", 1)
        vid, s, e = head.split()
        if not os.path.exists(os.path.join(video_root, f"{vid}.mp4")):
            m = VID_RE.search(vid)
            if m and os.path.exists(os.path.join(video_root, f"{m.group(1)}.mp4")):
                repaired.append((vid, m.group(1)))
                vid = m.group(1)
        rows.append((vid, float(s), float(e), sent.strip()))
    for old, new in repaired:
        print(f"[build] repaired broken vid token: {old!r} → {new!r}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default=f"{WS}/datasets/charades_sta/annotations/charades_sta_test.txt")
    ap.add_argument("--video_root", default=f"{WS}/datasets/charades_sta/Charades_v1")
    ap.add_argument("--audio_root", default=f"{WS}/datasets/charades_sta/audio")
    ap.add_argument("--out", default=f"{WS}/data/test/charades_sta_museg.json")
    ap.add_argument("--chunks_dir", default=f"{WS}/data/test/charades_sta_museg")
    ap.add_argument("--chunk_size", type=int, default=500)
    ap.add_argument("--sanity", type=int, default=30)
    args = ap.parse_args()

    rows = parse_ann(args.ann, args.video_root)
    print(f"[build] annotation rows: {len(rows)}")

    built, miss_v, miss_a = [], 0, 0
    for i, (vid, s, e, sent) in enumerate(rows):
        video = os.path.join(args.video_root, f"{vid}.mp4")
        audio = os.path.join(args.audio_root, f"{vid}.wav")
        miss_v += not os.path.exists(video)
        miss_a += not os.path.exists(audio)
        label = sent[:-1].strip() if sent.endswith(".") else sent   # 문장 끝 마침표 제거
        seg = [[round(s, 2), round(e, 2)]]
        q = STEM.format(sent=label)
        built.append({
            "id": f"{vid}_{i:05d}",
            "vid": vid,
            "video": video,
            "audio": audio,
            "use_audio": False,          # MUSEG = 비디오 전용
            "question": q,               # build_museg_inputs.py / infer 용 (stem only)
            "conversations": [
                {"from": "human", "value": "<video>\n" + q},
                {"from": "gpt", "value": f"{s:.2f}-{e:.2f}"},
            ],
            "gt_label": label,
            "gt_segments": seg,
            "groundtruth": seg,
        })
    print(f"[build] samples: {len(built)}  missing video: {miss_v}  missing audio: {miss_a}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(built, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[build] wrote {args.out} ({len(built)})")

    if args.sanity:
        step = max(1, len(built) // args.sanity)
        sub = built[::step][:args.sanity]
        p = args.out.replace(".json", f"_sanity{args.sanity}.json")
        json.dump(sub, open(p, "w"), ensure_ascii=False, indent=2)
        print(f"[build] wrote {p} ({len(sub)})")

    if args.chunks_dir:
        os.makedirs(args.chunks_dir, exist_ok=True)
        n = (len(built) + args.chunk_size - 1) // args.chunk_size
        for i in range(n):
            p = os.path.join(args.chunks_dir, f"chunk_{i:04d}.json")
            seg = built[i*args.chunk_size:(i+1)*args.chunk_size]
            json.dump(seg, open(p, "w"), ensure_ascii=False, indent=2)
        print(f"[build] wrote {n} chunks → {args.chunks_dir}")


if __name__ == "__main__":
    main()
