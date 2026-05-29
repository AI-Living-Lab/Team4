#!/usr/bin/env python3
"""
PU-VALOR only v5 학습/평가 데이터 준비:
  1) puvalor_train_v5.json   → video/audio path 추가 (in-place)
  2) puvalor_test_v5.json    → T1+T2 만 필터링 + paths + gt_segments
                                → puvalor_test_v5_t1t2.json 신규 생성
  3) tokenize (4+1 / 3+2)

mIoU 평가 대상은 T1+T2 (시간 예측 task). T3 (segment-conditioned captioning)
는 시간 예측이 없어서 mIoU 비대상 → inference 자체에서 제외.
"""
import json
import os
import re

from merge_puvalor_unav import normalize_puvalor

DATA_DIR = "/home/aix23102/audiolm/vS2_eunji/data"


def extract_gt_segments(sample):
    """meta.token 의 <s_i>/<e_i> 쌍을 [[start, end], ...] 로 변환."""
    token_map = sample.get("meta", {}).get("token", {})
    pairs = {}
    for k, v in token_map.items():
        m = re.match(r"^<([se])(\d+)>$", k)
        if not m: continue
        which, idx = m.group(1), int(m.group(2))
        pairs.setdefault(idx, {})[which] = float(v)
    segments = []
    for idx in sorted(pairs.keys()):
        if "s" in pairs[idx] and "e" in pairs[idx]:
            segments.append([pairs[idx]["s"], pairs[idx]["e"]])
    return segments


def main():
    # --- 1) train: paths 추가 ---
    src = os.path.join(DATA_DIR, "puvalor_train_v5.json")
    with open(src) as f:
        train = json.load(f)
    print(f"[train] {len(train)} samples loaded")
    train = [normalize_puvalor(s) for s in train]
    # 확인
    s0 = train[0]
    assert "video" in s0 and "audio" in s0 and "use_audio" in s0
    print(f"  sample[0].video = {s0['video']}")
    with open(src, "w") as f:
        json.dump(train, f)
    sz = os.path.getsize(src) / 1024 / 1024
    print(f"[train] saved {src} ({sz:.1f} MB)")
    print()

    # --- 2) test: filter T1+T2 + paths + gt_segments ---
    src = os.path.join(DATA_DIR, "puvalor_test_v5.json")
    with open(src) as f:
        test = json.load(f)
    print(f"[test] {len(test)} samples loaded (T1+T2+T3 mixed)")

    from collections import Counter
    before_dist = Counter(s["task_type"] for s in test)
    print(f"[test] before filter: {dict(before_dist)}")

    test_t1t2 = []
    skipped_no_seg = 0
    for s in test:
        if s.get("task_type") not in ("T1", "T2"):
            continue
        s = normalize_puvalor(s)
        gt = extract_gt_segments(s)
        if not gt:
            skipped_no_seg += 1
            continue
        s["gt_segments"] = gt
        s["gt_label"] = s.get("event", "")  # eval 스크립트 호환용 alias
        test_t1t2.append(s)

    after_dist = Counter(s["task_type"] for s in test_t1t2)
    print(f"[test] T1+T2 filtered + paths + gt_segments: {len(test_t1t2)}")
    print(f"[test] after filter: {dict(after_dist)}")
    print(f"[test] skipped (no segments): {skipped_no_seg}")

    out_test = os.path.join(DATA_DIR, "puvalor_test_v5_t1t2.json")
    with open(out_test, "w") as f:
        json.dump(test_t1t2, f)
    sz = os.path.getsize(out_test) / 1024 / 1024
    print(f"[test] saved {out_test} ({sz:.1f} MB)")
    print()

    # 샘플 검증
    print("=== test sample 검증 ===")
    for task in ("T1", "T2"):
        s = next(x for x in test_t1t2 if x["task_type"] == task)
        print(f"\n[{task}] id={s['id']}")
        print(f"  video: {s['video']}")
        print(f"  audio: {s['audio']}")
        print(f"  use_audio: {s['use_audio']}")
        print(f"  meta.token (first 4): {dict(list(s['meta']['token'].items())[:4])}")
        print(f"  gt_segments ({len(s['gt_segments'])}): {s['gt_segments'][:3]}{'...' if len(s['gt_segments'])>3 else ''}")
        print(f"  human[0]: {s['conversations'][0]['value'][:120]}")
        print(f"  gpt[1]:   {s['conversations'][1]['value'][:120]}")


if __name__ == "__main__":
    main()
