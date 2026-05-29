#!/usr/bin/env python3
"""
v5 학습/평가용 보조 JSON 생성:
  1) puvalor_val_v5_sub500.json — puvalor_val_v5 에서 500 sample 균등 추출
     → in-training eval_loss 용. tokenize_time 로 tok4_1/tok3_2 각각 생성.
  2) unav_test_t2.json — UnAV-100 test set 의 T2-style (caption→time) 단일 turn.
     gt_segments 보존. tok4_1/tok3_2 각각 생성.
"""
import json
import os
import random
from collections import defaultdict
from convert_unav_to_stage3_format import (
    T2_PROMPTS, VIDEO_DIR, AUDIO_DIR, ANN_PATH
)

DATA_DIR = "/home/aix23102/audiolm/vS2_eunji/data"


def build_val_sub500(seed=42):
    """puvalor_val_v5.json → 500 sample subset (task_type 분포 유지)."""
    with open(os.path.join(DATA_DIR, "puvalor_val_v5.json")) as f:
        val = json.load(f)
    print(f"puvalor_val_v5.json: {len(val)} samples")

    # task type 별 stratified sample
    by_task = defaultdict(list)
    for s in val:
        by_task[s.get("task_type", "unknown")].append(s)

    rng = random.Random(seed)
    target_total = 500
    n_per_task = {k: max(1, int(target_total * len(v) / len(val))) for k, v in by_task.items()}
    # 총합 조정
    diff = target_total - sum(n_per_task.values())
    if diff > 0:
        # 가장 큰 task 에 추가
        biggest = max(n_per_task, key=lambda k: len(by_task[k]))
        n_per_task[biggest] += diff
    elif diff < 0:
        biggest = max(n_per_task, key=lambda k: n_per_task[k])
        n_per_task[biggest] += diff

    out = []
    for k, n in n_per_task.items():
        out.extend(rng.sample(by_task[k], min(n, len(by_task[k]))))
    rng.shuffle(out)

    from collections import Counter
    print(f"sub500 task distribution: {dict(Counter(s['task_type'] for s in out))}")

    out_path = os.path.join(DATA_DIR, "puvalor_val_v5_sub500.json")
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"Saved → {out_path}\n")


def build_unav_test_t2(seed=42):
    """UnAV test 각 (video, unique label) 마다 한 sample 만들기 (single-turn T2).

    eval 용 2-turn 포맷:
      [human] "Where can we observe {label}?"
      [gpt]   "From <s0> to <e0>. From <s1> to <e1>. ..."   ← multi-instance
    + gt_segments (raw [start, end] list)
    """
    with open(ANN_PATH) as f:
        ann = json.load(f)
    db = ann["database"]

    rng = random.Random(seed)
    out = []
    for vid in sorted(db.keys()):
        info = db[vid]
        if info.get("subset") != "test":
            continue
        anns = info.get("annotations", [])
        if not anns:
            continue
        by_label = defaultdict(list)
        for a in anns:
            by_label[a["label"].strip()].append(a["segment"])
        for lbl_idx, (label, segs) in enumerate(sorted(by_label.items())):
            segs_sorted = sorted(segs, key=lambda s: s[0])
            token_map = {}
            time_parts = []
            for i, (s, e) in enumerate(segs_sorted):
                s_ph = f"<s{i}>"
                e_ph = f"<e{i}>"
                token_map[s_ph] = float(s)
                token_map[e_ph] = float(e)
                time_parts.append(f"From {s_ph} to {e_ph}.")
            time_response = " ".join(time_parts)
            prompt_q = rng.choice(T2_PROMPTS).format(event=label)

            out.append({
                "id": f"unav_{vid}_test_t2_{lbl_idx}",
                "video": os.path.join(VIDEO_DIR, f"{vid}.mp4"),
                "audio": os.path.join(AUDIO_DIR, f"{vid}.wav"),
                "use_audio": True,
                "conversations": [
                    {"from": "human", "value": f"<video>\n{prompt_q}"},
                    {"from": "gpt", "value": time_response},
                ],
                "meta": {
                    "duration": float(info["duration"]),
                    "token": token_map,
                },
                "source": "unav",
                "source_id": "unav",
                "task_type": "T2",
                "event": label,
                "gt_label": label,
                "gt_segments": [[float(s), float(e)] for s, e in segs_sorted],
            })

    print(f"unav_test_t2: {len(out)} samples")
    out_path = os.path.join(DATA_DIR, "unav_test_t2.json")
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"Saved → {out_path}\n")


if __name__ == "__main__":
    build_val_sub500()
    build_unav_test_t2()
