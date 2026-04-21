#!/usr/bin/env python3
"""
[03] rope2d.py 타임토큰 활성화 경로 검증
-----------------------------------------
get_rope_index_25(..., time_token_id_range=(lo, hi)) 가 Case 1(오디오+비디오)
에서 타임토큰을 설계대로 배치하는지 확인.

설계 규칙:
  - 청크 k 의 6 타임토큰은 모두 동일 3D position 을 가짐
      t = w = k * second_per_grid_t * 2   (비디오 첫 프레임과 동일)
      h = 0
  - 타임토큰 6 개 → 비디오(H*W/merge²) → 오디오(audio_len[k]) 의 인터리빙 순서대로
    배치되며 RoPE position_ids shape = (3, batch, seq_len).

시나리오:
  llm_grid_t=3, llm_grid_h=llm_grid_w=2 (비디오 청크당 4 토큰)
  청크별 audio 토큰 2 개, 전체 audio_len = 6
  second_per_grid_t = 2.0   → 청크 시작시각 0s, 2s, 4s
  타임토큰 ID 구간 = [151666, 151676]

체크포인트 로딩 없이 빠르게 동작한다.

사용: python _tools/tti/03_check_rope_with_time_tokens.py
종료 코드: 0=PASS, 1=FAIL
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "video_SALMONN2_plus"))

import torch
from qwenvl.data.rope2d import get_rope_index_25


# --- 토큰 ID (실제 체크포인트 config 값) ---
VISION_START = 151652
VIDEO_TOKEN = 151656
AUDIO_TOKEN = 151665
VISION_END = 151653
TIME_LO, TIME_HI = 151666, 151676
T0, TDOT = 151666, 151676
TEXT = 42

def time_tokens_for(chunk_idx: int, sec_per_grid_t: float):
    """청크 시작시각 → 6 토큰 ID 시퀀스."""
    secs = chunk_idx * sec_per_grid_t
    tenths = round(secs * 10)
    i = tenths // 10
    f = tenths % 10
    d = [(i // 1000) % 10, (i // 100) % 10, (i // 10) % 10, i % 10]
    return [T0 + d[0], T0 + d[1], T0 + d[2], T0 + d[3], TDOT, T0 + f]

# --- 시퀀스 조립 (dataset.py 인터리빙 스펙과 동일) ---
sec_per_grid_t = 2.0
T = 3
prefix = [TEXT] * 5 + [VISION_START]   # 길이 6 → 블록 ed = 6
chunks = []
for k in range(T):
    chunks += time_tokens_for(k, sec_per_grid_t)   # 6
    chunks += [VIDEO_TOKEN] * 4                    # llm_grid_h*llm_grid_w = 4
    chunks += [AUDIO_TOKEN] * 2                    # 청크당 audio_len=2
suffix = [VISION_END, TEXT, TEXT]
seq = prefix + chunks + suffix
assert len(seq) == 45, f"seq_len={len(seq)}"

input_ids = torch.tensor([seq], dtype=torch.long)
attention_mask = torch.ones_like(input_ids)

position_ids, _ = get_rope_index_25(
    spatial_merge_size=2,
    input_ids=input_ids,
    video_grid_thw=torch.tensor([[3, 4, 4]], dtype=torch.long),  # (T,H,W) — merge 전
    audio_lengths=[6],
    second_per_grid_ts=torch.tensor([[sec_per_grid_t]], dtype=torch.float),
    attention_mask=attention_mask,
    time_token_id_range=(TIME_LO, TIME_HI),
)
p = position_ids[:, 0, :]  # (3, 45)
t_pos = p[0].tolist()
h_pos = p[1].tolist()
w_pos = p[2].tolist()

all_ok = True
def check(name, idx_range, axis_vals, expected):
    global all_ok
    actual = [axis_vals[i] for i in idx_range]
    ok = actual == expected
    all_ok &= ok
    print(f"  {'OK ' if ok else 'FAIL'}  {name:<40s}  got={actual}  exp={expected}")

assert position_ids.shape == (3, 1, 45), position_ids.shape
print(f"shape = {position_ids.shape}  OK")

# --- prefix text [0:6] : t=h=w=[0..5] ---
check("prefix text t[0:6]",   range(0, 6),   t_pos, [0,1,2,3,4,5])
check("prefix text h[0:6]",   range(0, 6),   h_pos, [0,1,2,3,4,5])

# --- 청크 0 타임 [6:12] : 전부 t=6, h=6, w=6 ---
check("chunk0 time t[6:12]",  range(6, 12),  t_pos, [6]*6)
check("chunk0 time h[6:12]",  range(6, 12),  h_pos, [6]*6)
check("chunk0 time w[6:12]",  range(6, 12),  w_pos, [6]*6)

# --- 청크 0 비디오 [12:16] : t=6, h=[6,6,7,7], w=[6,7,6,7] ---
check("chunk0 video t[12:16]", range(12,16), t_pos, [6,6,6,6])
check("chunk0 video h[12:16]", range(12,16), h_pos, [6,6,7,7])
check("chunk0 video w[12:16]", range(12,16), w_pos, [6,7,6,7])

# --- 청크 0 오디오 [16:18] : t=[6,7], h=[6,6], w=[6,7] ---
check("chunk0 audio t[16:18]", range(16,18), t_pos, [6,7])

# --- 청크 1 타임 [18:24] : t=10 (= 1 * 2 * 2 + 6), h=6, w=10 ---
check("chunk1 time t[18:24]",  range(18,24), t_pos, [10]*6)
check("chunk1 time h[18:24]",  range(18,24), h_pos, [6]*6)
check("chunk1 time w[18:24]",  range(18,24), w_pos, [10]*6)

# --- 청크 1 비디오 [24:28] : t=10 ---
check("chunk1 video t[24:28]", range(24,28), t_pos, [10]*4)

# --- 청크 1 오디오 [28:30] : t=[8,9] (audio_pos 는 전체 6 을 연속 증가) ---
check("chunk1 audio t[28:30]", range(28,30), t_pos, [8,9])

# --- 청크 2 타임 [30:36] : t=14 ---
check("chunk2 time t[30:36]",  range(30,36), t_pos, [14]*6)

# --- 청크 2 비디오 [36:40] : t=14 ---
check("chunk2 video t[36:40]", range(36,40), t_pos, [14]*4)

# --- 청크 2 오디오 [40:42] : t=[10,11] ---
check("chunk2 audio t[40:42]", range(40,42), t_pos, [10,11])

# --- suffix [42:45] : 이전 max t = 14 이므로 15,16,17 ---
check("suffix text t[42:45]",  range(42,45), t_pos, [15,16,17])

print("\n" + "=" * 60)
print("RESULT:", "PASS" if all_ok else "FAIL")
sys.exit(0 if all_ok else 1)
