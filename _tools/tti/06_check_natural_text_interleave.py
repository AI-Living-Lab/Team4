#!/usr/bin/env python3
"""
[06] rope2d.py natural_text 모드 검증
-------------------------------------
get_rope_index_25(..., time_marker_token_len=9, time_token_id_range=None) 가
청크당 9개의 자연어 마커 토큰을 designed 3D 위치로 배치하는지 확인.

설계 규칙 (special_token 모드와 동일하지만 마커 길이만 9):
  - 청크 k 의 9 마커 토큰은 모두 동일 3D position 을 가짐
      t = w = k * second_per_grid_t * 2   (비디오 첫 프레임과 동일)
      h = 0  (st_idx + text_len 보정 후 6)
  - 9 마커 → 비디오(H*W/merge²) → 오디오(audio_len[k]) 의 순서로 인터리빙되며
    natural_text 모드에서는 ID 매칭 대신 ~(is_video | is_audio) 로 마커를 식별.

시나리오:
  llm_grid_t=3, llm_grid_h=llm_grid_w=2 (비디오 청크당 4 토큰)
  청크별 audio 토큰 2 개, 전체 audio_len = 6
  second_per_grid_t = 2.0   → 청크 시작시각 0s, 2s, 4s
  마커 토큰: 'second{0000.0}', 'second{0002.0}', 'second{0004.0}' 각각 9 토큰

체크포인트 로딩 없이 빠르게 동작한다.

사용: python _tools/tti/06_check_natural_text_interleave.py
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
TEXT = 42

# 자연어 마커 'second{XXXX.Y}' 토크나이즈 결과 (Qwen2.5 BPE):
#   ['second', '{', D, D, D, D, '.', D, '}'] = 9 토큰
# 실제 토크나이저로부터 검증된 ID. 여기서는 임의 텍스트 ID (TEXT_M*) 로 대체해도 무방
# (rope2d 가 ID 자체를 보지 않고 ~(is_video|is_audio) 로 마커를 식별하므로).
NATURAL_MARKER_LEN = 9

def natural_marker_for(chunk_idx: int, sec_per_grid_t: float):
    """청크 k → 9 토큰의 (의사) ID 시퀀스. video/audio token 과 겹치지 않으면 OK."""
    # text token IDs 50..58 — video/audio/vision_start/end 와 겹치지 않음
    base = 50
    return [base + i for i in range(NATURAL_MARKER_LEN)]

# --- 시퀀스 조립 (dataset.py 인터리빙 스펙과 동일, M=9) ---
sec_per_grid_t = 2.0
T = 3
H = W = 2  # llm_grid (merge 후)

prefix = [TEXT] * 5 + [VISION_START]   # 길이 6 → 블록 ed = 6
chunks = []
for k in range(T):
    chunks += natural_marker_for(k, sec_per_grid_t)   # 9
    chunks += [VIDEO_TOKEN] * (H * W)                 # 4
    chunks += [AUDIO_TOKEN] * 2                       # 청크당 audio_len=2
suffix = [VISION_END, TEXT, TEXT]
seq = prefix + chunks + suffix
expected_total = 6 + (NATURAL_MARKER_LEN + H*W + 2) * T + 3
assert len(seq) == expected_total, f"seq_len={len(seq)} expected {expected_total}"

input_ids = torch.tensor([seq], dtype=torch.long)
attention_mask = torch.ones_like(input_ids)

position_ids, _ = get_rope_index_25(
    spatial_merge_size=2,
    input_ids=input_ids,
    video_grid_thw=torch.tensor([[3, 4, 4]], dtype=torch.long),  # (T,H,W) — merge 전
    audio_lengths=[6],
    second_per_grid_ts=torch.tensor([[sec_per_grid_t]], dtype=torch.float),
    attention_mask=attention_mask,
    time_token_id_range=None,           # natural_text: ID 매칭 안 함
    time_marker_token_len=NATURAL_MARKER_LEN,
)
p = position_ids[:, 0, :]
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

assert position_ids.shape == (3, 1, expected_total), position_ids.shape
print(f"shape = {position_ids.shape}  OK")

# 각 청크의 시작 인덱스
def chunk_start(k):
    return 6 + k * (NATURAL_MARKER_LEN + H*W + 2)

# --- prefix text [0:6] : t=h=w=[0..5] ---
check("prefix text t[0:6]",   range(0, 6),   t_pos, [0,1,2,3,4,5])
check("prefix text h[0:6]",   range(0, 6),   h_pos, [0,1,2,3,4,5])

# --- 청크 0 마커 [6:15] : 전부 t=6, h=6, w=6 ---
c0 = chunk_start(0)
check("chunk0 marker t",  range(c0, c0+NATURAL_MARKER_LEN),  t_pos, [6]*NATURAL_MARKER_LEN)
check("chunk0 marker h",  range(c0, c0+NATURAL_MARKER_LEN),  h_pos, [6]*NATURAL_MARKER_LEN)
check("chunk0 marker w",  range(c0, c0+NATURAL_MARKER_LEN),  w_pos, [6]*NATURAL_MARKER_LEN)

# --- 청크 0 비디오 [15:19] : t=6, h=[6,6,7,7], w=[6,7,6,7] ---
v0 = c0 + NATURAL_MARKER_LEN
check("chunk0 video t",   range(v0, v0+H*W),  t_pos, [6,6,6,6])
check("chunk0 video h",   range(v0, v0+H*W),  h_pos, [6,6,7,7])
check("chunk0 video w",   range(v0, v0+H*W),  w_pos, [6,7,6,7])

# --- 청크 0 오디오 [19:21] : t=[6,7] ---
a0 = v0 + H*W
check("chunk0 audio t",   range(a0, a0+2),    t_pos, [6,7])

# --- 청크 1 마커 : t=10, h=6, w=10 (= 1 * sec_per_grid_t * 2 + 6 = 4+6) ---
c1 = chunk_start(1)
check("chunk1 marker t",  range(c1, c1+NATURAL_MARKER_LEN),  t_pos, [10]*NATURAL_MARKER_LEN)
check("chunk1 marker h",  range(c1, c1+NATURAL_MARKER_LEN),  h_pos, [6]*NATURAL_MARKER_LEN)
check("chunk1 marker w",  range(c1, c1+NATURAL_MARKER_LEN),  w_pos, [10]*NATURAL_MARKER_LEN)

# --- 청크 1 비디오 : t=10 ---
v1 = c1 + NATURAL_MARKER_LEN
check("chunk1 video t",   range(v1, v1+H*W),  t_pos, [10]*4)

# --- 청크 1 오디오 : t=[8,9] (audio_pos 는 audio_lengths=6 을 0..5 로 연속 증가) ---
a1 = v1 + H*W
check("chunk1 audio t",   range(a1, a1+2),    t_pos, [8,9])

# --- 청크 2 마커 : t=14, h=6, w=14 ---
c2 = chunk_start(2)
check("chunk2 marker t",  range(c2, c2+NATURAL_MARKER_LEN),  t_pos, [14]*NATURAL_MARKER_LEN)
check("chunk2 marker h",  range(c2, c2+NATURAL_MARKER_LEN),  h_pos, [6]*NATURAL_MARKER_LEN)
check("chunk2 marker w",  range(c2, c2+NATURAL_MARKER_LEN),  w_pos, [14]*NATURAL_MARKER_LEN)

# --- 청크 2 비디오 : t=14 ---
v2 = c2 + NATURAL_MARKER_LEN
check("chunk2 video t",   range(v2, v2+H*W),  t_pos, [14]*4)

# --- 청크 2 오디오 : t=[10,11] ---
a2 = v2 + H*W
check("chunk2 audio t",   range(a2, a2+2),    t_pos, [10,11])

# --- suffix : 이전 max t = 14 → [15,16,17] ---
suf = a2 + 2
check("suffix text t",    range(suf, suf+3),  t_pos, [15,16,17])

print("\n" + "=" * 60)
print("RESULT:", "PASS" if all_ok else "FAIL")
sys.exit(0 if all_ok else 1)
