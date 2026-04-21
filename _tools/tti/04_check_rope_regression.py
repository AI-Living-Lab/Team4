#!/usr/bin/env python3
"""
[04] rope2d.py 회귀 검증 (타임토큰 OFF 경로)
----------------------------------------------
time_token_id_range 를 전달하지 않았을 때 (= 기존 비디오+오디오 인터리빙만)
get_rope_index_25 의 결과가 수정 전과 동일한지 확인한다.

왜 중요한가?
  - Step 3 에서 rope2d 에 타임토큰 분기를 추가했지만, 기본값 None 이 유지돼야
    기존 체크포인트/데이터로 학습하던 플로우가 절대 깨지지 않음이 보장된다.
  - 회귀가 생기면 TTI 분기 자체가 opt-out 이 아니게 되어 위험.

본 스크립트는 기존 로직이 생성해야 하는 전 시퀀스 t-axis 값을 하드코딩해서
비교한다. 시퀀스 구조/파라미터는 03 과 동일하되 타임토큰만 제거.

사용: python _tools/tti/04_check_rope_regression.py
종료 코드: 0=PASS, 1=FAIL
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "video_SALMONN2_plus"))

import torch
from qwenvl.data.rope2d import get_rope_index_25

VISION_START = 151652
VIDEO_TOKEN = 151656
AUDIO_TOKEN = 151665
VISION_END = 151653
TEXT = 42

# 타임토큰 없는 전통 인터리빙
prefix = [TEXT] * 5 + [VISION_START]
chunks = []
for k in range(3):
    chunks += [VIDEO_TOKEN] * 4
    chunks += [AUDIO_TOKEN] * 2
suffix = [VISION_END, TEXT, TEXT]
seq = prefix + chunks + suffix

input_ids = torch.tensor([seq], dtype=torch.long)
attention_mask = torch.ones_like(input_ids)
video_grid_thw = torch.tensor([[3, 4, 4]], dtype=torch.long)
audio_lengths = [6]
second_per_grid_ts = torch.tensor([[2.0]], dtype=torch.float)

kwargs_no = dict(
    spatial_merge_size=2,
    input_ids=input_ids,
    video_grid_thw=video_grid_thw,
    audio_lengths=audio_lengths,
    second_per_grid_ts=second_per_grid_ts,
    attention_mask=attention_mask,
)

# --- (A) 명시적 None vs 생략 → 동일한 결과 ---
pos_explicit, _ = get_rope_index_25(**kwargs_no, time_token_id_range=None)
pos_default, _ = get_rope_index_25(**kwargs_no)
same = torch.equal(pos_explicit, pos_default)
print(f"[A] explicit None vs default: {'OK' if same else 'FAIL'}  (shape={pos_default.shape})")
all_ok = same

# --- (B) t-axis 값이 기존 스펙과 정확히 일치 ---
# 기대 (rope2d.py 기존 Case 1 로직으로 직접 계산):
#   prefix [0:6]         : [0,1,2,3,4,5]
#   chunk0 video [6:10]  : t=6*4 (T=0 베이스 * 2*2=4 → 0, +text_len 6)
#   chunk0 audio [10:12] : [6,7]
#   chunk1 video [12:16] : 10*4
#   chunk1 audio [16:18] : [8,9]
#   chunk2 video [18:22] : 14*4
#   chunk2 audio [22:24] : [10,11]
#   suffix [24:27]       : 이전 max=14 → 15,16,17
expected_t = (
    [0, 1, 2, 3, 4, 5]
    + [6, 6, 6, 6, 6, 7]
    + [10, 10, 10, 10, 8, 9]
    + [14, 14, 14, 14, 10, 11]
    + [15, 16, 17]
)
got_t = pos_default[0, 0].tolist()
same_t = got_t == expected_t
all_ok &= same_t
print(f"[B] t-axis 전체 일치: {'OK' if same_t else 'FAIL'}")
if not same_t:
    print(f"  got: {got_t}")
    print(f"  exp: {expected_t}")

print("\n" + "=" * 60)
print("RESULT:", "PASS" if all_ok else "FAIL")
sys.exit(0 if all_ok else 1)
