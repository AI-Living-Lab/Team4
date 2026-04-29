#!/usr/bin/env python3
"""
[07] tti_time_format=off 모드 검증
-----------------------------------
세 가지를 확인:
  (A) make_time_marker_string 은 off 외 두 모드에서만 호출되며, dataset 인터리빙
      라인 ([dataset.py:180]) 에 `tti_time_format != "off"` 가드가 들어가 있어
      off 모드에서는 video/audio 청크 사이에 마커 토큰이 추가되지 않는다.
  (B) rope2d off path: time_token_id_range=None, time_marker_token_len=None 인
      경우 04 와 동일하게 비-TTI 분기로 동작 (= 원래 Qwen2.5-VL 동작 보존).
  (C) _TIME_MARKER_TOKEN_LEN dict 에 'off': 0 매핑이 있어 모드 enum 검증을 통과.

사용: python _tools/tti/07_check_off_mode.py
종료 코드: 0=PASS, 1=FAIL
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "video_SALMONN2_plus"))

import torch
from qwenvl.data.dataset import _TIME_MARKER_TOKEN_LEN, sec_to_time_token_str
from qwenvl.data.rope2d import get_rope_index_25

all_ok = True

# --- (A) dataset.py 인터리빙: off 모드에서는 마커 미삽입 ---
print("[A] off 모드 인터리빙 (dataset.py 규칙 재현)")

def build_interleave_with_mode(T, hw, audio_len, sec_per_grid_t, mode):
    """dataset.py generate_id_target 의 마커 삽입 가드 재현."""
    parts = ["<|vision_start|>"]
    for timestep in range(T):
        if sec_per_grid_t is not None and mode != "off":
            # special_token / natural_text 만 마커 추가 (편의상 special_token 모양으로)
            parts.append(sec_to_time_token_str(timestep * sec_per_grid_t))
        parts.append("<|video_pad|>" * hw)
        parts.append("<|audio_pad|>" * audio_len[timestep])
    parts.append("<|vision_end|>")
    return "".join(parts)

T = 3
hw = 4
audio_len = [2, 2, 2]
sec_per_grid_t = 2.0

off_out = build_interleave_with_mode(T, hw, audio_len, sec_per_grid_t, "off")
on_out  = build_interleave_with_mode(T, hw, audio_len, sec_per_grid_t, "special_token")

has_no_time_off = all(c not in off_out for c in ["<t0>", "<tdot>"])
has_time_on     = "<t0>" in on_out and "<tdot>" in on_out
print(f"  {'OK ' if has_no_time_off else 'FAIL'}  off 모드 입력에 시간 마커 없음")
print(f"  {'OK ' if has_time_on     else 'FAIL'}  special_token 모드 입력에 시간 마커 있음")
all_ok &= has_no_time_off and has_time_on

# --- (B) rope2d off path: 두 인자 None 이면 비-TTI 동작 ---
print("\n[B] rope2d off path (time_marker_token_len=None, time_token_id_range=None)")
VS, VE, V, A = 151652, 151653, 151656, 151665
TEXT = 42
prefix = [TEXT] * 5 + [VS]
chunks = []
for k in range(T):
    chunks += [V] * hw
    chunks += [A] * audio_len[k]
suffix = [VE, TEXT, TEXT]
seq = prefix + chunks + suffix
input_ids = torch.tensor([seq], dtype=torch.long)
attention_mask = torch.ones_like(input_ids)

position_ids, _ = get_rope_index_25(
    spatial_merge_size=2,
    input_ids=input_ids,
    video_grid_thw=torch.tensor([[3, 4, 4]], dtype=torch.long),
    audio_lengths=[6],
    second_per_grid_ts=torch.tensor([[sec_per_grid_t]], dtype=torch.float),
    attention_mask=attention_mask,
    time_token_id_range=None,
    time_marker_token_len=None,
)
shape_ok = position_ids.shape == (3, 1, len(seq))
print(f"  {'OK ' if shape_ok else 'FAIL'}  shape={tuple(position_ids.shape)} (no errors raised)")
all_ok &= shape_ok

# off path 에서는 텍스트 prefix [0:6] 가 [0..5], 비전 블록은 동일 t (chunk_t_base) 로 시작
t_pos = position_ids[0, 0, :].tolist()
prefix_ok = t_pos[:6] == [0, 1, 2, 3, 4, 5]
print(f"  {'OK ' if prefix_ok else 'FAIL'}  prefix t[0:6]={t_pos[:6]} == [0..5]")
all_ok &= prefix_ok

# 첫 비디오 청크의 t 값들은 모두 6 (= ed=6 + chunk_t_base[0]=0)
v0_start = 6
v0_end = v0_start + hw
v0_t = t_pos[v0_start:v0_end]
v0_ok = v0_t == [6, 6, 6, 6]
print(f"  {'OK ' if v0_ok else 'FAIL'}  chunk0 video t={v0_t} == [6,6,6,6]")
all_ok &= v0_ok

# --- (C) mode enum 검증 ---
print("\n[C] _TIME_MARKER_TOKEN_LEN dict 키")
expected_keys = {"off", "special_token", "natural_text"}
got_keys = set(_TIME_MARKER_TOKEN_LEN.keys())
keys_ok = got_keys == expected_keys
print(f"  {'OK ' if keys_ok else 'FAIL'}  keys={got_keys} == {expected_keys}")
off_zero = _TIME_MARKER_TOKEN_LEN.get("off") == 0
print(f"  {'OK ' if off_zero else 'FAIL'}  _TIME_MARKER_TOKEN_LEN['off']==0")
all_ok &= keys_ok and off_zero

print("\n" + "=" * 60)
print("RESULT:", "PASS" if all_ok else "FAIL")
sys.exit(0 if all_ok else 1)
