#!/usr/bin/env python3
"""
[05] modeling_qwen2_5_vl.py shim 검증
--------------------------------------
Step 4 에서 modeling 의 get_rope_index (기존 424줄 복제본) 를 rope2d 위임
shim 으로 바꾼 결과가 아래 조건을 만족하는지 확인.

  (A) 파일 import 시 순환 의존/문법 오류 없음
  (B) shim 시그니처가 기존 positional 호출부(sft_forward 내부)와 호환
      즉 (input_ids, image_grid_thw, video_grid_thw, audio_lengths,
           second_per_grid_ts, attention_mask, time_token_id_range) 순서 유지
  (C) time_token_id_range 를 명시하면 그 값으로 동작
  (D) 명시하지 않으면 self.config.time_token_id_range 를 자동으로 꺼내 사용
      (list → tuple 변환 포함)
  (E) config 에도 없고 인자에도 없으면 fallback (None) 으로 동작 — 에러 없음

체크포인트 weight 를 로드하지 않는다 (클래스 정의만 import + 가짜 self).

사용: python _tools/tti/05_check_modeling_delegate.py
종료 코드: 0=PASS, 1=FAIL
"""
import inspect
import os
import sys
import types

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "video_SALMONN2_plus"))

import torch

# --- (A) import ---
print("[A] import modeling_qwen2_5_vl ...")
from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus
print("  OK")

# --- (B) 시그니처 ---
print("\n[B] get_rope_index signature")
sig = inspect.signature(video_SALMONN2_plus.get_rope_index)
params = list(sig.parameters.keys())
expected = ["self", "input_ids", "image_grid_thw", "video_grid_thw",
            "audio_lengths", "second_per_grid_ts", "attention_mask",
            "time_token_id_range"]
all_ok = params == expected
print(f"  params: {params}")
print(f"  {'OK' if all_ok else 'FAIL'}  (expected {expected})")

# --- 공통: rope2d 위임이 제대로 도는지 확인용 mock 시퀀스 ---
VS, VE, V, A = 151652, 151653, 151656, 151665
T0, TDOT = 151666, 151676
TEXT = 42

def build_seq():
    # 청크 3 개 × (6 time + 4 video + 2 audio), sec_per_grid_t=2.0
    def time_ids(k):
        secs = k * 2.0
        tenths = round(secs * 10); i = tenths // 10; f = tenths % 10
        d = [(i//1000)%10, (i//100)%10, (i//10)%10, i%10]
        return [T0+d[0], T0+d[1], T0+d[2], T0+d[3], TDOT, T0+f]
    body = []
    for k in range(3):
        body += time_ids(k) + [V]*4 + [A]*2
    return [TEXT]*5 + [VS] + body + [VE, TEXT, TEXT]

input_ids = torch.tensor([build_seq()], dtype=torch.long)
video_grid_thw = torch.tensor([[3, 4, 4]], dtype=torch.long)
audio_lengths = [6]
second_per_grid_ts = torch.tensor([[2.0]], dtype=torch.float)
attention_mask = torch.ones_like(input_ids)

def make_self(cfg_range):
    """가짜 self 객체(video_SALMONN2_plus 인스턴스화를 피하기 위한 types.SimpleNamespace)."""
    vision_cfg = types.SimpleNamespace(spatial_merge_size=2)
    cfg = types.SimpleNamespace(vision_config=vision_cfg,
                                time_token_id_range=cfg_range)
    return types.SimpleNamespace(config=cfg)

def call(self_obj, **overrides):
    return video_SALMONN2_plus.get_rope_index(
        self_obj,
        input_ids=input_ids,
        video_grid_thw=video_grid_thw,
        audio_lengths=audio_lengths,
        second_per_grid_ts=second_per_grid_ts,
        attention_mask=attention_mask,
        **overrides,
    )

# --- (C) 인자로 명시한 경우 ---
print("\n[C] explicit time_token_id_range argument")
pos_explicit, _ = call(make_self(None), time_token_id_range=(151666, 151676))
chunk0_t = pos_explicit[0, 0, 6:12].tolist()
ok_c = chunk0_t == [6]*6
all_ok &= ok_c
print(f"  chunk0 time t[6:12] = {chunk0_t} -> {'OK' if ok_c else 'FAIL'}")

# --- (D) config.time_token_id_range 자동 추출 (list → tuple) ---
print("\n[D] config.time_token_id_range auto-pull (list → tuple)")
pos_auto, _ = call(make_self([151666, 151676]))  # 일부러 list 로 전달
same = torch.equal(pos_auto, pos_explicit)
all_ok &= same
print(f"  explicit vs auto-from-config 결과 동일: {'OK' if same else 'FAIL'}")

# --- (E) config/인자 모두 없으면 에러 없이 fallback 동작 ---
print("\n[E] fallback (both None) — should not raise, shape preserved")
try:
    pos_fb, _ = call(make_self(None))
    shape_ok = pos_fb.shape == (3, 1, input_ids.shape[1])
    all_ok &= shape_ok
    print(f"  shape={pos_fb.shape} -> {'OK' if shape_ok else 'FAIL'}")
except Exception as e:
    all_ok = False
    print(f"  FAIL  raised {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("RESULT:", "PASS" if all_ok else "FAIL")
sys.exit(0 if all_ok else 1)
