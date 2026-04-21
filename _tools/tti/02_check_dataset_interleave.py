#!/usr/bin/env python3
"""
[02] dataset.py 인터리빙 검증
------------------------------
dataset.py 의 두 가지 책임을 코드 변경 없이 확인한다.

  (A) sec_to_time_token_str(sec) 이 "XXXX.Y" 6개 타임토큰 문자열을 만드는지
      - 경계/반올림/클리핑 포함
  (B) 비디오+오디오 인터리빙 시 각 청크가
        <t*>×6  →  <|video_pad|>×(H*W/merge²)  →  <|audio_pad|>×audio_len
      순서로 붙는지

체크포인트/토크나이저를 로드하지 않는다 — 파이썬 문자열 레벨 검증.

사용:
  python _tools/tti/02_check_dataset_interleave.py

종료 코드: 0=PASS, 1=FAIL
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "video_SALMONN2_plus"))

from qwenvl.data.dataset import sec_to_time_token_str

all_ok = True

# --- (A) 변환 함수 단위 테스트 ---
# 기대 포맷: <tA><tB><tC><tD><tdot><tE>  (정수 4자리 + . + 소수 1자리)
print("[A] sec_to_time_token_str 단위 테스트")
cases = [
    # (입력초, 기대 문자열, 설명)
    (0.0,      "<t0><t0><t0><t0><tdot><t0>", "0초 → 0000.0"),
    (0.1,      "<t0><t0><t0><t0><tdot><t1>", "최소 소수점 1자리"),
    (1.2,      "<t0><t0><t0><t1><tdot><t2>", "문서 예시 1.2s"),
    (12.3,     "<t0><t0><t1><t2><tdot><t3>", "두 자리"),
    (123.4,    "<t0><t1><t2><t3><tdot><t4>", "세 자리"),
    (9999.9,   "<t9><t9><t9><t9><tdot><t9>", "상한"),
    (99999.0,  "<t9><t9><t9><t9><tdot><t9>", "상한 클리핑"),
    (-1.0,     "<t0><t0><t0><t0><tdot><t0>", "하한 클리핑"),
    (1.24,     "<t0><t0><t0><t1><tdot><t2>", "반올림(<.5) → 1.2"),
    # Python round() 는 banker's rounding — 0.5 는 짝수로 붙는다.
    # round(12.5)=12, round(13.5)=14 이므로 1.25 → 1.2, 1.35 → 1.4.
    (1.25,     "<t0><t0><t0><t1><tdot><t2>", "banker's round: 1.25 → 1.2"),
    (1.35,     "<t0><t0><t0><t1><tdot><t4>", "banker's round: 1.35 → 1.4"),
]
for sec, expected, note in cases:
    got = sec_to_time_token_str(sec)
    ok = got == expected
    all_ok &= ok
    print(f"  {'OK ' if ok else 'FAIL'}  sec={sec!s:<8s}  got={got!s:<30s}  "
          f"exp={expected!r}  # {note}")

# --- (B) 인터리빙 순서 ---
# dataset.py 의 실제 조립 로직을 그대로 흉내내서 결과 문자열을 생성.
# (dataset.py 는 tokenizer 까지 필요해서 여기서는 로직만 재현한다.)
print("\n[B] 청크별 인터리빙 조립 (dataset.py 와 동일 규칙)")

def build_interleave(T, hw_video_tokens_per_chunk, audio_len_per_chunk, sec_per_grid_t):
    """dataset.py L175-L187 를 그대로 재현."""
    parts = ["<|vision_start|>"]
    for timestep in range(T):
        if sec_per_grid_t is not None:
            parts.append(sec_to_time_token_str(timestep * sec_per_grid_t))
        parts.append("<|video_pad|>" * hw_video_tokens_per_chunk)
        parts.append("<|audio_pad|>" * audio_len_per_chunk[timestep])
    parts.append("<|vision_end|>")
    return "".join(parts)

T = 3
hw = 4                       # llm_grid_h*llm_grid_w = 2*2 = 4
audio_len = [2, 2, 2]
sec_per_grid_t = 2.0         # 청크 시작시각: 0s, 2s, 4s

out = build_interleave(T, hw, audio_len, sec_per_grid_t)

# 각 청크에 나와야 하는 시간 토큰 문자열
chunk_time_strs = [sec_to_time_token_str(k * sec_per_grid_t) for k in range(T)]
print(f"  expected chunk time strings: {chunk_time_strs}")

# 검증 1: 각 청크 문자열이 등장하는가
for i, s in enumerate(chunk_time_strs):
    occ = out.count(s)
    ok = occ >= 1
    all_ok &= ok
    print(f"  {'OK ' if ok else 'FAIL'}  chunk{i} time marker '{s}' 발견 수={occ}")

# 검증 2: 시간토큰 직후 video_pad, 그 직후 audio_pad 순서 유지
pos = 0
ok_order = True
for k, s in enumerate(chunk_time_strs):
    idx = out.find(s, pos)
    if idx < 0:
        ok_order = False
        break
    after = out[idx + len(s):]
    prefix_video = "<|video_pad|>" * hw
    prefix_audio = "<|audio_pad|>" * audio_len[k]
    if not after.startswith(prefix_video + prefix_audio):
        ok_order = False
        print(f"  FAIL  chunk{k}: expected video×{hw} + audio×{audio_len[k]} after time marker")
        break
    pos = idx + len(s) + len(prefix_video) + len(prefix_audio)

if ok_order:
    print(f"  OK    모든 청크에서 시간토큰 → video_pad → audio_pad 순서 유지")
else:
    all_ok = False

# 검증 3: sec_per_grid_t=None (타임토큰 없는 경로) 는 기존 포맷 유지
out_no = build_interleave(T, hw, audio_len, sec_per_grid_t=None)
has_no_time = all(c not in out_no for c in ["<t0>", "<tdot>"])
print(f"  {'OK ' if has_no_time else 'FAIL'}  sec_per_grid_t=None 일 때 타임토큰 미삽입")
all_ok &= has_no_time

print("\n" + "=" * 60)
print("RESULT:", "PASS" if all_ok else "FAIL")
sys.exit(0 if all_ok else 1)
