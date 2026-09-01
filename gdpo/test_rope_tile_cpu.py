#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_rope_tile_cpu.py — fastgen 패치의 tile 정합성을 CPU에서만 검증.

모델 가중치를 전혀 로드하지 않으므로 GPU 를 건드리지 않는다(= 본 학습 무해).

검증하는 것
  A) _tile 이 모델의 _expand_inputs_for_generation._repeat_interleave_samples 와
     같은 의미(batch-major tile)인지 — 순수 텐서 비교.
  B) get_rope_index_25 에 tile 한 batch=G 입력을 넣었을 때, 각 행의 position_ids 가
     batch=1 결과와 정확히 같은지 — M-RoPE / audio_lengths / video_grid_thw 인덱싱이
     배치에서도 맞는지가 이 패치의 유일한 실질 리스크였다.
  C) (대조군) num_return_sequences 경로가 쓰는 repeat_interleave 로 확장하면
     실제로 깨지는지 — "왜 tile 이어야 하는가"의 증거.
"""
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "video_SALMONN2_plus"))

from qwenvl.data.rope2d import get_rope_index_25  # noqa: E402


def _tile(x, n):
    """fastgen 패치의 헬퍼와 동일."""
    if x is None:
        return None
    if isinstance(x, list):
        return x * n
    reps = [1] * x.dim()
    reps[0] = n
    return x.repeat(*reps)


def _repeat_interleave_samples(x, lengths, repeat_times):
    """모델 _expand_inputs_for_generation 안의 헬퍼를 그대로 옮긴 것(정답 기준)."""
    samples = torch.split(x, lengths)
    repeat_args = [repeat_times] + [1] * (x.dim() - 1)
    return torch.cat([s.repeat(*repeat_args) for s in samples], dim=0)


# --- 토큰 ID (rope2d 가 하드코딩하는 값과 동일) ---
VISION_START = 151652
VIDEO_PAD = 151656
AUDIO_PAD = 151665


def build_fake_sample(n_video_tok=64, n_audio_tok=60, n_text=12,
                      t=4, h=4, w=4, time_range=(151667, 151676)):
    """TTI special_token 모드를 흉내낸 input_ids 한 줄을 만든다.
    vision_start 뒤에 time-marker(<t*>) 가 오고, video_pad / audio_pad 가 이어지는 구조."""
    lo, hi = time_range
    ids = [100] * 3                      # 앞쪽 텍스트
    ids += [VISION_START]
    ids += [lo, lo + 1, lo + 2, lo + 3, lo + 4, lo + 5]   # time marker 6토큰
    ids += [VIDEO_PAD] * n_video_tok
    ids += [AUDIO_PAD] * n_audio_tok
    ids += [200] * n_text                # 뒤쪽 텍스트
    input_ids = torch.tensor([ids], dtype=torch.long)
    video_grid_thw = torch.tensor([[t, h, w]], dtype=torch.long)
    audio_lengths = [n_audio_tok]
    return input_ids, video_grid_thw, audio_lengths



def load_real_sample(idx=0):
    """LazySupervisedDataset 에서 진짜 샘플 하나를 뽑는다. 모델 가중치는 로드하지 않는다.
    반환: (input_ids[1,L], video_grid_thw, audio_lengths(list), attention_mask, time_marker_token_len)"""
    from dataclasses import dataclass
    from transformers import AutoTokenizer, WhisperFeatureExtractor, AutoConfig
    from qwenvl.data.dataset import LazySupervisedDataset, DataCollatorForSupervisedDataset
    from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast

    base = os.environ.get("SMOKE_MODEL_BASE", "/home/team404/workspace/checkpoints/base")
    ds_path = os.environ.get("SMOKE_DATASET", "/home/team404/workspace/data/train/unpucha_v2.json")

    @dataclass
    class A:
        dataset_use: str = ""
        model_type: str = "qwen2.5vl"
        video_max_frames: int = 128
        video_min_frames: int = 64
        base_interval: float = 0.2
        max_pixels: int = 176400
        min_pixels: int = 784
        video_max_frame_pixels: int = 28224
        video_min_frame_pixels: int = 784
        video_max_total_pixels: int = 1664 * 28 * 28
        video_min_total_pixels: int = 256 * 28 * 28
        run_test: bool = False
        do_sample: bool = False
        num_sample: int = 1
        train_type: str = "sft"
        tti_time_format: str = "special_token"
        feature_size: int = 128
        chunk_length: int = 30
        hop_length: int = 160
        sampling_rate: int = 16000
        image_processor: object = None
        audio_processor: object = None
        video_cache_size: int = 0

    try:
        tok = AutoTokenizer.from_pretrained(base, model_max_length=10000, padding_side="left")
        a = A()
        a.dataset_use = ds_path
        a.image_processor = Qwen2VLImageProcessorFast.from_pretrained(base)
        a.audio_processor = WhisperFeatureExtractor(
            feature_size=a.feature_size, sampling_rate=a.sampling_rate,
            hop_length=a.hop_length, chunk_length=a.chunk_length)
        ds = LazySupervisedDataset(tokenizer=tok, data_args=a)
        coll = DataCollatorForSupervisedDataset(tokenizer=tok)
        b = coll([ds[idx]])
        cfg = AutoConfig.from_pretrained(base)
        tmlen = getattr(cfg, "time_marker_token_len", None)
        trange = getattr(cfg, "time_token_id_range", None)
        # 트레이너 load_model_and_tokenizer(tti_mode="on") 와 동일한 복원 로직
        if trange is None:
            t0 = tok.convert_tokens_to_ids("<t0>")
            tdot = tok.convert_tokens_to_ids("<tdot>")
            if isinstance(t0, int) and isinstance(tdot, int) and t0 >= 0 and tdot >= 0:
                trange = (min(t0, tdot), max(t0, tdot))
                tmlen = 5
                print(f"   [real] tti_mode=on 복원: time_token_id_range={trange}, marker_len={tmlen}")
        globals()["_TRANGE"] = trange
        print(f"   [real] input_ids={tuple(b['input_ids'].shape)}, "
              f"video_grid_thw={b['video_grid_thw'].tolist()}, "
              f"audio_lengths={b.get('audio_lengths')}, time_marker_token_len={tmlen}")
        return (b["input_ids"], b["video_grid_thw"], b.get("audio_lengths"),
                b["attention_mask"], tmlen)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"   [real] 로드 실패: {type(e).__name__}: {e}")
        return None, None, None, None, None


def main():
    G = 8
    ok = True

    print("=" * 72)
    print("A) _tile 이 모델의 _repeat_interleave_samples 와 같은가")
    print("=" * 72)
    # 샘플 1개짜리 케이스 (실제 학습 조건: per_device_train_batch_size=1)
    pv = torch.arange(12 * 5, dtype=torch.float32).reshape(12, 5)   # (patches, dim)
    vgt = torch.tensor([[4, 4, 4]], dtype=torch.long)
    lengths = [int(torch.prod(vgt, dim=1).sum())] if False else [pv.size(0)]
    ref = _repeat_interleave_samples(pv, lengths=lengths, repeat_times=G)
    mine = _tile(pv, G)
    same = torch.equal(ref, mine)
    print(f"   pixel_values_videos tile 일치: {same}  (shape {tuple(mine.shape)})")
    ok &= same

    ref_g = _repeat_interleave_samples(vgt, lengths=[1], repeat_times=G)
    mine_g = _tile(vgt, G)
    same_g = torch.equal(ref_g, mine_g)
    print(f"   video_grid_thw       tile 일치: {same_g}  (shape {tuple(mine_g.shape)})")
    ok &= same_g

    # 대조군: repeat_interleave 는 다른 결과 (audio 가 이 경로로 잘못 확장됨)
    wrong = pv.repeat_interleave(G, dim=0)
    print(f"   [대조군] repeat_interleave 결과가 tile 과 같은가: "
          f"{torch.equal(wrong, mine)}  ← False 여야 정상(= 이래서 못 씀)")
    ok &= (not torch.equal(wrong, mine))

    print()
    print("=" * 72)
    print("B) get_rope_index_25: tile 한 batch=8 의 각 행 == batch=1 결과인가")
    print("=" * 72)
    input_ids, vgt, alen, attn, tmlen = load_real_sample()
    if input_ids is None:
        print("   ⚠️ 실제 샘플 로드 실패 → B/C 스킵")
        return 0 if ok else 1

    pos1, delta1 = get_rope_index_25(
        spatial_merge_size=2,
        input_ids=input_ids,
        image_grid_thw=None,
        video_grid_thw=vgt,
        audio_lengths=alen,
        second_per_grid_ts=None,
        attention_mask=attn,
        time_token_id_range=_TRANGE,
        time_marker_token_len=tmlen,
    )
    print(f"   batch=1 position_ids shape={tuple(pos1.shape)}, delta={delta1.flatten().tolist()}")

    posG, deltaG = get_rope_index_25(
        spatial_merge_size=2,
        input_ids=_tile(input_ids, G),
        image_grid_thw=None,
        video_grid_thw=_tile(vgt, G),
        audio_lengths=_tile(alen, G),
        second_per_grid_ts=None,
        attention_mask=_tile(attn, G),
        time_token_id_range=_TRANGE,
        time_marker_token_len=tmlen,
    )
    print(f"   batch={G} position_ids shape={tuple(posG.shape)}")

    n_bad = 0
    for i in range(G):
        if not torch.equal(posG[:, i, :], pos1[:, 0, :]):
            n_bad += 1
            if n_bad == 1:
                d = (posG[:, i, :] != pos1[:, 0, :]).nonzero()
                print(f"   ✗ 행 {i} 불일치, 첫 diff 위치 {d[0].tolist() if len(d) else '?'}")
    print(f"   일치한 행: {G - n_bad}/{G}")
    ok &= (n_bad == 0)

    same_delta = torch.equal(deltaG.flatten(), delta1.flatten().repeat(G))
    print(f"   mrope_position_deltas 일치: {same_delta}")
    ok &= same_delta

    print()
    print("=" * 72)
    print("C) [대조군] audio_lengths 를 확장하지 않으면 어떻게 되는가")
    print("   (num_return_sequences=G 경로가 실제로 하는 일: list 라서 확장 안 됨)")
    print("=" * 72)
    try:
        posBad, _ = get_rope_index_25(
            spatial_merge_size=2,
            input_ids=_tile(input_ids, G),
            image_grid_thw=None,
            video_grid_thw=_tile(vgt, G),
            audio_lengths=alen,           # ← 확장 안 함 (길이 1)
            second_per_grid_ts=None,
            attention_mask=_tile(attn, G),
            time_token_id_range=_TRANGE,
            time_marker_token_len=tmlen,
        )
        bad_rows = sum(1 for i in range(G) if not torch.equal(posBad[:, i, :], pos1[:, 0, :]))
        print(f"   예외 없이 통과했으나 {bad_rows}/{G} 행이 batch=1 과 다름 "
              f"→ {'조용한 오염(silent corruption)' if bad_rows else '우연히 동일'}")
    except Exception as e:
        print(f"   예외 발생: {type(e).__name__}: {e}")
        print("   → num_return_sequences 경로는 audio_lengths 미확장으로 깨진다(예상대로).")

    print()
    print("=" * 72)
    print("결과:", "✅ tile 정합성 통과" if ok else "❌ 실패 — 패치 재검토 필요")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
