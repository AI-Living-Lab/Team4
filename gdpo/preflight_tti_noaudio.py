#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preflight_tti_noaudio.py — 학습 시작 전 검증.

  오디오를 빼면 마커 인터리빙이 dataset.py 의 audio 분기(236-263)가 아니라
  video-only 분기(205-235)를 탄다. 그 경로에서도 TTI 타임마커가 실제로
  삽입되는지 **모델 로드 없이** 데이터셋 레벨에서 확인한다.
  (tokenizer + image_processor 만 필요 — 마커 삽입은 모델과 무관.)

  같은 인덱스 샘플을 audio 판 / no-audio 판으로 각각 만들어 비교:
    - time marker 토큰(<t0>..<tdot>) 개수  → no-audio 에서도 > 0 이어야 한다
    - audio_pad(151665) 개수               → no-audio 는 반드시 0
    - video_pad(151656) 개수               → 두 판이 동일해야 한다 (불변식)
"""
import os
import sys

REPO = "/home/team404/workspace/Team4"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "video_SALMONN2_plus"))

import torch  # noqa: E402
import transformers  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from qwenvl.data.dataset import LazySupervisedDataset  # noqa: E402
from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast  # noqa: E402
from transformers import WhisperFeatureExtractor  # noqa: E402

TOK_SRC = os.environ.get(   # 토크나이저/이미지프로세서 출처 (= 학습에 쓸 모델)
    "PREFLIGHT_TOK_SRC",
    "/home/team404/workspace/checkpoints/base/salmonn2p_7b_unav_v8")
AUDIO_JSON = "/home/team404/workspace/data/train/unpucha_v2.json"
NOAUDIO_JSON = "/home/team404/workspace/data/train/unpucha_v2_noaudio.json"

VIDEO_PAD, AUDIO_PAD = 151656, 151665


@dataclass
class DA:
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


def build(path, tok, imgproc, audproc):
    da = DA()
    da.dataset_use = path
    da.image_processor = imgproc
    da.audio_processor = audproc
    return LazySupervisedDataset(tokenizer=tok, data_args=da)


def report(name, ds, idx, id_range):
    d = ds._get_item(idx)           # __getitem__ 의 except 폴백을 우회 — 진짜 예외를 본다
    ids = d["input_ids"]
    ids = ids[0] if ids.dim() > 1 else ids
    lo, hi = id_range
    n_mark = int(((ids >= lo) & (ids <= hi)).sum())
    n_vid = int((ids == VIDEO_PAD).sum())
    n_aud = int((ids == AUDIO_PAD).sum())
    has_af = d.get("audio_feature", None) is not None
    print(f"  [{name}] len={ids.numel():6d}  time_marker={n_mark:4d}  "
          f"video_pad={n_vid:6d}  audio_pad={n_aud:6d}  audio_feature={has_af}")
    return dict(n_mark=n_mark, n_vid=n_vid, n_aud=n_aud, has_af=has_af)


def main():
    tok = transformers.AutoTokenizer.from_pretrained(TOK_SRC, use_fast=True)
    imgproc = Qwen2VLImageProcessorFast.from_pretrained(TOK_SRC)
    audproc = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000,
                                      hop_length=160, chunk_length=30)

    ds_a = build(AUDIO_JSON, tok, imgproc, audproc)
    ds_n = build(NOAUDIO_JSON, tok, imgproc, audproc)
    assert len(ds_a) == len(ds_n), "두 데이터셋 길이가 다르다"

    t0, tdot = tok.convert_tokens_to_ids("<t0>"), tok.convert_tokens_to_ids("<tdot>")
    id_range = (min(t0, tdot), max(t0, tdot))
    print(f"time_token_id_range={id_range}  (dataset 이 tokenizer 에서 유도하는 값)\n")

    # ⚠️ dataset.py:489 이 random.shuffle(list_data_dict) 를 **시드 없이** 호출한다.
    #    → 두 데이터셋이 각각 다르게 섞이므로 같은 idx 가 같은 샘플이 아니다.
    #    비디오 경로로 매칭해서 동일 샘플끼리 비교한다.
    #    ⚠️ 비디오 경로만으로는 부족하다 — 8,310개 비디오에 10,358 샘플이라
    #       한 비디오에 이벤트가 최대 5개까지 붙는다. conversations 까지 키에 넣어야
    #       같은 샘플끼리 비교된다. (이걸 놓치면 GT 가 달라 마커 수가 안 맞는다.)
    import json as _json

    def _key(x):
        return (x["video"], _json.dumps(x["conversations"], sort_keys=True))

    idx_a = {_key(x): i for i, x in enumerate(ds_a.list_data_dict)}
    idx_n = {_key(x): i for i, x in enumerate(ds_n.list_data_dict)}
    common = [k for k in idx_a if k in idx_n]
    picks = []
    for tag in ("unav", "charades"):
        picks += [(tag, k) for k in common if tag in k[0]][:3]   # 태그당 3개씩 본다

    ok = True
    for tag, vid in picks:
        was = ds_a.list_data_dict[idx_a[vid]].get("use_audio")
        print(f"--- {tag}  (원래 use_audio={was})  {vid[0].split('/')[-1]} ---")
        a = report("audio   ", ds_a, idx_a[vid], id_range)
        n = report("no-audio", ds_n, idx_n[vid], id_range)
        checks = [
            ("no-audio 에 타임마커가 삽입됨", n["n_mark"] > 0),
            ("no-audio 에 audio_pad 없음", n["n_aud"] == 0),
            ("no-audio 에 audio_feature 없음", not n["has_af"]),
            ("video_pad 개수 불변", a["n_vid"] == n["n_vid"]),
            ("타임마커 개수 불변", a["n_mark"] == n["n_mark"]),
        ]
        for msg, good in checks:
            print(f"     {'PASS' if good else 'FAIL'}  {msg}")
            ok &= good
        print()

    # 실물 확인 — vision_start 직후 40토큰 디코드 (학습 로그의 ⑤ 와 같은 뷰)
    d = ds_n._get_item(idx_n[picks[0][1]])
    row = (d["input_ids"][0] if d["input_ids"].dim() > 1 else d["input_ids"]).tolist()
    vs = tok.convert_tokens_to_ids("<|vision_start|>")
    if vs in row:
        p = row.index(vs)
        print("no-audio prompt[vision_start:+40] =")
        print("  " + repr(tok.decode(row[p:p + 40], skip_special_tokens=False)))

    print("\n==== " + ("ALL PASS — no-audio 경로에서 타임마커 정상 삽입" if ok
                       else "FAIL — 학습 시작 금지") + " ====")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
