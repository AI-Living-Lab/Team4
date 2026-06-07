#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_app.py — video-SALMONN2+ temporal grounding 데모 (VideoMind 스타일, Gradio)

비디오 업로드 + 질문 → 답변 텍스트 + 타임라인에 예측 세그먼트 표시.
gdpo_trainer_clip.py 의 검증된 추론 경로(load_model_and_tokenizer +
LazySupervisedDataset + generate)를 그대로 재사용한다(= 학습 때와 동일 전처리).

실행 (GPU 서버에서):
  pip install gradio
  source paths.env   # SFT_CKPT / BASE_MODEL 등
  DEMO_MODEL_PATH=${SFT_CKPT} DEMO_MODEL_BASE=${BASE_MODEL} DEMO_TTI_MODE=off \
  CUDA_VISIBLE_DEVICES=0 python _tools/GDPO/demo_app.py
  → 콘솔의 https://xxxx.gradio.live 링크(72h 임시 공개) 로 접속

env:
  DEMO_MODEL_PATH  [필수] SFT/RL 체크포인트 (gdpo --model_path 와 동일)
  DEMO_MODEL_BASE  [필수] VS2+ base 모델 (gdpo --model_base 와 동일)
  DEMO_TTI_MODE    off|on (기본 off). 데이터/모델 짝 — 학습 때 쓴 값과 맞출 것.
  DEMO_SHARE       1|0 (기본 1). gradio 공개 링크 생성 여부.
  DEMO_SERVER_NAME 기본 0.0.0.0
  DEMO_SERVER_PORT 기본 7860

⚠️ 이 파일은 GPU/체크포인트가 없는 환경에서 작성됨 → 첫 실행 시 generate 시그니처/
   전처리 한두 군데를 네 환경에서 검증해야 할 수 있음(아래 [VERIFY] 주석 지점).
"""

import os
import sys
import re
import json
import tempfile
from dataclasses import dataclass

import torch

# ── 경로 셋업 (gdpo_trainer_clip 와 동일 패턴) ──
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "video_SALMONN2_plus"))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# 검증된 모델 로더 재사용 (gdpo 와 100% 동일한 모델 로딩 경로)
from gdpo_trainer_clip import load_model_and_tokenizer

from qwenvl.data.dataset import LazySupervisedDataset, DataCollatorForSupervisedDataset
from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
from transformers import WhisperFeatureExtractor

import gradio as gr

# ── gradio_client api_info 버그 우회 ─────────────────────────────────────────
# 일부 gradio 버전에서 bool schema(additionalProperties: true)를 api_info 생성 시
# 처리 못 해 "argument of type 'bool' is not iterable" 로 페이지 로드가 깨짐.
# get_type / _json_schema_to_python_type 가 dict 가 아닌 schema 를 만나면 'Any' 처리.
try:
    import gradio_client.utils as _gcu

    _orig_get_type = _gcu.get_type
    def _safe_get_type(schema):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_get_type(schema)
    _gcu.get_type = _safe_get_type

    _orig_js = _gcu._json_schema_to_python_type
    def _safe_js(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        return _orig_js(schema, defs)
    _gcu._json_schema_to_python_type = _safe_js
except Exception as _e:
    print(f"[DEMO] gradio_client api_info patch skip: {_e}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 설정 (env)
# ============================================================
MODEL_PATH = os.environ.get("DEMO_MODEL_PATH")
MODEL_BASE = os.environ.get("DEMO_MODEL_BASE")
TTI_MODE = os.environ.get("DEMO_TTI_MODE", "off")
SHARE = os.environ.get("DEMO_SHARE", "1") not in ("0", "false", "False")
SERVER_NAME = os.environ.get("DEMO_SERVER_NAME", "0.0.0.0")
SERVER_PORT = int(os.environ.get("DEMO_SERVER_PORT", "7860"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if not MODEL_PATH or not MODEL_BASE:
    raise SystemExit("[DEMO] DEMO_MODEL_PATH / DEMO_MODEL_BASE 환경변수 필요")


# ============================================================
# 시간 토큰 디코딩 (eval/eval_miou_multiseg.py 와 동일)
# ============================================================
def decode_vtg_time(token_str, max_time=9999.9):
    has_dot = "<tdot>" in token_str
    if has_dot:
        parts = token_str.split("<tdot>")
        int_part = re.findall(r"<t(\d)>", parts[0])
        dec_part = re.findall(r"<t(\d)>", parts[1]) if len(parts) > 1 else []
    else:
        int_part = re.findall(r"<t(\d)>", token_str)
        dec_part = []
    if not int_part:
        return None
    integer_part = int("".join(int_part))
    decimal_part = int(dec_part[0]) if dec_part else 0
    t = integer_part + decimal_part / 10.0
    return min(t, max_time)


def parse_multi_segments(raw, max_time=9999.9):
    """'From X to Y. From X to Y.' → [[start, end], ...] (초)."""
    segments = []
    pattern = r"[Ff]rom\s+((?:<t\d>)+(?:<tdot>(?:<t\d>)+)?)\s+to\s+((?:<t\d>)+(?:<tdot>(?:<t\d>)+)?)"
    for m in re.finditer(pattern, raw):
        start = decode_vtg_time(m.group(1), max_time)
        end = decode_vtg_time(m.group(2), max_time)
        if start is not None and end is not None:
            if end <= start:
                end = min(start + 0.1, max_time)
            segments.append([start, end])
    return segments


def clean_text(raw):
    """special 토큰 정리 (시간 토큰은 사람이 읽기 쉽게 초로 치환된 raw 는 따로 표시)."""
    txt = raw
    for tok in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        txt = txt.replace(tok, "")
    return txt.strip()


# ============================================================
# 데이터 전처리 args (gdpo_trainer_clip.py main() 의 GDPODataArgs 복제)
#   ※ 학습 때와 동일한 전처리를 쓰기 위해 값도 동일하게 유지.
# ============================================================
@dataclass
class DemoDataArgs:
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


# ============================================================
# 모델 로드 (1회)
# ============================================================
print(f"[DEMO] loading model (tti_mode={TTI_MODE}, device={DEVICE})")
model, tokenizer = load_model_and_tokenizer(MODEL_PATH, MODEL_BASE, tti_mode=TTI_MODE)
model = model.to(DEVICE).eval()
# PEFT wrapper 면 한 겹 벗긴 base 로 generate (gdpo 와 동일 — LoRA 는 inline 적용)
GEN_MODEL = model.get_base_model() if hasattr(model, "get_base_model") else model

data_args = DemoDataArgs()
data_args.tti_time_format = "special_token" if TTI_MODE == "on" else "off"
data_args.image_processor = Qwen2VLImageProcessorFast.from_pretrained(MODEL_BASE)
data_args.audio_processor = WhisperFeatureExtractor(
    feature_size=data_args.feature_size,
    sampling_rate=data_args.sampling_rate,
    hop_length=data_args.hop_length,
    chunk_length=data_args.chunk_length,
)
collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
_VIDEO_TOKEN_ID = 151656
print("[DEMO] model ready")


# ============================================================
# 추론
# ============================================================
@torch.no_grad()
def infer(video_path, audio_path, question, max_new_tokens, do_sample, temperature):
    if not video_path:
        return "[비디오를 업로드하세요]", "", []
    if not question or not question.strip():
        question = "When does the described event happen? Answer with time segments."

    # 1) 단일 샘플 JSON 구성 (LazySupervisedDataset 스키마)
    sample = {
        "video": video_path,
        "conversations": [
            {"from": "human", "value": "<image>\n" + question.strip()},
            {"from": "gpt", "value": ""},
        ],
    }
    if audio_path:
        sample["audio"] = audio_path

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump([sample], f, ensure_ascii=False)
        tmp_json = f.name

    try:
        data_args.dataset_use = tmp_json
        ds = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
        item = ds[0]
        batch = collator([item])
    finally:
        try:
            os.remove(tmp_json)
        except OSError:
            pass

    device = DEVICE
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # 2) 프롬프트만 추출 (gdpo compute_loss 와 동일: 빈 gpt 답변 → assistant 프롬프트까지로 컷)
    labels = batch.get("labels", None)
    if labels is not None:
        labels = labels.to(device)
        ans = (labels[0] != -100).nonzero(as_tuple=True)[0]
        if len(ans) > 0:
            cut = ans[0].item()
            input_ids = input_ids[:, :cut]
            attention_mask = attention_mask[:, :cut]

    # 3) 멀티모달 kwargs (gdpo gen_kwargs 와 동일 키)
    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=int(max_new_tokens),
    )
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=float(temperature), top_p=1.0, top_k=50,
                          repetition_penalty=1.0)
    else:
        gen_kwargs.update(do_sample=False)

    pvv = batch.get("pixel_values_videos", None)
    if pvv is not None:
        gen_kwargs["pixel_values_videos"] = pvv.to(device=device, dtype=torch.bfloat16)
    vgt = batch.get("video_grid_thw", None)
    if vgt is not None:
        gen_kwargs["video_grid_thw"] = vgt.to(device)
    af = batch.get("audio_feature", None)
    if af is not None:
        gen_kwargs["audio_feature"] = af.to(device=device, dtype=torch.bfloat16)
    al = batch.get("audio_lengths", None)
    if al is not None:
        gen_kwargs["audio_lengths"] = al
    spg = batch.get("second_per_grid_ts", None)
    if spg is not None:
        gen_kwargs["second_per_grid_ts"] = spg.to(device) if torch.is_tensor(spg) else spg

    # [VERIFY] VS2+ generate 시그니처 — gdpo 는 raw_model.generate(**gen_kwargs) 로 동작 확인됨.
    out = GEN_MODEL.generate(**gen_kwargs)
    gen_ids = out[0, input_ids.size(1):]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=False)
    return raw, None, None  # 후처리는 호출부에서


def render_timeline(segments, max_time):
    fig, ax = plt.subplots(figsize=(8, 1.6))
    ax.set_xlim(0, max(max_time, 0.1))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    ax.hlines(0.5, 0, max_time, color="#cccccc", lw=8, zorder=1)
    colors = plt.cm.tab10.colors
    for i, (s, e) in enumerate(segments):
        ax.hlines(0.5, s, e, color=colors[i % len(colors)], lw=14, zorder=2)
        ax.text((s + e) / 2, 0.78, f"{s:.1f}–{e:.1f}s", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"{len(segments)} predicted segment(s)")
    fig.tight_layout()
    return fig


def run(video_path, audio_path, question, max_time, max_new_tokens, do_sample, temperature):
    raw, _, _ = infer(video_path, audio_path, question, max_new_tokens, do_sample, temperature)
    segments = parse_multi_segments(raw, max_time=float(max_time))
    answer = clean_text(raw)
    seg_str = "\n".join(f"  • {s:.1f}s – {e:.1f}s" for s, e in segments) or "  (유효 세그먼트 없음)"
    fig = render_timeline(segments, float(max_time))
    return answer, seg_str, fig


# ============================================================
# Gradio UI
# ============================================================
with gr.Blocks(title="video-SALMONN2+ Temporal Grounding") as demo:
    gr.Markdown("## 🎬 video-SALMONN2+ — Temporal Grounding Demo\n"
                "비디오 업로드 + 질문 → 답변 + 타임라인에 예측 시간 구간 표시")
    with gr.Row():
        with gr.Column(scale=1):
            video = gr.Video(label="Video", sources=["upload"])
            audio = gr.Audio(label="Audio (선택, 미지정 시 비디오 오디오 사용 안 함)",
                             sources=["upload"], type="filepath")
            question = gr.Textbox(label="Question", lines=2,
                                  placeholder="When does the dog bark? ...")
            max_time = gr.Slider(5, 600, value=60, step=1, label="Video duration (타임라인 축, 초)")
            with gr.Accordion("Generation 설정", open=False):
                max_new = gr.Slider(64, 1024, value=512, step=64, label="max_new_tokens")
                do_sample = gr.Checkbox(value=False, label="sampling (off=greedy)")
                temperature = gr.Slider(0.1, 1.5, value=1.0, step=0.1, label="temperature (sampling 시)")
            btn = gr.Button("Run", variant="primary")
        with gr.Column(scale=1):
            answer = gr.Textbox(label="Answer", lines=4)
            seg_box = gr.Textbox(label="Predicted segments", lines=4)
            timeline = gr.Plot(label="Timeline")

    btn.click(run,
              inputs=[video, audio, question, max_time, max_new, do_sample, temperature],
              outputs=[answer, seg_box, timeline])


if __name__ == "__main__":
    demo.queue().launch(share=SHARE, server_name=SERVER_NAME, server_port=SERVER_PORT)
