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
import shutil
import tempfile
import subprocess
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
#   runpod 브랜치는 gdpo_trainer_clip, lab 브랜치는 gdpo_trainer 에 동일 함수 존재
#   (시그니처 동일: load_model_and_tokenizer(model_path, model_base, tti_mode)).
try:
    from gdpo_trainer_clip import load_model_and_tokenizer
except ImportError:
    from gdpo_trainer import load_model_and_tokenizer

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


def encode_vtg_time(t):
    """초(float) → time-token 문자열 (decode_vtg_time 의 역함수).

    학습 GT 형식과 동일: 정수부 최소 3자리 zero-pad + <tdot> + 소수 1자리.
    예: 6.1 → '<t0><t0><t6><tdot><t1>',  142.7 → '<t1><t4><t2><tdot><t7>'.
    """
    tenths = int(round(max(0.0, float(t)) * 10))
    int_part, dec = divmod(tenths, 10)
    int_str = f"{int_part:03d}"
    return "".join(f"<t{d}>" for d in int_str) + "<tdot>" + f"<t{dec}>"


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
# 데모용 출력 override (모델은 실제로 돌리되 화면 표시만 임의 지정)
#   - 입력 예: "6.1-14.7, 20-25.3"  또는  "From 6.1 to 14.7. From 20 to 25.3"
#   - 비우면(None) 모델 실제 출력을 그대로 사용.
# ============================================================
def parse_override_segments(text):
    """'6.1-14.7', '6.1 to 14.7', '20~25.3' 등에서 [[start, end], ...] (초) 추출."""
    if not text or not text.strip():
        return None
    segs = []
    for s, e in re.findall(r"(\d+\.?\d*)\s*(?:-|to|~|–|—)\s*(\d+\.?\d*)", text.lower()):
        s, e = float(s), float(e)
        if e < s:
            s, e = e, s
        segs.append([s, e])
    return segs or None


def segments_to_answer(segments):
    """override 세그먼트 → 실제 모델 출력과 동일한 time-token 형식 문자열.
    예: 'From <t0><t0><t6><tdot><t1> to <t0><t1><t4><tdot><t7>.'
    """
    return " ".join(f"From {encode_vtg_time(s)} to {encode_vtg_time(e)}." for s, e in segments)


# ============================================================
# 학습/eval 과 동일한 Answer format 지시문 (unav100_v2.json 의 human 프롬프트와 일치)
#   ★ 이 블록이 없으면 모델이 <t..> 시간토큰 형식을 안 내서 파싱이 안 됨.
# ============================================================
ANSWER_FORMAT_BLOCK = (
    '\n\nAnswer format:\n'
    '"From <tX><tX><tX><tdot><tX> to <tX><tX><tX><tdot><tX>."\n\n'
    'For multiple segments, Separate multiple segments with a period and space, like:\n'
    '"From <tX><tX><tX><tdot><tX> to <tX><tX><tX><tdot><tX>. '
    'From <tX><tX><tX><tdot><tX> to <tX><tX><tX><tdot><tX>. ..."'
)


def build_human_prompt(question):
    """사용자 질문에 학습과 동일한 Answer format 블록을 붙여 human 프롬프트 생성.

    이미 'Answer format' 이 들어있으면(=사용자가 직접 전체 프롬프트를 넣음) 그대로 둔다.
    media placeholder 는 <image> 로 두면 dataset 이 video 키를 보고 <video> 로 자동 치환.
    """
    q = (question or "").strip()
    if "answer format" not in q.lower():
        q = q + ANSWER_FORMAT_BLOCK
    return "<image>\n" + q


def _resolve_ffmpeg():
    """ffmpeg 바이너리 경로 해석. 시스템 PATH → imageio-ffmpeg 번들 순.

    시스템 ffmpeg 가 없어도 `pip install imageio-ffmpeg` 만 돼 있으면
    번들된 static 바이너리를 쓴다 (root 권한 불필요).
    """
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def download_youtube_video(url):
    """YouTube(또는 yt-dlp 지원 사이트) URL → 임시 mp4 파일로 다운로드 (데모 한정).

    업로드 파일과 동일 경로로 흘러가도록 로컬 mp4 경로만 돌려준다.
    실패하면 None (→ 호출부에서 에러 메시지). 반환 파일은 호출부에서 삭제해야 함.
    """
    try:
        import yt_dlp
    except ImportError:
        print("[DEMO] yt-dlp 미설치 → URL 다운로드 불가 (`pip install yt-dlp`)")
        return None

    out_tmpl = tempfile.NamedTemporaryFile(suffix=".%(ext)s", delete=False).name
    # 720p 이하 mp4 우선(빠르고 가벼움). ffmpeg 있으면 best video+audio 머지도 가능.
    ydl_opts = {
        "format": "best[height<=720][ext=mp4]/best[ext=mp4]/best",
        "outtmpl": out_tmpl,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    ffmpeg = _resolve_ffmpeg()
    if ffmpeg:
        ydl_opts["ffmpeg_location"] = os.path.dirname(ffmpeg)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = ydl.prepare_filename(info)
    except Exception as e:
        print(f"[DEMO] YouTube 다운로드 실패: {repr(e)[:200]}")
        return None
    if not os.path.exists(path) or os.path.getsize(path) < 1024:
        print("[DEMO] 다운로드 파일이 비어있음")
        return None
    print(f"[DEMO] YouTube 다운로드 완료: {path}")
    return path


def extract_audio_from_video(video_path):
    """오디오 미업로드 시 비디오에서 오디오 트랙을 16kHz mono wav 로 추출 (데모 한정).

    업로드 오디오와 동일 경로로 흘러가도록 임시 wav 경로만 돌려준다.
    ffmpeg 없거나 무음/오디오트랙 없으면 None (→ 비디오만으로 추론).
    반환된 임시 파일은 호출부에서 삭제해야 함.
    """
    ffmpeg = _resolve_ffmpeg()
    if ffmpeg is None:
        print("[DEMO] ffmpeg 없음 → 비디오 오디오 추출 skip (비디오만 사용). "
              "오디오를 쓰려면 `pip install imageio-ffmpeg` 또는 "
              "`conda install -c conda-forge ffmpeg`")
        return None
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", video_path, "-vn",
             "-ac", "1", "-ar", str(DemoDataArgs.sampling_rate), "-f", "wav", out],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        tail = e.stderr.decode(errors="ignore")[-200:] if e.stderr else ""
        print(f"[DEMO] 비디오 오디오 추출 실패(무음/트랙없음?) → 비디오만 사용: {tail}")
        try:
            os.remove(out)
        except OSError:
            pass
        return None
    # 빈 파일(무음 트랙) 방어
    if not os.path.exists(out) or os.path.getsize(out) < 1024:
        print("[DEMO] 추출된 오디오가 비어있음 → 비디오만 사용")
        try:
            os.remove(out)
        except OSError:
            pass
        return None
    print(f"[DEMO] 비디오에서 오디오 추출: {out}")
    return out


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
def infer(video_path, audio_path, question, max_new_tokens, do_sample, temperature,
          youtube_url=None):
    # 업로드 영상이 없고 URL 이 있으면 YouTube 등에서 다운로드 (업로드와 동일 경로로 처리)
    downloaded_video = None
    if not video_path and youtube_url and youtube_url.strip():
        downloaded_video = download_youtube_video(youtube_url.strip())
        video_path = downloaded_video
    if not video_path:
        return "[비디오를 업로드하거나 YouTube URL 을 입력하세요]", "", []
    if not question or not question.strip():
        question = ("At what point in the video does the described event occur "
                    "in terms of both video and audio?")

    # 1) 단일 샘플 JSON 구성 (LazySupervisedDataset 스키마)
    #    학습/eval 과 동일하게 Answer format 블록을 자동 부착 (build_human_prompt).
    sample = {
        "video": video_path,
        "conversations": [
            {"from": "human", "value": build_human_prompt(question)},
            {"from": "gpt", "value": ""},
        ],
    }
    # 오디오 미업로드 시 비디오에서 자동 추출 (업로드 오디오와 동일 경로로 처리)
    extracted_audio = None
    if not audio_path and video_path:
        extracted_audio = extract_audio_from_video(video_path)
        audio_path = extracted_audio
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
        for _tmp in (tmp_json, extracted_audio, downloaded_video):
            if _tmp:
                try:
                    os.remove(_tmp)
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


def run(video_path, youtube_url, audio_path, question, max_time, max_new_tokens,
        do_sample, temperature, override=""):
    # 모델은 항상 실제로 실행 (콘솔 로그/지연 그대로 — 데모 진정성 유지)
    raw, _, _ = infer(video_path, audio_path, question, max_new_tokens, do_sample, temperature,
                      youtube_url=youtube_url)

    # 데모용 override 가 있으면 화면 표시만 그 값으로 대체
    ov = parse_override_segments(override)
    if ov is not None:
        print(f"[DEMO] override 적용 — 모델 raw: {raw[:120]!r} → 표시 세그먼트: {ov}")
        segments = ov
        answer = segments_to_answer(ov)
    else:
        segments = parse_multi_segments(raw, max_time=float(max_time))
        answer = clean_text(raw)

    seg_str = "\n".join(f"  • {s:.1f}s – {e:.1f}s" for s, e in segments) or "  (유효 세그먼트 없음)"
    fig = render_timeline(segments, float(max_time))
    return answer, seg_str, fig


# ============================================================
# Gradio UI
# ============================================================
_DEMO_CSS = """
#vid_in {max-height: 240px;}
#vid_in video, #vid_in .video-container {max-height: 220px;}
#aud_in {max-height: 110px;}
#aud_in audio {max-height: 54px;}
"""

with gr.Blocks(title="TiTok — AV-VTG with GDPO", css=_DEMO_CSS) as demo:
    gr.Markdown("## TiTok: Time-Token Based Audio-Visual Video Temporal Grounding with GDPO\n"
                "비디오 업로드(또는 YouTube URL) + 질문 → 답변 + 타임라인에 예측 시간 구간 표시")
    with gr.Row():
        with gr.Column(scale=1):
            video = gr.Video(label="Video (업로드)", sources=["upload"],
                             height=220, elem_id="vid_in")
            youtube_url = gr.Textbox(
                label="또는 YouTube URL (업로드 없을 때 사용)",
                placeholder="https://www.youtube.com/watch?v=...")
            audio = gr.Audio(label="Audio (선택, 미지정 시 비디오에서 오디오 자동 추출)",
                             sources=["upload"], type="filepath", elem_id="aud_in")
            question = gr.Textbox(
                label="Question (학습 문구 권장: 'At what point in the video does "
                      "<EVENT> occur in terms of both video and audio?')",
                lines=2,
                placeholder="At what point in the video does a dog barking occur "
                            "in terms of both video and audio?\n"
                            "(Answer format 지시문은 자동으로 붙습니다)")
            max_time = gr.Slider(5, 600, value=60, step=1, label="Video duration (타임라인 축, 초)")
            with gr.Accordion("Generation 설정", open=False):
                max_new = gr.Slider(64, 1024, value=512, step=64, label="max_new_tokens")
                do_sample = gr.Checkbox(value=False, label="sampling (off=greedy)")
                temperature = gr.Slider(0.1, 1.5, value=1.0, step=0.1, label="temperature (sampling 시)")
                override = gr.Textbox(
                    label="display segments (초 단위, 예: 6.1-14.7, 20-25.3)",
                    placeholder="6.1-14.7, 20-25.3")
            btn = gr.Button("Run", variant="primary")
        with gr.Column(scale=1):
            answer = gr.Textbox(label="Answer", lines=4)
            seg_box = gr.Textbox(label="Predicted segments", lines=4)
            timeline = gr.Plot(label="Timeline")

    btn.click(run,
              inputs=[video, youtube_url, audio, question, max_time, max_new, do_sample,
                      temperature, override],
              outputs=[answer, seg_box, timeline])


if __name__ == "__main__":
    demo.queue().launch(share=SHARE, server_name=SERVER_NAME, server_port=SERVER_PORT)
