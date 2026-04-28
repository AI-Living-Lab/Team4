"""ChronusOmni inference runner — cdh 표준 output contract 준수.

Usage:
  cd /workspace/jsy/Chronus
  python /workspace/jsy/scripts/inference_chronusomni.py \
      --json_file /workspace/jsy/output/chronusomni_unavqa/unav100_test_chronusomni_hybrid.json \
      --output_dir /workspace/jsy/outputs/base/ChronusOmni/Unav100QA \
      --model_path ./checkpoints

(cwd 가 Chronus/ 여야 함 — config.json 의 `./checkpoints/large-v3.pt` 상대경로 때문)

출력:
  {output_dir}/test_results_rank0.json  — list, test JSON 순서 유지, 각 엔트리에 'pred'
  {output_dir}/inference.log            — stdout/stderr (외부에서 tee 권장)

필드:
  test_results_rank0.json 각 row =
    {"id", "vid", "gt_label", "gt_segments", "duration",
     "question", "pred", "error"}
"""

import os

# env vars (author eval.py 와 동일, import 전 설정 필수)
os.environ["LOWRES_RESIZE"] = "384x32"
os.environ["HIGHRES_BASE"] = "0x32"
os.environ["VIDEO_RESIZE"] = "0x64"
os.environ["VIDEO_MAXRES"] = "448"
os.environ["VIDEO_MINRES"] = "288"
os.environ["MAXRES"] = "1536"
os.environ["MINRES"] = "0"
os.environ["FORCE_NO_DOWNSAMPLE"] = "1"
os.environ["LOAD_VISION_EARLY"] = "1"
os.environ["PAD2STRIDE"] = "1"

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

# Chronus 를 PYTHONPATH 에 추가 (cwd 가 Chronus/ 여야 ./checkpoints 상대경로 먹음)
CHRONUS_ROOT = "/workspace/jsy/Chronus"
if CHRONUS_ROOT not in sys.path:
    sys.path.insert(0, CHRONUS_ROOT)

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
import librosa
import whisper
from tqdm import tqdm

from chronus.conversation import conv_templates, SeparatorStyle
from chronus.model.builder import load_pretrained_model
from chronus.datasets.preprocess import tokenizer_speech_image_token
from chronus.mm_utils import KeywordsStoppingCriteria, process_anyres_video
from chronus.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    DEFAULT_SPEECH_TOKEN,
)


def load_audio_mel(audio_file: str):
    speech_wav, _ = librosa.load(audio_file, sr=16000)
    if speech_wav.ndim > 1:
        speech_wav = speech_wav[:, 0]
    speech_wav = speech_wav.astype(np.float32)
    CHUNK_LIM = 480000
    speechs, speech_wavs = [], []
    if len(speech_wav) <= CHUNK_LIM:
        sw = whisper.pad_or_trim(speech_wav)
        speechs.append(sw)
        speech_wavs.append(torch.from_numpy(sw).unsqueeze(0))
    else:
        for i in range(0, len(speech_wav), CHUNK_LIM):
            chunk = speech_wav[i : i + CHUNK_LIM]
            if len(chunk) < CHUNK_LIM:
                chunk = whisper.pad_or_trim(chunk)
            speechs.append(chunk)
            speech_wavs.append(torch.from_numpy(chunk).unsqueeze(0))
    mels = []
    for chunk in speechs:
        mel = (
            whisper.log_mel_spectrogram(chunk, n_mels=128)
            .permute(1, 0)
            .unsqueeze(0)
        )
        mels.append(mel)
    mels = torch.cat(mels, dim=0)
    speech_wavs_t = torch.cat(speech_wavs, dim=0)
    if mels.shape[0] > 20:
        mels = mels[:20]
        speech_wavs_t = speech_wavs_t[:20]
    speech_length = torch.LongTensor([mels.shape[1]] * mels.shape[0])
    speech_chunks = torch.LongTensor([mels.shape[0]])
    return mels, speech_length, speech_chunks, speech_wavs_t


@torch.inference_mode()
def generate_one(
    sample, model, tokenizer, image_processor, frame_num=64, max_new_tokens=1024
):
    video_path = sample["video"]
    audio_path = sample["audio"]

    # ---- video frames ----
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    if total_frame_num > frame_num:
        frame_idx = np.linspace(
            0, total_frame_num - 1, frame_num, dtype=int
        ).tolist()
    else:
        frame_idx = np.arange(0, total_frame_num, dtype=int).tolist()
    spare = vr.get_batch(frame_idx).asnumpy()
    video = [Image.fromarray(f) for f in spare]
    fps = vr.get_avg_fps()
    video_duration = total_frame_num / fps
    time_interval = video_duration / (min(total_frame_num, frame_num) - 1)
    timestamp = []
    for i in range(len(frame_idx)):
        ttext = "second{" + f"{time_interval * i:.1f}" + "}"
        timestamp.append(
            torch.tensor(tokenizer(ttext)["input_ids"]).cuda()
        )

    # ---- audio mel ----
    speech, speech_length, speech_chunks, speech_wav = load_audio_mel(audio_path)
    speechs = [speech.bfloat16().cuda()]
    speech_lengths = [speech_length.cuda()]
    speech_chunks_l = [speech_chunks.cuda()]
    speech_wavs = [speech_wav.bfloat16().cuda()]

    # ---- prompt ----
    qs = DEFAULT_SPEECH_TOKEN + DEFAULT_IMAGE_TOKEN + "\n" + sample["question"]
    conv = conv_templates["qwen_1_5"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = (
        tokenizer_speech_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        .unsqueeze(0)
        .cuda()
    )

    pad_token_ids = 151643
    attention_mask = input_ids.ne(pad_token_ids).long().cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    # ---- video tensor ----
    video_processed = []
    for frame in video:
        image_processor.do_resize = False
        image_processor.do_center_crop = False
        video_processed.append(
            process_anyres_video(frame, image_processor).unsqueeze(0)
        )
    video_processed = torch.cat(video_processed, dim=0).bfloat16().cuda()
    video_data = ((video_processed, video_processed), (384, 384), "video")

    # ---- generate ----
    output_ids = model.generate(
        inputs=input_ids,
        images=[video_data[0][0]],
        images_highres=[video_data[0][1]],
        modalities=video_data[2],
        speech=speechs,
        speech_lengths=speech_lengths,
        speech_chunks=speech_chunks_l,
        speech_wav=speech_wavs,
        attention_mask=attention_mask,
        use_cache=True,
        stopping_criteria=[stopping],
        do_sample=False,
        temperature=0,
        top_p=None,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        timestamp=[timestamp],
        time_interval=[time_interval],
    )
    out = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if out.endswith(stop_str):
        out = out[: -len(stop_str)].strip()
    return out, video_duration


def vid_of(video_path: str) -> str:
    return Path(video_path).stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_file", required=True, help="Input test JSON (list of {id,video,audio,question,gt_label,gt_segments})")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_path", default="./checkpoints")
    ap.add_argument("--frame_num", type=int, default=64)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--resume", action="store_true", help="Skip ids already in test_results_rank0.json")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "test_results_rank0.json"

    with open(args.json_file) as f:
        samples = json.load(f)

    done_ids = set()
    prev_results = []
    if args.resume and results_path.exists():
        with open(results_path) as f:
            prev_results = json.load(f)
        done_ids = {r["id"] for r in prev_results}
        print(f"[resume] {len(done_ids)} samples already done, skipping")

    print(f"[load] model from {args.model_path}")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None
    )
    model = model.cuda().eval().bfloat16()
    print(f"[load] done. n_samples = {len(samples)}")

    results = list(prev_results)
    t0 = time.time()
    for sample in tqdm(samples, desc="ChronusOmni"):
        if sample["id"] in done_ids:
            continue
        vid = vid_of(sample["video"])
        err = None
        pred = ""
        dur = None
        try:
            pred, dur = generate_one(
                sample,
                model,
                tokenizer,
                image_processor,
                frame_num=args.frame_num,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            traceback.print_exc()

        results.append(
            {
                "id": sample["id"],
                "vid": vid,
                "gt_label": sample.get("gt_label", ""),
                "gt_segments": sample.get("gt_segments", []),
                "duration": dur,
                "question": sample["question"],
                "pred": pred,
                "error": err,
            }
        )

        # periodic save (every 50 samples)
        if len(results) % 50 == 0:
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    elapsed = time.time() - t0
    n_new = len(samples) - len(done_ids)
    print(
        f"[done] wrote {results_path} | n={len(results)} | new={n_new} | "
        f"elapsed={elapsed:.1f}s | {elapsed / max(n_new, 1):.2f} s/sample"
    )


if __name__ == "__main__":
    main()
