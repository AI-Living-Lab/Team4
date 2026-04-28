"""
Step 0: Extract CLIP (video) + CLAP (audio) features.

Supports two modes:
  - wav_dir provided: reads wav directly (no ffmpeg needed)
  - wav_dir not provided: converts mp4 → wav first

Usage (with existing wav):
    python extract_features.py \
        --video_dir /workspace/datasets/unav_100/videos \
        --wav_dir /workspace/datasets/unav_100/audio \
        --output_video_dir data/unav100/features/video_clip \
        --output_audio_dir data/unav100/features/audio_clap \
        --clip_path checkpoints/clip/ViT-L-14.pt \
        --clap_path checkpoints/clap/630k-fusion-best.pt \
        --annotation data/unav100_annotations.json
"""

import os
import json
import argparse
import subprocess
import tempfile

import torch
import numpy as np
import librosa
import clip
import laion_clap
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from avicuna.mm_utils import VideoExtractor


# ================================================================== #
#  Args
# ================================================================== #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract CLIP + CLAP features from UnAV-100"
    )
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing mp4 files")
    parser.add_argument("--wav_dir", type=str, default=None,
                        help="Directory containing wav files (if already extracted)")
    parser.add_argument("--output_video_dir", type=str, required=True,
                        help="Output dir for CLIP video features (.npy)")
    parser.add_argument("--output_audio_dir", type=str, required=True,
                        help="Output dir for CLAP audio features (.npy)")
    parser.add_argument("--annotation", type=str, required=True,
                        help="unav100_annotations.json")
    parser.add_argument("--clip_path", type=str,
                        default="checkpoints/clip/ViT-L-14.pt")
    parser.add_argument("--clap_path", type=str,
                        default="checkpoints/clap/630k-fusion-best.pt")
    parser.add_argument("--subset", type=str, default="test",
                        help="test / train / all")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip if output npy already exists")
    return parser.parse_args()


# ================================================================== #
#  Video IDs from annotation
# ================================================================== #

def get_video_ids(annotation_path: str, subset: str):
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    db = data["database"]
    if subset == "all":
        return list(db.keys())
    return [vid for vid, info in db.items() if info.get("subset") == subset]


# ================================================================== #
#  CLIP video feature extraction (same as get_clip.py)
# ================================================================== #

def extract_clip_features(video_path, clip_model, video_loader, transform, device):
    """mp4 → 100 frames → CLIP encode → (100, 768)"""
    _, images = video_loader.extract({"id": None, "video": video_path})
    images = transform(images / 255.0)
    images = images.to(torch.float16)

    with torch.no_grad():
        features = clip_model.encode_image(images.to(device))

    return features.cpu().numpy()


# ================================================================== #
#  CLAP audio feature extraction (same as test_clap.py)
# ================================================================== #

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

def extract_clap_features(wav_path, clap_model, sr=48000, stride=4):
    """wav → 10s segments with stride → CLAP encode → (M, 512)"""
    audio_data, _ = librosa.load(wav_path, sr=sr)

    segment_length = 10 * sr
    stride_length = stride * sr
    remaining_samples = len(audio_data) % segment_length

    if 0 < remaining_samples <= sr:
        audio_data = audio_data[:-(remaining_samples)]
    elif remaining_samples > sr:
        padding_length = segment_length - remaining_samples
        audio_data = np.pad(audio_data, (0, padding_length), mode='constant')

    num_segments = (len(audio_data) - segment_length) // stride_length + 1
    num_segments = 1 if num_segments < 1 else num_segments

    audio_segments = np.array([
        audio_data[i * stride_length : i * stride_length + segment_length]
        for i in range(num_segments)
    ])
    audio_segments = torch.from_numpy(
        int16_to_float32(float32_to_int16(audio_segments))
    ).float()

    audio_embed = clap_model.get_audio_embedding_from_data(
        x=audio_segments, use_tensor=True
    )
    return audio_embed.detach().cpu().numpy()


# ================================================================== #
#  mp4 → wav (only if wav_dir not provided)
# ================================================================== #

def mp4_to_wav(mp4_path, wav_path):
    command = [
        "ffmpeg", "-y", "-i", mp4_path,
        "-acodec", "pcm_s16le", "-ar", "44100",
        wav_path,
    ]
    subprocess.run(
        command, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


# ================================================================== #
#  Main
# ================================================================== #

def main():
    args = parse_args()
    os.makedirs(args.output_video_dir, exist_ok=True)
    os.makedirs(args.output_audio_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Video IDs ----
    video_ids = get_video_ids(args.annotation, args.subset)
    print(f"Target videos ({args.subset}): {len(video_ids)}")

    # ---- CLIP (video) ----
    print("Loading CLIP ViT-L-14...")
    clip_model, _ = clip.load(args.clip_path, device=device)
    clip_model.eval()

    video_loader = VideoExtractor(N=100)

    transform = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    # ---- CLAP (audio) ----
    print("Loading CLAP 630k-fusion...")
    clap_model = laion_clap.CLAP_Module(enable_fusion=True)
    if os.path.exists(args.clap_path):
        clap_model.load_ckpt(args.clap_path)
    else:
        print(f"  WARNING: {args.clap_path} not found, using default checkpoint")
        clap_model.load_ckpt()
    clap_model = clap_model.to(device)
    clap_model.eval()

    # ---- Process ----
    success = 0
    skipped = 0
    errors = []

    for video_id in tqdm(video_ids, desc="Extracting features"):
        mp4_path = os.path.join(args.video_dir, f"{video_id}.mp4")
        video_out = os.path.join(args.output_video_dir, f"{video_id}.npy")
        audio_out = os.path.join(args.output_audio_dir, f"{video_id}.npy")

        if not os.path.exists(mp4_path):
            errors.append((video_id, "mp4 not found"))
            continue

        if args.skip_existing and os.path.exists(video_out) and os.path.exists(audio_out):
            skipped += 1
            continue

        try:
            # --- Video: mp4 → CLIP features ---
            v_feat = extract_clip_features(
                mp4_path, clip_model, video_loader, transform, device
            )
            np.save(video_out, v_feat)

            # --- Audio: wav → CLAP features ---
            if args.wav_dir:
                # wav already exists
                wav_path = os.path.join(args.wav_dir, f"{video_id}.wav")
                if not os.path.exists(wav_path):
                    errors.append((video_id, "wav not found"))
                    continue
                a_feat = extract_clap_features(wav_path, clap_model, sr=48000, stride=4)
            else:
                # convert mp4 → wav → CLAP
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_wav_path = tmp.name
                mp4_to_wav(mp4_path, tmp_wav_path)
                a_feat = extract_clap_features(tmp_wav_path, clap_model, sr=48000, stride=4)
                os.remove(tmp_wav_path)

            np.save(audio_out, a_feat)
            success += 1

        except Exception as e:
            errors.append((video_id, str(e)))
            continue

    # ---- Summary ----
    print(f"\n{'='*50}")
    print(f"  Done!")
    print(f"  Success: {success}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Errors: {len(errors)}")
    if errors[:10]:
        print(f"  First errors:")
        for vid, err in errors[:10]:
            print(f"    {vid}: {err}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
