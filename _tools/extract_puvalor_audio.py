"""
extract_pu_valor_audio.py

PU-VALOR 합성 비디오에서 오디오를 추출하는 스크립트.
출력: 16kHz mono WAV (av_dataset.py 요구사항)

사용법:
    python extract_pu_valor_audio.py \
        --video_dir  /data0/aix23102/PU-VALOR/videos \
        --audio_dir  /data0/aix23102/PU-VALOR/audios \
        --workers    16
"""

import os
import glob
import argparse
import logging
import subprocess
import traceback
from multiprocessing import Pool
from tqdm import tqdm

# ────────────────────────────────────────────────
# 로깅
# ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("extract_pu_valor_audio.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────
# 전역 worker 초기화
# ────────────────────────────────────────────────
_AUDIO_DIR = ""

def worker_init(audio_dir):
    global _AUDIO_DIR
    _AUDIO_DIR = audio_dir

# ────────────────────────────────────────────────
# 단일 비디오 처리
# ────────────────────────────────────────────────
def extract_audio(video_path):
    try:
        stem      = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = os.path.join(_AUDIO_DIR, f"{stem}.wav")

        # 이미 존재하면 스킵
        if os.path.exists(audio_path):
            return {"status": "skip", "id": stem}

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-ar", "16000",   # 16kHz (av_dataset.py 요구)
            "-ac", "1",        # mono
            "-vn",             # 비디오 스트림 제외
            "-loglevel", "error",
            audio_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return {
                "status": "error",
                "id": stem,
                "stderr": result.stderr[-300:],
            }

        return {"status": "ok", "id": stem}

    except Exception:
        return {
            "status": "exception",
            "id": os.path.basename(video_path),
            "error": traceback.format_exc(),
        }

# ────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir",  required=True)
    parser.add_argument("--audio_dir",  required=True)
    parser.add_argument("--workers",    type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.audio_dir, exist_ok=True)

    # 비디오 파일 목록
    video_paths = glob.glob(os.path.join(args.video_dir, "*.mp4"))
    # ls로는 안 잡히는 숨김파일 포함
    video_paths += glob.glob(os.path.join(args.video_dir, ".*.mp4"))
    video_paths = sorted(set(video_paths))
    logger.info(f"총 비디오 수: {len(video_paths)}")

    results = {"ok": 0, "skip": 0, "error": 0, "exception": 0}
    failed  = []

    with Pool(
        processes=args.workers,
        initializer=worker_init,
        initargs=(args.audio_dir,),
    ) as pool:
        for res in tqdm(
            pool.imap_unordered(extract_audio, video_paths, chunksize=8),
            total=len(video_paths),
            desc="오디오 추출",
        ):
            status = res["status"]
            results[status] = results.get(status, 0) + 1
            if status not in ("ok", "skip"):
                failed.append(res)
                if status == "exception":
                    logger.error(f"[exception] id={res['id']}\n{res.get('error','')}")

    logger.info("=" * 50)
    logger.info(f"완료:   {results['ok']}")
    logger.info(f"스킵:   {results['skip']}")
    logger.info(f"오류:   {results['error']}")
    logger.info(f"예외:   {results['exception']}")
    logger.info("=" * 50)

    # 추출된 오디오 수 확인
    n_audio = len(glob.glob(os.path.join(args.audio_dir, "*.wav")))
    logger.info(f"추출된 WAV 파일 수: {n_audio}")

    if failed:
        import json
        fail_path = os.path.join(args.audio_dir, "failed_audio.json")
        with open(fail_path, "w") as f:
            json.dump(failed, f, indent=2, ensure_ascii=False)
        logger.info(f"실패 목록 저장: {fail_path}")


if __name__ == "__main__":
    main()
