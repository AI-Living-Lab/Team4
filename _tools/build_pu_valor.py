"""
build_pu_valor.py

stage3.json의 pseudo-valor 샘플과 VALOR-32K 원본 클립을 이용해
PU-VALOR pseudo-untrimmed 비디오를 합성하고 학습용 JSON을 생성하는 스크립트.

사용법:
    python build_pu_valor.py \
        --stage3_json  /home/aix23102/audiolm/vS2_eunji/data/stage3.json \
        --valor_dir    /data0/aix23102/VALOR-32K/raid/datasets/audioset/valor_videos \
        --output_dir   /data0/aix23102/PU-VALOR \
        --workers      8 \
        [--dry_run]    # 실제 ffmpeg 실행 없이 매핑만 검증할 때
"""

import os
import re
import json
import glob
import argparse
import logging
import subprocess
import traceback
from multiprocessing import Pool
from tqdm import tqdm

# ────────────────────────────────────────────────
# 로깅 설정
# ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("build_pu_valor.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────
# 출력 규격
# VALOR-32K 클립들이 해상도/fps가 제각각이므로 통일
# ────────────────────────────────────────────────
OUT_W   = 640
OUT_H   = 360
OUT_FPS = 25

# ────────────────────────────────────────────────
# 전역 worker 초기화
# ────────────────────────────────────────────────
_VALOR_INDEX = {}
_VIDEO_DIR   = ""
_DRY_RUN     = False


def worker_init(valor_index, video_dir, dry_run):
    global _VALOR_INDEX, _VIDEO_DIR, _DRY_RUN
    _VALOR_INDEX = valor_index
    _VIDEO_DIR   = video_dir
    _DRY_RUN     = dry_run


# ────────────────────────────────────────────────
# VALOR-32K 파일 인덱스 빌드
# ────────────────────────────────────────────────
def build_valor_index(valor_dir: str) -> dict:
    index = {}
    pattern = os.path.join(valor_dir, "*.mp4")
    files = glob.glob(pattern)
    re_suffix = re.compile(r"_\d+\.\d+_\d+\.\d+\.mp4$")
    for fpath in files:
        fname = os.path.basename(fpath)
        yt_id = re_suffix.sub("", fname)
        index[yt_id] = fpath
    logger.info(f"VALOR-32K 인덱스 빌드 완료: {len(index)}개 클립")
    return index


# ────────────────────────────────────────────────
# ffmpeg: scale + 해상도/fps 통일 + concat 한 번에
# ────────────────────────────────────────────────
def build_pseudo_video(clip_paths: list, scales: list, output_path: str) -> tuple:
    """
    여러 클립을 ffmpeg filter_complex로 한 번에 처리:
      - 각 클립: 해상도 통일(scale+pad) + fps 통일 + 속도 조정(setpts/atempo)
      - 전체: concat filter로 이어붙임
    재인코딩 1회로 화질 손실 최소화, 깨짐 없음.
    """
    n = len(clip_paths)

    input_args = []
    for p in clip_paths:
        input_args += ["-i", p]

    filter_parts = []
    v_labels = []
    a_labels = []

    for i, scale in enumerate(scales):
        vl = f"v{i}"
        al = f"a{i}"
        filter_parts.append(
            f"[{i}:v]"
            f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,"
            f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
            f"fps={OUT_FPS},"
            f"setpts=PTS/{scale}"
            f"[{vl}]"
        )
        filter_parts.append(
            f"[{i}:a]atempo={scale}[{al}]"
        )
        v_labels.append(f"[{vl}]")
        a_labels.append(f"[{al}]")

    concat_inputs = "".join(f"{v}{a}" for v, a in zip(v_labels, a_labels))
    filter_parts.append(
        f"{concat_inputs}concat=n={n}:v=1:a=1[outv][outa]"
    )

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        *input_args,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "[outa]",
        "-c:v", "mpeg4", "-q:v", "5",
        "-c:a", "aac", "-b:a", "128k",
        "-loglevel", "error",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stderr


# ────────────────────────────────────────────────
# 단일 샘플 처리 함수 (worker)
# ────────────────────────────────────────────────
def process_sample(sample):
    try:
        sample_id  = sample["id"]
        sub_videos = sample["meta"]["sub_videos"]
        scales     = sample["meta"]["scale"]

        safe_id     = sample_id.replace("/", "_")
        output_path = os.path.join(_VIDEO_DIR, f"{safe_id}.mp4")

        if os.path.exists(output_path):
            return {"status": "skip", "id": sample_id}

        clip_paths = []
        missing    = []
        for vid_id in sub_videos:
            if vid_id in _VALOR_INDEX:
                clip_paths.append(_VALOR_INDEX[vid_id])
            else:
                missing.append(vid_id)

        if missing:
            return {"status": "missing", "id": sample_id, "missing": missing}

        if _DRY_RUN:
            return {"status": "dry_run_ok", "id": sample_id}

        ok, stderr = build_pseudo_video(clip_paths, scales, output_path)
        if not ok:
            return {
                "status": "ffmpeg_error",
                "id": sample_id,
                "stderr": stderr[-500:],
            }

        return {"status": "ok", "id": sample_id}

    except Exception as e:
        return {
            "status": "exception",
            "id": sample.get("id", "unknown"),
            "error": traceback.format_exc(),
        }


# ────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3_json", required=True)
    parser.add_argument("--valor_dir",   required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--workers",     type=int, default=8)
    parser.add_argument("--dry_run",     action="store_true")
    args = parser.parse_args()

    video_dir = os.path.join(args.output_dir, "videos")
    anno_dir  = os.path.join(args.output_dir, "annotations")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(anno_dir,  exist_ok=True)

    logger.info("stage3.json 로딩 중...")
    with open(args.stage3_json, "r") as f:
        all_samples = json.load(f)

    pv_samples = [s for s in all_samples if s.get("source") == "pseudo-valor"]
    logger.info(f"pseudo-valor 샘플 수: {len(pv_samples)}")

    valor_index = build_valor_index(args.valor_dir)

    results = {
        "ok": 0, "skip": 0, "missing": 0,
        "ffmpeg_error": 0, "dry_run_ok": 0, "exception": 0,
    }
    failed_samples = []

    with Pool(
        processes=args.workers,
        initializer=worker_init,
        initargs=(valor_index, video_dir, args.dry_run),
    ) as pool:
        for res in tqdm(
            pool.imap_unordered(process_sample, pv_samples, chunksize=4),
            total=len(pv_samples),
            desc="합성 진행",
        ):
            status = res["status"]
            results[status] = results.get(status, 0) + 1
            if status not in ("ok", "skip", "dry_run_ok"):
                failed_samples.append(res)
                if status == "exception":
                    logger.error(f"[exception] id={res['id']}\n{res['error']}")

    logger.info("=" * 50)
    logger.info(f"완료:              {results['ok']}")
    logger.info(f"스킵 (기존 파일):  {results['skip']}")
    logger.info(f"dry_run OK:        {results['dry_run_ok']}")
    logger.info(f"누락 클립:         {results['missing']}")
    logger.info(f"ffmpeg 오류:       {results['ffmpeg_error']}")
    logger.info(f"예외:              {results['exception']}")
    logger.info("=" * 50)

    if failed_samples:
        fail_path = os.path.join(anno_dir, "failed_samples.json")
        with open(fail_path, "w") as f:
            json.dump(failed_samples, f, indent=2, ensure_ascii=False)
        logger.info(f"실패 샘플 저장: {fail_path}")

    if not args.dry_run:
        succeeded_ids = set()
        for s in pv_samples:
            safe_id  = s["id"].replace("/", "_")
            vid_path = os.path.join(video_dir, f"{safe_id}.mp4")
            if os.path.exists(vid_path):
                succeeded_ids.add(s["id"])

        anno_samples = [s for s in pv_samples if s["id"] in succeeded_ids]
        anno_path = os.path.join(anno_dir, "pu_valor_stage3.json")
        with open(anno_path, "w") as f:
            json.dump(anno_samples, f, indent=2, ensure_ascii=False)
        logger.info(f"어노테이션 저장: {anno_path}  ({len(anno_samples)}개)")


if __name__ == "__main__":
    main()