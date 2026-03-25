"""
stage3.json → InternVid 클립 다운로드 (이미 받은 파일 제외)
-------------------------------------------------------------
사용법:
    # 기본 실행 (이미 받은 파일 자동 감지 후 제외)
    python download_internvid_v2.py

    # 옵션
    python download_internvid_v2.py \
        --json    /home/aix23102/audiolm/vS2_eunji/data/stage3.json \
        --done-dir /data0/aix23102/internvid \
        --output  /data0/aix23102/internvid \
        --workers 32 \
        --resolution 360

설치:
    pip install video2dataset
"""

import re
import csv
import argparse
import subprocess
import sys
import os
from pathlib import Path
from collections import Counter


# ── 1. stage3.json 파싱 ───────────────────────────────────────────────────────

def parse_internvid_clips(json_path: str) -> list[dict]:
    with open(json_path) as f:
        content = f.read()

    # 파일 잘림 경고
    if not content.rstrip().endswith("]"):
        print("[경고] stage3.json이 잘려 있습니다 — 완전히 파싱된 항목까지만 처리합니다.")

    # source 분포 출력
    all_sources = re.findall(r'"source":\s*"([^"]+)"', content)
    print("[분포] source 현황:")
    for src, cnt in Counter(all_sources).most_common():
        print(f"       {src}: {cnt:,}개")

    # (id, body) → clips
    pattern = r'"id":\s*"([^"]+)"((?:(?!"id").){0,5000}?)"source":\s*"internvid"'
    matches = re.findall(pattern, content, re.DOTALL)

    clips = []
    for vid_id, body in matches:
        split_match = re.search(r'"split":\s*\[([^\]]+)\]', body)
        if split_match:
            times = [float(x.strip()) for x in split_match.group(1).split(",")]
            start, end = times[0], times[1]
        else:
            start, end = 0.0, -1.0
        clips.append({
            "videoID": vid_id,
            "url": f"https://www.youtube.com/watch?v={vid_id}",
            "start_time": start,
            "end_time": end,
        })

    return clips


# ── 2. 이미 받은 파일 목록 추출 ───────────────────────────────────────────────

def get_already_downloaded(done_dir: str) -> set[str]:
    """
    /data0/aix23102/internvid/ 안의 파일명에서 확장자 제거 → videoID set 반환
    예: 'Byl0g4B_Gao.mp4' → 'Byl0g4B_Gao'
    """
    p = Path(done_dir)
    if not p.exists():
        print(f"[경고] 폴더가 없습니다: {done_dir}")
        return set()

    ids = set()
    for f in p.iterdir():
        if f.is_file():
            ids.add(f.stem)   # 확장자 제거
    return ids


# ── 3. CSV 생성 ───────────────────────────────────────────────────────────────

def write_csv(clips: list[dict], csv_path: str):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["videoID", "url", "start_time", "end_time"]
        )
        writer.writeheader()
        writer.writerows(clips)
    print(f"[✓] CSV 저장: {csv_path}  ({len(clips):,}개)")


# ── 4. video2dataset 실행 ─────────────────────────────────────────────────────

def find_video2dataset_bin() -> str:
    """python 실행파일과 같은 bin 디렉토리에서 video2dataset 찾기"""
    bin_dir = Path(sys.executable).parent
    v2d = bin_dir / "video2dataset"
    if v2d.exists():
        return str(v2d)
    # PATH에서 찾기
    import shutil
    v2d_path = shutil.which("video2dataset")
    if v2d_path:
        return v2d_path
    raise FileNotFoundError("video2dataset 실행파일을 찾을 수 없습니다. pip install video2dataset 로 설치하세요.")


def run_video2dataset(csv_path: str, output_dir: str, workers: int, resolution: int, cookies: str = None):
    v2d_bin = find_video2dataset_bin()

    cmd = [
        v2d_bin,
        csv_path,                                    # positional: URL_LIST
        "--url_col",                 "url",
        "--start_time_col",          "start_time",
        "--end_time_col",            "end_time",
        "--save_additional_columns", '["videoID"]',
        "--output_folder",           output_dir,
        "--output_format",           "files",
        "--video_size",              str(resolution),
        "--number_sample_per_shard", "200",
        "--processes_count",         str(workers),
        "--encode_formats",          '{"video": "mp4"}',
        "--retries",                 "3",
        "--config",                  "default",
    ]

    # 쿠키 파일: --yt-dlp_args 로 전달
    if cookies and os.path.exists(cookies):
        cmd += ["--yt-dlp_args", f"--cookies '{cookies}'"]
        print(f"[✓] 쿠키 파일 사용: {cookies}")

    print(f"\n[→] video2dataset 시작")
    print(f"    workers={workers}, resolution={resolution}p")
    print(f"    출력 폴더: {output_dir}")
    print(f"    명령어: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json",       default="/home/aix23102/audiolm/vS2_eunji/data/stage3.json")
    parser.add_argument("--done-dir",   default="/data0/aix23102/internvid",  help="이미 다운받은 파일들이 있는 폴더")
    parser.add_argument("--output",     default="/data0/aix23102/internvid",  help="다운로드 출력 폴더")
    parser.add_argument("--csv",        default="internvid_remaining.csv")
    parser.add_argument("--workers",    type=int, default=32)
    parser.add_argument("--resolution", type=int, default=360)
    parser.add_argument("--cookies",    default="/home/aix23102/audiolm/vS2_eunji/www.youtube.com_cookies (1).txt",
                                        help="YouTube 쿠키 파일 경로")
    parser.add_argument("--csv-only",   action="store_true", help="CSV만 생성하고 다운로드는 건너뜀")
    args = parser.parse_args()

    # 1) 파싱
    print(f"[→] {args.json} 파싱 중...")
    clips = parse_internvid_clips(args.json)
    print(f"[✓] 파싱된 InternVid 클립: {len(clips):,}개\n")

    # 2) 이미 받은 파일 제외
    print(f"[→] 이미 다운받은 파일 확인 중: {args.done_dir}")
    already_done = get_already_downloaded(args.done_dir)
    print(f"[✓] 이미 받은 파일: {len(already_done):,}개")

    remaining = [c for c in clips if c["videoID"] not in already_done]
    skipped = len(clips) - len(remaining)
    print(f"[✓] 제외: {skipped:,}개  |  남은 클립: {len(remaining):,}개\n")

    if not remaining:
        print("[✓] 모두 다운로드 완료 상태입니다!")
        return

    # 3) CSV 생성
    write_csv(remaining, args.csv)

    if args.csv_only:
        print("\n[--csv-only] CSV만 생성하고 종료합니다.")
        return

    # 4) video2dataset 설치 확인
    try:
        import video2dataset  # noqa
    except ImportError:
        print("\n[!] video2dataset 미설치 → 설치 중...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "video2dataset"],
            check=True
        )

    # 5) 다운로드
    run_video2dataset(args.csv, args.output, args.workers, args.resolution, args.cookies)
    print(f"\n[✓] 완료! → {args.output}/")


if __name__ == "__main__":
    main()
