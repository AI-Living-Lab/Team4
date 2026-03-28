#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_data_paths.py
  학습/평가 데이터 JSON 파일 내 하드코딩된 서버 경로를
  현재 환경(paths.env)에 맞게 치환.

Usage:
  source paths.env
  python _tools/fix_data_paths.py --data_dir data/

  또는 직접 지정:
  python _tools/fix_data_paths.py \
    --data_dir data/ \
    --video_dir /root/datasets/unav_100/videos \
    --audio_dir /root/datasets/unav_100/audio
"""

import argparse
import glob
import json
import os
import re


# 치환 대상 경로 패턴들
OLD_PATTERNS = [
    ("/data0/aix23102/unav_100/videos", "video"),
    ("/data0/aix23102/unav_100/audio", "audio"),
    ("/data0/aix23102/PU-VALOR/videos", "puvalor_video"),
    ("/data0/aix23102/PU-VALOR/audios", "puvalor_audio"),
]


def fix_json_file(filepath: str, replacements: dict, dry_run: bool = False) -> int:
    """JSON 파일 내 경로를 치환. 변경된 횟수 반환."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    count = 0
    for old_path, new_path in replacements.items():
        if old_path in content:
            n = content.count(old_path)
            content = content.replace(old_path, new_path)
            count += n

    if count > 0 and not dry_run:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    return count


def main():
    parser = argparse.ArgumentParser(
        description="데이터 JSON 내 하드코딩 경로 치환"
    )
    parser.add_argument("--data_dir", default="data/",
                        help="JSON 파일이 있는 디렉토리")
    parser.add_argument("--video_dir", default=None,
                        help="UnAV-100 비디오 디렉토리 (미지정 시 UNAV_VIDEO_DIR 환경변수)")
    parser.add_argument("--audio_dir", default=None,
                        help="UnAV-100 오디오 디렉토리 (미지정 시 UNAV_AUDIO_DIR 환경변수)")
    parser.add_argument("--puvalor_video_dir", default=None,
                        help="PU-VALOR 비디오 (미지정 시 PUVALOR_DIR/videos)")
    parser.add_argument("--puvalor_audio_dir", default=None,
                        help="PU-VALOR 오디오 (미지정 시 PUVALOR_DIR/audios)")
    parser.add_argument("--dry_run", action="store_true",
                        help="실제 파일을 수정하지 않고 변경 사항만 출력")
    args = parser.parse_args()

    # 환경변수에서 기본값 로드
    video_dir = args.video_dir or os.environ.get("UNAV_VIDEO_DIR", "")
    audio_dir = args.audio_dir or os.environ.get("UNAV_AUDIO_DIR", "")
    puvalor_video = args.puvalor_video_dir or os.environ.get("PUVALOR_DIR", "") + "/videos"
    puvalor_audio = args.puvalor_audio_dir or os.environ.get("PUVALOR_DIR", "") + "/audios"

    replacements = {}
    if video_dir:
        replacements["/data0/aix23102/unav_100/videos"] = video_dir
    if audio_dir:
        replacements["/data0/aix23102/unav_100/audio"] = audio_dir
    if puvalor_video and puvalor_video != "/videos":
        replacements["/data0/aix23102/PU-VALOR/videos"] = puvalor_video
    if puvalor_audio and puvalor_audio != "/audios":
        replacements["/data0/aix23102/PU-VALOR/audios"] = puvalor_audio

    if not replacements:
        print("[ERROR] 치환할 경로가 없습니다. --video_dir/--audio_dir를 지정하거나 paths.env를 source하세요.")
        return

    print("[fix_data_paths] 치환 규칙:")
    for old, new in replacements.items():
        print(f"  {old} → {new}")
    if args.dry_run:
        print("  (DRY RUN - 파일 수정 안 함)")

    json_files = glob.glob(os.path.join(args.data_dir, "*.json"))
    total = 0
    for fpath in sorted(json_files):
        count = fix_json_file(fpath, replacements, dry_run=args.dry_run)
        if count > 0:
            print(f"  {os.path.basename(fpath)}: {count}건 치환")
            total += count

    print(f"[fix_data_paths] 총 {total}건 치환 완료 ({len(json_files)}개 파일)")


if __name__ == "__main__":
    main()
