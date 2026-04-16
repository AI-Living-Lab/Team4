"""
verify_pu_valor_duration.py

합성된 PU-VALOR 비디오의 실제 duration과
stage3.json 메타데이터의 duration을 비교 검증하는 스크립트.

사용법:
    python verify_pu_valor_duration.py \
        --stage3_json /home/aix23102/audiolm/vS2_eunji/data/stage3.json \
        --video_dir   /data0/aix23102/PU-VALOR/videos \
        --num_samples 200
"""

import os
import json
import argparse
import subprocess

def get_video_duration(path: str):
    """ffprobe로 비디오의 실제 duration(초)을 반환."""
    ffprobe_bin = os.environ.get(
        "FFPROBE_BIN",
        "/home/aix23102/anaconda3/envs/avf/bin/ffprobe",
    )
    cmd = [
        ffprobe_bin, "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3_json", required=True)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=200,
                        help="검증할 샘플 수 (0이면 전체)")
    args = parser.parse_args()

    with open(args.stage3_json) as f:
        all_samples = json.load(f)

    pv_samples = [s for s in all_samples if s.get("source") == "pseudo-valor"]
    print(f"pseudo-valor 샘플 수: {len(pv_samples)}")

    if args.num_samples > 0:
        pv_samples = pv_samples[:args.num_samples]

    diffs = []
    missing = 0
    checked = 0

    for s in pv_samples:
        safe_id = s["id"].replace("/", "_")
        vpath = os.path.join(args.video_dir, f"{safe_id}.mp4")
        if not os.path.exists(vpath):
            missing += 1
            continue

        actual = get_video_duration(vpath)
        if actual is None:
            missing += 1
            continue

        expected = s["meta"]["duration"]
        diff = actual - expected
        rel_err = abs(diff) / expected * 100 if expected > 0 else float("inf")
        diffs.append({
            "id": s["id"],
            "expected": expected,
            "actual": round(actual, 2),
            "diff": round(diff, 2),
            "rel_err_%": round(rel_err, 2),
        })
        checked += 1

    print(f"\n검증 완료: {checked}개 | 누락/오류: {missing}개\n")

    if not diffs:
        print("검증된 샘플이 없습니다.")
        return

    abs_diffs = [abs(d["diff"]) for d in diffs]
    rel_errs = [d["rel_err_%"] for d in diffs]

    print("=== Duration 차이 통계 ===")
    print(f"  평균 절대 차이: {sum(abs_diffs)/len(abs_diffs):.2f}s")
    print(f"  최대 절대 차이: {max(abs_diffs):.2f}s")
    print(f"  평균 상대 오차: {sum(rel_errs)/len(rel_errs):.2f}%")
    print(f"  최대 상대 오차: {max(rel_errs):.2f}%")

    threshold = 2.0  # 2초 이상 차이나면 경고
    outliers = [d for d in diffs if abs(d["diff"]) > threshold]
    print(f"\n  |차이| > {threshold}s 인 샘플: {len(outliers)}개 / {checked}개")

    if outliers:
        print(f"\n=== 차이가 큰 상위 샘플 (최대 20개) ===")
        outliers.sort(key=lambda d: abs(d["diff"]), reverse=True)
        for d in outliers[:20]:
            print(f"  {d['id']:30s}  기대={d['expected']:7.1f}s  "
                  f"실제={d['actual']:7.1f}s  차이={d['diff']:+.2f}s  "
                  f"({d['rel_err_%']:.1f}%)")


if __name__ == "__main__":
    main()
