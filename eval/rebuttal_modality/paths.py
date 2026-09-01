#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paths.py — 이 작업에서 쓰는 실제 경로 한 곳 정의.

지시문의 <PATH> 자리표시자를 이 서버의 실제 파일로 해석한 결과.
해석 근거는 log.txt 의 PATH RESOLUTION 절에 남긴다.
"""
import os

WS = "/home/team404/workspace"
EVAL_DIR = f"{WS}/Team4/eval"
HERE = os.path.dirname(os.path.abspath(__file__))

# 재구성 프로토콜 테스트 split (3,455 샘플).
#   지시문의 unav100_multiseg_test.json 에 해당.
#   id/vid/gt_label/gt_segments 를 모두 가진 유일한 형태라 이 파일을 정본으로 쓴다.
TEST_SPLIT = f"{WS}/data/test/unav100/museg.json"

# 카테고리 정본 목록 (100개)
CLASS_LABELS = f"{WS}/data/avicuna_unav100/labels/unav100_class_labels.txt"

# 영상 길이 소스: 원본에서 추출한 wav 헤더(=영상 길이와 동일). ffprobe 가 없어서 이렇게 잰다.
AUDIO_DIR = f"{WS}/datasets/unav_100/audio"

# 예측 결과 4종. 지시문의 preds/{...}.json 에 해당.
PREDS = {
    # (경로, pred 파서, GT 출처)
    "titok": (
        f"{WS}/outputs/gdpo/sft_7b_unav_v8_rl_rMsep3_unpucha_batch4_noscaling/"
        f"checkpoint-2000/fps5_tti/unav100_titok/test_results_rank0.json",
        "token", "ref"),
    "titok_wo_audio": (
        f"{WS}/outputs/gdpo/sft_7b_unav_v8_rl_rMsep3_unpucha_batch4_noscaling/"
        f"checkpoint-2000/fps5_tti/unav100_noaudio/test_results_rank0.json",
        "token", "ref"),
    # audio 없이 학습된 별도 런 (noaudio ckpt-200). titok_wo_audio 와 달리
    # 체크포인트 자체가 다르다 — 학습량(2000 vs 200 step) 차이가 같이 섞인다.
    "titok_noaudio_trained": (
        f"{WS}/outputs/gdpo/sft_7b_unav_v8_rl_rMsep3_unpucha_batch4_noscaling_noaudio/"
        f"checkpoint-200/fps5_tti/unav100_titok_noaudio/test_results_rank0.json",
        "token", "ref"),
    "museg": (
        f"{WS}/outputs/base/MUSEG/unav100_multiseg/eval/test_results_rank0.json",
        "plain", "embedded"),
    "chronusomni": (
        f"{WS}/outputs/sft/ChronusOmni/unav100_chronus/eval/test_results_rank0.json",
        "plain", "embedded"),
}

MAX_TIME = 999.9

# MODALITY_CSV 를 주면 그 라벨로 Step 3~6 을 돌리고, 산출물에 .reviewed 접미사를 붙여
# 자동 라벨 결과를 덮어쓰지 않는다.
_REVIEW = os.environ.get("MODALITY_CSV")
SUF = ".reviewed" if _REVIEW else ""

OUT_CATEGORIES = f"{HERE}/categories_stats.csv"
OUT_MODALITY = _REVIEW if _REVIEW else f"{HERE}/unav100_modality_split.csv"
OUT_GROUPS = f"{HERE}/group_stats{SUF}.csv"
OUT_RESULTS = f"{HERE}/results_by_modality{SUF}.csv"
OUT_LATEX = f"{HERE}/results_by_modality{SUF}.tex"
LOG = f"{HERE}/log.txt"


def log(step, msg):
    """log.txt 에 타임스탬프와 함께 실행 순서를 남긴다."""
    import datetime
    tag = step + ("*" if SUF else "")
    line = f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {tag:<8} {msg}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)
