"""
convert_pu_valor_to_vs2.py

PU-VALOR stage3.json → vS2 학습용 JSON 변환 스크립트.

변환 전략:
  원본 PU-VALOR 대화 구조를 유지하면서, 시간 표현만 vS2 time token으로 변환.
  샘플1,2 (ce_only): 대화 구조 유지, <sN>/<eN> → time token 치환
  샘플3 (single):    대화 구조 유지, <sN>/<eN> → time token 치환
  샘플4 (dense):     mode=dense, events + PLACEHOLDER (av_dataset.py가 time token 생성)

타임스탬프 변환:
  token_value (0~100 퍼센트) → 실제 초: actual_sec = (token_value / 100.0) * duration

사용법:
    python convert_pu_valor_to_vs2.py \
        --stage3_json  /home/aix23102/audiolm/vS2_eunji/data/stage3.json \
        --video_dir    /data0/aix23102/PU-VALOR/videos \
        --audio_dir    /data0/aix23102/PU-VALOR/audios \
        --output_json  /home/aix23102/audiolm/vS2_eunji/data/pu_valor_train.json
"""

import os
import re
import json
import argparse
import logging
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────
# 타임스탬프 변환
# ────────────────────────────────────────────────
def token_to_sec(token_value: float, duration: float) -> float:
    """PU-VALOR 퍼센트 토큰값 → 실제 초"""
    return round((token_value / 100.0) * duration, 3)


def resolve_tokens(token_dict: dict, duration: float) -> dict:
    """{'<s0>': 65.0, '<e0>': 75.0, ...} → {'<s0>': 실제초, ...}"""
    return {k: token_to_sec(v, duration) for k, v in token_dict.items()}


# ────────────────────────────────────────────────
# 대화 유형 판별
# ────────────────────────────────────────────────
def classify_sample(sample: dict) -> str:
    """
    반환값:
      'ce_only' : 마지막 gpt가 순수 텍스트 (샘플1, 2)
      'single'  : 마지막 gpt가 "From <sN> to <eN>." 단일 구간 (샘플3)
      'dense'   : 마지막 gpt가 여러 구간 포함 (샘플4)
    """
    last_gpt = sample["conversations"][-1]["value"]

    # <sN> 토큰 등장 횟수로 판별
    s_tokens = re.findall(r"<s\d+>", last_gpt)

    if len(s_tokens) == 0:
        return "ce_only"
    elif len(s_tokens) == 1:
        return "single"
    else:
        return "dense"


# ────────────────────────────────────────────────
# 공통 헬퍼
# ────────────────────────────────────────────────
def _sec_to_time_tokens(sec: float) -> str:
    """초 → vS2 time token 문자열 (e.g., 61.8 → '<t0><t0><t6><t1><tdot><t8>')"""
    sec = round(sec, 1)
    integer_part = int(sec)
    frac = int(round((sec - integer_part) * 10))
    digits = f"{integer_part:04d}"
    return (
        f"<t{digits[0]}><t{digits[1]}><t{digits[2]}><t{digits[3]}>"
        f"<tdot><t{frac}>"
    )


def _replace_tokens_with_time_tokens(conversations: list,
                                     token_sec: dict) -> list:
    """conversations 내 <video>→<image>, <sN>/<eN>→time token 치환"""
    result = []
    for conv in conversations:
        c = dict(conv)
        val = c["value"].replace("<video>", "<image>")
        for tok, sec in token_sec.items():
            val = val.replace(tok, _sec_to_time_tokens(sec))
        c["value"] = val
        result.append(c)
    return result


def _make_dense_convs() -> list:
    """unav100과 동일한 dense 대화 템플릿"""
    return [
        {
            "from": "human",
            "value": (
                "<image>\nYou are an audio-visual event localization model.\n"
                "Given the video and audio, localize all audio-visual events.\n\n"
                "Output a JSON list. Each element must have:\n"
                "  \"event\": event label (string)\n"
                "  \"start\": six time tokens (4-digit integer part, <tdot>, 1-digit decimal)\n"
                "  \"end\":   six time tokens (same format)\n"
                "Example: [{\"event\": \"dog barking\", "
                "\"start\": \"<t0><t0><t1><t2><tdot><t3>\", "
                "\"end\": \"<t0><t0><t4><t5><tdot><t6>\"}]"
            ),
        },
        {"from": "gpt", "value": "PLACEHOLDER"},
    ]


# ────────────────────────────────────────────────
# 샘플 유형별 변환
# ────────────────────────────────────────────────
def convert_ce_only(sample: dict, video_path: str, audio_path: str,
                    token_sec: dict) -> dict:
    """
    샘플1,2 (ce_only): 대화 구조 유지, <sN>/<eN> → time token 치환.
    """
    convs = _replace_tokens_with_time_tokens(sample["conversations"], token_sec)
    return {
        "video": video_path,
        "audio": audio_path,
        "ce_only": True,
        "conversations": convs,
    }


def convert_single(sample: dict, video_path: str, audio_path: str,
                   token_sec: dict) -> dict:
    """
    샘플3 (single): 대화 구조 유지, <sN>/<eN> → time token 치환.
    """
    convs = _replace_tokens_with_time_tokens(sample["conversations"], token_sec)
    return {
        "video": video_path,
        "audio": audio_path,
        "conversations": convs,
    }


def convert_dense(sample: dict, video_path: str, audio_path: str,
                  token_sec: dict) -> dict:
    """
    샘플4: 마지막 gpt = "From <s0> to <e0>, 설명. From <s1> to <e1>, ..."
    → mode=dense, events=[{label, timestamps}, ...]
    av_dataset.py의 dense 처리:
      events의 label과 timestamps를 보고 time token 응답 생성
    """
    last_gpt = sample["conversations"][-1]["value"]

    # "From <sN> to <eN>, 설명." 패턴 파싱
    pattern = re.compile(
        r"[Ff]rom\s+<(s\d+)>\s+to\s+<(e\d+)>,?\s*(.+?)(?=\s+[Ff]rom\s+<s\d+>|$)",
        re.DOTALL
    )
    matches = pattern.findall(last_gpt)

    events = []
    for s_key, e_key, label in matches:
        t0 = token_sec.get(f"<{s_key}>", 0.0)
        t1 = token_sec.get(f"<{e_key}>", 0.0)
        events.append({
            "label": label.strip().rstrip("."),
            "timestamps": [t0, t1],
        })

    return {
        "video": video_path,
        "audio": audio_path,
        "mode": "dense",
        "events": events,
        "conversations": _make_dense_convs(),
    }


# ────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3_json", required=True)
    parser.add_argument("--video_dir",   required=True)
    parser.add_argument("--audio_dir",   required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    # stage3.json 로드
    logger.info("stage3.json 로딩 중...")
    with open(args.stage3_json) as f:
        all_data = json.load(f)

    pv_samples = [s for s in all_data if s.get("source") == "pseudo-valor"]
    logger.info(f"pseudo-valor 샘플 수: {len(pv_samples)}")

    # id별로 그룹핑 (4개씩)
    by_id = defaultdict(list)
    for s in pv_samples:
        by_id[s["id"]].append(s)

    stats = {"ce_only": 0, "single": 0, "dense": 0, "skip": 0}
    output = []

    for sample_id, samples in by_id.items():
        safe_id    = sample_id.replace("/", "_")
        video_path = os.path.join(args.video_dir, f"{safe_id}.mp4")
        audio_path = os.path.join(args.audio_dir, f"{safe_id}.wav")

        # 비디오/오디오 파일 존재 확인
        if not os.path.exists(video_path):
            stats["skip"] += len(samples)
            continue
        if not os.path.exists(audio_path):
            # 오디오 없으면 audio 키 생략
            audio_path = None

        for sample in samples:
            duration   = sample["meta"]["duration"]
            token_dict = sample["meta"]["token"]
            token_sec  = resolve_tokens(token_dict, duration)

            sample_type = classify_sample(sample)

            try:
                if sample_type == "ce_only":
                    item = convert_ce_only(sample, video_path, audio_path, token_sec)
                elif sample_type == "single":
                    item = convert_single(sample, video_path, audio_path, token_sec)
                else:  # dense
                    item = convert_dense(sample, video_path, audio_path, token_sec)

                # dense 모드에서 이벤트 파싱 실패 시 스킵
                if item is None:
                    stats["skip"] += 1
                    continue
                if item.get("mode") == "dense" and not item.get("events"):
                    stats["skip"] += 1
                    continue

                # audio_path가 None이면 키 제거
                if audio_path is None:
                    item.pop("audio", None)

                output.append(item)
                stats[sample_type] += 1

            except Exception as e:
                logger.warning(f"변환 실패 id={sample_id}: {e}")
                stats["skip"] += 1

    # 결과 저장
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info("=" * 50)
    logger.info(f"총 출력 샘플: {len(output)}")
    logger.info(f"  ce_only (time token 치환): {stats['ce_only']}")
    logger.info(f"  single  (time token 치환): {stats['single']}")
    logger.info(f"  dense   (PLACEHOLDER):     {stats['dense']}")
    logger.info(f"  skip:                      {stats['skip']}")
    logger.info(f"저장 완료: {args.output_json}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
