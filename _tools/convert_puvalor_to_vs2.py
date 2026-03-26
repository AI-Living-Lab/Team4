"""
convert_pu_valor_to_vs2.py

PU-VALOR stage3.json → vS2 학습용 JSON 변환 스크립트.

변환 전략:
  샘플1,2: 마지막 gpt가 순수 텍스트       → ce_only=True (time token 변환 없음)
  샘플3:   마지막 gpt가 "From <sN> to <eN>." → mode="single", timestamps=[t0, t1]
  샘플4:   마지막 gpt가 여러 "From <sN>..."  → mode="dense",  events=[{label, timestamps}]

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
# conversations 변환
# ────────────────────────────────────────────────
def replace_video_token(conversations: list) -> list:
    """<video> → <image> 치환"""
    result = []
    for conv in conversations:
        c = dict(conv)
        c["value"] = c["value"].replace("<video>", "<image>")
        result.append(c)
    return result


def replace_time_tokens_in_conv(conversations: list, token_sec: dict) -> list:
    """
    conversations 내 모든 <sN>, <eN> 토큰을 실제 초 숫자로 치환.
    (학습 시 av_dataset.py가 마지막 gpt의 timestamps 필드를 보고
     time token으로 변환하므로, 나머지 turn의 <sN>/<eN>은
     그냥 숫자 텍스트로 바꿔두는 게 자연스러움)
    """
    result = []
    for conv in conversations:
        c = dict(conv)
        val = c["value"]
        for tok, sec in token_sec.items():
            val = val.replace(tok, f"{sec:.1f}s")
        c["value"] = val
        result.append(c)
    return result


# ────────────────────────────────────────────────
# 샘플 유형별 변환
# ────────────────────────────────────────────────
def convert_ce_only(sample: dict, video_path: str, audio_path: str,
                    token_sec: dict) -> dict:
    """샘플1, 2: 순수 텍스트 답변 → ce_only"""
    convs = replace_video_token(sample["conversations"])
    convs = replace_time_tokens_in_conv(convs, token_sec)
    return {
        "video": video_path,
        "audio": audio_path,
        "ce_only": True,
        "conversations": convs,
    }


def convert_single(sample: dict, video_path: str, audio_path: str,
                   token_sec: dict) -> dict:
    """
    샘플3: 마지막 gpt = "From <sN> to <eN>."
    → mode=single, timestamps=[sN_sec, eN_sec]
    마지막 gpt의 <sN>/<eN>은 av_dataset.py가 time token으로 교체하므로
    그대로 두고, timestamps 필드만 추가.
    """
    last_gpt = sample["conversations"][-1]["value"]
    s_match = re.search(r"<(s\d+)>", last_gpt)
    e_match = re.search(r"<(e\d+)>", last_gpt)

    s_key = f"<{s_match.group(1)}>"
    e_key = f"<{e_match.group(1)}>"
    t0 = token_sec[s_key]
    t1 = token_sec[e_key]

    # 마지막 gpt 이전 turn의 <sN>/<eN>만 숫자로 치환
    convs = replace_video_token(sample["conversations"])
    new_convs = []
    for i, conv in enumerate(convs):
        c = dict(conv)
        if i < len(convs) - 1:  # 마지막 gpt 제외
            val = c["value"]
            for tok, sec in token_sec.items():
                val = val.replace(tok, f"{sec:.1f}s")
            c["value"] = val
        new_convs.append(c)

    return {
        "video": video_path,
        "audio": audio_path,
        "mode": "single",
        "timestamps": [t0, t1],
        "conversations": new_convs,
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

    # unav100과 동일한 질문 형식 사용
    convs = [
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
        {
            "from": "gpt",
            "value": "PLACEHOLDER",
        },
    ]

    return {
        "video": video_path,
        "audio": audio_path,
        "mode": "dense",
        "events": events,
        "conversations": convs,
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
                    if not item["events"]:  # 파싱 실패 시 스킵
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
    logger.info(f"  ce_only:  {stats['ce_only']}")
    logger.info(f"  single:   {stats['single']}")
    logger.info(f"  dense:    {stats['dense']}")
    logger.info(f"  skip:     {stats['skip']}")
    logger.info(f"저장 완료: {args.output_json}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
