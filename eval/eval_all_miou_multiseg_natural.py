#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_all_miou_multiseg_natural.py
  - outputs/base 의 자연어/이종(異種) 포맷 base 모델 출력에 대해 All/Best/Union mIoU 계산.
  - eval_all_miou_multiseg.py 의 집계(compute_core_summary)·저장(write_and_report) 을
    그대로 재사용하고, pred 파싱과 GT 추출만 모델별로 갈아끼운다.
    → 수치 집계 로직은 토큰 버전과 100% 동일, 포맷 차이만 흡수.
  - 출력: 각 test_results_rank0.json 옆에 eval_all_miou_summary.json

모델 자동 감지(경로 키워드) → (pred_format, gt_source) 매핑:
  arc(hunyuan) : pred=arc_hunyuan  (<answer>HH:MM:SS - HH:MM:SS</answer>)  gt=embedded
  avicuna      : pred=avicuna      ("from X to Y" = 영상 길이 대비 %)        gt=embedded (duration 필요)
  museg        : pred=museg        ("X.XX-X.XX" 초)                          gt=embedded
  chronus      : pred=chronusomni  ("second{X} to second{Y}")               gt=embedded
  salmonn      : pred=salmonn      ("from M:SS to M:SS")                     gt=ref (토큰 GT)
--pred_format / --gt_source / --model 로 수동 지정 가능 (자동 감지 실패 시).

사용:
  python3 eval_all_miou_multiseg_natural.py outputs/base/ArcHunyuan
  python3 eval_all_miou_multiseg_natural.py outputs/base/Avicuna/unav100/unav100_tail_B
  python3 eval_all_miou_multiseg_natural.py outputs/base/MUSEG/Unav100
  python3 eval_all_miou_multiseg_natural.py outputs/base/salmonn2p_7b_timetoken
"""
import argparse
import re

import eval_all_miou_multiseg as base
from pred_parsers import get_parser, parse_tokens, _fix, _hms_to_sec


# ---------------------------------------------------------------- salmonn(개선)
# 공유 pred_parsers.parse_salmonn 은 "M:SS to M:SS" 만 잡아 salmonn2p 자연어 출력의
# 상당수(dash/정수초/first N seconds)를 놓친다. reeval 호환을 위해 공유 파서는 두고
# 여기서 확장 파서를 둔다. 복구 대상(명확한 range)만 파싱:
#   "0:12-0:14" / "0:14 to 0:25"  (M:SS, 구분자 to|-|–|—|~)
#   "from 10 to 20 seconds" / "10-20 seconds"   (정수/소수 초)
#   "first 10 seconds" -> [0,10]
# 단일 시점("at 0:28" 등)은 GT 가 range 라 IoU≈0 이고 의미가 모호해 의도적으로 제외.
_HMS = r"(?:\d+:)?\d{1,2}:\d{2}"


def parse_salmonn_plus(pred, duration=None):
    text = pred or ""
    consumed = [False] * len(text)
    segs = []

    def grab(pattern, conv):
        for m in re.finditer(pattern, text, re.IGNORECASE):
            a, b = m.span()
            if any(consumed[a:b]):
                continue
            r = conv(m)
            if r is not None:
                segs.append(r)
                for i in range(a, b):
                    consumed[i] = True

    # 구체적인 패턴부터 (consumed 마스킹으로 중복 매칭 방지)
    grab(rf"({_HMS})\s*(?:to|-|–|—|~)\s*({_HMS})",
         lambda m: _fix(_hms_to_sec(m.group(1)), _hms_to_sec(m.group(2))))
    grab(r"from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*(?:seconds|secs|sec)\b",
         lambda m: _fix(float(m.group(1)), float(m.group(2))))
    grab(r"\b(\d+(?:\.\d+)?)\s*(?:to|-|–|—)\s*(\d+(?:\.\d+)?)\s*(?:seconds|secs|sec)\b",
         lambda m: _fix(float(m.group(1)), float(m.group(2))))
    grab(r"first\s+(\d+(?:\.\d+)?)\s*(?:seconds|secs|sec)\b",
         lambda m: _fix(0.0, float(m.group(1))))
    return segs


# 모델키 → 파서/GT 소스 설정. parser 가 명시되면 그 함수를, 아니면 pred_parsers 사용.
MODEL_CONFIGS = {
    "arc":     {"pred_format": "arc_hunyuan", "gt_source": "embedded"},
    "avicuna": {"pred_format": "avicuna",     "gt_source": "embedded"},
    "museg":   {"pred_format": "museg",       "gt_source": "embedded"},
    "chronus": {"pred_format": "chronusomni", "gt_source": "embedded"},
    "salmonn": {"pred_format": "salmonn",     "gt_source": "ref", "parser": parse_salmonn_plus},
}

# pred_format 이름 → 로컬 개선 파서 (있으면 pred_parsers 보다 우선)
LOCAL_PARSERS = {"salmonn": parse_salmonn_plus}

# 경로 키워드(소문자) → 모델키  (순서 = 우선순위)
_DETECT = [
    ("archunyuan", "arc"), ("arc_hunyuan", "arc"), ("hunyuan", "arc"), ("arc", "arc"),
    ("avicuna", "avicuna"),
    ("museg", "museg"),
    ("chronus", "chronus"),
    ("salmonn", "salmonn"),
]


def detect_model(path):
    p = path.lower()
    for kw, key in _DETECT:
        if kw in p:
            return key
    return None


def extract_gt(result, gt_source):
    """GT 세그먼트 [[s,e],...] 추출."""
    if gt_source == "embedded":
        return base._coerce_segments(result.get("gt_segments"))
    if gt_source == "ref":
        return parse_tokens(result.get("ref", "") or "")
    raise ValueError(f"bad gt_source '{gt_source}' (embedded|ref)")


def resolve_parser(pred_format):
    """로컬 개선 파서 우선, 없으면 공유 pred_parsers."""
    return LOCAL_PARSERS.get(pred_format) or get_parser(pred_format)


def build_samples(results, pred_format, gt_source, duration_key="duration"):
    """results → ([(gt_segs, pred_segs)], matched, unmatched)."""
    parse_pred = resolve_parser(pred_format)
    samples, matched, unmatched = [], 0, 0
    for r in results:
        dur = r.get(duration_key)
        pred_segs = parse_pred(r.get("pred", "") or "", dur)
        gt_segs = extract_gt(r, gt_source)
        if gt_segs:
            matched += 1
        else:
            unmatched += 1
        samples.append((gt_segs, pred_segs))
    return samples, matched, unmatched


def evaluate_one(results_file, args):
    results = base._load_json(results_file)

    model_key = args.model or detect_model(results_file)
    cfg = MODEL_CONFIGS.get(model_key, {})
    pred_format = args.pred_format or cfg.get("pred_format")
    gt_source = args.gt_source or cfg.get("gt_source")
    if not pred_format or not gt_source:
        raise ValueError(
            f"모델 자동 감지 실패 (path={results_file}). "
            f"--model {{{'/'.join(MODEL_CONFIGS)}}} 또는 --pred_format/--gt_source 를 지정하세요."
        )

    samples, matched, unmatched = build_samples(
        results, pred_format, gt_source, args.duration_key)

    summary = base.compute_core_summary(samples)
    summary["pred_format"] = pred_format
    summary["gt_source"] = gt_source
    summary["gt_matched"] = matched
    summary["gt_unmatched"] = unmatched

    base.write_and_report(results_file, summary, args)
    return summary


def main():
    ap = argparse.ArgumentParser(
        description="outputs/base 이종 포맷 모델에 대해 All/Best/Union mIoU 계산")
    ap.add_argument("results_dir",
                    help="출력 폴더 (하위 test_results_rank0.json 재귀 탐색) 또는 단일 json 경로")
    ap.add_argument("--model", default=None,
                    choices=list(MODEL_CONFIGS) + [None],
                    help="모델 강제 지정 (생략 시 경로 키워드로 자동 감지)")
    ap.add_argument("--pred_format", default=None,
                    help="pred 파서 강제 지정 (pred_parsers.PARSERS 키)")
    ap.add_argument("--gt_source", default=None, choices=["embedded", "ref", None],
                    help="GT 출처 강제 지정 (embedded=gt_segments / ref=토큰 ref)")
    ap.add_argument("--duration_key", default="duration",
                    help="avicuna(%%) 변환용 duration 필드명")
    ap.add_argument("--max_time", type=float, default=9999.9)
    ap.add_argument("--progress_log", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    results_files = base.find_results_files(args.results_dir)
    if not results_files:
        raise SystemExit(f"[에러] test_results_rank0.json 을 찾지 못함: {args.results_dir}")

    if not args.quiet:
        print(f"[INFO] {len(results_files)} 개의 test_results_rank0.json 발견")

    for rf in results_files:
        try:
            evaluate_one(rf, args)
        except Exception as e:  # noqa: BLE001
            print(f"[SKIP] {rf}: {e}")


if __name__ == "__main__":
    main()
