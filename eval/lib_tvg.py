"""
lib_tvg.py — Team4 temporal video grounding 분석 공통 라이브러리 (2026-05-29)

목적:
    /workspace/outputs/gdpo 안의 두 RL 모델(F1, rM) 추론 결과를 분석하기 위한
    공통 함수 모음. 평가 원본 로직(/workspace/hyj/Team4/eval/eval_miou.py,
    /workspace/tools/eval/analyze_segments.py)을 그대로 재현하여, 분석 수치가
    eval_miou_summary.json / segment_analysis.json 과 일치하도록 한다.

핵심 규칙(원본 eval에서 가져옴):
    - 토큰 파싱: "<t0><t3><t5><tdot><t9>" → 정수부 digit 이어붙이기 + 소수부 첫 digit/10
                 즉 (0*100? 아니라 "035"=35) + 9/10 = 35.9.  -> int("".join(int_digits)) + dec[0]/10
    - IoU(tIoU): inter/union, union<=0 이면 0
    - mIoU: 각 GT segment마다 (같은 라벨) pred 중 best IoU → 모든 GT segment 평균
            (sample 평균이 아니라 GT-segment 평균; n_gt_segments=844 기준)
    - R@1@θ: best_iou>=θ 인 GT segment 비율
    - max_time clamp = 60.0 (원본 decode_vtg_time default)

이 파일은 단독 실행하지 않고 import 해서 쓴다.
"""
import re

MAX_TIME = 60.0  # 원본 eval_miou.py decode_vtg_time default


def decode_vtg_time(token_str, max_time=MAX_TIME):
    """단일 시각 토큰열을 초로 디코드. 원본 eval_miou.py:53-67 재현.
    반환: float 초, 파싱 실패 시 None."""
    has_dot = "<tdot>" in token_str
    if has_dot:
        parts = token_str.split("<tdot>")
        int_part = re.findall(r"<t(\d)>", parts[0])
        dec_part = re.findall(r"<t(\d)>", parts[1]) if len(parts) > 1 else []
    else:
        int_part = re.findall(r"<t(\d)>", token_str)
        dec_part = []
    if not int_part:
        return None
    integer_part = int("".join(int_part))
    decimal_part = int(dec_part[0]) if dec_part else 0
    t = integer_part + decimal_part / 10.0
    return min(t, max_time)


# "From <...> to <...>." 한 세그먼트를 잡는 정규식. 여러 개면 반복 매칭.
_SEG_RE = re.compile(r"From\s*(.*?)\s*to\s*(.*?)\s*\.", re.IGNORECASE | re.DOTALL)


def parse_pred_segments(pred_text, max_time=MAX_TIME):
    """예측 텍스트에서 (start,end) 세그먼트 리스트 추출.
    멀티세그는 'From A to B. From C to D.' 형태를 모두 잡는다.
    각 raw (start_str,end_str)도 함께 반환하여 형식 분석에 사용.
    반환: list of dict {start,end,start_str,end_str, ok(bool)}"""
    out = []
    for m in _SEG_RE.finditer(pred_text):
        s_str, e_str = m.group(1), m.group(2)
        s = decode_vtg_time(s_str, max_time)
        e = decode_vtg_time(e_str, max_time)
        ok = (s is not None) and (e is not None)
        out.append({
            "start": s, "end": e,
            "start_str": s_str.strip(), "end_str": e_str.strip(),
            "ok": ok,
        })
    return out


def clean_segments(seglist):
    """파싱된 세그먼트 중 유효(ok, start<=end)한 것만 (s,e) 튜플로.
    start>end 면 swap 하지 않고 버린다(형식 오류로 카운트하려고 원본은 그대로 둘 수 있으나
    IoU 계산엔 유효 구간만 사용). 반환: list[(s,e)]"""
    res = []
    for g in seglist:
        if g["ok"] and g["start"] is not None and g["end"] is not None:
            if g["end"] >= g["start"]:
                res.append((g["start"], g["end"]))
            else:
                # start>end: 버리지 않고 swap 하면 IoU에 유리해지므로 원본 의도 보존 위해 그대로 둠
                res.append((g["start"], g["end"]))
    return res


def tiou(seg1, seg2):
    """temporal IoU. 원본 eval_miou.py:217-222 재현."""
    inter_s = max(seg1[0], seg2[0])
    inter_e = min(seg1[1], seg2[1])
    inter = max(0.0, inter_e - inter_s)
    union = (seg1[1] - seg1[0]) + (seg2[1] - seg2[0]) - inter
    return inter / (union + 1e-8) if union > 0 else 0.0


def best_ious_for_sample(gt_segs, pred_segs):
    """한 sample에서 각 GT segment의 best IoU 리스트 반환.
    pred 없으면 모든 GT는 0.0. (라벨은 sample당 하나이므로 라벨 필터 불필요)"""
    res = []
    for g in gt_segs:
        if not pred_segs:
            res.append(0.0)
        else:
            res.append(max(tiou(p, g) for p in pred_segs))
    return res


def seg_len(seg):
    return max(0.0, seg[1] - seg[0])


def total_len(segs):
    return sum(seg_len(s) for s in segs)
