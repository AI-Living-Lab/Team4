#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pred_parsers.py — outputs/base 의 이종(異種) 모델 출력 포맷 → (start,end) 초 세그먼트.

각 base 모델은 추론 출력 포맷이 제각각이라 Team4 기본 파서
(`From <t..> to <t..>` SALMONN 토큰)로는 파싱이 안 된다. 이 모듈은 모델별
파서를 한 곳에 모아 reeval_from_results.py 가 골라 쓰도록 한다.

모든 파서 시그니처: parse(pred_text: str, duration: float|None) -> list[[s,e]] (초)
  - duration: 퍼센트 기반(avicuna) 변환에만 필요. 그 외는 무시.
  - 잘못된 구간(e<=s)은 보수적으로 최소 0.1s 만 부여(점수 이득 최소화), Team4 파서와 동일.

지원 포맷(2026-06-03 outputs/base 관측):
  arc_hunyuan : <answer><span>HH:MM:SS - HH:MM:SS</span>...</answer>  (절대 시:분:초)
  avicuna     : "... from XX to YY ..."                              (영상 길이 대비 %)
  chronusomni : "From second{X} to second{Y}" / "second{X}-second{Y}"(절대 초, 멀티 가능)
  museg       : "X.XX-X.XX" (prose 안에 포함 가능, 멀티 가능)        (절대 초)
  salmonn     : "... from M:SS to M:SS, M:SS to M:SS ..."            (자연어 M:SS, 멀티 가능)
  tokens      : "From <t..> to <t..>. ..." (SALMONN 토큰; salmonn GT(ref)용)
"""
import re

MAX_TIME = 9999.9  # 초 clamp 상한 (unav100 일부 영상 60s 초과 → 낮게 자르면 안 됨)


def _fix(s, e):
    if e <= s:
        e = min(s + 0.1, MAX_TIME)
    return [min(s, MAX_TIME), min(e, MAX_TIME)]


# ---------------------------------------------------------------- arc_hunyuan
_ARC_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_ARC_SPAN = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2}):(\d{2})"
)


def parse_arc_hunyuan(pred, duration=None):
    text = pred or ""
    m = _ARC_ANSWER.search(text)
    scope = m.group(1) if m else text  # answer 태그 우선, 없으면 전체
    out = []
    for h1, m1, s1, h2, m2, s2 in _ARC_SPAN.findall(scope):
        s = int(h1) * 3600 + int(m1) * 60 + int(s1)
        e = int(h2) * 3600 + int(m2) * 60 + int(s2)
        out.append(_fix(float(s), float(e)))
    return out


# ---------------------------------------------------------------- avicuna (%)
_AVICUNA = re.compile(r"from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_avicuna(pred, duration=None):
    if duration is None:
        return []
    out = []
    for a, b in _AVICUNA.findall(pred or ""):
        s = float(a) / 100.0 * float(duration)
        e = float(b) / 100.0 * float(duration)
        out.append(_fix(s, e))
    return out


# ---------------------------------------------------------------- chronusomni
_CHRONUS = re.compile(
    r"second\{([\d.]+)\}\s*(?:to|-)\s*second\{([\d.]+)\}", re.IGNORECASE
)


def parse_chronusomni(pred, duration=None):
    out = []
    for a, b in _CHRONUS.findall(pred or ""):
        out.append(_fix(float(a), float(b)))
    return out


# ---------------------------------------------------------------- museg (초)
_MUSEG = re.compile(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)")


def parse_museg(pred, duration=None):
    out = []
    for a, b in _MUSEG.findall(pred or ""):
        out.append(_fix(float(a), float(b)))
    return out


# ---------------------------------------------------------------- salmonn M:SS
# "0:00 to 0:28", "1:07 to 1:10", "1:23:45 to 1:24:00" 등. 콤마 나열 멀티 지원.
_HMS = r"(?:\d+:)?\d{1,2}:\d{2}"
_SALMONN = re.compile(rf"({_HMS})\s*to\s*({_HMS})", re.IGNORECASE)


def _hms_to_sec(tok):
    parts = [int(x) for x in tok.split(":")]
    s = 0
    for p in parts:
        s = s * 60 + p
    return float(s)


def parse_salmonn(pred, duration=None):
    out = []
    for a, b in _SALMONN.findall(pred or ""):
        out.append(_fix(_hms_to_sec(a), _hms_to_sec(b)))
    return out


# ---------------------------------------------------------------- tokens (GT)
_TOK_SEG = re.compile(
    r"[Ff]rom\s+((?:<t\d>)+(?:<tdot>(?:<t\d>)+)?)\s+to\s+((?:<t\d>)+(?:<tdot>(?:<t\d>)+)?)"
)


def _decode_tok(token_str):
    if "<tdot>" in token_str:
        a, _, b = token_str.partition("<tdot>")
        ip = re.findall(r"<t(\d)>", a)
        dp = re.findall(r"<t(\d)>", b)
    else:
        ip = re.findall(r"<t(\d)>", token_str)
        dp = []
    if not ip:
        return None
    return min(int("".join(ip)) + (int(dp[0]) / 10.0 if dp else 0.0), MAX_TIME)


def parse_tokens(pred, duration=None):
    out = []
    for sa, sb in _TOK_SEG.findall(pred or ""):
        s, e = _decode_tok(sa), _decode_tok(sb)
        if s is None or e is None:
            continue
        out.append(_fix(s, e))
    return out


PARSERS = {
    "arc_hunyuan": parse_arc_hunyuan,
    "avicuna": parse_avicuna,
    "chronusomni": parse_chronusomni,
    "museg": parse_museg,
    "salmonn": parse_salmonn,
    "tokens": parse_tokens,
}


def get_parser(name):
    if name not in PARSERS:
        raise ValueError(f"unknown pred_format '{name}'. choices={list(PARSERS)}")
    return PARSERS[name]
