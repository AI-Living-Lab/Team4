#!/usr/bin/env python3
"""
타임토큰이 추가된 체크포인트의 토크나이저를 검증.
- <t0>~<t9>, <tdot> 11개가 각각 단일 ID로 인코딩되는지
- sec_to_time_token_str() 결과가 정확히 6 토큰으로 분해되는지
- 대표 시각 값(0.0, 1.2, 59.9, 9999.9)의 ID 시퀀스를 출력
"""
import argparse
import sys
from transformers import AutoTokenizer


VTG_TIME_TOKENS = [f"<t{i}>" for i in range(10)] + ["<tdot>"]


def sec_to_time_token_str(sec: float) -> str:
    # dataset.py의 헬퍼와 동일한 로직. 범위 0.0~9999.9, 정밀도 0.1초.
    sec = max(0.0, min(9999.9, sec))
    tenths = round(sec * 10)
    i = tenths // 10
    f = tenths % 10
    d1, d2, d3, d4 = (i // 1000) % 10, (i // 100) % 10, (i // 10) % 10, i % 10
    return f"<t{d1}><t{d2}><t{d3}><t{d4}><tdot><t{f}>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="타임토큰이 추가된 체크포인트 경로")
    args = ap.parse_args()

    print(f"[1/4] Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print(f"  Vocab size: {len(tokenizer)}")

    print(f"\n[2/4] 각 타임토큰이 단일 ID로 인코딩되는지 확인")
    all_ok = True
    token_ids = {}
    for tok in VTG_TIME_TOKENS:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        status = "OK" if len(ids) == 1 else "FAIL"
        if len(ids) != 1:
            all_ok = False
        token_ids[tok] = ids[0] if len(ids) == 1 else ids
        print(f"  {tok:<8} -> {ids}  [{status}]")

    if not all_ok:
        print("\n[ERROR] 일부 토큰이 단일 ID로 인코딩되지 않았습니다.")
        print("        add_time_tokens_salmonn2plus.py를 먼저 실행했는지 확인하세요.")
        sys.exit(1)

    print(f"\n[3/4] sec_to_time_token_str 출력 검증")
    test_cases = [0.0, 0.1, 1.2, 12.3, 59.9, 123.4, 9999.9, 99999.0, -1.0]
    expected_id_count = 6
    for sec in test_cases:
        s = sec_to_time_token_str(sec)
        ids = tokenizer.encode(s, add_special_tokens=False)
        status = "OK" if len(ids) == expected_id_count else "FAIL"
        if len(ids) != expected_id_count:
            all_ok = False
        print(f"  {sec:>8.1f}s -> {s!r:<50} ids={ids} [{status}]")

    print(f"\n[4/4] 대표 시퀀스의 decode 확인 (1.2초)")
    s = sec_to_time_token_str(1.2)
    ids = tokenizer.encode(s, add_special_tokens=False)
    decoded = tokenizer.decode(ids)
    print(f"  encoded:  {s}")
    print(f"  ids:      {ids}")
    print(f"  decoded:  {decoded}")
    round_trip_ok = decoded.replace(" ", "") == s.replace(" ", "")
    print(f"  round-trip: {'OK' if round_trip_ok else 'FAIL'}")
    if not round_trip_ok:
        all_ok = False

    print(f"\n{'=' * 60}")
    print("RESULT:", "ALL PASSED" if all_ok else "FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
