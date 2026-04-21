#!/usr/bin/env python3
"""
[01] 체크포인트 토크나이저 / config 검증
----------------------------------------
Pre-baked 체크포인트가 TTI 경로에 쓸 준비가 됐는지 확인한다.

검사 항목:
  (A) <t0>~<t9>, <tdot> 11개가 각각 단일 ID 로 인코딩되는가
  (B) 11개 ID 가 연속 구간(예: 151666~151676) 을 형성하는가
      → 연속이어야 rope2d 의 (lo, hi) 튜플 기반 범위 비교가 성립
  (C) config.json 의 time_token_id_range 가 (B) 의 (lo, hi) 와 일치하는가
      → modeling.py 의 get_rope_index shim 이 config 값을 그대로 꺼내 쓰기 때문

사용:
  python _tools/tti/01_check_tokenizer_ids.py --model_path /path/to/checkpoint

종료 코드: 0=PASS, 1=FAIL
"""
import argparse
import json
import os
import sys

from transformers import AutoTokenizer

VTG_TIME_TOKENS = [f"<t{i}>" for i in range(10)] + ["<tdot>"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True,
                    help="add_time_tokens_salmonn2plus.py 로 구워진 체크포인트 경로")
    args = ap.parse_args()

    all_ok = True

    # --- (A) 11개 토큰이 단일 ID 로 인코딩되는가 ---
    print(f"[A] loading tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print(f"  vocab size: {len(tokenizer)}")

    ids = {}
    for tok in VTG_TIME_TOKENS:
        enc = tokenizer.encode(tok, add_special_tokens=False)
        if len(enc) != 1:
            print(f"  FAIL  {tok} -> {enc} (expected single ID)")
            all_ok = False
        else:
            ids[tok] = enc[0]
            print(f"  OK    {tok:<8s} -> {enc[0]}")

    if not all_ok:
        print("\n토큰 등록이 안 돼 있으면 _tools/sft/add_time_tokens_salmonn2plus.py 먼저 실행.")
        sys.exit(1)

    # --- (B) ID 구간이 연속인가 ---
    id_list = sorted(ids.values())
    lo, hi = id_list[0], id_list[-1]
    expected_contiguous = list(range(lo, hi + 1))
    is_contig = id_list == expected_contiguous and len(id_list) == 11
    print(f"\n[B] ID 구간: [{lo}, {hi}], 개수={len(id_list)}, 연속={is_contig}")
    if not is_contig:
        print(f"  FAIL  연속이 아님. rope2d 의 (lo, hi) 범위 비교가 깨짐.")
        all_ok = False
    else:
        print("  OK    연속 구간 확인 — rope2d (lo, hi) 튜플 전달 방식 유효")

    # --- (C) config.time_token_id_range 와 일치하는가 ---
    cfg_path = os.path.join(args.model_path, "config.json")
    print(f"\n[C] checking {cfg_path}")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg_range = cfg.get("time_token_id_range", None)
    print(f"  config.time_token_id_range = {cfg_range}")
    if cfg_range is None:
        print("  FAIL  config.time_token_id_range 없음.")
        print("        해결: add_time_tokens_salmonn2plus.py 의 최신본으로 다시 굽거나,")
        print(f"        config.json 에 직접 \"time_token_id_range\": [{lo}, {hi}] 추가.")
        all_ok = False
    elif list(cfg_range) != [lo, hi]:
        print(f"  FAIL  tokenizer ID 범위와 불일치: tokenizer=[{lo},{hi}] vs cfg={cfg_range}")
        all_ok = False
    else:
        print("  OK    tokenizer 범위와 config 범위 일치")

    print("\n" + "=" * 60)
    print("RESULT:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
