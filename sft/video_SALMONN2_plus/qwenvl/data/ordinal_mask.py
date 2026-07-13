# Copyright (2026)
# Ordinal loss 를 위한 마스크 생성 헬퍼.
#
# input_ids/labels 에서 time-digit 토큰 위치를 식별하고, source(puvalor/unav) x
# task(T1/T2/T3) 정책에 따라 ordinal loss 적용 위치를 결정한다.
#
# 핵심 정책:
#   - T3 (segment-conditioned captioning) 는 GPT 응답에 time token 없음 → mask 전부 0
#   - PU-VALOR: 정수부 자리만 ordinal loss (decimal slot 학습 신호 zero)
#   - UnAV:     모든 자리 ordinal loss (소수부까지 학습 신호 있음)
#   - assistant turn 만 대상 (labels != -100)
#
# anchor: <tdot> 위치. 좌측 ndig_int 자리 = j=1..ndig_int (정수, MSB→LSB),
#         우측 ndig_dec 자리 = j=ndig_int+1..ndig_int+ndig_dec (소수)

import torch


def build_time_token_maps(tokenizer):
    """tokenizer 에서 <t0>..<t9>, <tdot> 의 id 매핑 생성.
    Returns:
      time_token_id_map: {token_id: digit_value (0..9)}
      tdot_id: int or -1 if not found
    """
    time_token_id_map = {}
    for d in range(10):
        tid = tokenizer.convert_tokens_to_ids(f"<t{d}>")
        if tid is not None and tid >= 0:
            time_token_id_map[tid] = d
    tdot_id = tokenizer.convert_tokens_to_ids("<tdot>")
    if tdot_id is None:
        tdot_id = -1
    return time_token_id_map, tdot_id


def make_ordinal_mask(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    source_id: str,
    task_type: str,
    ndig_int: int,
    ndig_dec: int,
    time_token_id_map: dict,
    tdot_id: int,
):
    """1-D input_ids/labels 에 대해 Ordinal loss 적용 위치를 식별.

    Args:
      input_ids: LongTensor [L]
      labels:    LongTensor [L] (assistant turn 에서만 != -100)
      source_id: "puvalor" or "unav" (그 외는 mask 전부 0)
      task_type: "T1" / "T2" / "T3" / ...
      ndig_int / ndig_dec: 토큰화 스킴 (e.g. 3, 2 for 3+2)
      time_token_id_map: {token_id: digit_value 0..9}
      tdot_id: <tdot> token id

    Returns:
      ordinal_mask: BoolTensor [L]  — Ordinal loss 대상 위치
      digit_target: LongTensor [L]  — masked 위치의 digit (0..9), 나머지 -100
      digit_place:  LongTensor [L]  — masked 위치의 j (1..ndig_int+ndig_dec), 나머지 -1
    """
    L = input_ids.shape[0]
    ordinal_mask = torch.zeros(L, dtype=torch.bool)
    digit_target = torch.full((L,), -100, dtype=torch.long)
    digit_place = torch.full((L,), -1, dtype=torch.long)

    # T3 또는 알 수 없는 source/task → mask 전부 0
    if task_type == "T3":
        return ordinal_mask, digit_target, digit_place
    if tdot_id < 0 or not time_token_id_map:
        return ordinal_mask, digit_target, digit_place

    total_digits = ndig_int + ndig_dec
    if source_id == "puvalor":
        active_j = set(range(1, ndig_int + 1))  # 정수부만
    elif source_id == "unav":
        active_j = set(range(1, total_digits + 1))  # 전부
    else:
        return ordinal_mask, digit_target, digit_place

    tdot_positions = (input_ids == tdot_id).nonzero(as_tuple=True)[0].tolist()
    if not tdot_positions:
        return ordinal_mask, digit_target, digit_place

    labels_l = labels.tolist()
    input_l = input_ids.tolist()

    for p in tdot_positions:
        for j in range(1, total_digits + 1):
            if j not in active_j:
                continue
            # 정수부: j=1 → offset=-ndig_int, j=ndig_int → offset=-1
            # 소수부: j=ndig_int+1 → offset=+1, j=total → offset=+ndig_dec
            if j <= ndig_int:
                offset = -(ndig_int - j + 1)
            else:
                offset = j - ndig_int
            pos = p + offset
            if pos < 0 or pos >= L:
                continue
            if labels_l[pos] == -100:
                continue
            tid = input_l[pos]
            digit_val = time_token_id_map.get(tid)
            if digit_val is None:
                continue
            ordinal_mask[pos] = True
            digit_target[pos] = digit_val
            digit_place[pos] = j

    return ordinal_mask, digit_target, digit_place
