#!/usr/bin/env python3
"""
v6_b 학습 데이터 생성:
  T1+T2 zero-start (<s0>=0) 샘플을 다운샘플링하여 zero:nonzero = 1:3 비율 (≈25/75)로 맞춤.
  T3 (inverse) 는 모델이 timestamp 를 출력하지 않으므로 영향 없음 → 전부 유지.

Input:  puvalor_train_v5_tok3_1.json (90,972)
Output: puvalor_train_v6b_tok3_1.json (≈79,481 expected)

Seed: 42 (재현 가능)
"""
import json
import random
from collections import Counter

SRC = "/home/aix23102/audiolm/vS2_eunji/data/puvalor_train_v5_tok3_1.json"
DST = "/home/aix23102/audiolm/vS2_eunji/data/puvalor_train_v6b_tok3_1.json"
TARGET_ZERO_RATIO = 0.25  # zero_start / (zero_start + nonzero) target = 25%
SEED = 42

def first_start_seconds(sample):
    tk = sample.get("meta", {}).get("token", {})
    if "<s0>" not in tk:
        return None
    return float(tk["<s0>"])

def main():
    random.seed(SEED)
    data = json.load(open(SRC))
    print(f"input: {len(data)} samples")

    # bucket
    t3_keep = []      # T3: keep all
    t12_zero = []     # T1/T2 with first start = 0
    t12_nonzero = []  # T1/T2 with first start > 0
    other = []        # safety bucket

    for s in data:
        task = s.get("task_type", "?")
        st = first_start_seconds(s)
        if task == "T3":
            t3_keep.append(s)
        elif task in ("T1", "T2"):
            if st is None:
                other.append(s)
            elif st == 0.0:
                t12_zero.append(s)
            else:
                t12_nonzero.append(s)
        else:
            other.append(s)

    print(f"  T3 (keep all): {len(t3_keep)}")
    print(f"  T1+T2 first_start=0: {len(t12_zero)}")
    print(f"  T1+T2 first_start>0: {len(t12_nonzero)}")
    print(f"  other (no token map / unknown task): {len(other)}")

    # target keep size for zero-start to achieve TARGET_ZERO_RATIO
    # zero / (zero + nonzero) = TARGET_ZERO_RATIO
    # zero = TARGET_ZERO_RATIO * nonzero / (1 - TARGET_ZERO_RATIO)
    n_keep_zero = int(round(TARGET_ZERO_RATIO * len(t12_nonzero) / (1 - TARGET_ZERO_RATIO)))
    n_keep_zero = min(n_keep_zero, len(t12_zero))
    print(f"\n  target zero ratio: {TARGET_ZERO_RATIO*100:.1f}%")
    print(f"  keep zero-start: {n_keep_zero} (drop {len(t12_zero) - n_keep_zero})")

    random.shuffle(t12_zero)
    kept_zero = t12_zero[:n_keep_zero]

    out = t3_keep + t12_nonzero + kept_zero + other
    random.shuffle(out)
    print(f"\noutput: {len(out)} samples")
    # verify
    z = n = 0
    for s in out:
        if s.get("task_type") in ("T1","T2"):
            st = first_start_seconds(s)
            if st == 0.0: z += 1
            elif st is not None: n += 1
    print(f"  T1+T2 zero/nonzero in output: {z}/{n} (ratio = {z/(z+n)*100:.1f}%)")
    print(f"  T3 in output: {len(t3_keep)}")
    # task type
    print(f"  task counts: {Counter(s.get('task_type','?') for s in out)}")

    json.dump(out, open(DST, "w"), ensure_ascii=False)
    print(f"\nsaved → {DST}")

if __name__ == "__main__":
    main()
