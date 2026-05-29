"""Compare LoRA adapter (+ q_tokens) between checkpoint-10612 and checkpoint-12128
of salmonn2plus_puvalor_v4 to quantify per-module drift between epoch 0.7 and 0.8.

Per tensor reports:
  - ||A||_F, ||B||_F  : Frobenius norms
  - ||B - A||_F       : drift norm
  - rel_drift = ||B-A||/||A||  : relative drift
  - cos = <A, B> / (||A|| ||B||)  : cosine similarity (1.0 = identical)

Per group: weighted-mean rel_drift (weight = ||A||), mean cos, max rel_drift, and
total ||B-A|| / total ||A|| (norm-weighted) so a few large deltas don't get
diluted by many tiny tensors.

For embed_tokens / lm_head, only 11 time-token rows are actually trained, so
drift is also reported on those rows alone (the other ~152k rows are frozen
in this run, so their drift should be ~0).
"""

import json
import math
import os
from collections import defaultdict

import torch
from safetensors import safe_open

V4 = "/data0/aix23102/checkpoints_open_aligner/salmonn2plus_puvalor_v4"
CKPT_A = f"{V4}/checkpoint-10612"
CKPT_B = f"{V4}/checkpoint-12128"
MERGED_A = f"{CKPT_A}_merged"
MERGED_B = f"{CKPT_B}_merged"

TIME_TOKEN_IDS = list(range(151666, 151677))  # <t0>..<t9>, <tdot>


def classify(key: str) -> str:
    if "lora_A" in key or "lora_B" in key:
        if ".q_proj." in key:
            return "lora.q_proj"
        if ".k_proj." in key:
            return "lora.k_proj"
        if ".v_proj." in key:
            return "lora.v_proj"
        if ".o_proj." in key:
            return "lora.o_proj"
        return "lora.other"
    if "embed_tokens" in key:
        return "embed_tokens"
    if "lm_head" in key:
        return "lm_head"
    if ".visual." in key and ".merger." in key:
        return "visual.merger"
    if ".audio." in key:
        if "qformer" in key:
            return "audio.qformer"
        if "audio_proj" in key:
            return "audio.audio_proj"
        if "q_tokens" in key:
            return "audio.q_tokens"
    return "other"


def tensor_stats(a: torch.Tensor, b: torch.Tensor):
    a = a.to(torch.float32).flatten()
    b = b.to(torch.float32).flatten()
    nA = float(torch.linalg.norm(a))
    nB = float(torch.linalg.norm(b))
    nD = float(torch.linalg.norm(b - a))
    rel = nD / (nA + 1e-12)
    cos = float(torch.dot(a, b) / (nA * nB + 1e-12))
    return nA, nB, nD, rel, cos


def main():
    PATH_A = f"{CKPT_A}/adapter_model.safetensors"
    PATH_B = f"{CKPT_B}/adapter_model.safetensors"

    print(f"A = {PATH_A}")
    print(f"B = {PATH_B}")

    rows = []  # per-tensor stats
    with safe_open(PATH_A, framework="pt") as fa, safe_open(PATH_B, framework="pt") as fb:
        keys_a = set(fa.keys())
        keys_b = set(fb.keys())
        common = sorted(keys_a & keys_b)
        only_a = sorted(keys_a - keys_b)
        only_b = sorted(keys_b - keys_a)
        print(f"keys: A={len(keys_a)}  B={len(keys_b)}  common={len(common)}  only_A={len(only_a)}  only_B={len(only_b)}")
        if only_a:
            print("  only_A sample:", only_a[:3])
        if only_b:
            print("  only_B sample:", only_b[:3])

        for k in common:
            ta = fa.get_tensor(k)
            tb = fb.get_tensor(k)
            if ta.shape != tb.shape:
                print(f"[shape mismatch] {k}: {ta.shape} vs {tb.shape}")
                continue
            nA, nB, nD, rel, cos = tensor_stats(ta, tb)
            rows.append({
                "key": k, "group": classify(k), "shape": list(ta.shape),
                "n": int(ta.numel()),
                "normA": nA, "normB": nB, "drift": nD,
                "rel_drift": rel, "cos_sim": cos,
            })
            # Time-token row drift for embed/lm_head
            if "embed_tokens" in k or "lm_head" in k:
                if ta.dim() == 2 and ta.shape[0] >= 151677:
                    a_tt = ta[TIME_TOKEN_IDS].to(torch.float32)
                    b_tt = tb[TIME_TOKEN_IDS].to(torch.float32)
                    nAt = float(torch.linalg.norm(a_tt))
                    nBt = float(torch.linalg.norm(b_tt))
                    nDt = float(torch.linalg.norm(b_tt - a_tt))
                    rowt = {
                        "key": k + "[time_tokens]", "group": classify(k) + "[time_tokens]",
                        "shape": [11, ta.shape[1]], "n": 11 * ta.shape[1],
                        "normA": nAt, "normB": nBt, "drift": nDt,
                        "rel_drift": nDt / (nAt + 1e-12),
                        "cos_sim": float(torch.dot(a_tt.flatten(), b_tt.flatten()) / (nAt * nBt + 1e-12)),
                    }
                    rows.append(rowt)
                    # Also: which non-time rows had any drift? (sanity — should be exactly 0)
                    mask = torch.ones(ta.shape[0], dtype=torch.bool)
                    mask[TIME_TOKEN_IDS] = False
                    other_diff = (ta[mask].to(torch.float32) - tb[mask].to(torch.float32))
                    other_norm = float(torch.linalg.norm(other_diff))
                    print(f"  [{k}] non-time-row drift L2 = {other_norm:.6e}  (should be ~0)")

    # ---- q_tokens from merged shard 1 ----
    A_SHARD = f"{MERGED_A}/model-00001-of-00004.safetensors"
    B_SHARD = f"{MERGED_B}/model-00001-of-00004.safetensors"
    if os.path.exists(A_SHARD) and os.path.exists(B_SHARD):
        with safe_open(A_SHARD, framework="pt") as fa, safe_open(B_SHARD, framework="pt") as fb:
            if "audio.q_tokens" in fa.keys() and "audio.q_tokens" in fb.keys():
                ta = fa.get_tensor("audio.q_tokens")
                tb = fb.get_tensor("audio.q_tokens")
                nA, nB, nD, rel, cos = tensor_stats(ta, tb)
                rows.append({
                    "key": "audio.q_tokens", "group": "audio.q_tokens",
                    "shape": list(ta.shape), "n": int(ta.numel()),
                    "normA": nA, "normB": nB, "drift": nD,
                    "rel_drift": rel, "cos_sim": cos,
                })

    # ---- aggregate by group ----
    by_group = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)

    print()
    print("=" * 92)
    print(f"{'group':30s} {'#tensors':>9s} {'norm-wt rel_drift':>17s} {'mean cos':>10s} {'min cos':>9s} {'max rel':>9s}")
    print("=" * 92)
    summary = {}
    for g in sorted(by_group.keys()):
        lst = by_group[g]
        sumA = math.sqrt(sum(r["normA"] ** 2 for r in lst))
        sumD = math.sqrt(sum(r["drift"] ** 2 for r in lst))
        nw_rel = sumD / (sumA + 1e-12)
        mean_cos = sum(r["cos_sim"] for r in lst) / len(lst)
        min_cos = min(r["cos_sim"] for r in lst)
        max_rel = max(r["rel_drift"] for r in lst)
        print(f"{g:30s} {len(lst):>9d} {nw_rel*100:>15.4f}% {mean_cos:>10.6f} {min_cos:>9.6f} {max_rel*100:>7.3f}%")
        summary[g] = {
            "n_tensors": len(lst),
            "norm_weighted_rel_drift_%": round(nw_rel * 100, 6),
            "mean_cos_sim": round(mean_cos, 6),
            "min_cos_sim": round(min_cos, 6),
            "max_rel_drift_%": round(max_rel * 100, 4),
        }
    print("=" * 92)

    # Save
    out = "/home/aix23102/audiolm/vS2_eunji/_tools/eval_v4_quick_logs/weight_drift_10612_vs_12128.json"
    with open(out, "w") as f:
        json.dump({"per_tensor": rows, "summary": summary}, f, indent=2)
    print(f"\n[SAVED] {out}")


if __name__ == "__main__":
    main()
