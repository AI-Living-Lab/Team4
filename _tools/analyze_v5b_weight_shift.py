"""
Weight-shift analysis: v5_b merged vs base (video_salmonn2_plus_7B_time_tokens).

For each shared tensor key:
  delta = merged - base
  metrics:
    ||base||_2,  ||delta||_2,  rel_shift = ||delta|| / ||base||,  mean|delta| / mean|base|

Aggregates by module-group prefix. Special analysis for time tokens
(<t0>..<t9>, <tdot>; ids 151666..151676) rows in embed_tokens / lm_head.

Loads tensors lazily via safetensors.safe_open — never loads the full model.
"""

import argparse
import json
import os
from collections import defaultdict

import torch
from safetensors import safe_open

BASE = "/data0/aix23102/checkpoints_open_aligner/video_salmonn2_plus_7B_time_tokens"
MERGED = "/data0/aix23102/checkpoints_open_aligner/salmonn2plus_v5_b/checkpoint-11371_merged"
TIME_TOKEN_IDS = list(range(151666, 151677))  # <t0>..<t9>,<tdot>


def index_map(model_dir):
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    return idx["weight_map"]  # key -> shard file


def group_of(key):
    """Bucket every key into one coarse group for summary."""
    if "embed_tokens" in key:
        return "embed_tokens"
    if "lm_head" in key:
        return "lm_head"
    if ".self_attn." in key:
        for sub in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if sub in key:
                return f"llm.self_attn.{sub}"
        return "llm.self_attn.other"
    if ".mlp." in key:
        for sub in ("gate_proj", "up_proj", "down_proj"):
            if sub in key:
                return f"llm.mlp.{sub}"
        return "llm.mlp.other"
    if "layernorm" in key.lower() or "norm" in key.split(".")[-1]:
        return "llm.norm"
    if key.startswith("visual.merger") or "visual.merger" in key:
        return "visual.merger"
    if key.startswith("visual") or ".visual." in key:
        return "visual.other"
    if "audio.q_tokens" in key:
        return "audio.q_tokens"
    if "audio.qformer" in key or "audio.Qformer" in key or "qformer" in key.lower():
        return "audio.qformer"
    if "audio.audio_proj" in key or "audio_proj" in key:
        return "audio.proj"
    if key.startswith("audio") or ".audio." in key:
        return "audio.other"
    return "misc"


def load_key(model_dir, idx_map, key):
    shard = idx_map[key]
    with safe_open(os.path.join(model_dir, shard), framework="pt", device="cpu") as f:
        return f.get_tensor(key).to(torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--merged", default=MERGED)
    ap.add_argument("--top", type=int, default=20, help="show top-N per-key shifts")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base_idx = index_map(args.base)
    merged_idx = index_map(args.merged)

    base_keys = set(base_idx.keys())
    merged_keys = set(merged_idx.keys())
    shared = sorted(base_keys & merged_keys)
    only_base = sorted(base_keys - merged_keys)
    only_merged = sorted(merged_keys - base_keys)

    print(f"[meta] base keys={len(base_keys)}  merged keys={len(merged_keys)}  shared={len(shared)}")
    if only_base:
        print(f"[meta] only-in-base ({len(only_base)}): {only_base[:5]}")
    if only_merged:
        print(f"[meta] only-in-merged ({len(only_merged)}): {only_merged[:5]}")

    per_key = []  # (key, base_norm, delta_norm, rel)
    grp_agg = defaultdict(lambda: {"n_keys": 0, "n_params": 0,
                                   "base_sq": 0.0, "delta_sq": 0.0,
                                   "base_abs_sum": 0.0, "delta_abs_sum": 0.0})

    # Special time-token row analysis state
    time_token_results = {}

    for i, k in enumerate(shared):
        b = load_key(args.base, base_idx, k)
        m = load_key(args.merged, merged_idx, k)
        if b.shape != m.shape:
            print(f"[skip shape mismatch] {k}  base={tuple(b.shape)}  merged={tuple(m.shape)}")
            continue
        d = m - b
        bn = float(torch.linalg.norm(b))
        dn = float(torch.linalg.norm(d))
        rel = dn / (bn + 1e-12)
        per_key.append((k, bn, dn, rel))

        g = group_of(k)
        agg = grp_agg[g]
        agg["n_keys"] += 1
        agg["n_params"] += b.numel()
        agg["base_sq"] += float(b.pow(2).sum())
        agg["delta_sq"] += float(d.pow(2).sum())
        agg["base_abs_sum"] += float(b.abs().sum())
        agg["delta_abs_sum"] += float(d.abs().sum())

        # Time-token row analysis on embed_tokens / lm_head
        if "embed_tokens.weight" in k or k.endswith("lm_head.weight"):
            rows_b = b[TIME_TOKEN_IDS, :]
            rows_m = m[TIME_TOKEN_IDS, :]
            rows_d = rows_m - rows_b
            per_tok = []
            for j, tid in enumerate(TIME_TOKEN_IDS):
                rb = float(torch.linalg.norm(rows_b[j]))
                rd = float(torch.linalg.norm(rows_d[j]))
                per_tok.append((tid, rb, rd, rd / (rb + 1e-12)))
            time_token_results[k] = per_tok

        if (i + 1) % 20 == 0:
            print(f"  ...processed {i+1}/{len(shared)}")

    # ---- group summary ----
    print("\n================== GROUP SUMMARY ==================")
    print(f"{'group':<28} {'#keys':>5} {'params':>13} {'||base||':>10} {'||delta||':>10} {'rel':>8}")
    rows = []
    for g, agg in sorted(grp_agg.items()):
        bn = agg["base_sq"] ** 0.5
        dn = agg["delta_sq"] ** 0.5
        rel = dn / (bn + 1e-12)
        rows.append((g, agg["n_keys"], agg["n_params"], bn, dn, rel))
    rows.sort(key=lambda r: -r[5])
    for g, nk, np_, bn, dn, rel in rows:
        print(f"{g:<28} {nk:>5d} {np_:>13,d} {bn:>10.3f} {dn:>10.3f} {rel*100:>7.3f}%")

    # ---- top-N per-key shifts ----
    print(f"\n================== TOP-{args.top} per-key RELATIVE SHIFTS ==================")
    per_key.sort(key=lambda r: -r[3])
    print(f"{'key':<70} {'||base||':>9} {'||delta||':>10} {'rel':>8}")
    for k, bn, dn, rel in per_key[:args.top]:
        print(f"{k:<70} {bn:>9.3f} {dn:>10.3f} {rel*100:>7.3f}%")

    # ---- time-token analysis ----
    print("\n================== TIME-TOKEN ROW SHIFTS ==================")
    tok_names = ["<t0>","<t1>","<t2>","<t3>","<t4>","<t5>","<t6>","<t7>","<t8>","<t9>","<tdot>"]
    for k, per_tok in time_token_results.items():
        print(f"\n[{k}]")
        print(f"  {'tok':<8} {'id':>7} {'||base_row||':>14} {'||delta_row||':>15} {'rel':>8}")
        for (tid, rb, rd, rel), nm in zip(per_tok, tok_names):
            print(f"  {nm:<8} {tid:>7d} {rb:>14.5f} {rd:>15.5f} {rel*100:>7.2f}%")

    if args.out:
        out = {
            "groups": [{"group": g, "n_keys": nk, "n_params": np_,
                        "base_norm": bn, "delta_norm": dn, "rel_shift": rel}
                       for (g, nk, np_, bn, dn, rel) in rows],
            "top_keys": [{"key": k, "base_norm": bn, "delta_norm": dn, "rel_shift": rel}
                         for (k, bn, dn, rel) in per_key[:args.top]],
            "time_tokens": {k: [{"name": nm, "id": tid, "base_row_norm": rb,
                                 "delta_row_norm": rd, "rel_shift": rel}
                                for (tid, rb, rd, rel), nm in zip(v, tok_names)]
                            for k, v in time_token_results.items()},
        }
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
