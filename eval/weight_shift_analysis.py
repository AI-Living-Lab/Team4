#!/usr/bin/env python3
# Base 대비 LoRA(RL) weight shift 분석.
#   ΔW = scaling * (B @ A),  scaling = alpha/r (rslora면 alpha/sqrt(r))
#   ΔW_fro       = ||ΔW||_F
#   상대 shift   = ||ΔW||_F / ||W_base||_F
# proj 타입별 평균/최대 집계. 전부 CPU·float32.
import json, os, re, glob, sys
from collections import defaultdict
import torch
from safetensors import safe_open

ADAPTER = sys.argv[1] if len(sys.argv) > 1 else "/workspace/checkpoints/gdpo/sft_7b_puvalor_off_v2_rl_tti_prec_v2/checkpoint-500"

cfg = json.load(open(f"{ADAPTER}/adapter_config.json"))
BASE    = cfg["base_model_name_or_path"]
print(f"ADAPTER={ADAPTER}\nBASE={BASE}")
r, alpha = cfg["r"], cfg["lora_alpha"]
scaling = alpha / (r ** 0.5) if cfg.get("use_rslora") else alpha / r
print(f"r={r} alpha={alpha} use_rslora={cfg.get('use_rslora')} -> scaling={scaling}")

# base safetensors: weight name -> shard 파일 매핑
idx = json.load(open(f"{BASE}/model.safetensors.index.json"))["weight_map"]
_base_handles = {}
def base_norm(name):
    shard = idx[name]
    h = _base_handles.get(shard)
    if h is None:
        h = safe_open(f"{BASE}/{shard}", framework="pt", device="cpu")
        _base_handles[shard] = h
    w = h.get_tensor(name).float()
    return torch.linalg.norm(w).item()

# 어댑터 로드
af = safe_open(f"{ADAPTER}/adapter_model.safetensors", framework="pt", device="cpu")
keys = list(af.keys())
lora_a = {k for k in keys if k.endswith("lora_A.weight")}

PROJ_RE = re.compile(r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.")
rows = []          # (proj, layer, dW_fro, base_fro, rel)
for ka in sorted(lora_a):
    kb = ka.replace("lora_A.weight", "lora_B.weight")
    A = af.get_tensor(ka).float()      # [r, in]
    B = af.get_tensor(kb).float()      # [out, r]
    dW = scaling * (B @ A)             # [out, in]
    dW_fro = torch.linalg.norm(dW).item()
    # 대응 base weight 이름: base_model.model.<...>.lora_A.weight -> <...>.weight
    mod = ka.replace("base_model.model.", "").replace(".lora_A.weight", "")
    base_name = mod + ".weight"
    bf = base_norm(base_name)
    proj = mod.split(".")[-1]
    lyr = int(re.search(r"layers\.(\d+)\.", mod).group(1))
    rows.append((proj, lyr, dW_fro, bf, dW_fro / bf))

# modules_to_save (embed_tokens, lm_head) full-weight ΔW — vocab resize 가능성 고려
full = {}
for k in keys:
    if "lora" not in k and k.endswith(".weight"):
        mod = k.replace("base_model.model.", "").replace(".weight", "")
        full[mod] = k
for mod, k in full.items():
    base_name = mod + ".weight"
    if base_name not in idx:
        continue
    Wt = af.get_tensor(k).float()
    shard = idx[base_name]
    W0 = _base_handles.setdefault(shard, safe_open(f"{BASE}/{shard}", framework="pt", device="cpu")).get_tensor(base_name).float()
    n = min(Wt.shape[0], W0.shape[0])   # 추가 토큰 행 제외, 공통 vocab만 비교
    dW = Wt[:n] - W0[:n]
    rows.append((mod.split(".")[-1] + "(full)", -1,
                 torch.linalg.norm(dW).item(), torch.linalg.norm(W0[:n]).item(),
                 torch.linalg.norm(dW).item() / torch.linalg.norm(W0[:n]).item()))

# 집계
agg = defaultdict(list)
for proj, lyr, dW_fro, bf, rel in rows:
    agg[proj].append((dW_fro, rel, lyr))

order = ["q_proj","k_proj","v_proj","o_proj","embed_tokens(full)","lm_head(full)"]
print("\nproj\t\tn\tΔW_fro(평균)\t상대shift(평균)\t상대최대(layer)")
print("-"*78)
for proj in sorted(agg, key=lambda p: order.index(p) if p in order else 99):
    vals = agg[proj]
    n = len(vals)
    mean_dW = sum(v[0] for v in vals)/n
    mean_rel = sum(v[1] for v in vals)/n
    mx = max(vals, key=lambda v: v[1])
    print(f"{proj:<16}\t{n}\t{mean_dW:8.3f}\t{mean_rel*100:7.3f}%\t{mx[1]*100:7.3f}% (L{mx[2]})")

# 레이어별 상대shift (q/k/v/o 합산 ΔW_fro 기준) — 깊이별 경향
print("\n[레이어별 상대shift % — proj별]")
bylayer = defaultdict(dict)
for proj, lyr, dW_fro, bf, rel in rows:
    if lyr >= 0: bylayer[lyr][proj] = rel*100
print("layer\t" + "\t".join(["q","k","v","o"]))
for lyr in sorted(bylayer):
    d = bylayer[lyr]
    print(f"{lyr}\t" + "\t".join(f"{d.get(p+'_proj',0):.2f}" for p in ["q","k","v","o"]))
