# Copyright (2025) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adopted from https://github.com/LLaVA-VL/LLaVA-NeXT. The original license is located at 'third-party-license/llava_next.txt'.
# Adopted from https://github.com/lm-sys/FastChat. The original license is located at 'third-party-license/fastchat.txt'.
# Adopted from tatsu-lab@stanford_alpaca. The original license is located at 'third-party-license/stanford_alpaca.txt'.

import os
import re
import sys
import glob
import copy
from dataclasses import dataclass, field, asdict
import json
import pathlib
from typing import Dict, Optional
import ast
import torch
import random
import torch.distributed as dist
import transformers
import yaml
from tqdm.auto import tqdm
from llava.train.llava_trainer import LLaVATrainer
from llava.train.dpo_trainer import LLaVADPOTrainer
from llava import conversation as conversation_lib
from llava.model import VideoSALMONN2ForCausalLM
from llava.model.utils import load_qwen_lora_model
import numpy as np
from transformers import AutoConfig
import torch.nn.functional as F
from llava.dataset import make_supervised_data_module, make_test_data_module
from transformers import TrainerCallback

def _prompt_to_text(prompt):
    """
    prompt can be:
      - list[dict(from,value), ...]
      - list[list[dict...]] (nested)
      - dict / str
      - batch-style list of prompts
    Returns a stable text string for dedup key.
    """
    def _flatten(x):
        if isinstance(x, list):
            for y in x:
                yield from _flatten(y)
        else:
            yield x

    if prompt is None:
        return ""

    parts = []
    for item in _flatten(prompt):
        if isinstance(item, dict):
            v = item.get("value", None)
            if isinstance(v, str):
                parts.append(v)
        elif isinstance(item, str):
            parts.append(item)
        # 그 외 타입은 무시 (e.g., None, numbers)
    return " ".join(parts)

def _unwrap_model(m):
    return m.module if hasattr(m, "module") else m

def unfreeze_embeddings_and_lm_head(model):
    m = _unwrap_model(model)

    # input embeddings
    in_emb = m.get_input_embeddings()
    if in_emb is not None:
        for p in in_emb.parameters():
            p.requires_grad_(True)

    # output embeddings (tied) or lm_head
    out_emb = m.get_output_embeddings()
    if out_emb is not None:
        for p in out_emb.parameters():
            p.requires_grad_(True)
    elif hasattr(m, "lm_head"):
        for p in m.lm_head.parameters():
            p.requires_grad_(True)

    return model

def ensure_image_token(tokenizer, model):
    vocab = tokenizer.get_vocab()
    if "<image>" in vocab:
        image_id = tokenizer.convert_tokens_to_ids("<image>")
        print(f"[INFO] <image> already in vocab. id={image_id}")
    else:
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        print(f"[INFO] Added <image> token. num_added={num_added}")
        model.resize_token_embeddings(len(tokenizer))
        image_id = tokenizer.convert_tokens_to_ids("<image>")

    if hasattr(model.config, "image_token_id"):
        model.config.image_token_id = int(image_id)
        print(f"[INFO] model.config.image_token_id set to {model.config.image_token_id}")

    return image_id

def _resolve_tokenizer_path(training_args, model_args=None):
    # time token을 추가해야 하는 실험이면:
    #  - 기존 ckpt tokenizer를 쓰면 ID 충돌이 이미 포함돼 있을 수 있음
    #  - 반드시 model_base tokenizer에서 시작해야 안전
    if model_args is not None and getattr(model_args, "add_time_token", False):
        return training_args.model_base

    # (기존 로직 유지)
    outdir = getattr(training_args, "output_dir", None)
    if outdir and os.path.isdir(outdir):
        ckpts = sorted(glob.glob(os.path.join(outdir, "checkpoint-*")))
        if len(ckpts) > 0:
            return ckpts[-1]
        tok_file = os.path.join(outdir, "tokenizer.json")
        if os.path.exists(tok_file):
            return outdir
    return training_args.ckpt

def simple_predict_generate(
    model,
    tokenizer,
    trainer,
    eval_dataset,
    do_sample: bool = False,
    max_new_tokens: int = 512,
    max_time: float = 30.0,
):
    """
    Fallback inference when LLaVATrainer.predict() returns None.
    Uses trainer.get_eval_dataloader(eval_dataset) for consistent batching.
    Returns: List[Dict] with keys: id, prompt, pred
    """
    model.eval()
    results = []

    dl = trainer.get_eval_dataloader(eval_dataset)

    # =========================================================
    # ✅ POSTPROCESS: force outputs to strict JSON array with time-token strings
    # =========================================================
    _OBJ_RE = re.compile(r"\{.*?\}", re.DOTALL)
    _NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

    def _tokens_to_numeric_string(s: str) -> str:
        """
        Converts mixed "<t3>" "<tdot>" tokens to digits, strips non-numeric cruft.
        Example: "0 <t0>.0 sec" -> "0 0.0"
                 "0 0<tdot>2 sec" -> "0 0.2"
                 "0.<t6>" -> "0.6"
        """
        if not isinstance(s, str):
            return ""
        for i in range(10):
            s = s.replace(f"<t{i}>", str(i))
        s = s.replace("<tdot>", ".")
        # keep digits/dot/sign/exp, turn others into spaces
        s = re.sub(r"[^0-9\.\-\s eE\+]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _parse_float_any(s: str) -> Optional[float]:
        """
        Robust float parser for:
          - plain numbers ("3.7")
          - mixed tokens ("0 3.<t7> sec")
          - token-only ("<t0> <t3> <tdot> <t7>")
        """
        if s is None:
            return None
        if isinstance(s, (int, float)):
            try:
                return float(s)
            except Exception:
                return None
        if not isinstance(s, str):
            return None
        s = s.strip()
        if s == "":
            return None

        # 1) try direct numeric in original string
        m = _NUM_RE.search(s)
        if m:
            try:
                return float(m.group(0))
            except Exception:
                pass

        # 2) normalize tokens -> numeric-ish string, then parse
        tmp = _tokens_to_numeric_string(s)
        m2 = _NUM_RE.search(tmp)
        if m2:
            try:
                return float(m2.group(0))
            except Exception:
                return None
        return None

    def _sec_to_time_tokens(sec: float, *, width: int = 2, decimals: int = 1) -> str:
        """
        Convert seconds to VTG-LLM token string:
          3.7 -> "<t0> <t3> <tdot> <t7>" (width=2)
         19.3 -> "<t1> <t9> <tdot> <t3>"
        Clamps into [0, max_time].
        """
        try:
            sec = float(sec)
        except Exception:
            sec = 0.0
        if sec < 0:
            sec = 0.0
        if max_time is not None:
            try:
                mt = float(max_time)
                if mt > 0:
                    sec = min(sec, mt)
            except Exception:
                pass

        # round to desired decimals
        q = round(sec, decimals)
        fmt = f"{{:0{width + (1 + decimals if decimals > 0 else 0)}.{decimals}f}}"
        s = fmt.format(q)  # e.g. "03.7"

        out = []
        for ch in s:
            if ch.isdigit():
                out.append(f"<t{ch}>")
            elif ch == ".":
                out.append("<tdot>")
        return " ".join(out)

    def _normalize_pred_to_json_array(pred_text: str) -> str:
        """
        Force pred into:
          JSON array string of dicts:
            [{"label":..., "start":"<t.. ...>", "end":"<t.. ...>", "score":0.xx}, ...]
        Even if model produced:
          - multiple JSON objects separated by commas/newlines
         - times like "0 3.7 sec" or "0 <t0>.0 sec"
          - score like "0.<t6>"
        """
        if not isinstance(pred_text, str):
            return "[]"
        s = pred_text.strip()
        if s == "":
            return "[]"

        # remove stray special tokens
        s = s.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()

        # 1) if already a JSON list, try load
        if s.startswith("["):
            try:
                arr = json.loads(s)
                if isinstance(arr, list):
                    # best-effort sanitize fields
                    fixed = []
                    for o in arr:
                        if not isinstance(o, dict):
                            continue
                        label = o.get("label", None)
                        if not isinstance(label, str) or label.strip() == "":
                            continue
                        st = o.get("start", "")
                        ed = o.get("end", "")
                        sc = o.get("score", 0.0)

                        st_sec = _parse_float_any(st)
                        ed_sec = _parse_float_any(ed)
                        if st_sec is None:
                            st_sec = 0.0
                        if ed_sec is None:
                            ed_sec = st_sec + 0.1
                        ed_sec = max(ed_sec, st_sec + 0.1)

                        sc_f = _parse_float_any(sc)
                        if sc_f is None:
                            sc_f = 0.0
                        sc_f = max(0.0, min(1.0, float(sc_f)))

                        fixed.append({
                            "label": label,
                            "start": _sec_to_time_tokens(st_sec),
                            "end": _sec_to_time_tokens(ed_sec),
                            "score": sc_f,
                        })
                    return json.dumps(fixed, ensure_ascii=False)
            except Exception:
                pass

        # 2) recover by extracting JSON objects
        objs = []
        for m in _OBJ_RE.finditer(s):
            chunk = m.group(0)
            try:
                o = json.loads(chunk)
                if isinstance(o, dict):
                    objs.append(o)
            except Exception:
                continue

        if not objs:
            return "[]"

        # 3) sanitize + convert times/scores
        fixed = []
        for o in objs:
            label = o.get("label", None)
            if not isinstance(label, str) or label.strip() == "":
                continue

            st = o.get("start", "")
            ed = o.get("end", "")
            sc = o.get("score", 0.0)

            st_sec = _parse_float_any(st)
            ed_sec = _parse_float_any(ed)
            if st_sec is None:
                st_sec = 0.0
            if ed_sec is None:
                ed_sec = st_sec + 0.1
            ed_sec = max(ed_sec, st_sec + 0.1)

            sc_f = _parse_float_any(sc)
            if sc_f is None:
                sc_f = 0.0
            sc_f = max(0.0, min(1.0, float(sc_f)))

            fixed.append({
                "label": label,
                "start": _sec_to_time_tokens(st_sec),
                "end": _sec_to_time_tokens(ed_sec),
                "score": sc_f,
            })

        return json.dumps(fixed, ensure_ascii=False)

    def _to_cuda_if_tensor(x):
        if torch.is_tensor(x):
            return x.cuda(non_blocking=True)
        return x

    with torch.no_grad():

        def _safe_decode(tokenizer, ids, skip_special_tokens=False):
            # ids: Tensor or list[int]
            if torch.is_tensor(ids):
                ids = ids.tolist()
            # fast tokenizer는 음수 토큰 decode 시 OverflowError
            ids = [x for x in ids if isinstance(x, int) and x >= 0]
            return tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

        def _strip_trailing_im_end(input_ids_1d, im_end_id: int):
            # input_ids_1d: 1D Tensor
            ids = input_ids_1d
            # 마지막이 <|im_end|>로 닫혀있으면 제거 (한 개만 제거하면 충분)
            if ids.numel() > 0 and int(ids[-1].item()) == int(im_end_id):
                ids = ids[:-1]
            return ids

        # ---- tqdm: rank0에서만 출력 (멀티 GPU 로그 난장판 방지) ----
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        is_rank0 = (rank == 0)

        # total 추정 (가능하면 dataloader 길이 사용)
        try:
            total = len(dl)
        except Exception:
            try:
                total = len(eval_dataset)
            except Exception:
                total = None

        pbar = tqdm(
            dl,
            total=total,
            disable=(not is_rank0),
            desc="[TEST] Inference",
            dynamic_ncols=True,
        )

        for step, batch in enumerate(pbar):

            # ===== DEBUG START =====
            if step == 0:  # 첫 배치만 찍기
                print("\n================ DEBUG INPUT ================")
                print("[DBG] modalities:", batch.get("modalities"))
                print("[DBG] has images:", batch.get("images") is not None)
                print("[DBG] input_ids shape:", batch["input_ids"].shape)
                print("[DBG] last 50 token ids:",
                    batch["input_ids"][0][-50:].tolist())
                print("[DBG] decoded prompt tail:\n",
                    _safe_decode(
                        tokenizer, batch["input_ids"][0][-400:], 
                        skip_special_tokens=False)
                    )
                print("=============================================\n")
            # ===== DEBUG END =====

            # ---- metadata keys can vary depending on collator ----
            ids = batch.get("ids", batch.get("id", None))
            prompts = batch.get("prompts", batch.get("prompt", None))

            # ---- move to cuda only if needed ----
            for k in ["input_ids", "attention_mask"]:
                if k in batch:
                    batch[k] = _to_cuda_if_tensor(batch[k])

            # multimodal inputs (may already be on device)
            if "images" in batch and batch["images"] is not None:
                if isinstance(batch["images"], (list, tuple)):
                    batch["images"] = [it.to(torch.bfloat16).cuda(non_blocking=True) if torch.is_tensor(it) else it
                                      for it in batch["images"]]
                else:
                    batch["images"] = batch["images"].to(torch.bfloat16).cuda(non_blocking=True)

            if "spectrogram" in batch and batch["spectrogram"] is not None:
                if torch.is_tensor(batch["spectrogram"]):
                    batch["spectrogram"] = batch["spectrogram"].to(torch.bfloat16).cuda(non_blocking=True)


            # ---- IMPORTANT: normalize keys for VideoSALMONN2ForCausalLM.generate ----
            # This repo's collator/dataset often uses:
            #   image (Tensor) + modality (str) + real_time (float)
            # but model.generate expects:
            #   images (List[Tensor]) + modalities (List[str]) + real_time (Tensor/float)
            #
            # 1) image -> images
            if "images" not in batch and "image" in batch and batch["image"] is not None:
                img = batch["image"]
                # move to cuda + bf16
                if torch.is_tensor(img):
                    img = img.to(torch.bfloat16).cuda(non_blocking=True)
                batch["images"] = [img]
            elif "images" in batch and batch["images"] is not None:
                if isinstance(batch["images"], (list, tuple)):
                    batch["images"] = [
                        it.to(torch.bfloat16).cuda(non_blocking=True) if torch.is_tensor(it) else it
                        for it in batch["images"]
                    ]
                elif torch.is_tensor(batch["images"]):
                    batch["images"] = [batch["images"].to(torch.bfloat16).cuda(non_blocking=True)]

            # 2) modality -> modalities
            if "modalities" not in batch and "modality" in batch and batch["modality"] is not None:
                m = batch["modality"]
                # batch_size=1이면 보통 str, 혹은 list[str]
                if isinstance(m, str):
                    batch["modalities"] = [m]
                else:
                    batch["modalities"] = m

            # 3) real_time 그대로 넘기기 (있으면)
            #    (prepare_inputs_labels_for_multimodal에서 사용)
            # 4) org_groups (있으면)도 넘기기

            # ---- keep only keys generate() accepts in this repo ----
            allowed = {
                "input_ids",
                "attention_mask",
                "images",
                "modalities",
                "spectrogram",
                "org_groups",
                "real_time",
            }
            gen_batch = {k: v for k, v in batch.items() if k in allowed}

            # =========================================================
            # 🔥 SAFE GENERATION PROMPT FIX (string-normalize; Qwen-safe)
            # Goal: force prompt to end with "<|im_start|>assistant\n"
            # Also remove any trailing CLOSED assistant blocks:
            #   "<|im_start|>assistant ... <|im_end|>"
            # =========================================================

            GEN_PROMPT_STR = "<|im_start|>assistant\n"
            IM_START_ID = tokenizer.convert_tokens_to_ids("<|im_start|>")
            IM_END_ID   = tokenizer.convert_tokens_to_ids("<|im_end|>")

            # ids for "assistant\n" (Qwen tokenizer may encode as multiple tokens)
            ASSIST_HDR_IDS = tokenizer.encode("assistant\n", add_special_tokens=False)

            def _ensure_generation_prompt_ids_only(ids_1d: torch.Tensor) -> torch.Tensor:
                ids = ids_1d.tolist()

                IMAGE_PLACEHOLDER = -200  # LLaVA IMAGE_TOKEN_INDEX
                ids = [x for x in ids if isinstance(x, int) and (x >= 0 or x == IMAGE_PLACEHOLDER)]

                # ids for "assistant\n" (Qwen tokenizer may encode as multiple tokens)
                patA = ([IM_START_ID] if IM_START_ID is not None else []) + ASSIST_HDR_IDS
                patB = ([IM_START_ID, IM_END_ID] if (IM_START_ID is not None and IM_END_ID is not None) else []) + ASSIST_HDR_IDS

                # helper: reverse find pattern (no _rfind_pattern dependency)
                def _rfind(seq, pat):
                    if len(pat) == 0 or len(seq) < len(pat):
                        return None
                    for j in range(len(seq) - len(pat), -1, -1):
                        if seq[j:j+len(pat)] == pat:
                            return j
                    return None

                # 0) strip trailing IM_ENDs
                while len(ids) > 0 and IM_END_ID is not None and ids[-1] == IM_END_ID:
                    ids.pop()

                # 1) find last assistant header (A or B)
                posA = _rfind(ids, patA)
                posB = _rfind(ids, patB)

                pos = None
                if posA is not None:
                    pos = posA
                if posB is not None and (pos is None or posB > pos):
                    pos = posB

                # 2) define junk tokens to remove right before the header
                lbr = tokenizer.encode("[", add_special_tokens=False)
                LBRACK_ID = lbr[0] if len(lbr) > 0 else None

                def _strip_tail_junk(prefix):
                    junk = set()
                    if IM_END_ID is not None:
                        junk.add(IM_END_ID)
                    if LBRACK_ID is not None:
                        junk.add(LBRACK_ID)
                    # common whitespace tokens (Qwen often uses these)
                    junk |= {198, 220}  # '\n', ' '

                    while len(prefix) > 0 and prefix[-1] in junk:
                        prefix.pop()
                    return prefix

                if pos is not None:
                    # keep everything BEFORE the last header, drop anything AFTER it
                    prefix = ids[:pos]
                    prefix = _strip_tail_junk(prefix)

                    # enforce clean header (always use patA, not patB)
                    new_ids = prefix + patA
                    return ids_1d.new_tensor(new_ids, dtype=ids_1d.dtype)

                # 3) if no header exists, append clean header
                ids = _strip_tail_junk(ids)
                ids = ids + patA
                return ids_1d.new_tensor(ids, dtype=ids_1d.dtype)
                
            if "input_ids" in gen_batch and gen_batch["input_ids"] is not None:
                ids0 = gen_batch["input_ids"][0]
                fixed0 = _ensure_generation_prompt_ids_only(ids0)
                gen_batch["input_ids"] = fixed0.unsqueeze(0)

                # ✅ 원칙: attention_mask는 "원본"을 최대한 유지
                # 다만 길이가 달라졌으면 그때만 tail만 1로 붙이기
                if "attention_mask" in gen_batch and torch.is_tensor(gen_batch["attention_mask"]):
                    old_mask = gen_batch["attention_mask"][0]
                    old_len = old_mask.numel()
                    new_len = gen_batch["input_ids"].shape[1]
                    if new_len == old_len:
                        pass  # 그대로 둠
                    elif new_len > old_len:
                        pad = torch.ones(new_len - old_len, device=old_mask.device, dtype=old_mask.dtype)
                        gen_batch["attention_mask"] = torch.cat([old_mask, pad], dim=0).unsqueeze(0)
                    else:
                        gen_batch["attention_mask"] = old_mask[:new_len].unsqueeze(0)
                else:
                    # 없으면 pad 기반으로 생성(최후의 fallback)
                    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.convert_tokens_to_ids("<|im_end|>")
                    gen_batch["attention_mask"] = (gen_batch["input_ids"] != pad_id).to(torch.long)

                if step == 0 and is_rank0:
                    tail = _safe_decode(tokenizer, gen_batch["input_ids"][0][-120:], skip_special_tokens=False)
                    print("[DBG][AFTER_FIX] prompt tail:\n", tail, flush=True)
                    print("[CHK][PROMPT_ENDS_WITH_ASSIST_HDR] =", tail.endswith("assistant\n"))

            # =========================================================
            if step == 0 and is_rank0:
                print("[CHK][AFTER_FIX_TAIL_TEXT]",
                    _safe_decode(tokenizer, gen_batch["input_ids"][0][-80:], skip_special_tokens=False))
                print("[CHK][AFTER_FIX_TAIL_IDS]",
                    gen_batch["input_ids"][0][-30:].tolist())

            if step < 3 and is_rank0:
                n_img_idx = int((gen_batch["input_ids"] == -200).sum().item())
                print(f"[CHK][AFTER_FIX] count(-200)={n_img_idx}")

            if step < 1 and is_rank0:
                print("[CHK][FIX_TAIL_LAST_ID]", int(gen_batch["input_ids"][0, -1].item()))

            # --- MIN DEBUG (only first 3 steps, rank0) ---
            if step < 3 and is_rank0:
                print("[DBG][IDS]", ids)
                print("[DBG][GEN_BATCH_KEYS]", sorted(list(gen_batch.keys())))

                img = gen_batch.get("images", None)
                if isinstance(img, (list, tuple)) and len(img) > 0 and torch.is_tensor(img[0]):
                    print("[DBG][IMG0] mean/std:", float(img[0].mean()), float(img[0].std()))
                else:
                    print("[DBG][IMG0] None or not tensor-list")

                spec = gen_batch.get("spectrogram", None)
                if torch.is_tensor(spec):
                    print("[DBG][SPEC] mean/std:", float(spec.mean()), float(spec.std()))
                else:
                    print("[DBG][SPEC] None")

                # ===== 추가 디버그 (여기부터) =====
                # (A) <image> 토큰이 input_ids에 실제로 들어있는지
                if "input_ids" in gen_batch and torch.is_tensor(gen_batch["input_ids"]):
                    ids_t = gen_batch["input_ids"]
                    IMAGE_ID = tokenizer.convert_tokens_to_ids("<image>")  # 보통 151646
                    n_img_tok = int((ids_t == IMAGE_ID).sum().item())
                    n_img_idx = int((ids_t == -200).sum().item())          # LLaVA placeholder (IMAGE_TOKEN_INDEX)

                    print(f"[CHK][IMG_TOK] <image>_id={IMAGE_ID} count(id)={n_img_tok} count(-200)={n_img_idx}")
                    print(f"[CHK][IMG_TOK] input_ids min/max = {int(ids_t.min())}/{int(ids_t.max())}")
                else:
                    print("[CHK][IMG_TOK] gen_batch has no tensor input_ids")

                # (B) images/spectrogram shape/dtype/device (실제 텐서인지)
                imgs = gen_batch.get("images", None)
                print(f"[CHK][BATCH] images is None? {imgs is None}")
                if imgs is not None:
                    # VS2 계열에서 images가 list/tuple일 수도, tensor일 수도 있어 둘 다 대응
                    if torch.is_tensor(imgs):
                        print(f"[CHK][BATCH] images tensor shape={tuple(imgs.shape)} dtype={imgs.dtype} device={imgs.device}")
                    elif isinstance(imgs, (list, tuple)) and len(imgs) > 0 and torch.is_tensor(imgs[0]):
                        print(f"[CHK][BATCH] images[0] shape={tuple(imgs[0].shape)} dtype={imgs[0].dtype} device={imgs[0].device} (list/tuple)")
                    else:
                        print(f"[CHK][BATCH] images type={type(imgs)} (not tensor / not tensor-list)")

                sp = gen_batch.get("spectrogram", None)
                print(f"[CHK][BATCH] spectrogram is None? {sp is None}")
                if sp is not None:
                    if torch.is_tensor(sp):
                        print(f"[CHK][BATCH] spectrogram shape={tuple(sp.shape)} dtype={sp.dtype} device={sp.device}")
                    else:
                        print(f"[CHK][BATCH] spectrogram type={type(sp)} (not tensor)")
                # ===== 추가 디버그 (여기까지) =====
            # --- end debug ---

            # --- DIAG: does MM affect the FIRST generated token? (rank0, step<1 only) ---
            if step < 1 and is_rank0:
                ids_t = gen_batch["input_ids"]
                n_img_idx = int((ids_t == -200).sum().item())
                print(f"[DIAG][MM_EFFECT] count(-200)={n_img_idx}")

                # 1) normal run (only 1 token)
                out1 = model.generate(
                    **gen_batch,
                    do_sample=do_sample,
                    num_beams=1,
                    max_new_tokens=1,
                    eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                tok1 = int(out1.sequences[0, -1].item())
                print(f"[DIAG][MM_EFFECT] first_tok(normal) id={tok1} str={tokenizer.decode([tok1], skip_special_tokens=False)!r}")
                lbr_ids = tokenizer.encode("[", add_special_tokens=False)
                lbr_id = lbr_ids[0] if len(lbr_ids) > 0 else None
                print(f"[CHK][JSON_ANCHOR] '[' token_id={lbr_id}  first_tok_is_[?] {tok1 == lbr_id}")

                # 2) zero-image run (keep input_ids identical, only zero images/spectrogram)
                gen_batch2 = copy.deepcopy(gen_batch)
                if "images" in gen_batch2 and gen_batch2["images"] is not None:
                    imgs = gen_batch2["images"]
                    if isinstance(imgs, (list, tuple)):
                        gen_batch2["images"] = [x * 0 if torch.is_tensor(x) else x for x in imgs]
                    elif torch.is_tensor(imgs):
                        gen_batch2["images"] = imgs * 0

                if "spectrogram" in gen_batch2 and torch.is_tensor(gen_batch2["spectrogram"]):
                    gen_batch2["spectrogram"] = gen_batch2["spectrogram"] * 0

                out2 = model.generate(
                    **gen_batch2,
                    do_sample=do_sample,
                    num_beams=1,
                    max_new_tokens=1,
                    eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                tok2 = int(out2.sequences[0, -1].item())
                print(f"[DIAG][MM_EFFECT] first_tok(zeroMM) id={tok2} str={tokenizer.decode([tok2], skip_special_tokens=False)!r}")

                # 3) score delta (L1 mean) for the 1st token distribution
                s1 = out1.scores[0]  # (bs, vocab)
                s2 = out2.scores[0]
                delta = float((s1 - s2).abs().mean().item())
                print(f"[DIAG][MM_EFFECT] mean|score_delta|={delta:.6f}")
            # --- end DIAG ---

            # --- DIAG3: compare full generated sequences (rank0, step<1 only) ---
            if step < 1 and is_rank0:
                def _zero_mm(gb):
                    gb2 = copy.deepcopy(gb)
                    if "images" in gb2 and gb2["images"] is not None:
                        imgs = gb2["images"]
                        if isinstance(imgs, (list, tuple)):
                            gb2["images"] = [x * 0 if torch.is_tensor(x) else x for x in imgs]
                        elif torch.is_tensor(imgs):
                            gb2["images"] = imgs * 0
                    if "spectrogram" in gb2 and torch.is_tensor(gb2.get("spectrogram", None)):
                        gb2["spectrogram"] = gb2["spectrogram"] * 0
                    return gb2

                def _gen(gb, n):
                    return model.generate(
                        **gb,
                        do_sample=do_sample,
                        num_beams=1,
                        max_new_tokens=n,
                        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
                        pad_token_id=tokenizer.pad_token_id,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )

                N = 40
                out_n = _gen(gen_batch, N)
                out_z = _gen(_zero_mm(gen_batch), N)

                seq_n = out_n.sequences[0].tolist()
                seq_z = out_z.sequences[0].tolist()

                prompt_len = gen_batch["input_ids"].shape[1]

                # HF generate가 "prompt+gen"을 줄 수도 있고 "gen_only"만 줄 수도 있음
                if len(seq_n) >= prompt_len:
                    gen_n = seq_n[prompt_len:]
                else:
                    gen_n = seq_n

                if len(seq_z) >= prompt_len:
                    gen_z = seq_z[prompt_len:]
                else:
                    gen_z = seq_z

                txt_n = tokenizer.decode(gen_n, skip_special_tokens=False)
                txt_z = tokenizer.decode(gen_z, skip_special_tokens=False)

                print("\n[DIAG3][MM_SEQ] normal gen (head):", repr(txt_n[:200]))
                print("[DIAG3][MM_SEQ] zeroMM gen (head):", repr(txt_z[:200]))

                # find first differing token
                first_diff = None
                for i, (a, b) in enumerate(zip(gen_n, gen_z)):
                    if a != b:
                        first_diff = i
                        break

                if first_diff is None:
                    print("[DIAG3][MM_SEQ] ✅ first 40 tokens are IDENTICAL (normal == zeroMM)")
                else:
                    a = gen_n[first_diff]
                    b = gen_z[first_diff]
                    print(f"[DIAG3][MM_SEQ] ❗ first diff at gen_pos={first_diff}")
                    print("  normal tok:", a, repr(tokenizer.decode([a], skip_special_tokens=False)))
                    print("  zeroMM tok:", b, repr(tokenizer.decode([b], skip_special_tokens=False)))

                    # score delta at that generation step (scores indexed by generated step)
                    sdn = out_n.scores[first_diff]  # (bs, vocab)
                    sdz = out_z.scores[first_diff]
                    delta = float((sdn - sdz).abs().mean().item())
                    topn = torch.topk(sdn[0], k=5).indices.tolist()
                    topz = torch.topk(sdz[0], k=5).indices.tolist()
                    print(f"[DIAG3][MM_SEQ] mean|score_delta|@diff_step={delta:.6f}")
                    print("[DIAG3][MM_SEQ] top5(normal):", [(t, tokenizer.decode([t], skip_special_tokens=False)) for t in topn])
                    print("[DIAG3][MM_SEQ] top5(zeroMM):", [(t, tokenizer.decode([t], skip_special_tokens=False)) for t in topz])
            if step < 1 and is_rank0:
                prompt_len = gen_batch["input_ids"].shape[1]
                eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
                print("[CHK][DIAG3] eos_id:", eos_id)
                first_gen_id = gen_n[0] if len(gen_n) > 0 else None
                print("[CHK][DIAG3] out_n first gen id:", first_gen_id)
                if first_gen_id is not None:
                    print("[CHK][DIAG3] first gen tok str:", repr(tokenizer.decode([first_gen_id], skip_special_tokens=False)))
                if out_n.sequences.shape[1] > prompt_len:
                    tid = out_n.sequences[0][prompt_len].item()
                    print("[CHK][DIAG3] first gen tok str:", repr(tokenizer.decode([tid], skip_special_tokens=False)))
            if step < 1 and is_rank0:
                print("[CHK][DIAG3] gen_n first 10 ids:", gen_n[:10])
                print("[CHK][DIAG3] gen_n first 10 toks:",
                    [repr(tokenizer.decode([t], skip_special_tokens=False)) for t in gen_n[:10]])
            # --- end DIAG3 ---

            # ================== DIAG-A: generation args (rank0, step0 only) ==================
            if step == 0 and is_rank0:
                eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
                print("[CHK][GEN_ARGS] do_sample =", do_sample)
                print("[CHK][GEN_ARGS] num_beams =", 1)
                print("[CHK][GEN_ARGS] max_new_tokens =", max_new_tokens)
                print("[CHK][GEN_ARGS] eos_token_id(<|im_end|>) =", eos_id)
                print("[CHK][GEN_ARGS] pad_token_id =", tokenizer.pad_token_id)

                gc = getattr(model, "generation_config", None)
                if gc is not None:
                    print("[CHK][GENCFG] do_sample/temperature/top_p/top_k/max_new_tokens =",
                        getattr(gc, "do_sample", None),
                        getattr(gc, "temperature", None),
                        getattr(gc, "top_p", None),
                        getattr(gc, "top_k", None),
                        getattr(gc, "max_new_tokens", None))
            # ============================================================================== 

            # Generate
            out = model.generate(
                **gen_batch,
                do_sample=False,        # ✅ 강제
                temperature=None,       # ✅ sampling 파라미터 비활성화
                top_p=None,
                top_k=None,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
            gen = out.sequences
            gen_scores = out.scores  # list length == #generated tokens, each (bs, vocab)

            # ✅ [DIAG2] step0에서만: 멀티모달 키를 아예 제거하고 generate 비교
            if step == 0 and is_rank0:
                import copy as _copy
                ab_drop = _copy.deepcopy(gen_batch)

                # 멀티모달 키 제거
                ab_drop.pop("images", None)
                ab_drop.pop("spectrogram", None)
                ab_drop.pop("modalities", None)

                # ✅✅✅ 여기: -200(<image> placeholder)이 남아있으면 DROP_MM generate를 하면 안 됨
                if torch.is_tensor(ab_drop.get("input_ids", None)) and (ab_drop["input_ids"] == -200).any():
                    print("[DIAG][DROP_MM] skip: input_ids contains -200 but MM keys were dropped.", flush=True)
                else:
                    try:
                        out3 = model.generate(
                            **ab_drop,
                            do_sample=do_sample,
                            num_beams=1,
                            max_new_tokens=max_new_tokens,
                            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
                            pad_token_id=tokenizer.pad_token_id,
                            return_dict_in_generate=True,
                            output_scores=True,
                        )
                        txt3 = _safe_decode(tokenizer, out3.sequences[0], skip_special_tokens=False)
                        print("\n[DIAG][DROP_MM] decoded head:", repr(txt3[:300]), flush=True)
                    except Exception as e:
                        print("\n[DIAG][DROP_MM] raised:", repr(e), flush=True)

            # ===== (PATCH #1) Robust: gen이 prompt 포함인지/생성-only인지 자동 판별 =====
            inp_len = gen_batch["input_ids"].shape[1] if gen_batch.get("input_ids") is not None else 0
            gen_len = gen.shape[1]

            # Case A) gen이 prompt+생성을 함께 반환 (gen_len >= inp_len)
            # Case B) gen이 생성 토큰만 반환 (gen_len < inp_len)
            if gen_len >= inp_len:
                gen_only = gen[0, inp_len:]
            else:
                gen_only = gen[0]

            def _sequence_confidence(gen_only_ids_1d: torch.Tensor, scores_list) -> float:
                """
                평균 log-prob 기반 confidence.
                conf01 = exp(mean logprob) ∈ (0,1]
                """
                if scores_list is None:
                    return 0.0
                T = int(gen_only_ids_1d.numel())
                if T == 0 or len(scores_list) == 0:
                    return 0.0

                T2 = min(T, len(scores_list))
                ids = gen_only_ids_1d[:T2].to(torch.long)

                logps = []
                for t in range(T2):
                    logits = scores_list[t][0]  # (vocab,)
                    lp = F.log_softmax(logits, dim=-1)[ids[t]]
                    logps.append(lp)

                mean_logp = torch.stack(logps).mean()
                conf01 = float(torch.exp(mean_logp).item())
                return conf01

            # ================== 여기부터 디버그 ==================
            if step < 3:
                print("\n================ GEN DEBUG ================")
                print("[DBG] step:", step)
                print("[DBG] inp_len:", inp_len, "gen_len:", gen_len)
                print("[DBG] gen shape:", tuple(gen.shape))
                print("[DBG] gen_only len:", int(gen_only.numel()))
                print("[DBG] first 50 gen_only ids:", gen_only[:50].tolist())

                dbg_keep = tokenizer.decode(gen_only, skip_special_tokens=False)
                dbg_skip = tokenizer.decode(gen_only, skip_special_tokens=True)
                print("[DBG] decoded (keep special) head:", repr(dbg_keep[:200]))
                print("[DBG] decoded (skip special) head:", repr(dbg_skip[:200]))

                has_time = any(tok in dbg_keep for tok in ["<t0>", "<t1>", "<t2>", "<tdot>"])
                print("[DBG] contains time tokens?:", has_time)
                print("[DBG] gen decoded full head:", repr(tokenizer.decode(gen[0], skip_special_tokens=False)[:200]))
                print("[DBG] gen decoded full tail:", repr(tokenizer.decode(gen[0], skip_special_tokens=False)[-200:]))
                print("==========================================\n")
            # ================== 디버그 끝 ==================

            txt = _safe_decode(tokenizer, gen_only, skip_special_tokens=False)

            txt = txt.replace("<|im_start|>", "").replace("<|im_end|>", "").replace("assistant", "").strip()

            # ✅ (NEW) raw 저장용
            raw_txt = txt

            conf01 = _sequence_confidence(gen_only, gen_scores)

            gen_text = _normalize_pred_to_json_array(raw_txt)

            # score 덮어쓰기
            try:
                arr = json.loads(gen_text)
                if isinstance(arr, list):
                    for o in arr:
                        if isinstance(o, dict):
                            o["score"] = conf01
                    gen_text = json.dumps(arr, ensure_ascii=False)
            except Exception:
                pass

            results.append({
                "id": ids,
                "prompt": prompts,
                "pred_raw": raw_txt,   # ✅ (NEW) 모델이 실제 생성한 원문
                "pred": gen_text,      # 기존: 후처리된 JSON
            })

    return results

def _get_single_token_id(tokenizer, text: str):
    """Return a single token id for `text`. If it splits into multiple tokens, fall back to the first."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) == 0:
        raise ValueError(f"Tokenizer cannot encode '{text}' (empty).")
    if len(ids) > 1:
        # Qwen tokenizer sometimes splits; we fall back to the first for initialization.
        # You may change to averaging if you want, but keep it simple first.
        print(f"[WARN] '{text}' is encoded into multiple tokens {ids}. Using the first one for init.")
    return ids[0]


def add_time_tokens_and_init(tokenizer, model, add_sep_sync: bool = True):
    """
    VTG-LLM 스타일:
    - time tokens: <t0>...<t9>, <tdot>
    - (옵션) <tsep>, <tsync>
    - 임베딩/LM head를 기존 숫자/ '.' 토큰 임베딩으로 복사 초기화
    """
    time_tokens = [f"<t{i}>" for i in range(10)] + ["<tdot>"]
    if add_sep_sync:
        time_tokens += ["<tsep>", "<tsync>"]

    # tokenizer에 추가 (special token으로)
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": time_tokens})
    if num_added == 0:
        print("[INFO] Time tokens already exist in tokenizer. Skipping add_special_tokens().")
    else:
        print(f"[INFO] Added {num_added} time special tokens.")

    # 모델 임베딩 크기 확장
    model.resize_token_embeddings(len(tokenizer))

    # 입력 임베딩 / 출력 헤드 접근
    in_embed = model.get_input_embeddings()
    out_embed = model.get_output_embeddings()  # 보통 lm_head

    # base token id (digit, dot)
    base_digit_ids = {str(i): _get_single_token_id(tokenizer, str(i)) for i in range(10)}
    base_dot_id = _get_single_token_id(tokenizer, ".")

    # time token id
    t_ids = {f"<t{i}>": tokenizer.convert_tokens_to_ids(f"<t{i}>") for i in range(10)}
    t_ids["<tdot>"] = tokenizer.convert_tokens_to_ids("<tdot>")

    # 안전 체크
    for k, v in t_ids.items():
        if v is None or v < 0:
            raise ValueError(f"[ERROR] Time token {k} not found after adding. id={v}")

    # 임베딩 복사 초기화
    with torch.no_grad():
        for i in range(10):
            src_id = base_digit_ids[str(i)]
            dst_id = t_ids[f"<t{i}>"]
            in_embed.weight[dst_id].copy_(in_embed.weight[src_id])
            if out_embed is not None and hasattr(out_embed, "weight"):
                out_embed.weight[dst_id].copy_(out_embed.weight[src_id])

        # dot
        dst_id = t_ids["<tdot>"]
        in_embed.weight[dst_id].copy_(in_embed.weight[base_dot_id])
        if out_embed is not None and hasattr(out_embed, "weight"):
            out_embed.weight[dst_id].copy_(out_embed.weight[base_dot_id])

    print("[INFO] Initialized time token embeddings from digit/dot token embeddings.")

def _unwrap_core_for_emb(m):
    # DDP
    if hasattr(m, "module"):
        m = m.module
    # PEFT (PeftModel)
    if hasattr(m, "get_base_model"):
        try:
            m = m.get_base_model()
        except Exception:
            pass
    return m

def apply_time_token_grad_mask(model, tokenizer, *, only_time_tokens: bool = True):
    """
    임베딩/출력헤드의 grad를 time token row에만 남기도록 마스킹 hook을 겁니다.
    - only_time_tokens=True: time row만 업데이트
    - False: 전체 embedding 업데이트 (권장 X)
    """
    if not getattr(model.config, "add_time_token", False) and not getattr(getattr(model, "config", None), "add_time_token", False):
        # config에 플래그가 없더라도, tokenizer에 time token이 있으면 훅을 걸 수는 있음
        pass

    m = _unwrap_core_for_emb(model)

    time_tokens = [
        "<t0>","<t1>","<t2>","<t3>","<t4>","<t5>","<t6>","<t7>","<t8>","<t9>",
        "<tdot>","<tsep>","<tsync>"
    ]
    time_ids = [tokenizer.convert_tokens_to_ids(t) for t in time_tokens]
    time_ids = [i for i in time_ids if i is not None and i >= 0 and i != tokenizer.unk_token_id]
    time_id_set = set(time_ids)

    if len(time_id_set) == 0:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank == 0:
            print("[WARN] apply_time_token_grad_mask: time_ids is empty. skip hook.")
        return

    def _mask_grad(grad: torch.Tensor):
        if grad is None:
            return None
        if not only_time_tokens:
            return grad
        g = grad.clone()
        mask = torch.zeros(g.size(0), device=g.device, dtype=torch.bool)
        idx = torch.tensor(sorted(list(time_id_set)), device=g.device, dtype=torch.long)
        mask[idx] = True
        g[~mask] = 0
        return g

    # input embedding
    in_emb = m.get_input_embeddings() if hasattr(m, "get_input_embeddings") else None
    if in_emb is not None and hasattr(in_emb, "weight") and in_emb.weight is not None:
        in_emb.weight.requires_grad_(True)
        in_emb.weight.register_hook(_mask_grad)

    # output embedding / lm_head (tied일 가능성 높음)
    out_emb = m.get_output_embeddings() if hasattr(m, "get_output_embeddings") else None
    if out_emb is not None and hasattr(out_emb, "weight") and out_emb.weight is not None:
        out_emb.weight.requires_grad_(True)
        out_emb.weight.register_hook(_mask_grad)

    # rank0 디버그
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank == 0:
        t0 = tokenizer.convert_tokens_to_ids("<t0>")
        print("[PATCH][TIME_MASK] time_ids size =", len(time_id_set), " e.g. <t0> id =", t0, flush=True)
        if in_emb is not None:
            print("[PATCH][TIME_MASK] input_emb.requires_grad =", in_emb.weight.requires_grad, flush=True)
        if out_emb is not None:
            print("[PATCH][TIME_MASK] output_emb.requires_grad =", out_emb.weight.requires_grad, flush=True)

@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    image_processor: Optional[str] = field(default=None)
    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_spatial_pool_stride: Optional[int] = field(default=None)
    mm_spatial_pool_mode: str = field(default="average")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None)
    mm_pooling_position: Optional[str] = field(default="before")
    mm_newline_position: Optional[str] = field(default="grid")
    modality_max_length: Optional[str] = field(default="None")
    audio_visual: bool = False
    whisper_path: str = "openai/whisper-large-v3"
    freeze_whisper: bool = True
    num_speech_query_token: int = 1
    freeze_speech_QFormer: bool = False
    window_level_Qformer: bool = True
    second_per_window: float = 0.333333
    second_stride: float = 0.333333
    use_final_linear: bool = False
    freeze_final_linear: bool = False
    add_time_token: bool = False
    temporal_supervised: bool = False
    temporal_num_bins: int = 200
    temporal_loss_weight: float = 1.0

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    test_data_path: str = field(default=None, metadata={"help": "Path to the test data."})
    is_multimodal: bool = True
    video_fps: Optional[int] = field(default=1)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: int = 224
    image_split_resolution: int = 224
    audio_processor : str = "openai/whisper-large-v3"
    max_time: int = 30
    use_timestamps_crop: bool = field(default=True)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=32768,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    seed: int = 2024
    load_from_lora: bool = False
    do_test: bool = False
    test_output_dir: str = None
    evaluation_strategy: str = "steps"
    eval_steps: int = 1000
    disable_tqdm: bool = False
    lora_path: str = None
    do_sample: bool = False
    model_base: str = None
    max_new_tokens: int = 1024
    ckpt: str = None
    do_demo: bool = False
    pretrain_weight: str = None
    load_full: bool = False
    merge_and_new_lora: bool = False
    dpo_train: bool = False
    loss_type: str = "sigmoid"
    ce_loss_weight: float = 0.1
    with_ce_loss: bool = False
    beta: float = 0.1

class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 0:
            control.should_evaluate = True

class TrainGenProbeCallback(TrainerCallback):
    def __init__(self, tokenizer, every_steps=50, max_new_tokens=256):
        self.tokenizer = tokenizer
        self.every_steps = every_steps
        self.max_new_tokens = max_new_tokens
        self._cached_batch = None
        self.trainer = None  # ✅ trainer를 밖에서 주입받을 자리

    def _rank0_only(self):
        return (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)

    def _get_trainer(self, kwargs):
        # ✅ 우선 self.trainer 사용, 없으면 kwargs에서 fallback
        t = getattr(self, "trainer", None)
        if t is not None:
            return t
        return kwargs.get("trainer", None)

    def on_train_begin(self, args, state, control, **kwargs):
        if not self._rank0_only():
            return
        trainer = self._get_trainer(kwargs)
        if trainer is None:
            print("[WARN][GEN_PROBE] trainer is None on_train_begin; will try later.", flush=True)
            return
        dl = trainer.get_train_dataloader()
        self._cached_batch = next(iter(dl))

    def on_step_end(self, args, state, control, **kwargs):
        if not self._rank0_only():
            return
        if state.global_step % self.every_steps != 0:
            return

        trainer = self._get_trainer(kwargs)
        if trainer is None:
            print("[WARN][GEN_PROBE] trainer is None; cannot probe.", flush=True)
            return

        if self._cached_batch is None:
            dl = trainer.get_train_dataloader()
            self._cached_batch = next(iter(dl))

        model = kwargs["model"]
        model.eval()

        with torch.no_grad():
            b = self._cached_batch

            # ✅ generate에 넣을 키만 추려서 가져오기
            keys = ["input_ids", "attention_mask", "images", "modalities", "spectrogram", "org_groups", "real_time"]
            gen_batch = {k: b.get(k, None) for k in keys if k in b and b.get(k, None) is not None}

            # cuda 이동
            for k in ["input_ids", "attention_mask"]:
                if torch.is_tensor(gen_batch.get(k, None)):
                    gen_batch[k] = gen_batch[k].cuda(non_blocking=True)

            if "images" in gen_batch:
                if isinstance(gen_batch["images"], (list, tuple)):
                    gen_batch["images"] = [
                        it.to(torch.bfloat16).cuda(non_blocking=True) if torch.is_tensor(it) else it
                        for it in gen_batch["images"]
                    ]
                elif torch.is_tensor(gen_batch["images"]):
                    gen_batch["images"] = [gen_batch["images"].to(torch.bfloat16).cuda(non_blocking=True)]

            if "spectrogram" in gen_batch and torch.is_tensor(gen_batch["spectrogram"]):
                gen_batch["spectrogram"] = gen_batch["spectrogram"].to(torch.bfloat16).cuda(non_blocking=True)

            out = model.generate(
                **gen_batch,
                do_sample=getattr(args, "do_sample", False),
                num_beams=1,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                pad_token_id=self.tokenizer.pad_token_id,
            )
            inp_len = gen_batch["input_ids"].shape[1]
            seq = out[0]

            # generate가 prompt+gen을 같이 내는 경우가 대부분이라 slice
            gen_only = seq[inp_len:] if seq.numel() >= inp_len else seq

            txt_keep = self.tokenizer.decode(gen_only, skip_special_tokens=False)
            txt_skip = self.tokenizer.decode(gen_only, skip_special_tokens=True)

            print(f"\n[TRAIN][GEN_PROBE] step={state.global_step}")
            print("[GEN_ONLY][keep_special]:", repr(txt_keep[:800]))
            print("[GEN_ONLY][skip_special]:", repr(txt_skip[:800]))
            print("", flush=True)

        model.train()

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank == 0:
        print("[DEBUG argv]", " ".join(sys.argv))
        print("[DEBUG parsed] ckpt=", training_args.ckpt, " output_dir=", training_args.output_dir, " do_test=", training_args.do_test, " load_from_lora=", training_args.load_from_lora)


    if getattr(training_args, "gradient_checkpointing", False):
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    model_args.fps = data_args.video_fps
    model_args.lora_enable = training_args.lora_enable
    model_args.lora_r = training_args.lora_r
    model_args.lora_alpha = training_args.lora_alpha
    model_args.lora_dropout = training_args.lora_dropout

    seed = training_args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Seed: {seed}")

    audio_config = dict(
        audio_visual=model_args.audio_visual,
        video_fps=data_args.video_fps,
        whisper_path=model_args.whisper_path,
        num_speech_query_token = model_args.num_speech_query_token,
        window_level_Qformer = model_args.window_level_Qformer,
        second_per_window = model_args.second_per_window,
        second_stride = model_args.second_stride,
        use_final_linear=model_args.use_final_linear,
    )

    if not training_args.load_from_lora:
        cfg_pretrained = AutoConfig.from_pretrained(training_args.model_base)
        overwrite_config = {"model_args": vars(model_args), "add_time_token": model_args.add_time_token}

        print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(cfg_pretrained, k, v)
        model = VideoSALMONN2ForCausalLM.from_pretrained(
            training_args.ckpt,
            config=cfg_pretrained,
            cache_dir=training_args.cache_dir,
            # attn_implementation="flash_attention_2",
            attn_implementation="sdpa",
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **audio_config
        )

        if training_args.gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


        if training_args.lora_enable:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj"],
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            if training_args.bits == 16:
                if training_args.bf16:
                    model.to(torch.bfloat16)
                if training_args.fp16:
                    model.to(torch.float16)
            print("Adding LoRA adapters...")

            if model_args.audio_visual:
                speech_encoder = model.speech_encoder
                model.speech_encoder = None
                
                v_flag = False
                if hasattr(model, "vision_tower"):
                    vision_tower = model.vision_tower
                    model.vision_tower = None
                    v_flag = True

                model = get_peft_model(model, lora_config)
                model.model.speech_encoder = speech_encoder
                
                if v_flag:
                    model.model.model.vision_tower = vision_tower
                    
            else:
                v_flag = False
                if hasattr(model.model, "vision_tower"):
                    vision_tower = model.model.vision_tower
                    del model.model.vision_tower
                    v_flag = True

                model = get_peft_model(model, lora_config)
                
                if v_flag:
                    model.model.model.vision_tower = vision_tower

            if training_args.bf16:
                model.to(torch.bfloat16)

            # ====== [HERE] 임베딩(+출력 임베딩) 확실히 푸는 패치 ======
            def _unwrap(m):
                # DDP / DataParallel
                if hasattr(m, "module"):
                    m = m.module
                # PEFT wrapper (PeftModel)
                if hasattr(m, "get_base_model"):
                    try:
                        m = m.get_base_model()
                    except Exception:
                        pass
                return m

            def _unfreeze_token_embeddings_and_lm_head(model):
                m = _unwrap(model)

                # 1) input embeddings
                in_emb = m.get_input_embeddings() if hasattr(m, "get_input_embeddings") else None
                if in_emb is not None:
                    for p in in_emb.parameters():
                        p.requires_grad_(True)

                # 2) output embeddings (권장: 이게 lm_head 역할인 경우 많음)
                out_emb = m.get_output_embeddings() if hasattr(m, "get_output_embeddings") else None
                if out_emb is not None:
                    for p in out_emb.parameters():
                        p.requires_grad_(True)

                # 3) lm_head가 따로 존재하는 모델도 커버
                if hasattr(m, "lm_head") and m.lm_head is not None:
                    for p in m.lm_head.parameters():
                        p.requires_grad_(True)

                # 4) tie_weights는 보통 안전하지만, 실패해도 무시
                if hasattr(m, "tie_weights"):
                    try:
                        m.tie_weights()
                    except Exception:
                        pass

            _unfreeze_token_embeddings_and_lm_head(model)

            if rank == 0:
                mm = _unwrap(model)
                ie = mm.get_input_embeddings()
                oe = mm.get_output_embeddings() if hasattr(mm, "get_output_embeddings") else None
                print("[PATCH] input_emb grad =", (ie.weight.requires_grad if ie is not None else None))
                print("[PATCH] output_emb grad =", (oe.weight.requires_grad if oe is not None else None))
                if hasattr(mm, "lm_head") and mm.lm_head is not None and hasattr(mm.lm_head, "weight"):
                    print("[PATCH] lm_head grad =", mm.lm_head.weight.requires_grad)
            # ===========================================================

        tok_path = _resolve_tokenizer_path(training_args, model_args=model_args)
        if rank == 0:
            print("[TOK] loading tokenizer from:", tok_path)

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tok_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
        
        ensure_image_token(tokenizer, model)

        if model_args.add_time_token:
            add_time_tokens_and_init(tokenizer, model, add_sep_sync=True)
            model.config.add_time_token = True

            # ==============================
            # 🔥 CRITICAL TOKEN CONSISTENCY CHECK
            # ==============================
            img_id = tokenizer.convert_tokens_to_ids("<image>")
            t0_id  = tokenizer.convert_tokens_to_ids("<t0>")
            tdot_id = tokenizer.convert_tokens_to_ids("<tdot>")

            print("[CHK] <image> id:", img_id)
            print("[CHK] <t0> id:", t0_id)
            print("[CHK] <tdot> id:", tdot_id)

            assert img_id != t0_id, "FATAL: <image> id == <t0> id (collision)"
            assert img_id != tdot_id, "FATAL: <image> id == <tdot> id (collision)"

            if hasattr(model.config, "image_token_id"):
                print("[CHK] model.config.image_token_id:", model.config.image_token_id)
                assert int(model.config.image_token_id) == int(img_id), \
                    "FATAL: model.config.image_token_id != tokenizer('<image>')"
            # ==============================

            # ---- DEBUG PRINTS ----
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            if rank == 0:
                print("Tokenizer vocab size:", len(tokenizer))
                print("Time token id <t0>:", tokenizer.convert_tokens_to_ids("<t0>"))
                print("Time token id <tdot>:", tokenizer.convert_tokens_to_ids("<tdot>"))
                print("Embedding size:", model.get_input_embeddings().weight.shape)
                print("Time token exists?:", "<t0>" in tokenizer.get_vocab())
                print("Embedding rows == tokenizer size?:", model.get_input_embeddings().weight.shape[0] == len(tokenizer))
                print("[TOK] additional_special_tokens:", tokenizer.additional_special_tokens)
                print("[TOK] special_tokens_map:", tokenizer.special_tokens_map)
                print("[TOK] <t0> id:", tokenizer.convert_tokens_to_ids("<t0>"))
                print("[TOK] is <t0> in added vocab?:", "<t0>" in tokenizer.get_vocab())
            # ----------------------------------
                    
    else:
        # load_from_lora=True 인 경우:
        #   ckpt = base model dir
        #   lora_path = adapter checkpoint dir
        assert training_args.lora_path is not None, "ERROR: --lora_path must be set when --load_from_lora True"

        model, tokenizer = load_qwen_lora_model(
            training_args.lora_path,                 # ✅ adapter dir로 사용
            model_base=training_args.ckpt,           # ✅ base dir로 사용
            lora_enable=training_args.lora_enable,
            pretrain_weight=training_args.pretrain_weight,
            load_full=training_args.load_full,
            lora_r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            model_max_length=training_args.model_max_length,
            new_model_args=model_args,
            **audio_config
        )

        # --- [PATCH] Prefer tokenizer from lora_path if present (critical for time tokens) ---
        tok_from = None
        tok_ckpt = training_args.lora_path
        tok_base = training_args.ckpt

        if os.path.isdir(tok_ckpt) and (
            os.path.exists(os.path.join(tok_ckpt, "tokenizer_config.json")) or
            os.path.exists(os.path.join(tok_ckpt, "tokenizer.json")) or
            os.path.exists(os.path.join(tok_ckpt, "special_tokens_map.json"))
        ):
            tok_from = tok_ckpt
        else:
            tok_from = tok_base

        if dist.get_rank() == 0:
            print(f"[TOK][LOAD_FROM_LORA] reload tokenizer from: {tok_from}")

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tok_from,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )

        # tokenizer가 바뀌었으니, 모델 임베딩 크기도 항상 일치시킴
        model.resize_token_embeddings(len(tokenizer))
        # -------------------------------------------------------------------------------

        ensure_image_token(tokenizer, model)

        if model_args.add_time_token:
            add_time_tokens_and_init(tokenizer, model, add_sep_sync=True)
            model.config.add_time_token = True

            # ==============================
            # 🔥 CRITICAL TOKEN CONSISTENCY CHECK
            # ==============================
            img_id = tokenizer.convert_tokens_to_ids("<image>")
            t0_id  = tokenizer.convert_tokens_to_ids("<t0>")
            tdot_id = tokenizer.convert_tokens_to_ids("<tdot>")

            print("[CHK] <image> id:", img_id)
            print("[CHK] <t0> id:", t0_id)
            print("[CHK] <tdot> id:", tdot_id)

            assert img_id != t0_id, "FATAL: <image> id == <t0> id (collision)"
            assert img_id != tdot_id, "FATAL: <image> id == <tdot> id (collision)"

            if hasattr(model.config, "image_token_id"):
                print("[CHK] model.config.image_token_id:", model.config.image_token_id)
                assert int(model.config.image_token_id) == int(img_id), \
                    "FATAL: model.config.image_token_id != tokenizer('<image>')"
            # ==============================

            # ---- DEBUG PRINTS ----
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            if rank == 0:
                print("Tokenizer vocab size:", len(tokenizer))
                print("Time token id <t0>:", tokenizer.convert_tokens_to_ids("<t0>"))
                print("Time token id <tdot>:", tokenizer.convert_tokens_to_ids("<tdot>"))
                print("Embedding size:", model.get_input_embeddings().weight.shape)
                print("Time token exists?:", "<t0>" in tokenizer.get_vocab())
                print("Embedding rows == tokenizer size?:", model.get_input_embeddings().weight.shape[0] == len(tokenizer))
                print("[TOK] additional_special_tokens:", tokenizer.additional_special_tokens)
                print("[TOK] special_tokens_map:", tokenizer.special_tokens_map)
                print("[TOK] <t0> id:", tokenizer.convert_tokens_to_ids("<t0>"))
                print("[TOK] is <t0> in added vocab?:", "<t0>" in tokenizer.get_vocab())            
            # ----------------------------------

        # if True:
        #     tmp_dir = "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/video-SALMONN2/output/video_SALMONN_2"
        #     if dist.get_rank() == 0:
        #         import shutil
        #         if os.path.exists(tmp_dir):
        #             shutil.rmtree(tmp_dir)
        #         breakpoint()
        #         model = model.merge_and_unload()
        #         model.to(torch.bfloat16)
        #         model.save_pretrained(tmp_dir)
        #         exit()
        #     dist.barrier()
        #     model = VideoSALMONN2ForCausalLM.from_pretrained(tmp_dir, low_cpu_mem_usage=True, device_map="cuda", attn_implementation="flash_attention_2", **audio_config)
        #     model.to(torch.bfloat16)

        if training_args.merge_and_new_lora:
            print("Merging LoRA")
            model = model.merge_and_unload()
            if training_args.lora_enable:
                from peft import LoraConfig, get_peft_model
                lora_config = LoraConfig(
                    r=training_args.lora_r,
                    lora_alpha=training_args.lora_alpha,
                    target_modules=["q_proj", "k_proj", "v_proj"],
                    lora_dropout=training_args.lora_dropout,
                    bias=training_args.lora_bias,
                    task_type="CAUSAL_LM",
                )
                if model_args.audio_visual:
                    if True:
                        speech_encoder = model.speech_encoder
                        model.speech_encoder = None
                        v_flag = False
                        if hasattr(model.model, "vision_tower"):
                            vision_tower = model.model.vision_tower
                            del model.model.vision_tower
                            v_flag = True

                        model = get_peft_model(model, lora_config)

                        model.model.speech_encoder = speech_encoder
                        if v_flag:
                            model.model.model.vision_tower = vision_tower
                else:
                    v_flag = False
                    if hasattr(model.model, "vision_tower"):
                        vision_tower = model.model.vision_tower
                        del model.model.vision_tower
                        v_flag = True

                    model = get_peft_model(model, lora_config)
                    
                    if v_flag:
                        model.model.model.vision_tower = vision_tower
        else:
            print("Not merging LoRA")

        model.config.use_cache = False
        if model_args.freeze_backbone:
            model.model.requires_grad_(False)
            model.lm_head.requires_grad_(False)
        for n, p in model.named_parameters():
            if "lora" in n:
                p.requires_grad = True

        if training_args.gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if model_args.version == "qwen_1_5":
        # ✅ FORCE pad_token to be consistent across TRAIN/TEST
        # Qwen tokenizer usually has "<|endoftext|>" as a valid token.
        PAD = "<|endoftext|>"

        pad_id = tokenizer.convert_tokens_to_ids(PAD)
        if pad_id is None or pad_id < 0:
            # If somehow missing, add it explicitly (rare)
            tokenizer.add_special_tokens({"pad_token": PAD})
            pad_id = tokenizer.convert_tokens_to_ids(PAD)

            # ensure embeddings resized if vocab changed
            model.resize_token_embeddings(len(tokenizer))

        tokenizer.pad_token = PAD
        tokenizer.pad_token_id = pad_id
        tokenizer.padding_side = "right"

        # keep eos as <|im_end|> (do NOT overwrite eos)
        # tokenizer.eos_token should already be "<|im_end|>" in this repo

        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank == 0:
            print("[TOK][FORCE] pad_token:", tokenizer.pad_token, "pad_token_id:", tokenizer.pad_token_id)
    else:
        raise NotImplementedError
    
    # ================== DIAG-B: time token atom check (rank0 only) ==================
    if dist.get_rank() == 0:
        def _atom(tok: str):
            ids = tokenizer.encode(tok, add_special_tokens=False)
            back = tokenizer.decode(ids, skip_special_tokens=False)
            print(f"[CHK][TOK_ATOM] {tok} -> ids={ids} (len={len(ids)}) -> back={repr(back)}")

        for t in ["<t0>", "<t3>", "<t7>", "<tdot>"]:
            _atom(t)

        ats = getattr(tokenizer, "additional_special_tokens", None)
        print("[CHK][TOK_ATOM] has all time tokens?",
            ats is not None and all(x in ats for x in
                [f"<t{i}>" for i in range(10)] + ["<tdot>"]))
        print("[CHK][TOK_ATOM] additional_special_tokens sample:", ats[:30] if ats else None)
    # ============================================================================== 

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
    
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    data_args.image_processor = copy.deepcopy(vision_tower.image_processor)
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    if data_args.image_grid_pinpoints is not None:
        data_args.image_grid_pinpoints = ast.literal_eval(data_args.image_grid_pinpoints)
    model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
    model.config.image_crop_resolution = data_args.image_crop_resolution
    model.config.image_split_resolution = data_args.image_split_resolution
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.mm_newline_position = model_args.mm_newline_position
    assert model_args.mm_pooling_position in ["before", "after", "no"] # "mm_pooling_position must be either 'before' or 'after' or 'no'"
    model.config.mm_spatial_pool_stride = model_args.mm_spatial_pool_stride
    model.config.mm_pooling_position = model_args.mm_pooling_position
    model.config.mm_spatial_pool_mode = model_args.mm_spatial_pool_mode
    model.config.modality_max_length = model_args.modality_max_length

    # ===== Freeze policy (single source of truth) =====
    if training_args.lora_enable:
        # 1) freeze all
        model.requires_grad_(False)

        # 2) enable LoRA params
        for n, p in model.named_parameters():
            if "lora" in n.lower():
                p.requires_grad_(True)

        # 3) if time-token training: enable emb/lm_head (but we'll mask grads to time rows)
        if model_args.add_time_token:
            unfreeze_embeddings_and_lm_head(model)  # <- 위에 정의된 함수 사용
            # 주의: grad mask는 trainer 생성 직전에 다시 걸어도 되지만,
            # 여기서 한 번 걸어두면 "실수로 다시 freeze"되는 위험을 줄입니다.
            apply_time_token_grad_mask(model, tokenizer, only_time_tokens=True)

    else:
        # non-LoRA: follow freeze_backbone
        if model_args.freeze_backbone:
            model.model.requires_grad_(False)
            if hasattr(model, "lm_head") and model.lm_head is not None:
                model.lm_head.requires_grad_(False)
        else:
            model.model.requires_grad_(True)
            if hasattr(model, "lm_head") and model.lm_head is not None:
                model.lm_head.requires_grad_(True)
    # =========================================

    if model_args.audio_visual:
        if not model_args.freeze_whisper:
            for p in model.speech_encoder.parameters():
                p.requires_grad = True
            for p in model.ln_speech.parameters():
                p.requires_grad = True
        else:
            for p in model.speech_encoder.parameters():
                p.requires_grad = False
            for p in model.ln_speech.parameters():
                p.requires_grad = False
            model.speech_encoder.eval()

        if model_args.freeze_speech_QFormer:
            for name, param in model.speech_Qformer.named_parameters():
                param.requires_grad = False
            model.speech_Qformer.eval()
            model.speech_query_tokens.requires_grad = False
        else:
            for name, param in model.speech_Qformer.named_parameters():
                param.requires_grad = True
            model.speech_Qformer.train()
            model.speech_query_tokens.requires_grad = True

        if model_args.use_final_linear:
            for p in model.final_linear.parameters():
                p.requires_grad = True
        if model_args.freeze_final_linear:
            for p in model.final_linear.parameters():
                p.requires_grad = False

    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    if training_args.freeze_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False
        model.get_model().image_newline.requires_grad = False
    else:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
        if hasattr(model.get_model(), "image_newline"):
            model.get_model().image_newline.requires_grad = True

    model.config.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
    if model_args.unfreeze_mm_vision_tower:
        vision_tower.requires_grad_(True)
    else:
        vision_tower.requires_grad_(False)

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token

    if training_args.do_test or training_args.do_demo:
        data_module = make_test_data_module(tokenizer=tokenizer, data_args=data_args)
        test_dataset = data_module["eval_dataset"]
        print("[TEST] test_data_path =", data_args.test_data_path)
        print("[TEST] len(test_dataset) =", len(test_dataset))
        
        model.to(torch.bfloat16).cuda()
        model.eval()

        # ===== [PATCH] FORCE deterministic generation config (TEST) =====
        # generate() 인자로 do_sample=False를 주더라도,
        # model.generation_config가 do_sample=True로 남아 있으면 내부에서 혼선/경고가 발생할 수 있어
        # TEST에서는 아예 config 자체를 deterministic으로 고정합니다.
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.do_sample = False
            model.generation_config.temperature = None
            model.generation_config.top_p = None
            model.generation_config.top_k = None
            model.generation_config.num_beams = 1
        # ===============================================================

        # ✅ CHECKPOINT/LoRA 로드 여부 확인 (TEST에서만)
        if dist.get_rank() == 0:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"[CHECK] trainable params: {trainable} / {total}")

            # LoRA가 붙어있으면 보통 lora 관련 파라미터 이름이 존재합니다.
            lora_names = [n for n, _ in model.named_parameters() if "lora" in n.lower()]
            print(f"[CHECK] #lora params (by name): {len(lora_names)}")
            if len(lora_names) > 0:
                print("[CHECK] sample lora param name:", lora_names[0])

        if training_args.lora_enable:
            model.model.tokenizer = tokenizer
        else:
            model.tokenizer = tokenizer

        if training_args.do_demo:
            from llava.conversation import conv_templates, SeparatorStyle
            from llava.mm_utils import KeywordsStoppingCriteria
            from transformers import set_seed
            test_dataset = data_module["eval_dataset"]
            data_collator = data_module["data_collator"]
            while True:
                try:
                    yaml_file = input("yaml file: ")
                    with open(yaml_file, 'r') as file:
                        yaml_data = yaml.safe_load(file)
                    if model_args.audio_visual:
                        audio_path = yaml_data.get('audio_path', None)
                    text_only = yaml_data.get("text_only", False)
                    if text_only:
                        video_path = ""
                    else:
                        video_path = yaml_data['video_path']
                    if not text_only:
                        assert os.path.exists(video_path)

                    qs = yaml_data['question']
                    max_time = yaml_data.get("max_time", 30)
                    fps = yaml_data.get("fps", 1)
                    max_new_tokens = yaml_data.get("max_new_tokens", 1024)
                    do_sample = yaml_data.get("do_sample", False)
                    top_p = yaml_data.get("top_p", 0.9)
                    seed = yaml_data.get("seed", 2024)
                    prefix = yaml_data.get("prefix", "")

                    test_dataset.max_time = max_time
                    test_dataset.data_args.video_fps = fps
                    test_dataset.max_frame_num = round(test_dataset.max_time * test_dataset.data_args.video_fps)

                    test_dataset.list_data_dict = [{}]
                    if not text_only:
                        if video_path != "":
                            test_dataset.list_data_dict[0]["video"] = video_path

                        if model_args.audio_visual and not text_only:
                            test_dataset.list_data_dict[0]["audio"] = audio_path

                        test_dataset.list_data_dict[0]["conversations"] = [
                            {
                                "from": "human",
                                "value": "<image>\n" + qs.strip(),
                            },
                            {
                                "from": "gpt",
                                "value": "",
                                "prefix": prefix,
                            }
                        ]
                    else:
                        test_dataset.list_data_dict[0]["conversations"] = [
                            {
                                "from": "human",
                                "value": qs.strip(),
                                "prefix": prefix,
                            },
                            {
                                "from": "gpt",
                                "value": ""
                            }
                        ]
                    item = test_dataset._get_item(0)

                    batch = data_collator([item])
                    
                    batch["input_ids"] = batch["input_ids"].cuda()
                    batch["labels"] = batch["labels"].cuda()
                    batch["attention_mask"] = batch["attention_mask"].cuda()
                    if not text_only:
                        batch["images"] = [it.to(torch.bfloat16).cuda() for it in batch["images"]]
                        batch["spectrogram"] = batch["spectrogram"].to(torch.bfloat16).cuda()

                    batch.pop("ids")
                    batch.pop("prompts")
                    batch.pop("ce_only")
                    batch.pop("texts")

                    conv = conv_templates['qwen_1_5'].copy()
                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    keywords = [stop_str]
                    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, batch["input_ids"])

                    set_seed(seed)
                    _ = batch.pop("ori_item", None)
                    result = model.generate(
                        do_sample=do_sample,
                        num_beams=1,
                        stopping_criteria=[stopping_criteria],
                        max_new_tokens=max_new_tokens,
                        top_p=top_p,
                        **batch
                    )
                    res_ids = result.tolist()
                    res_text = [tokenizer.decode(it) for it in res_ids]
                    print("======================")
                    print(res_text[0])
                    print("======================")                        

                except Exception as e:
                    # raise e
                    print(e, e.__traceback__.tb_lineno)
                    breakpoint()

        else:
            test_output_dir = training_args.test_output_dir
            if dist.get_rank() == 0:
                os.makedirs(test_output_dir, exist_ok=True)

            # --- DEBUG: dataset sanity check ---
            test_ds = data_module.get("eval_dataset", None)
            print("[TEST][DEBUG] eval_dataset type:", type(test_ds))
            try:
                print("[TEST][DEBUG] eval_dataset len:", len(test_ds))
                if len(test_ds) > 0:
                    ex0 = test_ds[0]
                    if isinstance(ex0, dict):
                        print("[TEST][DEBUG] first item keys:", list(ex0.keys()))
                    else:
                        print("[TEST][DEBUG] first item type:", type(ex0))
            except Exception as e:
                print("[TEST][DEBUG] cannot inspect eval_dataset:", repr(e))
            # ----------------------------------

            # ===== [TIME TOKEN GRAD MASK] apply RIGHT BEFORE trainer creation (works for DPO too) =====
            if model_args.add_time_token:
                apply_time_token_grad_mask(model, tokenizer, only_time_tokens=True)
            # ================================================================================

            if training_args.dpo_train:
                training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
                trainer = LLaVADPOTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

                # ✅ DPO여도 생성 결과를 JSON으로 저장하려면 generate 기반으로 통일하는 게 안전
                print("[TEST][INFO] DPO test: using simple_generate loop for JSON-serializable outputs.")
                results = simple_predict_generate(
                    model=model,
                    tokenizer=tokenizer,
                    trainer=trainer,
                    eval_dataset=data_module["eval_dataset"],
                    do_sample=training_args.do_sample,
                    max_new_tokens=training_args.max_new_tokens,
                    max_time=float(getattr(data_args, "max_time", 30)),
                )
            else:
                trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

                # 해결책 3: output_dir 강제
                orig_outdir = training_args.output_dir
                training_args.output_dir = test_output_dir

                print("[TEST][INFO] Using simple_generate loop for deterministic generation prompt handling.")
                results = simple_predict_generate(
                    model=model,
                    tokenizer=tokenizer,
                    trainer=trainer,
                    eval_dataset=data_module["eval_dataset"],
                    do_sample=training_args.do_sample,
                    max_new_tokens=training_args.max_new_tokens,
                    max_time=float(getattr(data_args, "max_time", 30)),
                )
                print("[TEST][INFO] results len =", len(results))

            print(f"rank {dist.get_rank()} finish predict")

            output_path = os.path.join(test_output_dir, f"test_results_rank{dist.get_rank()}.json")
            with open(output_path, 'w') as fp:
                json.dump(results, fp, ensure_ascii=False)
            dist.barrier()
            # ---- single-process shortcut: no merge needed ----
            if (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_world_size() == 1):
                if dist.get_rank() == 0:
                    with open(output_path, "r") as fp:
                        res0 = json.load(fp) or []
                    tp_path = os.path.join(test_output_dir, "test_results.json")
                    with open(tp_path, "w") as fp:
                        json.dump(res0, fp, indent=4, ensure_ascii=False)
                    print(os.path.abspath(tp_path))
                return
            # --------------------------------------------------

            if dist.get_rank() == 0:
                res = []
                print("start merging")
                for i in range(dist.get_world_size()):
                    print(f"rank {i} start merging")
                    with open(os.path.join(test_output_dir, f"test_results_rank{i}.json"), 'r') as fp:
                        res_i = json.load(fp)
                    if not res_i:
                        continue
                    res += res_i

                temp_dict = {}
                new_res = []
                for it in res:
                    key_id = str(it.get("id")) + _prompt_to_text(it.get("prompt"))
                    if key_id not in temp_dict:
                        temp_dict[key_id] = 1
                        new_res.append(it)

                res = new_res
                with open(tp_path := os.path.join(test_output_dir, f"test_results.json"), 'w') as fp:
                    json.dump(res, fp, indent=4, ensure_ascii=False)
                print(os.path.abspath(tp_path))

    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
        if training_args.lora_enable:
            model.model.tokenizer = tokenizer
        else:
            model.tokenizer = tokenizer

        # ===== [TIME TOKEN GRAD MASK] apply RIGHT BEFORE trainer creation (works for DPO too) =====
        if model_args.add_time_token:
            apply_time_token_grad_mask(model, tokenizer, only_time_tokens=True)
        # ================================================================================

        if training_args.dpo_train:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
            trainer = LLaVADPOTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
        else:
            trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
            cb = TrainGenProbeCallback(tokenizer, every_steps=50, max_new_tokens=256)
            cb.trainer = trainer  # ✅ 여기서 trainer를 직접 주입
            trainer.add_callback(cb)

        if training_args.evaluation_strategy != "no":
            trainer.add_callback(EvaluateFirstStepCallback())

        temp_cnt, temp_total = 0, 0
        if dist.get_rank() == 0:
            for k, p in model.named_parameters():
                temp_total += 1
                if p.requires_grad:
                    print(k)
                    temp_cnt += 1

            print(temp_cnt, temp_total)

        # ================== DEBUG: direct forward on 1 batch (TRAIN PATH) ==================
        if dist.get_rank() == 0:
            dl = trainer.get_train_dataloader()
            first_batch = next(iter(dl))

            # batch를 model forward에 필요한 키만 남기고 cuda로 이동
            keys = ["input_ids", "labels", "attention_mask", "images", "modalities", "spectrogram", "org_groups", "real_time"]
            b = {k: first_batch.get(k, None) for k in keys if k in first_batch}

            def _to_cuda(x):
                if torch.is_tensor(x):
                    return x.cuda(non_blocking=True)
                return x

            for k in ["input_ids", "labels", "attention_mask"]:
                if k in b and b[k] is not None:
                    b[k] = _to_cuda(b[k])

            if "images" in b and b["images"] is not None:
                if isinstance(b["images"], (list, tuple)):
                    b["images"] = [it.to(torch.bfloat16).cuda(non_blocking=True) if torch.is_tensor(it) else it for it in b["images"]]
                elif torch.is_tensor(b["images"]):
                    b["images"] = b["images"].to(torch.bfloat16).cuda(non_blocking=True)

            if "spectrogram" in b and b["spectrogram"] is not None and torch.is_tensor(b["spectrogram"]):
                b["spectrogram"] = b["spectrogram"].to(torch.bfloat16).cuda(non_blocking=True)

            model.train()
            with torch.no_grad():
                lab = b["labels"]
                print("[DBG][LABEL] dtype:", lab.dtype, "min/max:", int(lab.min()), int(lab.max()), flush=True)
                valid = (lab != -100).sum().item()
                print("[DBG][LABEL] #valid(!=-100):", valid, " / total:", lab.numel(), flush=True)
                print("[DBG][LABEL] tail raw:", lab[0, -60:].tolist(), flush=True)
                out = model(**{k: v for k, v in b.items() if v is not None})
                logits_len = out.logits.shape[1]
                lab = b["labels"]
                valid_total = (lab != -100).sum().item()
                valid_used  = (lab[:, :logits_len] != -100).sum().item()

                print("[DBG][LEN] labels_len =", lab.shape[1], "logits_len =", logits_len, flush=True)
                print("[DBG][VALID] total =", valid_total, " used(first logits_len) =", valid_used, flush=True)
                
                print("[DBG][TRAIN-DIRECT] out.loss =", getattr(out, "loss", None), flush=True)
                if getattr(out, "loss", None) is not None:
                    print(f"[DBG][TRAIN-DIRECT] out.loss.item = {out.loss.item():.8e}", flush=True)
        # ================================================================================"

        ckpts = sorted(glob.glob(os.path.join(training_args.output_dir, "checkpoint-*")))
        if len(ckpts) > 0:
            trainer.train(resume_from_checkpoint=ckpts[-1])
        else:
            trainer.train()
        trainer.save_state()
        
        if dist.get_rank() == 0:
            tokenizer.save_pretrained(training_args.output_dir)
            # 모델은 trainer가 저장할 수도 있지만, 안전하게 같이 저장하고 싶으면:
            # trainer.save_model(training_args.output_dir)
            print("[SAVE] tokenizer saved to", training_args.output_dir)

if __name__ == "__main__":
    train()