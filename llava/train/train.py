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
from llava.dataset import make_supervised_data_module, make_test_data_module
from transformers import TrainerCallback

# ==============================================================================
# VTG-LLM 방식 time token 상수 (전역)
# <t0>~<t9>: digit 0~9, <tdot>: 소수점 → 총 11개 고정
# ==============================================================================
VTG_TIME_TOKENS = [f"<t{i}>" for i in range(10)] + ["<tdot>"]


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
    return " ".join(parts)


def _unwrap_model(m):
    return m.module if hasattr(m, "module") else m


def unfreeze_embeddings_and_lm_head(model):
    m = _unwrap_model(model)

    # PeftModel을 완전히 unwrap
    if hasattr(m, "get_base_model"):
        try:
            m = m.get_base_model()
        except Exception:
            pass

    in_emb = m.get_input_embeddings()
    if in_emb is not None:
        for p in in_emb.parameters():
            p.requires_grad_(True)

    out_emb = m.get_output_embeddings()
    if out_emb is not None:
        for p in out_emb.parameters():
            p.requires_grad_(True)
    elif hasattr(m, "lm_head"):
        for p in m.lm_head.parameters():
            p.requires_grad_(True)

    # fallback: PeftModel 구조에서 lm_head 직접 탐색
    if out_emb is None and not hasattr(m, "lm_head"):
        for name, mod in model.named_modules():
            if name.endswith("lm_head"):
                for p in mod.parameters():
                    p.requires_grad_(True)
                break

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
    if model_args is not None and getattr(model_args, "add_time_token", False):
        return training_args.model_base

    outdir = getattr(training_args, "output_dir", None)
    if outdir and os.path.isdir(outdir):
        ckpts = sorted(
            glob.glob(os.path.join(outdir, "checkpoint-*")),
            key=lambda p: int(os.path.basename(p).split("-")[-1])
        )
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

    def _to_cuda_if_tensor(x):
        if torch.is_tensor(x):
            return x.cuda(non_blocking=True)
        return x

    with torch.no_grad():

        def _safe_decode(tokenizer, ids, skip_special_tokens=False):
            if torch.is_tensor(ids):
                ids = ids.tolist()
            ids = [x for x in ids if isinstance(x, int) and x >= 0]
            return tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        is_rank0 = (rank == 0)

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
            if step == 0:
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

            ids = batch.get("ids", batch.get("id", None))
            prompts = batch.get("prompts", batch.get("prompt", None))

            for k in ["input_ids", "attention_mask"]:
                if k in batch:
                    batch[k] = _to_cuda_if_tensor(batch[k])

            if "images" in batch and batch["images"] is not None:
                if isinstance(batch["images"], (list, tuple)):
                    batch["images"] = [it.to(torch.bfloat16).cuda(non_blocking=True) if torch.is_tensor(it) else it
                                      for it in batch["images"]]
                else:
                    batch["images"] = batch["images"].to(torch.bfloat16).cuda(non_blocking=True)

            if "spectrogram" in batch and batch["spectrogram"] is not None:
                if torch.is_tensor(batch["spectrogram"]):
                    batch["spectrogram"] = batch["spectrogram"].to(torch.bfloat16).cuda(non_blocking=True)

            if "images" not in batch and "image" in batch and batch["image"] is not None:
                img = batch["image"]
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

            if "modalities" not in batch and "modality" in batch and batch["modality"] is not None:
                m = batch["modality"]
                if isinstance(m, str):
                    batch["modalities"] = [m]
                else:
                    batch["modalities"] = m

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

            if step == 0 and is_rank0:
                print("\n[TEST][FINAL_INPUT_TAIL]")
                print(repr(_safe_decode(tokenizer, gen_batch["input_ids"][0][-300:], skip_special_tokens=False)))
                print("[TEST][FINAL_INPUT_IDS_TAIL]")
                print(gen_batch["input_ids"][0][-40:].tolist())
                print("[TEST][COUNT_-200]", int((gen_batch["input_ids"] == -200).sum().item()))

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

            if step < 3 and is_rank0:
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

                if "input_ids" in gen_batch and torch.is_tensor(gen_batch["input_ids"]):
                    ids_t = gen_batch["input_ids"]
                    IMAGE_ID = tokenizer.convert_tokens_to_ids("<image>")
                    n_img_tok = int((ids_t == IMAGE_ID).sum().item())
                    n_img_idx = int((ids_t == -200).sum().item())
                    print(f"[CHK][IMG_TOK] <image>_id={IMAGE_ID} count(id)={n_img_tok} count(-200)={n_img_idx}")
                    print(f"[CHK][IMG_TOK] input_ids min/max = {int(ids_t.min())}/{int(ids_t.max())}")

                # VTG-LLM 토큰 체크
                t0_id   = tokenizer.convert_tokens_to_ids("<t0>")
                t9_id   = tokenizer.convert_tokens_to_ids("<t9>")
                tdot_id = tokenizer.convert_tokens_to_ids("<tdot>")
                print(f"[CHK][TIME_ANCHOR] <t0>={t0_id} <t9>={t9_id} <tdot>={tdot_id}")

                imgs = gen_batch.get("images", None)
                print(f"[CHK][BATCH] images is None? {imgs is None}")
                if imgs is not None:
                    if torch.is_tensor(imgs):
                        print(f"[CHK][BATCH] images tensor shape={tuple(imgs.shape)} dtype={imgs.dtype} device={imgs.device}")
                    elif isinstance(imgs, (list, tuple)) and len(imgs) > 0 and torch.is_tensor(imgs[0]):
                        print(f"[CHK][BATCH] images[0] shape={tuple(imgs[0].shape)} dtype={imgs[0].dtype} device={imgs[0].device}")

                sp = gen_batch.get("spectrogram", None)
                print(f"[CHK][BATCH] spectrogram is None? {sp is None}")
                if sp is not None and torch.is_tensor(sp):
                    print(f"[CHK][BATCH] spectrogram shape={tuple(sp.shape)} dtype={sp.dtype} device={sp.device}")

            if step < 1 and is_rank0:
                ids_t = gen_batch["input_ids"]
                n_img_idx = int((ids_t == -200).sum().item())
                print(f"[DIAG][MM_EFFECT] count(-200)={n_img_idx}")

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
                print(f"[DIAG][MM_EFFECT] first_tok id={tok1} str={tokenizer.decode([tok1], skip_special_tokens=False)!r}")

                t0_id   = tokenizer.convert_tokens_to_ids("<t0>")
                t9_id   = tokenizer.convert_tokens_to_ids("<t9>")
                tdot_id = tokenizer.convert_tokens_to_ids("<tdot>")
                print(f"[CHK][TIME_ANCHOR] <t0>={t0_id} <t9>={t9_id} <tdot>={tdot_id}")

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

                s1 = out1.scores[0]
                s2 = out2.scores[0]
                delta = float((s1 - s2).abs().mean().item())
                print(f"[DIAG][MM_EFFECT] mean|score_delta|={delta:.6f}")

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

                gen_n = seq_n[prompt_len:] if len(seq_n) >= prompt_len else seq_n
                gen_z = seq_z[prompt_len:] if len(seq_z) >= prompt_len else seq_z

                txt_n = tokenizer.decode(gen_n, skip_special_tokens=False)
                txt_z = tokenizer.decode(gen_z, skip_special_tokens=False)

                print("\n[DIAG3][MM_SEQ] normal gen (head):", repr(txt_n[:200]))
                print("[DIAG3][MM_SEQ] zeroMM gen (head):", repr(txt_z[:200]))

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

                print("[CHK][DIAG3] gen_n first 10 ids:", gen_n[:10])
                decoded_first = [tokenizer.decode([t], skip_special_tokens=False) for t in gen_n[:20]]
                print("[CHK][TIME_TRACE] first 20 decoded tokens:", decoded_first)
                decoded_join = "".join(decoded_first)
                # VTG-LLM 토큰 기준 체크
                print("[CHK][TIME_TRACE] has_any_time_token?",
                    any(tok in decoded_join for tok in ["<t0>", "<t9>", "<tdot>"]))

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

            # Generate
            out = model.generate(
                **gen_batch,
                do_sample=False,
                temperature=None,
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

            if step == 0 and is_rank0:
                import copy as _copy
                ab_drop = _copy.deepcopy(gen_batch)
                ab_drop.pop("images", None)
                ab_drop.pop("spectrogram", None)
                ab_drop.pop("modalities", None)

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

            inp_len = gen_batch["input_ids"].shape[1] if gen_batch.get("input_ids") is not None else 0
            gen_len = gen.shape[1]

            if gen_len >= inp_len:
                gen_only = gen[0, inp_len:]
            else:
                gen_only = gen[0]

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

                # VTG-LLM 토큰 기준 체크
                has_time = any(tok in dbg_keep for tok in ["<t0>", "<t9>", "<tdot>"])
                print("[DBG] contains VTG-LLM time tokens?:", has_time)
                print("[DBG] gen decoded full head:", repr(tokenizer.decode(gen[0], skip_special_tokens=False)[:200]))
                print("[DBG] gen decoded full tail:", repr(tokenizer.decode(gen[0], skip_special_tokens=False)[-200:]))
                print("==========================================\n")

            txt = _safe_decode(tokenizer, gen_only, skip_special_tokens=False)
            txt = txt.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
            raw_txt = txt.strip()

            results.append({
                "id": ids,
                "prompt": prompts,
                "pred": raw_txt,
            })

    return results


def _get_single_token_id(tokenizer, text: str):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) == 0:
        raise ValueError(f"Tokenizer cannot encode '{text}' (empty).")
    if len(ids) > 1:
        print(f"[WARN] '{text}' is encoded into multiple tokens {ids}. Using the first one for init.")
    return ids[0]


def add_time_tokens_and_init(tokenizer, model):
    """
    VTG-LLM 방식: <t0>~<t9> (digit 0~9) + <tdot> (소수점) 총 11개 추가.

    초기화 전략 (VTG-LLM 논문 권장):
      - <t0>~<t9>: 기존 숫자 문자 '0'~'9' 토큰 임베딩으로 초기화
      - <tdot>:    기존 '.' 토큰 임베딩으로 초기화
    → 랜덤 초기화 대비 학습 안정성 대폭 향상
    """
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": VTG_TIME_TOKENS}
    )
    model.resize_token_embeddings(len(tokenizer))

    if num_added == 0:
        print("[INFO] VTG-LLM time tokens already exist in tokenizer. Keep existing rows.")
        return 0

    print(f"[INFO] Added {num_added} VTG-LLM time tokens: {VTG_TIME_TOKENS}")

    in_embed  = model.get_input_embeddings().weight.data
    out_embed = (model.get_output_embeddings().weight.data
                 if model.get_output_embeddings() is not None else None)

    # digit 토큰 → 대응하는 숫자 문자 임베딩으로 초기화
    # <tdot> → '.' 문자 임베딩으로 초기화
    init_map = {f"<t{i}>": str(i) for i in range(10)}
    init_map["<tdot>"] = "."

    with torch.no_grad():
        # fallback으로 사용할 mean embedding (새로 추가된 행 제외)
        in_mean  = in_embed[:-num_added].float().mean(dim=0).to(in_embed.dtype)
        out_mean = (out_embed[:-num_added].float().mean(dim=0).to(out_embed.dtype)
                    if out_embed is not None else None)

        for new_tok, orig_char in init_map.items():
            new_id   = tokenizer.convert_tokens_to_ids(new_tok)
            orig_ids = tokenizer.encode(orig_char, add_special_tokens=False)

            if len(orig_ids) == 0:
                # fallback: mean embedding
                in_embed[new_id] = in_mean
                if out_embed is not None:
                    out_embed[new_id] = out_mean
                print(f"[WARN] '{orig_char}' not found in tokenizer vocab. Used mean for {new_tok}.")
            else:
                orig_id = orig_ids[0]
                in_embed[new_id] = in_embed[orig_id].clone()
                if out_embed is not None:
                    out_embed[new_id] = out_embed[orig_id].clone()
                print(f"[INFO] {new_tok} (id={new_id}) ← '{orig_char}' (id={orig_id})")

    print("[INFO] VTG-LLM time token embeddings initialized from digit/dot character embeddings.")
    return num_added


def _unwrap_core_for_emb(m):
    if hasattr(m, "module"):
        m = m.module
    if hasattr(m, "get_base_model"):
        try:
            m = m.get_base_model()
        except Exception:
            pass
    return m


def apply_time_token_grad_mask(model, tokenizer, *, only_time_tokens: bool = True):
    """
    VTG-LLM 방식: VTG_TIME_TOKENS(11개)에 해당하는 임베딩 행만 gradient를 통과시키는 hook.
    only_time_tokens=False이면 mask 없이 전체 임베딩 학습 (SFT 후반부 권장).
    """
    if getattr(model, "_time_token_grad_mask_installed", False):
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank == 0:
            print("[PATCH][TIME_MASK] hook already installed. skip.", flush=True)
        return

    m = _unwrap_core_for_emb(model)

    time_ids = [tokenizer.convert_tokens_to_ids(t) for t in VTG_TIME_TOKENS]
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

    in_emb = m.get_input_embeddings() if hasattr(m, "get_input_embeddings") else None
    if in_emb is not None and hasattr(in_emb, "weight") and in_emb.weight is not None:
        in_emb.weight.requires_grad_(True)
        in_emb.weight.register_hook(_mask_grad)

    out_emb = m.get_output_embeddings() if hasattr(m, "get_output_embeddings") else None
    if out_emb is not None and hasattr(out_emb, "weight") and out_emb.weight is not None:
        out_emb.weight.requires_grad_(True)
        out_emb.weight.register_hook(_mask_grad)

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank == 0:
        print("[PATCH][TIME_MASK] VTG_TIME_TOKENS count =", len(time_id_set),
              " e.g. <t0>=", tokenizer.convert_tokens_to_ids("<t0>"),
              "<tdot>=", tokenizer.convert_tokens_to_ids("<tdot>"), flush=True)

    model._time_token_grad_mask_installed = True


def _load_time_rows_if_exist(model, ckpt_dir):
    path = os.path.join(ckpt_dir, "time_token_rows.pt")
    if not os.path.exists(path):
        return
    payload = torch.load(path, map_location="cpu")
    time_ids = payload.get("time_ids", [])
    if not time_ids:
        return

    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "get_base_model"):
        try:
            m = m.get_base_model()
        except Exception:
            pass

    in_emb  = m.get_input_embeddings()
    out_emb = m.get_output_embeddings()

    idx     = torch.tensor(time_ids, dtype=torch.long, device=in_emb.weight.device)
    in_rows = payload["input_emb_rows"].to(in_emb.weight.device)
    in_emb.weight.data.index_copy_(0, idx, in_rows)

    if "output_emb_rows" in payload and out_emb is not None and hasattr(out_emb, "weight"):
        out_rows = payload["output_emb_rows"].to(out_emb.weight.device)
        out_emb.weight.data.index_copy_(0, idx, out_rows)

    print(f"[RESUME][TIME_ROWS] loaded from {path}", flush=True)


@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    image_processor: Optional[str] = field(default=None)
    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)
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
    freeze_mm_projector: bool = True        # visual aligner freeze 여부 (default: freeze)
    freeze_speech_qformer: bool = True      # audio aligner freeze 여부 (default: freeze)
    temporal_supervised: bool = False
    # temporal_num_bins 제거: VTG-LLM 방식은 11개 고정 (VTG_TIME_TOKENS)
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
    audio_processor: str = "openai/whisper-large-v3"
    max_time: int = 30
    use_timestamps_crop: bool = field(default=False)
    # num_time_bins 제거: VTG-LLM 방식은 고정 11개


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=32768,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
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
        self.trainer = None

    def _rank0_only(self):
        return (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)

    def _get_trainer(self, kwargs):
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
            keys = ["input_ids", "attention_mask", "images", "modalities", "spectrogram", "org_groups", "real_time"]
            gen_batch = {k: b.get(k, None) for k in keys if k in b and b.get(k, None) is not None}

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
            gen_only = seq[inp_len:] if seq.numel() >= inp_len else seq

            txt_keep = self.tokenizer.decode(gen_only, skip_special_tokens=False)
            txt_skip = self.tokenizer.decode(gen_only, skip_special_tokens=True)

            print(f"\n[TRAIN][GEN_PROBE] step={state.global_step}")
            print("[GEN_ONLY][keep_special]:", repr(txt_keep[:800]))
            print("[GEN_ONLY][skip_special]:", repr(txt_skip[:800]))
            print("", flush=True)

        model.train()


class SaveTimeTokenRowsCallback(TrainerCallback):
    """VTG-LLM 방식: 11개 time token 임베딩 행을 체크포인트마다 저장."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _rank0_only(self):
        return (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)

    def _get_time_ids(self):
        ids = [self.tokenizer.convert_tokens_to_ids(t) for t in VTG_TIME_TOKENS]
        ids = [i for i in ids if i is not None and i >= 0 and i != self.tokenizer.unk_token_id]
        return sorted(set(ids))

    def on_save(self, args, state, control, **kwargs):
        if not self._rank0_only():
            return
        model = kwargs.get("model", None)
        if model is None:
            return

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)

        time_ids = self._get_time_ids()
        if len(time_ids) == 0:
            print("[SAVE][TIME_ROWS] time_ids empty. skip.", flush=True)
            return

        m = model.module if hasattr(model, "module") else model
        if hasattr(m, "get_base_model"):
            try:
                m = m.get_base_model()
            except Exception:
                pass

        in_emb  = m.get_input_embeddings()
        out_emb = m.get_output_embeddings()

        idx = torch.tensor(time_ids, dtype=torch.long, device=in_emb.weight.device)

        payload = {
            "time_ids": time_ids,
            "input_emb_rows": in_emb.weight.index_select(0, idx).detach().cpu(),
        }
        if out_emb is not None and hasattr(out_emb, "weight") and out_emb.weight is not None:
            payload["output_emb_rows"] = out_emb.weight.index_select(0, idx).detach().cpu()

        torch.save(payload, os.path.join(ckpt_dir, "time_token_rows.pt"))
        print(f"[SAVE][TIME_ROWS] saved {len(time_ids)} VTG-LLM token rows -> {ckpt_dir}/time_token_rows.pt", flush=True)


class SaveTrainerStatePerCheckpointCallback(TrainerCallback):
    def _rank(self):
        return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    def _world_size(self):
        return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

    def _rank0_only(self):
        return self._rank() == 0

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)

        rank = self._rank()
        world_size = self._world_size()

        if self._rank0_only():
            state_dict = asdict(state)
            ckpt_state_path = os.path.join(ckpt_dir, "trainer_state.json")
            with open(ckpt_state_path, "w", encoding="utf-8") as f:
                json.dump(state_dict, f, indent=2, ensure_ascii=False)
            root_state_path = os.path.join(args.output_dir, "trainer_state.json")
            with open(root_state_path, "w", encoding="utf-8") as f:
                json.dump(state_dict, f, indent=2, ensure_ascii=False)
            print(f"[SAVE][TRAINER_STATE] saved -> {ckpt_state_path}", flush=True)

        if self._rank0_only():
            optimizer = kwargs.get("optimizer", None)
            if optimizer is not None:
                opt_path = os.path.join(ckpt_dir, "optimizer.pt")
                torch.save(optimizer.state_dict(), opt_path)
                print(f"[SAVE][OPTIMIZER] saved -> {opt_path}", flush=True)
            else:
                print("[SAVE][OPTIMIZER] optimizer not found in kwargs; skip.", flush=True)

            lr_scheduler = kwargs.get("lr_scheduler", None)
            if lr_scheduler is None:
                lr_scheduler = kwargs.get("scheduler", None)
            if lr_scheduler is not None:
                sch_path = os.path.join(ckpt_dir, "scheduler.pt")
                torch.save(lr_scheduler.state_dict(), sch_path)
                print(f"[SAVE][SCHEDULER] saved -> {sch_path}", flush=True)
            else:
                print("[SAVE][SCHEDULER] scheduler not found in kwargs; skip.", flush=True)

        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            try:
                rng_state["cuda"] = torch.cuda.get_rng_state_all()
            except Exception:
                pass

        rng_path = (os.path.join(ckpt_dir, "rng_state.pth")
                    if world_size == 1
                    else os.path.join(ckpt_dir, f"rng_state_{rank}.pth"))
        torch.save(rng_state, rng_path)
        if self._rank0_only():
            print(f"[SAVE][RNG] saved -> {rng_path}", flush=True)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # VTG-LLM 방식에서는 num_time_bins 개념이 없으므로 data_args 동기화 불필요
    # (기존: data_args.num_time_bins = model_args.temporal_num_bins → 제거)

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank == 0:
        print("[DEBUG argv]", " ".join(sys.argv))
        print("[DEBUG parsed] ckpt=", training_args.ckpt,
              " output_dir=", training_args.output_dir,
              " do_test=", training_args.do_test,
              " load_from_lora=", training_args.load_from_lora)
        print(f"[INFO] VTG_TIME_TOKENS ({len(VTG_TIME_TOKENS)}): {VTG_TIME_TOKENS}")

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
        num_speech_query_token=model_args.num_speech_query_token,
        window_level_Qformer=model_args.window_level_Qformer,
        second_per_window=model_args.second_per_window,
        second_stride=model_args.second_stride,
        use_final_linear=model_args.use_final_linear,
    )

    if not training_args.load_from_lora:
        cfg_pretrained = AutoConfig.from_pretrained(training_args.model_base)
        overwrite_config = {
            "model_args": vars(model_args),
            "add_time_token": model_args.add_time_token,
        }
        print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(cfg_pretrained, k, v)

        model = VideoSALMONN2ForCausalLM.from_pretrained(
            training_args.ckpt,
            config=cfg_pretrained,
            cache_dir=training_args.cache_dir,
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

            def _unwrap(m):
                if hasattr(m, "module"):
                    m = m.module
                if hasattr(m, "get_base_model"):
                    try:
                        m = m.get_base_model()
                    except Exception:
                        pass
                return m

            def _unfreeze_token_embeddings_and_lm_head(model):
                m = _unwrap(model)
                in_emb = m.get_input_embeddings() if hasattr(m, "get_input_embeddings") else None
                if in_emb is not None:
                    for p in in_emb.parameters():
                        p.requires_grad_(True)
                out_emb = m.get_output_embeddings() if hasattr(m, "get_output_embeddings") else None
                if out_emb is not None:
                    for p in out_emb.parameters():
                        p.requires_grad_(True)
                if hasattr(m, "lm_head") and m.lm_head is not None:
                    for p in m.lm_head.parameters():
                        p.requires_grad_(True)
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

        tok_path = _resolve_tokenizer_path(training_args, model_args=model_args)
        if rank == 0:
            print("[TOK] loading tokenizer from:", tok_path)

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tok_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )

        # ============ CHK1: TOKENIZER BASIC ============
        if rank == 0:
            def _atom(tok: str):
                ids = tokenizer.encode(tok, add_special_tokens=False)
                back = tokenizer.decode(ids, skip_special_tokens=False)
                print(f"[TOK_ATOM] {tok} -> ids={ids} (len={len(ids)}) -> back={repr(back)}")

            print("\n==== [CHK1] TOKENIZER BASIC ====")
            print("[CHK1] vocab_size =", len(tokenizer))
            print("[CHK1] unk_token_id =", tokenizer.unk_token_id)
            print("[CHK1] pad_token/pad_id =", tokenizer.pad_token, tokenizer.pad_token_id)
            print("[CHK1] im_start_id =", tokenizer.convert_tokens_to_ids("<|im_start|>"))
            print("[CHK1] im_end_id   =", tokenizer.convert_tokens_to_ids("<|im_end|>"))
            img_id = tokenizer.convert_tokens_to_ids("<image>")
            print("[CHK1] <image> id =", img_id, " (is_unk?", img_id == tokenizer.unk_token_id, ")")
            _atom("<image>")
            # VTG-LLM 토큰 체크
            for t in ["<t0>", "<t5>", "<t9>", "<tdot>"]:
                print(f"[CHK1] {t} id =", tokenizer.convert_tokens_to_ids(t))
                _atom(t)
            print("[CHK1] additional_special_tokens(head) =", (tokenizer.additional_special_tokens or [])[:30])
            print("================================\n")
        # ==============================================

        ensure_image_token(tokenizer, model)

        if model_args.add_time_token:
            # VTG-LLM 방식: num_bins 인자 없음
            add_time_tokens_and_init(tokenizer, model)
            model.config.add_time_token = True

            # ============ CHK2: EMB INIT CHECK ============
            if rank == 0:
                m = model.module if hasattr(model, "module") else model
                if hasattr(m, "get_base_model"):
                    try:
                        m = m.get_base_model()
                    except Exception:
                        pass

                emb = m.get_input_embeddings().weight.detach()
                print("\n==== [CHK2] EMB INIT CHECK (VTG-LLM) ====")
                print("[CHK2] emb.shape =", tuple(emb.shape))
                for t in VTG_TIME_TOKENS:
                    tid = tokenizer.convert_tokens_to_ids(t)
                    print(f"[CHK2] {t} id={tid}")
                print("==========================================\n")
            # ==============================================

            img_id   = tokenizer.convert_tokens_to_ids("<image>")
            t0_id    = tokenizer.convert_tokens_to_ids("<t0>")
            tdot_id  = tokenizer.convert_tokens_to_ids("<tdot>")

            print("[CHK] <image> id:", img_id)
            print("[CHK] <t0> id:", t0_id)
            print("[CHK] <tdot> id:", tdot_id)

            assert img_id != t0_id,   "FATAL: <image> id == <t0> id (collision)"
            assert img_id != tdot_id, "FATAL: <image> id == <tdot> id (collision)"

            if hasattr(model.config, "image_token_id"):
                print("[CHK] model.config.image_token_id:", model.config.image_token_id)
                assert int(model.config.image_token_id) == int(img_id), \
                    "FATAL: model.config.image_token_id != tokenizer('<image>')"

            if rank == 0:
                print("Tokenizer vocab size:", len(tokenizer))
                for t in VTG_TIME_TOKENS:
                    print(f"Time token {t} id:", tokenizer.convert_tokens_to_ids(t))
                print("Embedding size:", model.get_input_embeddings().weight.shape)
                print("Embedding rows == tokenizer size?:",
                      model.get_input_embeddings().weight.shape[0] == len(tokenizer))
                print("[TOK] additional_special_tokens:", tokenizer.additional_special_tokens)

    else:
        # load_from_lora=True
        assert training_args.lora_path is not None, \
            "ERROR: --lora_path must be set when --load_from_lora True"

        model, tokenizer = load_qwen_lora_model(
            training_args.lora_path,
            model_base=training_args.ckpt,
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

        if rank == 0:
            print(f"[TOK][LOAD_FROM_LORA] reload tokenizer from: {tok_from}")

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tok_from,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )

        model.resize_token_embeddings(len(tokenizer))
        ensure_image_token(tokenizer, model)

        if model_args.add_time_token:
            # VTG-LLM 방식: num_bins 인자 없음
            add_time_tokens_and_init(tokenizer, model)
            model.config.add_time_token = True

            _load_time_rows_if_exist(model, training_args.lora_path)

            img_id  = tokenizer.convert_tokens_to_ids("<image>")
            t0_id   = tokenizer.convert_tokens_to_ids("<t0>")
            tdot_id = tokenizer.convert_tokens_to_ids("<tdot>")

            print("[CHK] <image> id:", img_id)
            print("[CHK] <t0> id:", t0_id)
            print("[CHK] <tdot> id:", tdot_id)

            assert img_id != t0_id,   "FATAL: <image> id == <t0> id (collision)"
            assert img_id != tdot_id, "FATAL: <image> id == <tdot> id (collision)"

            if hasattr(model.config, "image_token_id"):
                print("[CHK] model.config.image_token_id:", model.config.image_token_id)
                assert int(model.config.image_token_id) == int(img_id), \
                    "FATAL: model.config.image_token_id != tokenizer('<image>')"

            if rank == 0:
                print("Tokenizer vocab size:", len(tokenizer))
                for t in VTG_TIME_TOKENS:
                    print(f"Time token {t} id:", tokenizer.convert_tokens_to_ids(t))
                print("Embedding size:", model.get_input_embeddings().weight.shape)
                print("Embedding rows == tokenizer size?:",
                      model.get_input_embeddings().weight.shape[0] == len(tokenizer))

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
        PAD = "<|endoftext|>"
        pad_id = tokenizer.convert_tokens_to_ids(PAD)
        if pad_id is None or pad_id < 0:
            tokenizer.add_special_tokens({"pad_token": PAD})
            pad_id = tokenizer.convert_tokens_to_ids(PAD)
            model.resize_token_embeddings(len(tokenizer))

        tokenizer.pad_token    = PAD
        tokenizer.pad_token_id = pad_id
        tokenizer.padding_side = "right"

        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]

        if rank == 0:
            print("[TOK][FORCE] pad_token:", tokenizer.pad_token, "pad_token_id:", tokenizer.pad_token_id)
    else:
        raise NotImplementedError

    # ================== DIAG-B: VTG-LLM time token atom check ==================
    if rank == 0:
        def _atom(tok: str):
            ids = tokenizer.encode(tok, add_special_tokens=False)
            back = tokenizer.decode(ids, skip_special_tokens=False)
            print(f"[CHK][TOK_ATOM] {tok} -> ids={ids} (len={len(ids)}) -> back={repr(back)}")

        print("\n==== [DIAG-B] VTG-LLM TIME TOKEN ATOM CHECK ====")
        for t in VTG_TIME_TOKENS:
            _atom(t)
        ats = getattr(tokenizer, "additional_special_tokens", None)
        all_present = ats is not None and all(x in ats for x in VTG_TIME_TOKENS)
        print("[CHK][TOK_ATOM] all VTG_TIME_TOKENS present?", all_present)
        print("[CHK][TOK_ATOM] additional_special_tokens sample:", ats[:30] if ats else None)
        print("=================================================\n")
    # ===========================================================================

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    data_args.image_processor = copy.deepcopy(vision_tower.image_processor)
    data_args.is_multimodal = True

    model.config.image_aspect_ratio        = data_args.image_aspect_ratio
    if data_args.image_grid_pinpoints is not None:
        data_args.image_grid_pinpoints = ast.literal_eval(data_args.image_grid_pinpoints)
    model.config.image_grid_pinpoints      = data_args.image_grid_pinpoints
    model.config.image_crop_resolution     = data_args.image_crop_resolution
    model.config.image_split_resolution    = data_args.image_split_resolution
    model.config.tokenizer_padding_side    = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.mm_newline_position       = model_args.mm_newline_position
    assert model_args.mm_pooling_position in ["before", "after", "no"]
    model.config.mm_spatial_pool_stride   = model_args.mm_spatial_pool_stride
    model.config.mm_pooling_position      = model_args.mm_pooling_position
    model.config.mm_spatial_pool_mode     = model_args.mm_spatial_pool_mode
    model.config.modality_max_length      = model_args.modality_max_length

    # ===== Freeze policy =====
    if training_args.lora_enable:
        model.requires_grad_(False)
        for n, p in model.named_parameters():
            if "lora" in n.lower():
                p.requires_grad_(True)
        if model_args.add_time_token:
            unfreeze_embeddings_and_lm_head(model)
            apply_time_token_grad_mask(model, tokenizer, only_time_tokens=False)
    else:
        if model_args.freeze_backbone:
            model.model.requires_grad_(False)
            if hasattr(model, "lm_head") and model.lm_head is not None:
                model.lm_head.requires_grad_(False)
        else:
            model.model.requires_grad_(True)
            if hasattr(model, "lm_head") and model.lm_head is not None:
                model.lm_head.requires_grad_(True)
        if model_args.add_time_token:
            unfreeze_embeddings_and_lm_head(model)
            apply_time_token_grad_mask(model, tokenizer, only_time_tokens=False)
    # =========================

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

        if model_args.freeze_speech_qformer:
            for name, param in model.speech_Qformer.named_parameters():
                param.requires_grad = False
            model.speech_Qformer.eval()
            model.speech_query_tokens.requires_grad = False
            print("[INFO] speech_Qformer FROZEN.")
        else:
            for name, param in model.speech_Qformer.named_parameters():
                param.requires_grad = True
            model.speech_query_tokens.requires_grad = True
            print("[INFO] speech_Qformer UNFROZEN for training.")

        if model_args.use_final_linear:
            for p in model.final_linear.parameters():
                p.requires_grad = True
        if model_args.freeze_final_linear:
            for p in model.final_linear.parameters():
                p.requires_grad = False

    if model_args.freeze_mm_projector:
        model.config.freeze_mm_mlp_adapter = True
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False
        if hasattr(model.get_model(), "image_newline"):
            model.get_model().image_newline.requires_grad = False
        print("[INFO] mm_projector FROZEN.")
    else:
        model.config.freeze_mm_mlp_adapter = False
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
        if hasattr(model.get_model(), "image_newline"):
            model.get_model().image_newline.requires_grad = True
        print("[INFO] mm_projector UNFROZEN for training.")

    model.config.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
    if model_args.unfreeze_mm_vision_tower:
        vision_tower.requires_grad_(True)
    else:
        vision_tower.requires_grad_(False)

    # ============ CHK6: TRAINABLE PARAMS SUMMARY ============
    if rank == 0:
        total     = sum(p.numel() for p in model.parameters())
        trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
        print("\n==== [CHK6] TRAINABLE PARAMS ====")
        print("[CHK6] total params     =", total)
        print("[CHK6] trainable params =", sum(sz for _, sz in trainable))
        for n, sz in trainable[:30]:
            print("[CHK6] trainable:", n, sz)
        print("[CHK6] ... (#trainable tensors =", len(trainable), ")")
        print("=================================\n")
    # ========================================================

    model.config.mm_use_im_start_end   = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr       = training_args.mm_projector_lr
    model.config.mm_vision_tower_lr    = training_args.mm_vision_tower_lr
    training_args.use_im_start_end     = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token

    if training_args.do_test or training_args.do_demo:
        data_module  = make_test_data_module(tokenizer=tokenizer, data_args=data_args)
        test_dataset = data_module["eval_dataset"]
        print("[TEST] test_data_path =", data_args.test_data_path)
        print("[TEST] len(test_dataset) =", len(test_dataset))

        model.to(torch.bfloat16).cuda()
        model.eval()

        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.do_sample   = False
            model.generation_config.temperature = None
            model.generation_config.top_p       = None
            model.generation_config.top_k       = None
            model.generation_config.num_beams   = 1

        if rank == 0:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total     = sum(p.numel() for p in model.parameters())
            print(f"[CHECK] trainable params: {trainable} / {total}")
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
            test_dataset  = data_module["eval_dataset"]
            data_collator = data_module["data_collator"]
            while True:
                try:
                    yaml_file = input("yaml file: ")
                    with open(yaml_file, 'r') as file:
                        yaml_data = yaml.safe_load(file)
                    if model_args.audio_visual:
                        audio_path = yaml_data.get('audio_path', None)
                    text_only = yaml_data.get("text_only", False)
                    video_path = "" if text_only else yaml_data['video_path']
                    if not text_only:
                        assert os.path.exists(video_path)

                    qs             = yaml_data['question']
                    max_time       = yaml_data.get("max_time", 30)
                    fps            = yaml_data.get("fps", 1)
                    max_new_tokens = yaml_data.get("max_new_tokens", 1024)
                    do_sample      = yaml_data.get("do_sample", False)
                    top_p          = yaml_data.get("top_p", 0.9)
                    seed           = yaml_data.get("seed", 2024)
                    prefix         = yaml_data.get("prefix", "")

                    test_dataset.max_time              = max_time
                    test_dataset.data_args.video_fps   = fps
                    test_dataset.max_frame_num         = round(test_dataset.max_time * test_dataset.data_args.video_fps)
                    test_dataset.list_data_dict        = [{}]

                    if not text_only:
                        if video_path != "":
                            test_dataset.list_data_dict[0]["video"] = video_path
                        if model_args.audio_visual:
                            test_dataset.list_data_dict[0]["audio"] = audio_path
                        test_dataset.list_data_dict[0]["conversations"] = [
                            {"from": "human", "value": "<image>\n" + qs.strip()},
                            {"from": "gpt",   "value": "", "prefix": prefix},
                        ]
                    else:
                        test_dataset.list_data_dict[0]["conversations"] = [
                            {"from": "human", "value": qs.strip(), "prefix": prefix},
                            {"from": "gpt",   "value": ""},
                        ]

                    item  = test_dataset._get_item(0)
                    batch = data_collator([item])

                    batch["input_ids"]     = batch["input_ids"].cuda()
                    batch["labels"]        = batch["labels"].cuda()
                    batch["attention_mask"] = batch["attention_mask"].cuda()
                    if not text_only:
                        batch["images"]      = [it.to(torch.bfloat16).cuda() for it in batch["images"]]
                        batch["spectrogram"] = batch["spectrogram"].to(torch.bfloat16).cuda()

                    batch.pop("ids")
                    batch.pop("prompts")
                    batch.pop("ce_only")
                    batch.pop("texts")

                    conv          = conv_templates['qwen_1_5'].copy()
                    stop_str      = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    keywords      = [stop_str]
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
                    res_ids  = result.tolist()
                    res_text = [tokenizer.decode(it) for it in res_ids]
                    print("======================")
                    print(res_text[0])
                    print("======================")

                except Exception as e:
                    print(e, e.__traceback__.tb_lineno)
                    breakpoint()

        else:
            test_output_dir = training_args.test_output_dir
            if rank == 0:
                os.makedirs(test_output_dir, exist_ok=True)

            test_ds = data_module.get("eval_dataset", None)
            print("[TEST][DEBUG] eval_dataset type:", type(test_ds))
            try:
                print("[TEST][DEBUG] eval_dataset len:", len(test_ds))
                if len(test_ds) > 0:
                    ex0 = test_ds[0]
                    if isinstance(ex0, dict):
                        print("[TEST][DEBUG] first item keys:", list(ex0.keys()))
            except Exception as e:
                print("[TEST][DEBUG] cannot inspect eval_dataset:", repr(e))

            if model_args.add_time_token:
                apply_time_token_grad_mask(model, tokenizer, only_time_tokens=True)

            if training_args.dpo_train:
                training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
                trainer = LLaVADPOTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
                print("[TEST][INFO] DPO test: using simple_generate loop.")
                results = simple_predict_generate(
                    model=model,
                    tokenizer=tokenizer,
                    trainer=trainer,
                    eval_dataset=data_module["eval_dataset"],
                    do_sample=training_args.do_sample,
                    max_new_tokens=training_args.max_new_tokens,
                    max_time=float(getattr(data_args, "max_time", 30)),
                    # num_time_bins 제거
                )
            else:
                trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
                print("[TEST][INFO] Using simple_generate loop.")
                results = simple_predict_generate(
                    model=model,
                    tokenizer=tokenizer,
                    trainer=trainer,
                    eval_dataset=data_module["eval_dataset"],
                    do_sample=training_args.do_sample,
                    max_new_tokens=training_args.max_new_tokens,
                    max_time=float(getattr(data_args, "max_time", 30)),
                    # num_time_bins 제거
                )
                print("[TEST][INFO] results len =", len(results))

            print(f"rank {rank} finish predict")

            output_path = os.path.join(test_output_dir, f"test_results_rank{rank}.json")
            with open(output_path, 'w') as fp:
                json.dump(results, fp, ensure_ascii=False)

            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            if (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_world_size() == 1):
                if rank == 0:
                    with open(output_path, "r") as fp:
                        res0 = json.load(fp) or []
                    tp_path = os.path.join(test_output_dir, "test_results.json")
                    with open(tp_path, "w") as fp:
                        json.dump(res0, fp, indent=4, ensure_ascii=False)
                    print(os.path.abspath(tp_path))
                return

            if rank == 0:
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
                new_res   = []
                for it in res:
                    key_id = str(it.get("id")) + _prompt_to_text(it.get("prompt"))
                    if key_id not in temp_dict:
                        temp_dict[key_id] = 1
                        new_res.append(it)

                res = new_res
                with open(tp_path := os.path.join(test_output_dir, "test_results.json"), 'w') as fp:
                    json.dump(res, fp, indent=4, ensure_ascii=False)
                print(os.path.abspath(tp_path))

    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
        if training_args.lora_enable:
            model.model.tokenizer = tokenizer
        else:
            model.tokenizer = tokenizer

        if model_args.add_time_token:
            apply_time_token_grad_mask(model, tokenizer, only_time_tokens=False)

        if training_args.dpo_train:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
            trainer = LLaVADPOTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
            trainer.add_callback(SaveTrainerStatePerCheckpointCallback())
        else:
            trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
            trainer.add_callback(SaveTrainerStatePerCheckpointCallback())

            if model_args.add_time_token:
                # VTG-LLM 방식: num_bins 인자 없음
                trainer.add_callback(SaveTimeTokenRowsCallback(tokenizer))

            cb = TrainGenProbeCallback(tokenizer, every_steps=50, max_new_tokens=256)
            cb.trainer = trainer
            trainer.add_callback(cb)

        if training_args.evaluation_strategy != "no":
            trainer.add_callback(EvaluateFirstStepCallback())

        temp_cnt, temp_total = 0, 0
        if rank == 0:
            for k, p in model.named_parameters():
                temp_total += 1
                if p.requires_grad:
                    print(k)
                    temp_cnt += 1
            print(temp_cnt, temp_total)

        # ============ CHK3: FIRST TRAIN BATCH QUICK CHECK ============
        if rank == 0:
            def _safe_decode_local(ids_like, skip_special_tokens=False):
                if torch.is_tensor(ids_like):
                    ids_like = ids_like.tolist()
                ids_like = [x for x in ids_like if isinstance(x, int) and x >= 0]
                return tokenizer.decode(ids_like, skip_special_tokens=skip_special_tokens)

            print("\n==== [CHK3] FIRST TRAIN BATCH ====")
            dl = trainer.get_train_dataloader()
            b  = next(iter(dl))

            ids = b["input_ids"][0]
            lab = b["labels"][0]
            print("[CHK3] input_ids len =", ids.numel())
            print("[CHK3] labels len    =", lab.numel())
            print("[CHK3] count(-200)   =", int((ids == -200).sum().item()))
            print("[CHK3] valid_labels  =", int((lab != -100).sum().item()))

            tail = _safe_decode_local(ids[-200:], skip_special_tokens=False)
            print("[CHK3] tail decode:\n", repr(tail))

            valid_ids = lab[lab != -100]
            tail_lab  = (_safe_decode_local(valid_ids[-200:], skip_special_tokens=False)
                         if valid_ids.numel() > 0 else "")
            print("[CHK3] label(valid) tail decode:\n", repr(tail_lab))

            # VTG-LLM 토큰 존재 여부 확인
            has_time = any(tok in tail_lab for tok in ["<t0>", "<t9>", "<tdot>"])
            print("[CHK3] label contains VTG-LLM time tokens?:", has_time)
            print("=================================\n")
        # ============================================================

        if rank == 0:
            dl_check    = trainer.get_train_dataloader()
            total_valid = 0
            total_tokens = 0
            for i, b_check in enumerate(dl_check):
                if i >= 5:
                    break
                lab_check = b_check.get("labels", None)
                if lab_check is not None:
                    total_valid  += int((lab_check != -100).sum().item())
                    total_tokens += lab_check.numel()

            print(f"\n==== [LABEL SANITY] first 5 batches ====")
            print(f"[LABEL SANITY] valid labels: {total_valid} / {total_tokens}")
            if total_valid == 0:
                raise RuntimeError(
                    "[FATAL] 학습 데이터의 labels가 전부 -100입니다!\n"
                    "VTG-LLM time token(<t0>~<t9>, <tdot>)이 tokenizer에 올바르게 등록됐는지, "
                    "conversation preprocess에서 response 구간 계산이 올바른지 확인하세요."
                )
            else:
                print(f"[LABEL SANITY] valid ratio: {total_valid / total_tokens:.3f} ✅")
            print("=========================================\n")

        if rank == 0:
            dl          = trainer.get_train_dataloader()
            first_batch = next(iter(dl))
            keys        = ["input_ids", "labels", "attention_mask", "images", "modalities", "spectrogram", "org_groups", "real_time"]
            b           = {k: first_batch.get(k, None) for k in keys if k in first_batch}

            def _to_cuda(x):
                return x.cuda(non_blocking=True) if torch.is_tensor(x) else x

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

            lab   = b["labels"]

        ckpts = sorted(
            glob.glob(os.path.join(training_args.output_dir, "checkpoint-*")),
            key=lambda p: int(os.path.basename(p).split("-")[-1])
        )

        def _is_real_resume_checkpoint(ckpt_dir):
            required_files = [
                os.path.join(ckpt_dir, "trainer_state.json"),
                os.path.join(ckpt_dir, "optimizer.pt"),
                os.path.join(ckpt_dir, "scheduler.pt"),
            ]
            return all(os.path.exists(p) for p in required_files)

        if len(ckpts) > 0:
            last_ckpt = ckpts[-1]
            _load_time_rows_if_exist(model, last_ckpt)
            if _is_real_resume_checkpoint(last_ckpt):
                print(f"[RESUME] real resume from {last_ckpt}", flush=True)
                trainer.train(resume_from_checkpoint=last_ckpt)
            else:
                print(f"[RESUME] optimizer/scheduler state missing in {last_ckpt}; warm-start only.", flush=True)
                trainer.train()
        else:
            trainer.train()

        trainer.save_state()

        if rank == 0:
            tokenizer.save_pretrained(training_args.output_dir)
            print("[SAVE] tokenizer saved to", training_args.output_dir)


if __name__ == "__main__":
    train()
