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

from transformers import AutoConfig
from llava.model import VideoSALMONN2ForCausalLM
from transformers import AutoConfig, AutoTokenizer
import torch
import os
import json
from collections import OrderedDict
from peft import LoraConfig, get_peft_model, PeftModel
from dataclasses import make_dataclass

def _is_hf_weight_dir(p: str) -> bool:
    if not (p and os.path.isdir(p)):
        return False
    # HF safetensors shard index or single weight
    if os.path.exists(os.path.join(p, "model.safetensors.index.json")):
        return True
    if os.path.exists(os.path.join(p, "model.safetensors")):
        return True
    if os.path.exists(os.path.join(p, "pytorch_model.bin")):
        return True
    # some repos use these shard names
    if len([f for f in os.listdir(p) if f.startswith("model-") and f.endswith(".safetensors")]) > 0:
        return True
    if len([f for f in os.listdir(p) if f.startswith("pytorch_model-") and f.endswith(".bin")]) > 0:
        return True
    return False

def load_qwen_lora_model(model_path, model_base=None, lora_enable=False, pretrain_weight=None,
                         load_full=False, lora_r=128, lora_alpha=256, lora_dropout=0.05,
                         model_max_length=32768, new_model_args=None, **audio_config):

    # model_path can be either:
    #  - a directory (HF-style checkpoint folder: checkpoint-5/)
    #  - a file path (legacy .bin)
    # Normalize to a directory that contains config.json
    model_ckpt_path = model_path

    if os.path.isdir(model_ckpt_path):
        model_path = model_ckpt_path
    else:
        model_path = os.path.dirname(model_ckpt_path)

    # Prefer checkpoint-local all_parameters.bin if exists
    lora_ckpt = os.path.join(model_path, "all_parameters.bin")

    with open(os.path.join(model_path, "config.json"), "r") as fp:
        config = json.load(fp)

    if model_base is None:
        model_base = config["_name_or_path"]
        while os.path.exists(os.path.join(model_base, "all_parameters.bin")):
            with open(os.path.join(model_base, 'config.json'), 'r') as fp:
                config = json.load(fp)
            model_base = config["_name_or_path"]

    def _has_tokenizer_files(d: str) -> bool:
        if not d or not os.path.isdir(d):
            return False
        for fn in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "added_tokens.json"]:
            if os.path.exists(os.path.join(d, fn)):
                return True
        return False

    # ✅ prefer checkpoint tokenizer if files exist
    tok_src = model_path if _has_tokenizer_files(model_path) else model_base
    print(f"[TOK] Loading tokenizer from: {tok_src}")

    tokenizer = AutoTokenizer.from_pretrained(
        tok_src,
        model_max_length=model_max_length,
        padding_side="right"
    )

    # ✅ optional: assert / warn if time token experiment but tokenizer lacks them
    if new_model_args is not None and getattr(new_model_args, "add_time_token", False):
        t0 = tokenizer.convert_tokens_to_ids("<t0>")
        if t0 is None or t0 < 0 or t0 == tokenizer.unk_token_id:
            print("[WARN][TOK] add_time_token=True but <t0> not in tokenizer yet. It must be added later in train.py and saved to checkpoint.")
            
    cfg_pretrained = AutoConfig.from_pretrained(model_base)
    if "model_args" in config:
        model_args = config["model_args"]

        model_args["lora_r"] = lora_r
        model_args["lora_alpha"] = lora_alpha
        model_args["lora_dropout"] = lora_dropout

        TempData = make_dataclass('TempData', model_args)
        model_args = TempData(**model_args)
        # ---- OVERRIDE from CLI(new_model_args) if provided ----
        if new_model_args is not None:
            for k in [
                "mm_patch_merge_type",
                "mm_use_im_start_end",
                "mm_use_im_patch_token",
                "mm_pooling_position",
                "mm_newline_position",
                "mm_spatial_pool_stride",
                "mm_spatial_pool_mode",
                "mm_vision_select_layer",
                "mm_projector_type",
            ]:
                if hasattr(new_model_args, k) and getattr(new_model_args, k) is not None:
                    setattr(model_args, k, getattr(new_model_args, k))

            # ✅ [PATCH] time-token flag must follow CLI
            if hasattr(new_model_args, "add_time_token"):
                model_args.add_time_token = bool(getattr(new_model_args, "add_time_token"))
        # --------------------------------------

        overwrite_config = {"model_args": vars(model_args), "add_time_token": model_args.add_time_token}
    else:
        model_args = new_model_args
        overwrite_config = {"model_args": vars(model_args), "add_time_token": model_args.add_time_token}

    for k, v in overwrite_config.items():
        setattr(cfg_pretrained, k, v)

    # ✅ prefer full HF checkpoint weights in model_path if present
    base_to_load = model_path if _is_hf_weight_dir(model_path) else model_base

    print(f"[INFO] Loading base model from: {base_to_load}")
    model = VideoSALMONN2ForCausalLM.from_pretrained(
        base_to_load,
        config=cfg_pretrained,
        cache_dir=None,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        **audio_config
    )
    model.resize_token_embeddings(len(tokenizer))
    model.get_model().initialize_vision_modules(model_args=model_args)
    model = model.to(torch.bfloat16)

    # -------- NEW: PEFT adapter checkpoint support --------
    adapter_safetensors = os.path.join(model_path, "adapter_model.safetensors")
    adapter_bin = os.path.join(model_path, "adapter_model.bin")
    adapter_config = os.path.join(model_path, "adapter_config.json")

    has_peft_adapter = (os.path.exists(adapter_config) and (os.path.exists(adapter_safetensors) or os.path.exists(adapter_bin)))

    if has_peft_adapter:
        print(f"[INFO] Detected PEFT adapter checkpoint at: {model_path}")
        # is_trainable=True면 학습 재개도 가능, 테스트면 False로 둬도 됨
        model = PeftModel.from_pretrained(model, model_path, is_trainable=lora_enable)
        model = model.to(torch.bfloat16)
        return model, tokenizer
    # ------------------------------------------------------

    if load_full and lora_enable:
        # --- legacy fallback (all_parameters.bin) only if it exists ---
        if not os.path.exists(lora_ckpt):
            print(f"[INFO] No all_parameters.bin at {lora_ckpt}. Skipping legacy state_dict load.")
            return model, tokenizer
        # ------------------------------------------------------------

        ckpt = torch.load(lora_ckpt, map_location='cpu')
        if model_ckpt_path != lora_ckpt:
            ckpt_2 = torch.load(model_ckpt_path, map_location='cpu')
            for k in ckpt_2.keys():
                ckpt[k] = ckpt_2[k]
        
        new_ckpt = OrderedDict()
        for k in ckpt.keys():
            if 'vision_tower' not in k:
                new_ckpt[k[len('module.'):]] = ckpt[k]
            else:
                new_ckpt[k[len('module.'):]] = ckpt[k]

        kk = model.load_state_dict(new_ckpt, strict=False)
        print("Load full: ", len(kk.unexpected_keys), len(kk.missing_keys))

    if lora_enable:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj"],
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        print("Lora Config: ", model_args.lora_r, model_args.lora_alpha, model_args.lora_dropout)
        model.to(torch.bfloat16)
        if audio_config.get("audio_visual", False):
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
        model.to(torch.bfloat16)
    
    if load_full and lora_enable:
        if pretrain_weight is not None and pretrain_weight != "None":
            ckpt = OrderedDict()
            if pretrain_weight is not None and pretrain_weight != "None":
                ckpt_3 = torch.load(pretrain_weight, map_location='cpu')
                for k in ckpt_3.keys():
                    if "speech" in k or "final_linear" in k:
                        key = k.replace("module.", "base_model.model.")
                        ckpt[key] = ckpt_3[k]
                print("Load Pretrain Weight")

            kk = model.load_state_dict(ckpt, strict=False)
            print(len(kk.unexpected_keys), len(kk.missing_keys))
            print(len([it for it in kk.missing_keys if 'lora' in it or 'final_linear' in it]))

    else:
        # --- legacy fallback (all_parameters.bin) only if it exists ---
        if not os.path.exists(lora_ckpt):
            print(f"[INFO] No all_parameters.bin at {lora_ckpt}. Skipping legacy state_dict load.")
            return model, tokenizer
        # ------------------------------------------------------------

        ckpt = torch.load(lora_ckpt, map_location='cpu')
        
        if pretrain_weight is not None and pretrain_weight != "None":
            ckpt_3 = torch.load(pretrain_weight, map_location='cpu')
            for k in ckpt_3.keys():
                if "speech" in k or "final_linear" in k:
                    key = k.replace("module.", "module.base_model.model.")
                    ckpt[key] = ckpt_3[k]
            print("Load Pretrain Weight")

        if model_ckpt_path != lora_ckpt:
            ckpt_2 = torch.load(model_ckpt_path, map_location='cpu')
            for k in ckpt_2.keys():
                ckpt[k] = ckpt_2[k]
        
        new_ckpt = OrderedDict()
        for k in ckpt.keys():
            if 'vision_tower' not in k:
                new_ckpt[k[len('module.'):]] = ckpt[k]
            else:
                new_ckpt[k[len('module.'):]] = ckpt[k]

        kk = model.load_state_dict(new_ckpt, strict=False)

        print(len(kk.unexpected_keys), len(kk.missing_keys))
        print(len([it for it in kk.missing_keys if 'lora' in it or 'final_linear' in it]))

    return model, tokenizer

