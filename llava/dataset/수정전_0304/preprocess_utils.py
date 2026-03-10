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

# Adapted from https://github.com/LLaVA-VL/LLaVA-NeXT. The original license is located at 'third-party-license/llava_next.txt'.

from typing import Dict, Optional, Sequence, List
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from llava import conversation as conversation_lib
import transformers
import torch
from packaging import version
import tokenizers
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


def preprocess_multimodal(
    sources: Sequence[str],
    data_args,
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value'] and not sentence['value'].startswith(DEFAULT_IMAGE_TOKEN):
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()

            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
            
    return sources

def preprocess_qwen(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful assistant.",
) -> Dict:
    roles = {
        "human": "<|im_start|>user",
        "gpt": "<|im_start|>assistant",
        "reject_gpt": "<|im_start|>assistant",
        "gt": "<|im_start|>assistant",
        "history_gpt": "<|im_start|>assistant",
    }

    # Robustly resolve Qwen image special token ids.
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # Fallback: if tokenizer doesn't know them, fall back to the first two additional specials.
    if (
        im_start is None
        or im_start < 0
        or im_start == tokenizer.unk_token_id
        or im_end is None
        or im_end < 0
        or im_end == tokenizer.unk_token_id
    ):
        ids = tokenizer.additional_special_tokens_ids
        if len(ids) < 2:
            raise ValueError(
                f"Cannot resolve image start/end token ids. additional_special_tokens_ids={ids}"
            )
        im_start, im_end = ids[0], ids[1]

    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens

    def _build_chunk(sentence: Dict):
        role = roles[sentence["from"]]
        role_ids = tokenizer(role).input_ids  # may include <|im_start|>

        content = sentence.get("value", "")
        content_ids = tokenizer(content, add_special_tokens=False).input_ids

        img_tok_id = tokenizer.convert_tokens_to_ids("<image>")
        if img_tok_id is not None and img_tok_id >= 0:
            content_ids = [IMAGE_TOKEN_INDEX if tid == img_tok_id else tid for tid in content_ids]

        _input_id = role_ids + nl_tokens + content_ids + [im_end]

        _target = [IGNORE_INDEX] * len(_input_id)

        if role == "<|im_start|>assistant" and sentence["from"] != "history_gpt":
            start = len(role_ids) + len(nl_tokens)
            end = start + len(content_ids)

            # never supervise IMAGE_TOKEN_INDEX(-200)
            content_targets = [
                (IGNORE_INDEX if tid == IMAGE_TOKEN_INDEX else tid)
                for tid in content_ids
            ]
            _target[start:end] = content_targets

        return _input_id, _target

    # Apply prompt templates
    input_ids, targets = [], []
    reject_input_ids, reject_targets = [], []
    gt_input_ids, gt_targets = [], []

    for i, source in enumerate(sources):
        # ensure starts with human
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        dpo_ver = any(s["from"] == "reject_gpt" for s in source)

        # -------------------------
        # DPO: build reject + gt
        # -------------------------
        if dpo_ver:
            # 1) reject branch: include reject_gpt, exclude gpt/gt
            input_id, target = [], []
            system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
            input_id += system
            target += [IGNORE_INDEX] * len(system)

            for sentence in source:
                if sentence["from"] in ("gpt", "gt"):
                    continue
                _in, _tg = _build_chunk(sentence)
                input_id += _in
                target += _tg

            assert len(input_id) == len(target), f"reject input len {len(input_id)} != target len {len(target)}"
            reject_input_ids.append(input_id)
            reject_targets.append(target)

            # 2) gt branch: include gt, exclude gpt/reject_gpt
            input_id, target = [], []
            system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
            input_id += system
            target += [IGNORE_INDEX] * len(system)

            for sentence in source:
                if sentence["from"] in ("gpt", "reject_gpt"):
                    continue
                _in, _tg = _build_chunk(sentence)
                input_id += _in
                target += _tg

            assert len(input_id) == len(target), f"gt input len {len(input_id)} != target len {len(target)}"
            gt_input_ids.append(input_id)
            gt_targets.append(target)

        # -------------------------
        # Main CE branch: exclude reject_gpt, gt
        # -------------------------
        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [IGNORE_INDEX] * len(system)

        for sentence in source:
            if sentence["from"] in ("reject_gpt", "gt"):
                continue
            _in, _tg = _build_chunk(sentence)
            input_id += _in
            target += _tg

        assert len(input_id) == len(target), f"ce input len {len(input_id)} != target len {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    # pack tensors (NO-DPO default should be None, not [None])
    reject_input_ids_out = [None]
    reject_targets_out = [None]
    if len(input_ids) == len(reject_input_ids) and len(reject_input_ids) > 0:
        reject_input_ids_out = torch.tensor(reject_input_ids, dtype=torch.long)
        reject_targets_out = torch.tensor(reject_targets, dtype=torch.long)

    gt_input_ids_out = [None]
    gt_targets_out = [None]
    if len(input_ids) == len(gt_input_ids) and len(gt_input_ids) > 0:
        gt_input_ids_out = torch.tensor(gt_input_ids, dtype=torch.long)
        gt_targets_out = torch.tensor(gt_targets, dtype=torch.long)
    else:
        gt_input_ids_out = torch.tensor(input_ids, dtype=torch.long)
        gt_targets_out = torch.tensor(targets, dtype=torch.long)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
        reject_input_ids=reject_input_ids_out,
        reject_labels=reject_targets_out,
        gt_input_ids=gt_input_ids_out,
        gt_labels=gt_targets_out,
    )

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # print(conversation_lib.default_conversation.version)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=has_image)
    else:
        raise NotImplementedError
