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

    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
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
                f"Cannot resolve Qwen start/end token ids. additional_special_tokens_ids={ids}"
            )
        im_start, im_end = ids[0], ids[1]

    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens

    def _build_chunk(sentence: Dict, is_last: bool = False):
        role = roles[sentence["from"]]
        role_ids = tokenizer(role).input_ids

        content = sentence.get("value", "")
        content_ids = tokenizer(content, add_special_tokens=False).input_ids

        img_tok_id = tokenizer.convert_tokens_to_ids("<image>")
        if img_tok_id is not None and img_tok_id >= 0:
            content_ids = [IMAGE_TOKEN_INDEX if tid == img_tok_id else tid for tid in content_ids]

        if role == "<|im_start|>assistant" and is_last and len(content_ids) == 0:
            _input_id = role_ids + nl_tokens
            _target = [IGNORE_INDEX] * len(_input_id)
            return _input_id, _target

        _input_id = role_ids + nl_tokens + content_ids + [im_end]
        _target = [IGNORE_INDEX] * len(_input_id)

        if role == "<|im_start|>assistant" and sentence["from"] != "history_gpt":
            start = len(role_ids) + len(nl_tokens)
            end = start + len(content_ids)
            _target[start:end] = [
                (IGNORE_INDEX if tid == IMAGE_TOKEN_INDEX else tid)
                for tid in content_ids
            ]
            _target[end] = im_end

        return _input_id, _target

    input_ids, targets = [], []

    for source in sources:
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [IGNORE_INDEX] * len(system)

        filtered = [s for s in source if s["from"] not in ("reject_gpt", "gt")]
        for j, sentence in enumerate(filtered):
            _in, _tg = _build_chunk(sentence, is_last=(j == len(filtered) - 1))
            input_id += _in
            target += _tg

        assert len(input_id) == len(target), f"ce input len {len(input_id)} != target len {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
        reject_input_ids=[None],
        reject_labels=[None],
        gt_input_ids=input_ids.clone(),
        gt_labels=targets.clone(),
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
