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

from llava.dataset.preprocess_utils import preprocess_multimodal, preprocess
import json
from torch.utils.data import Dataset
import torch.distributed as dist
from decord import VideoReader, cpu
from dataclasses import dataclass
import copy
from typing import Dict, Sequence, Optional
from llava.constants import IGNORE_INDEX
import numpy as np
import transformers
import torch
from transformers import WhisperFeatureExtractor
import random
import soundfile as sf
import random
import math

def _sec_to_vtgllm_tokens(sec: float) -> str:
    sec = max(0.0, sec)
    # 소수 1자리로 반올림한 뒤 정수부/소수부 분리
    # → 12.95 같은 경우 13.0으로 올바르게 처리
    rounded = round(sec * 10)          # 129 (0.1초 단위 정수)
    int_part = rounded // 10           # 12
    frac_digit = rounded % 10          # 9
    int_str = f"{int_part:04d}"
    tokens = "".join(f"<t{d}>" for d in int_str)
    tokens += f"<tdot><t{frac_digit}>"
    return tokens

def _build_single_time_token_answer(t0: float, t1: float) -> str:
    return f"start: {_sec_to_vtgllm_tokens(t0)} end: {_sec_to_vtgllm_tokens(t1)}"

def _build_dense_time_token_answer(events: list) -> str:
    """events: [{"label": ..., "timestamps": [start, end]}, ...]"""
    result = []
    for ev in sorted(events, key=lambda x: x["timestamps"][0]):
        result.append({
            "event": ev["label"],
            "start": _sec_to_vtgllm_tokens(ev["timestamps"][0]),
            "end":   _sec_to_vtgllm_tokens(ev["timestamps"][1]),
        })
    return json.dumps(result, ensure_ascii=False)

class LazyAVSupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args, is_test=False):
        super(LazyAVSupervisedDataset, self).__init__()

        list_data_dict = json.load(open(data_path, "r"))
    
        print("Formatting inputs...Skip in lazy mode. Audio visual dataset")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

        whisper_path = self.data_args.audio_processor
        self.wav_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)
        self.max_time = data_args.max_time
        self.use_timestamps_crop = getattr(data_args, "use_timestamps_crop", False)

        self.is_test = is_test

        self.max_frame_num = round(self.max_time * self.data_args.video_fps)
        print("Max frame num: ", self.max_frame_num)

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if ('image' in sample) or ('video' in sample) or ("audio" in sample) else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self._get_item(i)

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        try:
            sources = self.list_data_dict[i]
            if isinstance(i, int):
                sources = [sources]
            assert len(sources) == 1  # "Don't know why it is wrapped to a list"

            ori_item = copy.deepcopy(sources[0])
            prompt = [sources[0]["conversations"][k] for k in range(len(sources[0]["conversations"]) - 1)]

            text = sources[0]["conversations"][-1]["value"]

            # ----------------------------
            # TIME-TOKEN SUPERVISION (TRAIN)
            # - train에서는 gpt.value(JSON string)의 start/end 숫자를 time-token 문자열로 변환
            # - test(is_test=True)에서는 건드리지 않음
            # ----------------------------
            if not self.is_test:
                mode = sources[0].get("mode", "single")
                skip_time_build = sources[0].get("ce_only", False)

                # timestamps/events가 없고 답변에 이미 time token이 있으면 변환 skip
                if not skip_time_build and mode != "dense" and "timestamps" not in sources[0]:
                    skip_time_build = True

                if not skip_time_build:
                    if mode == "dense":
                        events = sources[0].get("events")
                        if not events:
                            raise ValueError(f"[TIME-TOKEN] dense mode requires 'events' field.")
                        new = _build_dense_time_token_answer(events)

                    else:  # single
                        t0 = float(sources[0]["timestamps"][0])
                        t1 = float(sources[0]["timestamps"][1])

                        if self.use_timestamps_crop:
                            # crop 모드는 single에서만 유효
                            t1 = max(0.0, t1 - t0)
                            t0 = 0.0

                        new = _build_single_time_token_answer(t0, t1)

                    sources[0]["conversations"][-1]["value"] = new
                    text = new
            # ----------------------------

            if self.is_test:
                last = sources[0]["conversations"][-1]
                if "prefix" in last and isinstance(last["prefix"], str):
                    sources[0]["conversations"][-1]["value"] = last["prefix"]
                else:
                    # ✅ 테스트는 "생성용 prompt"로 끝나야 합니다.
                    # assistant의 content를 비워두면 preprocess에서 <|im_start|>assistant\n 로 열린 상태로 끝낼 수 있습니다.
                    sources[0]["conversations"][-1]["value"] = ""

            use_crop = (
                self.use_timestamps_crop
                and sources[0].get("mode", "single") != "dense"
                and "timestamps" in sources[0]
            )

            if 'video' in sources[0]:
                video_file = sources[0]['video']
                vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)

                total_frame_num = len(vr)
                ori_fps = vr.get_avg_fps()
                avg_fps = max(round(ori_fps / self.data_args.video_fps), 1)
                real_time = total_frame_num / vr.get_avg_fps()
                
                max_frames = min(self.max_frame_num, 120)

                if not use_crop:
                    frame_idx = [k for k in range(0, total_frame_num, round(avg_fps))]
                    if len(frame_idx) > max_frames:
                        frame_idx = np.linspace(0, total_frame_num - 1, max_frames, dtype=int).tolist()
                else:
                    start = round(sources[0]["timestamps"][0] * vr.get_avg_fps())
                    end = round(sources[0]["timestamps"][1] * vr.get_avg_fps())
                    end = min(end, total_frame_num - 1)
                    frame_idx = [k for k in range(start, end + 1, round(avg_fps))]
                    if len(frame_idx) > max_frames:
                        frame_idx = np.linspace(start, end, max_frames, dtype=int).tolist()
                    real_time = sources[0]["timestamps"][1] - sources[0]["timestamps"][0]

                video = vr.get_batch(frame_idx).asnumpy() # video: (F, H, W, C)
                video = np.array(video)

                processor = self.data_args.image_processor
                image = processor.preprocess(video, return_tensors='pt')['pixel_values']

                assert len(image) > 1

                image = [(image, video[0].size, "video")]
                process_sources = preprocess_multimodal(
                    copy.deepcopy([e["conversations"] for e in sources]),
                    self.data_args,
                )

            else:
                # sources = copy.deepcopy([e["conversations"] for e in sources])
                process_sources = preprocess_multimodal(
                    copy.deepcopy([e["conversations"] for e in sources]),
                    self.data_args,
                )
            
            if 'audio' in sources[0]:
                audio_file = sources[0]['audio']
                audio, sr = sf.read(audio_file)
                assert sr == 16000  # only support 16kHz audio
                if len(audio.shape) == 2: # stereo to mono
                    audio = audio[:, 0]

                if use_crop:   # 비디오 섹션에서 계산한 use_crop 재사용
                    t0 = float(sources[0]["timestamps"][0])
                    t1 = float(sources[0]["timestamps"][1])
                    s0 = int(math.floor(t0 * sr))
                    s1 = int(math.ceil(t1 * sr))
                    s0 = max(0, min(s0, len(audio)))
                    s1 = max(0, min(s1, len(audio)))
                    if s1 <= s0:
                        s1 = min(len(audio), s0 + sr)
                    audio = audio[s0:s1]

                if len(audio) < sr: # pad audio to at least 1s
                    sil = np.zeros(sr - len(audio), dtype=float)
                    audio = np.concatenate((audio, sil), axis=0)

                if 'video' in sources[0]:
                    audio = audio[:round(sr * real_time)] 
                else:
                    audio = audio[:round(sr * self.max_time)] # truncate audio to at most 30s
                audio_lst = [audio[k: k + 30 * sr] for k in range(0, len(audio), 30 * sr)]
                spectrogram_lst = [self.wav_processor(a, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze() for a in audio_lst]

            else:
                audio_file = None

            has_image = ('image' in sources[0]) or ('video' in sources[0])
            if "video" in sources[0]:
                data_id = "['{}', '{}']".format(sources[0]["video"], audio_file)
            else:
                data_id = "['{}', '{}']".format(None, audio_file)

            data_dict = preprocess(process_sources, self.tokenizer, has_image=has_image)

            # ===== [DBG] label mask sanity check (DATASET) =====
            is_rank0 = (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)

            if is_rank0 and (i < 3):  # 처음 3개만
                labels0 = data_dict["labels"][0]  # shape: (seq_len,)
                valid = (labels0 != IGNORE_INDEX).sum().item()
                total = labels0.numel()

                print(f"[DBG][DATASET] i={i} valid_label_tokens={valid}/{total}")
                print("[DBG][DATASET] text_tail:\n", str(text)[-500:])
            # ===== [DBG] end =====

            lab = data_dict["labels"][0].tolist()
            valid = [x for x in lab if x != IGNORE_INDEX]
            dec = self.tokenizer.decode(valid[-200:], skip_special_tokens=False)

            if is_rank0 and (i < 3):
                print("[DBG][DATASET] valid_label_decode_tail:\n", dec)

            # if self.is_test:
            #     data_dict["input_ids"] = data_dict["input_ids"][:, :-2]
            #     data_dict["labels"] = data_dict["labels"][:, :-2]

            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                labels=data_dict["labels"][0],
                reject_input_ids=data_dict["reject_input_ids"][0],
                reject_labels=data_dict["reject_labels"][0],
                gt_input_ids=data_dict["gt_input_ids"][0],
                gt_labels=data_dict["gt_labels"][0]
            )

            if 'video' in sources[0]:
                data_dict['image'] = image
                if 'audio' not in sources[0]:
                    data_dict['modality'] = "video"
                else:
                    data_dict['modality'] = "audio-video"
            elif "audio" in sources[0]:
                data_dict['modality'] = "audio"
                data_dict['image'] = None
            else:
                data_dict['modality'] = "text"
                data_dict['image'] = None

            if audio_file is not None:
                data_dict["spectrogram"] = torch.stack(spectrogram_lst, dim=0)
            else:
                data_dict["spectrogram"] = None

            if data_dict['modality'] != "audio" and data_dict['modality'] != "text":
                data_dict["real_time"] = real_time
            else:
                data_dict["real_time"] = 30 * len(audio_lst)

            data_dict['prompt'] = prompt

            data_dict['id'] = data_id

            data_dict["ori_item"] = ori_item
            data_dict["ce_only"] = sources[0].get("ce_only", False)
            data_dict["text"] = text

            if sources[0].get("mode", "single") == "dense":
                # dense는 단일 timestamps 대신 events 전체를 저장
                data_dict["gt_timestamps"] = None
                data_dict["gt_events"] = sources[0].get("events", [])
            elif "timestamps" in sources[0]:
                data_dict["gt_timestamps"] = [float(sources[0]["timestamps"][0]), float(sources[0]["timestamps"][1])]
                data_dict["gt_events"] = None
            else:
                data_dict["gt_timestamps"] = None
                data_dict["gt_events"] = None

            return data_dict
        
        except Exception as e:
            import traceback
            print(f'GGGG {i}. Line: {e.__traceback__.tb_lineno}, Exception:', e)
            traceback.print_exc()
            if self.is_test:
                raise e
            else:
                return self._get_item(random.choice(range(len(self))))
        

@dataclass
class DataCollatorForAVSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids] 
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=batch_first,
            padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, ids, reject_input_ids, reject_labels, gt_input_ids, gt_labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels", 'id', "reject_input_ids", "reject_labels", "gt_input_ids", "gt_labels"))
        input_ids = [_input_ids[:self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[:self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            if "qwen" in self.tokenizer.name_or_path.lower():
                self.tokenizer.pad_token_id = 151643
            else:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME
        input_ids = self.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels,
                                   batch_first=True,
                                   padding_value=IGNORE_INDEX)

        for input in reject_input_ids:
            if input is None:
                reject_input_ids = None
                reject_labels = None
                reject_attention_mask = None
                dpo_forward = False
                gt_input_ids = None
                gt_labels = None
                gt_attention_mask = None
                break
        else:
            reject_input_ids = [_input_ids[:self.tokenizer.model_max_length] for _input_ids in reject_input_ids]
            reject_labels = [_labels[:self.tokenizer.model_max_length] for _labels in reject_labels]
            reject_input_ids = self.pad_sequence(
                reject_input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id
            )
            reject_labels = self.pad_sequence(
                reject_labels,
                batch_first=True,
                padding_value=IGNORE_INDEX
            )
            reject_attention_mask = reject_input_ids.ne(self.tokenizer.pad_token_id)
            dpo_forward = True

            gt_input_ids = [_input_ids[:self.tokenizer.model_max_length] for _input_ids in gt_input_ids]
            gt_labels = [_labels[:self.tokenizer.model_max_length] for _labels in gt_labels]
            gt_input_ids = self.pad_sequence(
                gt_input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id
            )
            gt_labels = self.pad_sequence(
                gt_labels,
                batch_first=True,
                padding_value=IGNORE_INDEX
            )
            gt_attention_mask = gt_input_ids.ne(self.tokenizer.pad_token_id)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            reject_input_ids=reject_input_ids,
            reject_labels=reject_labels,
            gt_input_ids=gt_input_ids,
            gt_labels=gt_labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            ids=ids,
            reject_attention_mask=reject_attention_mask,
            gt_attention_mask=gt_attention_mask,
            dpo_forward=dpo_forward,
        )

        batch['modalities'] = [im['modality'] for im in instances]

        # --- FIX: only create batch["images"] when there is at least one real image/video tensor ---
        has_any_image = any(inst.get("image", None) is not None for inst in instances)

        if has_any_image:
            images = [instance.get('image', None) for instance in instances]

            # ✅ 항상 batch 크기와 동일한 길이를 유지 (instance당 1개 tensor or None)
            true_images = []
            for im_list in images:
                if im_list is None or len(im_list) == 0:
                    true_images.append(None)
                else:
                    # (pixel_values, size, "video") 중 pixel_values만
                    true_images.append(im_list[0][0])
            batch['images'] = true_images
            
        else:
            # 중요: images 키를 아예 안 넣거나 None으로 넣어야 모델이 이미지 경로로 안 들어갑니다.
            batch['images'] = None
        # --- END FIX ---
                
        if 'prompt' in instances[0]:
            batch['prompts'] = [instance['prompt'] for instance in instances]

        if instances[0]["spectrogram"] is not None:  # assert batchsize = 1
            samples_spectrogram = [s["spectrogram"] for s in instances]
            cat_spectrogram = torch.cat(samples_spectrogram, dim=0)
            org_groups = [s.size(0) for s in samples_spectrogram]

            batch["spectrogram"] = cat_spectrogram
            batch["org_groups"] = org_groups
        else:
            batch["spectrogram"] = None
            batch["org_groups"] = None

        batch['real_time'] = [s["real_time"] for s in instances]
        batch["ori_item"] = [s["ori_item"] for s in instances]
        batch["ce_only"] = [s["ce_only"] for s in instances]
        batch["texts"] = [s["text"] for s in instances]

        batch["gt_timestamps"] = [s.get("gt_timestamps", None) for s in instances]
        batch["gt_events"]     = [s.get("gt_events",     None) for s in instances]

        # ===== [DBG] label mask sanity check (COLLATOR) =====
        is_rank0 = (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)

        # ===== [DBG] end =====

        return batch