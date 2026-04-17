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

# Adopted from https://github.com/QwenLM/Qwen2.5-VL. The original license is located at 'third-party-license/qwenvl.txt'.

# ============================================================================
# [파일 개요]
# 이 파일은 Qwen2.5-VL(비전-언어 모델)에서 사용하는 "3D RoPE(Rotary Position
# Embedding) 인덱스"를 계산하는 함수들을 정의한다.
#
# 보통의 LLM은 토큰 순서대로 0,1,2,... 1차원 위치 ID를 쓰지만,
# 비디오/이미지/오디오가 섞인 멀티모달 입력에서는 "시간(T), 높이(H), 너비(W)"
# 3개의 축으로 위치를 표현해야 한다. (이를 MRoPE = Multimodal RoPE 라고 부름)
#
# - get_rope_index_25 : Qwen2.5-VL용 (오디오까지 포함하는 확장판)
# - get_rope_index_2  : Qwen2-VL용 (이미지/비디오만, 이 저장소에서도 같이 쓰임)
#
# 각 함수는 결과로 position_ids (3, batch, seq_len) 와
# mrope_position_deltas(배치별 MRoPE 최대 index와 시퀀스 길이 차) 를 반환한다.
# ============================================================================

import os
import copy
import json
import random
import logging
import re
import time
import math
import ast
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple
from io import BytesIO
import base64

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
# from decord import VideoReader  # 비디오 프레임 읽기 라이브러리 (여기선 사용 안 함)
import transformers


def get_rope_index_25(
    spatial_merge_size: Optional[int] = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    audio_lengths: Optional[list] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embedding for text part.
        Examples:
            Temporal (Time): 3 patches, representing different segments of the video in time.
            Height: 2 patches, dividing each frame vertically.
            Width: 2 patches, dividing each frame horizontally.
            We also have some important parameters:
            fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
            tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
            temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
            interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
            input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
            vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text temporal position_ids: [101, 102, 103, 104, 105]
            text height position_ids: [101, 102, 103 104, 105]
            text width position_ids: [101, 102, 103, 104, 105]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """
    # -------------------------------------------------------------
    # [한글 설명]
    # Qwen2.5-VL의 3D RoPE 위치 ID 계산 함수.
    #
    # 반환되는 position_ids는 (3, batch, seq_len) 모양이고,
    #   0번 축: 시간(Temporal)     위치
    #   1번 축: 높이(Height)       위치
    #   2번 축: 너비(Width)        위치
    # 를 각각 담는다. 텍스트 토큰 구간에서는 3축 값이 모두 같아지고,
    # 비전/오디오 토큰 구간에서는 각각 다른 값이 들어간다.
    # -------------------------------------------------------------

    # 특수 토큰 ID (토크나이저에서 고정적으로 부여된 값)
    image_token_id = 151655          # 이미지 토큰
    video_token_id = 151656          # 비디오 토큰
    audio_token_id = 151665          # 오디오 토큰
    vision_start_token_id = 151652   # 비전/오디오 블록의 시작을 알리는 토큰

    mrope_position_deltas = []   # 배치별 (최대 MRoPE 인덱스 + 1 - 시퀀스 길이) 저장

    # ==========================================================================
    # CASE 1) 오디오 + 비디오 가 섞여있는 입력
    #   - audio_lengths  : 각 오디오 블록의 토큰 수 리스트
    #   - video_grid_thw : 각 비디오의 (T, H, W) 격자 크기
    # ==========================================================================
    if input_ids is not None and (
        audio_lengths is not None and video_grid_thw is not None
    ):
        # 오디오·비디오가 교차(interleaved) 로 들어오는 경우 처리
        total_input_ids = input_ids
        # attention_mask가 없으면 "모두 1" (전부 유효한 토큰)로 간주
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)

        # 결과 텐서 초기화: shape (3, batch, seq_len). 나중에 실제 값으로 채워짐.
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        video_index = 0   # 현재까지 처리한 비디오 개수 카운터
        attention_mask = attention_mask.to(total_input_ids.device)

        # 배치 내 시퀀스를 하나씩 순회
        for i, input_ids in enumerate(total_input_ids):
            # 패딩 제외: attention_mask == 1 인 토큰만 남김
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0

            # vision_start 토큰 위치를 모두 찾고, 그 바로 다음 토큰이 무엇인지로
            # 이미지/비디오/오디오를 구분한다.
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id
            ).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            video_nums = (vision_tokens == video_token_id).sum()  # 이 샘플의 비디오 수

            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []   # 구간별 position id 블록을 모으는 리스트
            st = 0                         # 현재 처리 위치 (start index)
            remain_videos = video_nums     # 남은 비디오 수

            # 비디오 블록 한 개씩 처리
            for _ in range(video_nums):
                # 다음 비디오 토큰 블록의 시작 위치(= 직전까지는 텍스트)
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1

                # 이 비디오의 (시간, 높이, 너비) 격자 크기
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                # 각 시간 그리드가 몇 초를 담는지 (fps에 따라 달라짐)
                if second_per_grid_ts is not None:
                    second_per_grid_t = second_per_grid_ts[video_index]
                else:
                    second_per_grid_t = 1.0
                ed = ed_video

                # LLM이 실제로 보게 되는 그리드 수:
                # H/W는 spatial_merge_size 만큼 축소(merge) 된 값이 사용됨.
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st  # 비디오 블록 직전까지의 텍스트 길이

                # 새 구간의 position id는 직전 구간 max+1 부터 이어감
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                # [텍스트 부분] 3축 position id를 모두 동일한 1D 카운터로 채움
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

                # ------- 비디오 토큰들의 3D position id 계산 -------
                # t 축: 시간 프레임 인덱스(0..T-1) * (second_per_grid * 2)
                #   => 몇 초가 지난 시점인지를 position에 반영
                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
                time_tensor = expanded_range * second_per_grid_t * 2
                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()

                # h 축: 0..H-1 을 (T,H,W)로 확장 후 평탄화
                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                # w 축: 0..W-1 을 (T,H,W)로 확장 후 평탄화
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                # 비디오 토큰 전체에 대응하는 (3, T*H*W) 형태의 position id
                video_pos = torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                # llm_pos_ids_list.append(video_pos)  # 아래서 오디오와 합쳐서 append

                # ------- 오디오 토큰들의 position id 계산 -------
                audio_len = audio_lengths[video_index]  # 이 구간 오디오 토큰 수

                time_index_audio = torch.arange(audio_len, device=input_ids.device)
                w_index_audio = time_index_audio                      # w축은 시간처럼 증가
                h_index_audio = torch.zeros_like(time_index_audio)    # h축은 0 고정 (1차원 신호)
                audio_pos = torch.stack([time_index_audio, h_index_audio, w_index_audio]) + st_idx + text_len

                # ------- 비디오·오디오 토큰이 섞인 구간의 position id 재배열 -------
                # 실제 input_ids 에서 audio/video 토큰이 나오는 위치에 맞춰 채워 넣는다.
                audio_visual_pos = torch.zeros_like(torch.cat((video_pos, audio_pos), dim=1))
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w + audio_len
                audio_visual_pos[:, input_ids[ed:st] == audio_token_id] = audio_pos
                audio_visual_pos[:, input_ids[ed:st] == video_token_id] = video_pos
                llm_pos_ids_list.append(audio_visual_pos)

                video_index += 1
                remain_videos -= 1

            # 마지막 비디오 이후에 남은 텍스트가 있다면 1D position id로 마무리
            if st < len(input_tokens):
                if len(llm_pos_ids_list) > 0:
                    # 마지막 블록 time 축(0번 축) 최대값 + 1 부터 이어 쓴다
                    last_time_max = llm_pos_ids_list[-1][0].max()
                    st_idx = last_time_max + 1
                else:
                    st_idx = 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            # 여러 구간의 position id 블록을 seq_len 축으로 이어 붙임
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            # 패딩이 아닌(=실제 토큰) 위치에만 값 대입
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                position_ids.device
            )
            # 이 샘플의 MRoPE 최대 index+1 에서 원래 시퀀스 길이를 뺀 값
            # (generation 때 cache_position 보정용으로 사용)
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(total_input_ids[i])
            )

        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas

    # ==========================================================================
    # CASE 2) 이미지 또는 비디오만 있는 입력 (오디오 없음) — 원본 Qwen2.5-VL 경로
    # ==========================================================================
    elif input_ids is not None and (
        image_grid_thw is not None or video_grid_thw is not None
    ):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0   # 이미지/비디오 진행 포인터
        attention_mask = attention_mask.to(total_input_ids.device)

        # 배치 내 시퀀스마다 처리
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]   # 패딩 제거
            image_nums, video_nums = 0, 0

            # vision_start 토큰 다음 토큰을 보고 이미지/비디오 개수 카운트
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id
            ).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()

            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums

            # 이미지/비디오 블록을 등장 순서대로 하나씩 처리
            for _ in range(image_nums + video_nums):
                # 다음 이미지 토큰 / 다음 비디오 토큰 위치를 각각 찾고,
                # 더 앞쪽에 있는 것(ed_image vs ed_video)을 이번 루프에서 처리
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1

                if ed_image < ed_video:
                    # 이미지 블록 처리
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    second_per_grid_t = 0   # 이미지는 시간 축이 없으므로 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    # 비디오 블록 처리
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                # merge 후의 실제 LLM 그리드 크기
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st   # 비전 블록 직전까지 텍스트 길이

                # 다음 position id의 시작값(이전 블록 max + 1)
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                # 텍스트 구간의 position id (3축 모두 동일)
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

                # 비전(이미지/비디오) 구간의 3D position id
                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                # 시간 축: 초당 토큰 수 * temporal_patch(=2) 반영
                time_tensor = expanded_range * second_per_grid_t * 2
                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()

                # 높이/너비 축도 각각 T,H,W 순서로 평탄화해서 만듦
                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                # (3, T*H*W) 형태의 비전 position id 추가
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                # 다음 시작 위치 = 비전 토큰들 끝난 뒤
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            # 마지막 비전 블록 이후의 꼬리 텍스트 처리
            if st < len(input_tokens):
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            # 여러 구간을 이어붙여 최종 position_ids 완성
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                position_ids.device
            )
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(total_input_ids[i])
            )
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas

    # ==========================================================================
    # CASE 3) 오디오만 있는 입력 (이미지/비디오 없음)
    # ==========================================================================
    elif input_ids is not None and audio_lengths is not None:
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        audio_index = 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]   # 패딩 제거
            audio_nums = 0
            # 오디오 시작 위치 찾기 (비전 블록과 같은 vision_start 토큰을 재사용)
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id
            ).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            audio_nums = (vision_tokens == audio_token_id).sum()   # 이 샘플의 오디오 수
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_audios = audio_nums

            # 오디오 블록 하나씩 처리
            for _ in range(audio_nums):
                # 현재 오디오 토큰 블록의 시작 위치 탐색
                if audio_token_id in input_tokens and remain_audios > 0:
                    ed_audio = input_tokens.index(audio_token_id, st)
                else:
                    ed_audio = len(input_tokens) + 1
                ed = ed_audio

                # audio_lengths 값은 "이미 토큰 단위"의 길이로 들어온다고 가정
                llm_grid_t = audio_lengths[audio_index]  # 시간축 토큰 수
                llm_grid_h = 1                           # 오디오는 높이 개념 없음
                llm_grid_w = 1                           # 오디오는 너비 개념 없음

                text_len = ed - st
                st_idx = (llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0)
                # 텍스트 구간 position id
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

                # 오디오 구간의 3축 position id
                # (t: 0..L-1, h: t와 동일, w: 0 고정) 형태로 붙여 L개의 토큰 표현
                time_index = torch.arange(llm_grid_t, device=input_ids.device)
                h_index = time_index
                w_index = torch.zeros_like(time_index)
                audio_pos = torch.stack([time_index, h_index, w_index]) + st_idx + text_len
                llm_pos_ids_list.append(audio_pos)

                st = ed + llm_grid_t    # 다음 위치로 이동
                audio_index += 1
                remain_audios -= 1

            # 꼬리 텍스트 처리 (중요한 버그 수정 구간)
            if st < len(input_tokens):
                if len(llm_pos_ids_list) > 0:
                    # 마지막 오디오 블록의 time 축 max 값을 기준으로 이어 씀
                    last_time_max = llm_pos_ids_list[-1][0].max()
                    st_idx = last_time_max + 1
                else:
                    st_idx = 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            # 모든 구간 position id 합치기
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas

    # ==========================================================================
    # CASE 4) 멀티모달 입력이 전혀 없는 경우 (순수 텍스트)
    #  → 일반 LLM처럼 단순 1D position id를 만들고 (3, ...) 로 복제한다.
    # ==========================================================================
    else:
        if attention_mask is not None:
            # attention_mask로부터 누적합 - 1 형태의 position id 생성
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)   # 패딩 위치는 1로 마스킹
            position_ids = (
                position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            )
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                -1, keepdim=True
            )[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            # attention_mask도 없으면 0..seq-1 단순 나열
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas

# ============================================================================
# [아래의 주석 처리된 블록]
# 원본 Qwen2.5-VL 의 "이미지/비디오만" 처리하는 버전(get_rope_index_25)이다.
# 참고용으로 남겨져 있으며 실제 호출되지는 않는다. (위의 함수가 오디오까지 확장)
# ============================================================================
# original visual only rope index
# def get_rope_index_25(
#     spatial_merge_size: Optional[int] = 2,
#     input_ids: Optional[torch.LongTensor] = None,
#     image_grid_thw: Optional[torch.LongTensor] = None,
#     video_grid_thw: Optional[torch.LongTensor] = None,
#     second_per_grid_ts: Optional[torch.Tensor] = None,
#     attention_mask: Optional[torch.Tensor] = None,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

#     Explanation:
#         Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

#         For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
#         Examples:
#             input_ids: [T T T T T], here T is for text.
#             temporal position_ids: [0, 1, 2, 3, 4]
#             height position_ids: [0, 1, 2, 3, 4]
#             width position_ids: [0, 1, 2, 3, 4]

#         For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
#         and 1D rotary position embedding for text part.
#         Examples:
#             Temporal (Time): 3 patches, representing different segments of the video in time.
#             Height: 2 patches, dividing each frame vertically.
#             Width: 2 patches, dividing each frame horizontally.
#             We also have some important parameters:
#             fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
#             tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
#             temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
#             interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
#             input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
#             vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
#             vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
#             vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
#             text temporal position_ids: [101, 102, 103, 104, 105]
#             text height position_ids: [101, 102, 103, 104, 105]
#             text width position_ids: [101, 102, 103, 104, 105]
#             Here we calculate the text start position_ids as the max vision position_ids plus 1.

#     Args:
#         input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
#             Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
#             it.
#         image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
#             The temporal, height and width of feature shape of each image in LLM.
#         video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
#             The temporal, height and width of feature shape of each video in LLM.
#         second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
#             The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
#         attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

#             - 1 for tokens that are **not masked**,
#             - 0 for tokens that are **masked**.

#     Returns:
#         position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
#         mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
#     """
#     image_token_id = 151655
#     video_token_id = 151656
#     vision_start_token_id = 151652
#     mrope_position_deltas = []
#     if input_ids is not None and (
#         image_grid_thw is not None or video_grid_thw is not None
#     ):
#         total_input_ids = input_ids
#         if attention_mask is None:
#             attention_mask = torch.ones_like(total_input_ids)
#         position_ids = torch.ones(
#             3,
#             input_ids.shape[0],
#             input_ids.shape[1],
#             dtype=input_ids.dtype,
#             device=input_ids.device,
#         )
#         image_index, video_index = 0, 0
#         attention_mask = attention_mask.to(total_input_ids.device)
#         for i, input_ids in enumerate(total_input_ids):
#             input_ids = input_ids[attention_mask[i] == 1]
#             image_nums, video_nums = 0, 0
#             vision_start_indices = torch.argwhere(
#                 input_ids == vision_start_token_id
#             ).squeeze(1)
#             vision_tokens = input_ids[vision_start_indices + 1]
#             image_nums = (vision_tokens == image_token_id).sum()
#             video_nums = (vision_tokens == video_token_id).sum()
#             input_tokens = input_ids.tolist()
#             llm_pos_ids_list: list = []
#             st = 0
#             remain_images, remain_videos = image_nums, video_nums
#             for _ in range(image_nums + video_nums):
#                 if image_token_id in input_tokens and remain_images > 0:
#                     ed_image = input_tokens.index(image_token_id, st)
#                 else:
#                     ed_image = len(input_tokens) + 1
#                 if video_token_id in input_tokens and remain_videos > 0:
#                     ed_video = input_tokens.index(video_token_id, st)
#                 else:
#                     ed_video = len(input_tokens) + 1
#                 if ed_image < ed_video:
#                     t, h, w = (
#                         image_grid_thw[image_index][0],
#                         image_grid_thw[image_index][1],
#                         image_grid_thw[image_index][2],
#                     )
#                     second_per_grid_t = 0
#                     image_index += 1
#                     remain_images -= 1
#                     ed = ed_image

#                 else:
#                     t, h, w = (
#                         video_grid_thw[video_index][0],
#                         video_grid_thw[video_index][1],
#                         video_grid_thw[video_index][2],
#                     )
#                     if second_per_grid_ts is not None:
#                         second_per_grid_t = second_per_grid_ts[video_index]
#                     else:
#                         second_per_grid_t = 1.0
#                     video_index += 1
#                     remain_videos -= 1
#                     ed = ed_video
#                 llm_grid_t, llm_grid_h, llm_grid_w = (
#                     t.item(),
#                     h.item() // spatial_merge_size,
#                     w.item() // spatial_merge_size,
#                 )
#                 text_len = ed - st

#                 st_idx = (
#                     llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
#                 )
#                 llm_pos_ids_list.append(
#                     torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
#                 )

#                 range_tensor = torch.arange(llm_grid_t).view(-1, 1)
#                 expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

#                 time_tensor = expanded_range * second_per_grid_t * 2

#                 time_tensor_long = time_tensor.long()
#                 t_index = time_tensor_long.flatten()

#                 h_index = (
#                     torch.arange(llm_grid_h)
#                     .view(1, -1, 1)
#                     .expand(llm_grid_t, -1, llm_grid_w)
#                     .flatten()
#                 )
#                 w_index = (
#                     torch.arange(llm_grid_w)
#                     .view(1, 1, -1)
#                     .expand(llm_grid_t, llm_grid_h, -1)
#                     .flatten()
#                 )
#                 llm_pos_ids_list.append(
#                     torch.stack([t_index, h_index, w_index]) + text_len + st_idx
#                 )
#                 st = ed + llm_grid_t * llm_grid_h * llm_grid_w

#             if st < len(input_tokens):
#                 st_idx = (
#                     llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
#                 )
#                 text_len = len(input_tokens) - st
#                 llm_pos_ids_list.append(
#                     torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
#                 )

#             llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
#             position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
#                 position_ids.device
#             )
#             mrope_position_deltas.append(
#                 llm_positions.max() + 1 - len(total_input_ids[i])
#             )
#         mrope_position_deltas = torch.tensor(
#             mrope_position_deltas, device=input_ids.device
#         ).unsqueeze(1)
#         return position_ids, mrope_position_deltas
#     else:
#         if attention_mask is not None:
#             position_ids = attention_mask.long().cumsum(-1) - 1
#             position_ids.masked_fill_(attention_mask == 0, 1)
#             position_ids = (
#                 position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
#             )
#             max_position_ids = position_ids.max(0, keepdim=False)[0].max(
#                 -1, keepdim=True
#             )[0]
#             mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
#         else:
#             position_ids = (
#                 torch.arange(input_ids.shape[1], device=input_ids.device)
#                 .view(1, 1, -1)
#                 .expand(3, input_ids.shape[0], -1)
#             )
#             mrope_position_deltas = torch.zeros(
#                 [input_ids.shape[0], 1],
#                 device=input_ids.device,
#                 dtype=input_ids.dtype,
#             )

#         return position_ids, mrope_position_deltas


# ============================================================================
# [Qwen2-VL(구버전)용 3D RoPE index 계산 함수]
# 위의 get_rope_index_25와 구조는 같지만,
#   - 오디오를 다루지 않음
#   - 시간축 position id가 "second_per_grid_t * 2" 배수 없이
#     단순히 0,1,2,... 로 부여된다는 점이 다르다.
# 즉 시간 축을 "초 단위"가 아닌 "프레임 인덱스" 자체로 쓴다.
# ============================================================================
def get_rope_index_2(
    spatial_merge_size: Optional[int] = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embeddin for text part.
        Examples:
            Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
            input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
            vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text temporal position_ids: [3, 4, 5, 6, 7]
            text height position_ids: [3, 4, 5, 6, 7]
            text width position_ids: [3, 4, 5, 6, 7]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """
    # 특수 토큰 ID (Qwen2-VL)
    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652
    mrope_position_deltas = []

    # 이미지 또는 비디오가 포함된 멀티모달 입력 처리
    if input_ids is not None and (
        image_grid_thw is not None or video_grid_thw is not None
    ):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)

        # 결과 position_ids: (3, batch, seq_len)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]   # 패딩 제거
            image_nums, video_nums = 0, 0

            # vision_start 다음 토큰으로 이미지/비디오 개수 카운트
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id
            ).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()

            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums

            # 이미지/비디오 블록을 등장 순서대로 처리
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1

                # 더 앞쪽에 등장하는 블록을 이번 루프에서 처리
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                # LLM에서 실제 보게 되는 그리드 크기 (H/W는 merge 후)
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                # 이전 블록의 max+1 부터 이어 쓰기
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                # 텍스트 position id (3축 동일)
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

                # 비전 position id: Qwen2-VL은 시간축을 그냥 0..T-1 로 사용
                t_index = (
                    torch.arange(llm_grid_t)
                    .view(-1, 1)
                    .expand(-1, llm_grid_h * llm_grid_w)
                    .flatten()
                )
                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            # 꼬리 텍스트 처리
            if st < len(input_tokens):
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            # 구간 이어붙여 최종 position id 완성
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                position_ids.device
            )
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(total_input_ids[i])
            )
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        # 순수 텍스트 입력: 일반 LLM처럼 1D 위치를 (3,...)로 확장
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = (
                position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            )
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                -1, keepdim=True
            )[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas
