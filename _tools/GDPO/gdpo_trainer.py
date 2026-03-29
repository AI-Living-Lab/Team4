#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gdpo_trainer.py
  trl의 GRPOTrainer를 상속하여 GDPO(Group reward-Decoupled normalization
  Policy Optimization) advantage 계산을 구현.

  GDPO 핵심:
    1. 각 리워드를 독립적으로 그룹 내 정규화
    2. 가중 합산
    3. 배치 정규화

  참고: https://github.com/NVlabs/GDPO
"""

import torch
from trl import GRPOTrainer, GRPOConfig


class GDPOTrainer(GRPOTrainer):
    """GRPOTrainer에 GDPO advantage 계산을 추가한 트레이너.

    Args:
        reward_weights: 각 리워드 함수의 가중치 리스트.
                        예: [0.1, 0.3, 0.6] (format, label, iou)
        apply_gdpo: True면 GDPO 정규화, False면 기본 GRPO.
        **kwargs: GRPOTrainer에 전달되는 모든 인자.
    """

    def __init__(self, reward_weights=None, apply_gdpo=True, **kwargs):
        super().__init__(**kwargs)
        self.apply_gdpo = apply_gdpo

        if reward_weights is not None:
            self.reward_weights = torch.tensor(
                reward_weights, dtype=torch.float32
            )
        else:
            self.reward_weights = None

    def _compute_advantages(
        self, rewards_per_func: torch.Tensor
    ) -> torch.Tensor:
        """GDPO advantage 계산.

        Args:
            rewards_per_func: shape (batch_size, num_reward_funcs)

        Returns:
            advantages: shape (batch_size,)
        """
        num_reward_funcs = rewards_per_func.shape[1]

        # 리워드 함수가 1개이거나 GDPO 비활성화면 기본 GRPO
        if not self.apply_gdpo or num_reward_funcs <= 1:
            return super()._compute_advantages(rewards_per_func)

        device = rewards_per_func.device
        reward_weights = self.reward_weights
        if reward_weights is None:
            reward_weights = torch.ones(num_reward_funcs, device=device)
        reward_weights = reward_weights.to(device)

        # [개선 1] 가중치 개수 검사
        assert len(reward_weights) == num_reward_funcs, \
        f"reward_weights 개수({len(reward_weights)})와 " \
        f"reward_funcs 개수({num_reward_funcs})가 다릅니다."

        rewards_per_func = torch.nan_to_num(rewards_per_func)
        num_generations = self.num_generations

        all_advantages = []
        for i in range(num_reward_funcs):
            reward_i = rewards_per_func[:, i]

            # [개선 2] 공식과 동일하게 1D 유지 방식으로 변경
            mean_g = reward_i.view(-1, num_generations).mean(dim=1)
            std_g  = reward_i.view(-1, num_generations).std(dim=1)

            mean_g = mean_g.repeat_interleave(num_generations, dim=0)
            std_g  = std_g.repeat_interleave(num_generations, dim=0)

            each_advantage = (reward_i - mean_g) / (std_g + 1e-4)
            all_advantages.append(each_advantage)

        # 가중 합산
        stacked = torch.stack(all_advantages, dim=1)  # (batch, num_funcs)

        # [개선 3] sum → nansum으로 변경하여 NaN이 있는 경우에도 계산 가능하도록
        combined = (stacked * reward_weights.unsqueeze(0)).nansum(dim=1)

        # 배치 정규화
        advantages = (combined - combined.mean()) / (combined.std() + 1e-4)

        return advantages
