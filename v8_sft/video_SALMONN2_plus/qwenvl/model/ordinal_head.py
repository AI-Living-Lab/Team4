# Copyright (2026)
# Ordinal head for time-token digit prediction (CORN conditional decomposition).
#
# 자리당 K_max=10 digits 에 대해 K_max-1=9 개의 sigmoid 확률을 출력한다:
#   p_0 = P(d > 0)
#   p_k = P(d > k | d > k-1),  k = 1..8
# Target:  t_k = 1[d > k]
# Loss:    BCE(p_k, t_k) per (position, k)
#
# 5(또는 4) 개 자리 모두 같은 head 를 공유 (위치 정보는 모델 hidden state 에 implicit).

import torch
import torch.nn as nn


class OrdinalHead(nn.Module):
    """단일 Linear head (D -> 9), CORN 분해용.

    Args:
      hidden_dim: int. LM hidden state dimension.
      n_thresholds: int = 9. K_max - 1 (digits 0..9 → 9 thresholds).
      init_std: float = 0.01. weight init std. 작게 유지해서 학습 초반
                LM hidden state 에 큰 영향을 주지 않도록 한다.
    """

    def __init__(self, hidden_dim: int, n_thresholds: int = 9, init_std: float = 0.01):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, n_thresholds, bias=True)
        nn.init.normal_(self.linear.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.linear.bias)
        self.n_thresholds = n_thresholds
        self.hidden_dim = hidden_dim

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
          hidden_states: [..., hidden_dim]
        Returns:
          logits: [..., n_thresholds] — sigmoid 입력 (CORN BCE)
        """
        return self.linear(hidden_states)
