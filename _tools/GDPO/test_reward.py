#!/usr/bin/env python3
"""GDPO_v2(revised) 리워드 함수 + convert 임포트 테스트."""
import sys, os

# 프로젝트 루트 + 현재 폴더를 path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))

from reward_functions import format_reward, label_reward, iou_reward, combined_reward
from convert_to_gdpo import convert_sample
print("reward_functions: OK")
print("convert_to_gdpo: OK")

# 리워드 함수 테스트
test_completion = '[{"event": "skateboarding", "start": 13.0, "end": 15.8}]'
gt_events = [{"label": "skateboarding", "timestamps": [13.01, 15.78]}]

r_format = format_reward(test_completion)
r_label = label_reward(test_completion, gt_events)
r_iou = iou_reward(test_completion, gt_events)
r_combined = combined_reward(test_completion, gt_events)

print(f"format_reward: {r_format}")
print(f"label_reward: {r_label}")
print(f"iou_reward: {r_iou:.4f}")
print(f"combined_reward: {r_combined:.4f}")

# gdpo_trainer 임포트 테스트
try:
    from gdpo_trainer import GDPOTrainer
    print("gdpo_trainer: OK")
except ImportError as e:
    print(f"gdpo_trainer: SKIP (dependencies not installed) - {e}")

print("\nAll tests passed!")
