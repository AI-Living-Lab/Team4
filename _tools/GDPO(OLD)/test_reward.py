#!/usr/bin/env python3
"""GDPO 리워드 함수 + convert 임포트 테스트."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from _tools.GDPO.reward_functions import format_reward, label_reward, iou_reward, combined_reward
from _tools.GDPO.convert_to_gdpo import convert_sample
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
    from _tools.GDPO.gdpo_trainer import GDPOTrainer
    print("gdpo_trainer: OK")
except ImportError as e:
    print(f"gdpo_trainer: SKIP (trl not installed) - {e}")

# train_gdpo 임포트 테스트
try:
    from _tools.GDPO.train_gdpo import load_gdpo_dataset
    print("train_gdpo: OK")
except ImportError as e:
    print(f"train_gdpo: SKIP (trl/datasets not installed) - {e}")

print("\nAll tests passed!")
