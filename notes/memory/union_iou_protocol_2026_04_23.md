---
name: Union-IoU + FP_rate + FN_rate 통일 (2026-04-23 팀 지표 표준화)
description: 모든 baseline eval 을 Best-IoU → Union-IoU + FP/FN 로 일괄 교체. 추론 재실행 불필요, eval 만 재계산
type: feedback
originSessionId: 7cf60908-ffa2-425c-9445-c628fd4bc0af
---
**2026-04-23 팀 피드백 반영**: mIoU 지표를 **Union-IoU** 로 통일하고, multi-seg 예측의 over/under-prediction 분석용 **FP_rate / FN_rate** 를 필수 보조 지표로 추가.

**Why:**
- 기존 Best-IoU: 각 GT 마다 가장 잘 맞는 pred 1개만 반영 → **multi-seg prediction 불리**. GT 를 여러 pred 로 쪼개서 잘 커버해도 점수는 하나만.
- 개선 Union-IoU: 각 GT 와 겹치는 pred **모두의 합집합** 과 IoU → multi-seg 예측 공정 평가. 쪼개서 맞춰도 합산 반영.
- FP_rate = (어떤 GT 와도 안 겹치는 pred) / 전체 pred → over-prediction 측정
- FN_rate = (어떤 pred 와도 안 겹치는 GT) / 전체 GT → under-prediction 측정
- 둘 다 낮을수록 좋음. IoU 만으로는 "과장" vs "놓침" 구분 안 됨.

**How to apply:**
- 공통 라이브러리: `/workspace/jsy/scripts/eval_utils.py` (`compute_union_iou`, `score_sample`, `summarize`, `print_report`)
- 팀 표준 파서 (cdh f7920fd): `/workspace/jsy/Team4-cdh/eval/eval_miou_multiseg.py` (Union-IoU + FP + FN, 내가 FN 추가)
- 팀 NL 파서: `/workspace/jsy/Team4-cdh/eval/eval_miou_nl_parser.py` (Union-IoU + FP + FN 추가됨)
- baseline eval 스크립트 전부 재작성 완료 (`eval_miou_multiseg_chronusomni.py`, `_arc_hunyuan.py`, `_crab_plus.py`, `eval_unavqa_tvg.py`, `eval_longvale_tvg.py`)

**재측정 결과 (2026-04-23, UnAV-100 full 3455 / 5900 GT)**:
| 모델 | FSR | mIoU(union) | R@0.3 | R@0.5 | R@0.7 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| ChronusOmni hybrid | 100.00% | 36.78% | 43.61% | 33.49% | 25.58% | 5.46% | 22.00% |
| ARC-Hunyuan hybrid | 100.00% | 35.46% | 41.88% | 31.64% | 24.59% | 4.78% | 18.59% |
| Avicuna stage4 | 100.00% | 30.42% | 35.51% | 28.20% | 22.92% | 22.92% | 40.93% |
| SALMONN base V2 (+hint) | 97.2% | 16.89% | 21.31% | 12.08% | 6.22% | 28.34% | 38.17% |
| SALMONN base V1 (no hint) | 1.68% | 0.28% | 0.32% | 0.15% | 0.12% | 18.64% | 98.93% |

LongVALE 1k subset:
| 모델 | FSR | mIoU(union) | FP | FN |
|---|---:|---:|---:|---:|
| Avicuna × LongVALE 1k | 99.50% | 16.56% | 54.47% | 54.70% |
| SALMONN base × LongVALE 1k (NL) | 96.6% | 7.76% | 62.53% | 63.80% |

**핵심 관찰:**
- Union-IoU 값이 Best-IoU 와 ±0.1%p 차이 — 현 baseline 들은 대부분 **single-seg 출력** 이라 multi-seg 이점 없음
- Union-IoU 이득은 **팀 SFT+GDPO (multi-seg 출력 가능)** 에서 본격 발휘될 것. Best-IoU 리포트 모델 vs Union-IoU 리포트 모델 mix 하면 unfair → **전수 Union-IoU 통일** 필수
- FP/FN 로 현 baseline 특성 드러남: ARC-Hunyuan 이 가장 균형 (FP 낮음 + FN 낮음), Avicuna 는 FP/FN 모두 높음(과장+놓침)

**재측정 시간 (참고)**: eval 스크립트 재실행만으로 전부 몇 분 내 완료. 추론은 안 다시 돌림 (저장된 `test_results_rank0.json` / `predictions.jsonl` 재활용).

**TODO (미해결)**:
- Crab+ partial 200 samples Union-IoU 재측정 — 필요시 `eval_miou_multiseg_crab_plus.py` 로 돌리면 끝. 현재 limitation 문구 에 포함 안 함 (추후 포함 여부 결정)
- 팀 SFT/GDPO 모델 eval 시 `eval_miou_multiseg.py` (cdh 버전, FN 포함됨) 사용 → 자동으로 Union-IoU + FP + FN 표 생성
