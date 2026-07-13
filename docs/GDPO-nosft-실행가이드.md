# nosft GDPO(바로 RL) 학습 실행 가이드

**run:** `sft_7b_unav_v8_rl_rMsep3_unpucha_batch4_nosft`
SFT 체크포인트가 아니라 **time-token base 에서 바로 GDPO(RL)**. sep3(noscaling)과 그 외 100% 동일.

## sep3 대비 차이 (model 만 교체)
| 항목 | sep3(noscaling) | nosft |
|---|---|---|
| model_path/base | `salmonn2p_7b_unav_v8` (SFT-머지) | **`video_salmonn2_plus_7B_time_tokens` (base)** |
| output/run | ...noscaling | **...nosft** |
| trainer | `gdpo_trainer_batch.py` | 동일 |
| config | `config_sep3.yaml` | 동일 (재사용) |
| reward | `reward_functions_rM_sep3` | 동일 |
| dataset | `unpucha_v2.json` | 동일 |
| TTI | special_token (on) | 동일 |

- base 엔 adapter 없음 → **fresh RL LoRA(경로 B)** 로 base 위에 바로 RL.
- config 에 time_token_id_range 없음 → 트레이너가 토크나이저에서 `(<t0>..<tdot>)=(151666,151676)`, marker_len=5 복원.
- 출력=special_token 이라 **natural 실험 때의 format reward 콜드스타트 없음** (base 가 이미 `<t..>` 포맷을 냄).
- 신규 파일은 런처 1개(`run_rMsep3_nosft.sh`) 뿐. config/trainer/reward 는 sep3 것 재사용(CLI 오버라이드).

## GPU
현재 0,1 은 다른 학습 점유중 → 런처는 **2,3** 사용(`CUDA_VISIBLE_DEVICES=2,3`), master_port=29524.
바뀌면 런처 안에서 조정.
```bash
nvidia-smi   # 2,3 이 비어있는지 확인
```

## tmux 실행 절차
```bash
tmux new -s nosft
cd /home/team404/workspace/master/Team4
bash gdpo/run_rMsep3_nosft.sh
# 분리: Ctrl+b 그다음 d   재접속: tmux attach -t nosft
```

## 로그 모니터링
```bash
tail -f /home/team404/workspace/checkpoints/gdpo/sft_7b_unav_v8_rl_rMsep3_unpucha_batch4_nosft/train.log
```
정상 신호:
- `[GDPO] tti_mode=on ... time_token_id_range=(151666, 151676) ... marker_len=5` (토크나이저 복원)
- `data_args.tti_time_format=special_token`
- `[GDPO SAMPLES] GT sec: (..)` 파싱됨
- `rewards/format` 이 **0 이 아님**(base 가 special_token 포맷을 내므로 초반부터 신호 있어야 정상)

## nosft 관전 포인트 (SFT warm-start 가치 측정)
- SFT 없이 시작하므로 초반 `reward`/`seg_miou` 가 sep3(SFT-start)보다 낮게 출발할 것.
- 핵심 질문: RL 만으로 어디까지 따라잡나 → sep3(SFT→RL) 대비 val combined 곡선 비교.
- format reward 가 계속 낮으면(base 가 TVG 포맷 지시를 잘 못 따르면) 그때 SFT 필요성 입증.

## 학습 후 best ckpt 선택 (sep3 동일)
```bash
python gdpo/select_best_ckpt.py \
  /home/team404/workspace/checkpoints/gdpo/sft_7b_unav_v8_rl_rMsep3_unpucha_batch4_nosft
```
