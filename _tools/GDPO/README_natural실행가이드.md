# natural mode GDPO 학습 실행 가이드

**run:** `sft_7b_unav_v8_rl_rMsep3_unpucha_batch4_noscaling_natural`
sep3(noscaling) 과 그 외 설정 100% 동일. 다른 점은 **입력마커=natural_text, 출력/GT=chronus** 뿐.

- 입력마커: `second{XXX.Y}` (zero-pad 3자리, **8토큰**) — special_token(XXX.Y)과 자릿수 정합
- 출력/GT: `second{start}-second{end}. second{start}-second{end}. ...` (chronus, unpadded)

## 생성된 별도 파일 (기존 파일·실행중 job 미변경)
| 종류 | 파일 |
|---|---|
| 트레이너 | `_tools/GDPO/gdpo_trainer_batch_natural.py` |
| 리워드 | `_tools/GDPO/reward_functions_rM_sep3_natural.py` |
| config | `_tools/GDPO/config_sep3_natural.yaml` |
| 런처 | `_tools/GDPO/run_rMsep3_natural.sh` |
| 학습 데이터 | `data/train/unpucha_chronus.json` |
| val 데이터 | `data/val/unav100_tail_val100_chronus.json`, `data/val/charades_tail_val100_chronus.json` |
※ 입력마커 XXX.X(8토큰)는 공유 `dataset.py`(`sec_to_natural_text_str` + `_TIME_MARKER_TOKEN_LEN`)에서 조정 —
  natural_text 모드에서만 호출되므로 실행중 special_token job 엔 무영향.

## ⚠️ 시작 전 필수 확인 — GPU
이 랩엔 **GPU가 2장(0,1)뿐**이고, 준비 시점 기준 **둘 다 학습으로 100% 점유중**이다.
→ **기존 학습이 끝나 GPU가 빌 때까지 대기**하거나, 빈 GPU가 생기면 실행.
```bash
watch -n 30 nvidia-smi        # 0,1 이 idle(0%/저메모리) 되면 실행 가능
```

## tmux 실행 절차
```bash
# 1) 세션 생성 (분리돼도 학습 유지)
tmux new -s natural

# 2) (세션 안에서) 저장소로 이동 후 런처 실행
cd /home/team404/workspace/master/Team4
bash _tools/GDPO/run_rMsep3_natural.sh
#   └ conda activate salmonn2plus / paths.env source / torchrun 2GPU 까지 런처가 다 함.
#     로그는 stdout + $CKPT_DIR/gdpo/<run>/train.log 에 동시 기록(tee).

# (선택) 첫 3스텝 TTI 디버그 계측을 켜서 마커 삽입을 눈으로 확인하고 싶으면:
TTI_DEBUG=1 TTI_DEBUG_STEPS=3 bash _tools/GDPO/run_rMsep3_natural.sh

# 3) 분리(detach): Ctrl+b 누른 뒤 d
# 4) 재접속:       tmux attach -t natural
# 5) 세션 목록:    tmux ls
```

## 초기 정상동작 확인 (attach 하거나 train.log tail)
```bash
tail -f /home/team404/workspace/checkpoints/gdpo/sft_7b_unav_v8_rl_rMsep3_unpucha_batch4_noscaling_natural/train.log
```
아래가 보이면 natural 배선이 맞은 것:
- `[GDPO] tti_mode=on + tti_time_format=natural_text → time_token_id_range=None, time_marker_token_len=8`
- `[GDPO] tti_mode=on → data_args.tti_time_format=natural_text`
- `[GDPO SAMPLES] GT sec: (..)` 가 `[none]` 이 아님 (chronus GT 파싱 성공)

**TTI_DEBUG=1 로 켰을 때** 추가로:
- `② main: tti_mode=on, tti_time_format=natural_text, model.config.time_token_id_range=None`  (desync 경고 없어야 정상)
- `⑤ step N: ... marker_len=8, #special_markers(id)=0, #natural_markers(second{})=<>0>`
  - `#natural_markers(second{})` 가 0 이 아니면 입력마커 삽입 성공.
  - `prompt[vision_start:+40]= 'second{072.0}...'` 로 마커 실물 확인.
  - `⚠️ WARNING: natural_text 인데 ... 'second{...}' 마커 0개` 가 뜨면 마커 미삽입 → 데이터/collator 점검.

## 🔴 이 실험의 핵심 리스크 — 포맷 콜드스타트 (반드시 모니터링)
SFT 베이스(v8)는 **special_token 출력·입력**에 맞춰 학습됨. 이번엔 출력=chronus, 입력마커=natural_text 라
초기에는 모델이 못 맞출 수 있다. 프롬프트가 chronus 를 지시하므로 instruction-following 으로 적응할 여지는 있음.
**초반 스텝에서 확인:**
- wandb `rewards/format` 이 0 근처에서 안 오르면 → 모델이 chronus 포맷을 못 뽑는 것 (학습 신호 약함)
- val `n_parse_ok` 가 0 근처면 → 출력 파싱 실패 (같은 원인)
- `[GDPO SAMPLES]` pred 가 여전히 `From <t..>` 형태면 → 베이스 special_token 관성 → format reward 로 교정 기대

format reward 가 몇백 스텝 내 상승하면 정상. 계속 0 이면 SFT 단계에서 chronus 로 데운(warm-start) ckpt 필요.

## GPU/포트 조정 (필요시, 런처 안에서 수정)
- `export CUDA_VISIBLE_DEVICES=0,1` → 빈 GPU 번호로
- `--master_port=29523` → 다른 job 과 겹치면 변경 (기존 sep3=29522 회피값)
- 1장만 쓸 거면 `--nproc_per_node=1` + config 의 `gradient_accumulation_steps` 를 4 로 (eff batch 4 유지)

## 학습 후 best ckpt 선택 (sep3 와 동일)
```bash
python _tools/GDPO/select_best_ckpt.py \
  /home/team404/workspace/checkpoints/gdpo/sft_7b_unav_v8_rl_rMsep3_unpucha_batch4_noscaling_natural
# val_metrics.jsonl 의 combined(=sample_miou+f1_avg 평균) 기준 best step 선택.
```
