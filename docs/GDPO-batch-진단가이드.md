# GDPO batch(Tier1) 테스트 & 진단 가이드

`gdpo_trainer_batch.py` (배치 실험 + 헛디코드 제거 + 자가진단 계측)를 **A100에서** 돌려
"batch를 키울 수 있는가 / RAM OOM이 해소되는가"를 확정하기 위한 문서.
진단 후 삭제

---

## 3. 테스트 실행 (2단계 권장)

### 3-1. Smoke test (3 step) — "안 깨지고 캐시 켜지는지"

```bash
python gdpo/gdpo_trainer_batch.py \
    --config       gdpo/config_sep2fp_lr_mlp_headoff.yaml \
    --model_path   $CKPT_DIR/base/salmonn2p_7b_unav_v8 \
    --model_base   $CKPT_DIR/base/salmonn2p_7b_unav_v8 \
    --dataset_path $TRAIN_DIR/unav100_v2.json \
    --reward_module reward_functions_sep2fp \
    --tti_mode on --max_steps 3 \
    --output_dir $CKPT_DIR/gdpo/_smoke_batch
```

확인:
- 시작 로그에 `[GDPO Tier1] video_cache_size=2 (μ=2, grad_accum=1 ...)` (현 config는 grad_accum=1)
- `[GDPO Tier1-DBG] process_video 계측 부착 ...`
- 3 step 크래시 없이 완료 + `[GDPO STEP]` 수치가 기존과 동일(디코드만 생략, 결과 불변)

### 3-2. Stage A — 현재 batch 그대로(grad_accum=1, eff batch 2)로 "RAM 해소" 검증

config 그대로(μ=2, grad_accum=1) 본 런. **batch는 안 키우고** Tier1 효과만 분리 확인.

```bash
OUT=$CKPT_DIR/gdpo/batch_stageA_ga1
CUDA_VISIBLE_DEVICES=0,4 setsid nohup torchrun --standalone --nproc_per_node=2 \
    --master_port=29517 gdpo/gdpo_trainer_batch.py \
    --config       gdpo/config_sep2fp_lr_mlp_headoff.yaml \
    --model_path   $CKPT_DIR/base/salmonn2p_7b_unav_v8 \
    --model_base   $CKPT_DIR/base/salmonn2p_7b_unav_v8 \
    --dataset_path $TRAIN_DIR/unav100_v2.json \
    --reward_module reward_functions_sep2fp --tti_mode on \
    --output_dir "$OUT" --run_name batch_stageA_ga1 \
    < /dev/null > /dev/null 2>&1 &
tail -f "$OUT/train.log"   # 또는 콘솔 [GDPO RSS] 확인
```

### 3-3. Stage B — batch 키우기(grad_accum=2, eff batch 4)

config의 `training.gradient_accumulation_steps: 1 → 2` 로 바꾼 사본을 만들어 실행
(그러면 `video_cache_size`도 자동 3). 고유 prompt 수 보존하려면 `max_steps`를 절반으로:
`고유 prompt = max_steps × grad_accum × world / μ` 동일하게 유지.

```bash
cp gdpo/config_sep2fp_lr_mlp_headoff.yaml gdpo/config_batch_ga2.yaml
#   → config_batch_ga2.yaml 에서 gradient_accumulation_steps: 2 로 수정 (필요시 max_steps 조정)
OUT=$CKPT_DIR/gdpo/batch_stageB_ga2
CUDA_VISIBLE_DEVICES=0,4 setsid nohup torchrun --standalone --nproc_per_node=2 \
    --master_port=29518 gdpo/gdpo_trainer_batch.py \
    --config       gdpo/config_batch_ga2.yaml \
    --model_path   $CKPT_DIR/base/salmonn2p_7b_unav_v8 \
    --model_base   $CKPT_DIR/base/salmonn2p_7b_unav_v8 \
    --dataset_path $TRAIN_DIR/unav100_v2.json \
    --reward_module reward_functions_sep2fp --tti_mode on \
    --output_dir "$OUT" --run_name batch_stageB_ga2 \
    < /dev/null > /dev/null 2>&1 &
```

> ⚠️ **캐시는 μ>1이면 grad_accum=1에서도 켜집니다**(현 config의 [p0,p0] 인접 재사용도 제거).
> 그래서 Stage A만으로도 "헛디코드 제거가 RAM을 잡는가"를 먼저 확인할 수 있습니다.

### 모니터링 (실행 중)

```bash
tail -f "$OUT/train.log"                  # [GDPO RSS] step=.. rss=..GB cache_hit_rate=..
nvidia-smi                                # VRAM (80GB 대비 여유)
free -g                                   # 호스트 RAM 잔여
# wandb 패널: rss_gb, cache_hit_rate, reward, ratio, clip_frac
```

---

## 4. 봐야 할 핵심 신호 2개

| 신호 | 어디서 | 정상값 | 의미 |
|------|--------|--------|------|
| **`cache_hit_rate`** | wandb / `[GDPO RSS]` 콘솔 | **μ=2 → ≈0.5** | 캐시가 실제로 재디코드를 막는가 |
| **`rss_gb`** | wandb / `[GDPO RSS]` 콘솔 | **평탄(우상향 X)** | 호스트 RAM 누수 여부 |

---

## 5. 진단 결정 트리 (결과별)

### ① 캐시가 작동하는가 — `cache_hit_rate`

- **≈ 0.5 (μ=2)** → ✅ 캐시 정상. 헛디코드가 절반 제거됨.
- **≈ 0 인데 μ=2** → ❌ **캐시 미작동**. 원인 후보:
  - `video_cache_size`가 0으로 들어옴 → 시작 로그 `[GDPO Tier1] video_cache_size=` 확인
    (μ가 2로 안 잡혔거나, config의 `clip.num_iterations`가 1).
  - block-repeat가 안 켜짐(grad_accum=1 & μ=1) → 이 경우 캐시 의미 없음(정상).
  - 같은 prompt가 같은 `video_file`이 아님(데이터 경로 형식 이슈) → 키 불일치.
- **0 < rate < 0.5** → 일부만 hit. DDP 순서 교란/캐시 크기 부족 가능. 보통 무해(miss=재디코드).

### ② 호스트 RAM이 새는가 — `rss_gb` 기울기

- **평탄(거의 일정)** → ✅ 누수 아님(working-set). **A100 + Tier1로 OOM 해소 확정.**
- **우상향(계속 증가)** → ⚠️ **누수/단편화 존재.** 추가 판단:
  - 기울기(GB/100step) × 총 step < (노드 RAM total − 시작 RSS) 이면 → **완주 가능**(천장 안 닿음).
  - 그렇지 않으면 → **§7 Tier2** 적용 필요.
  - 참고: Tier1 ON(이번) vs 이전(OFF) 기울기를 비교하면 헛디코드 제거 효과가 정량으로 보임.

### ③ 그래도 OOM이 나면 — 에러 타입부터 구분

| 에러 메시지 | 종류 | 해석/처방 |
|-------------|------|-----------|
| `CUDA out of memory. Tried to allocate ...` | **VRAM** | A100 80GB에선 드묾. `num_generations`/`max_completion_length`/`video_max_frames`↓ 또는 긴-비디오 샘플 이슈 |
| `Cannot allocate memory` / `DefaultCPUAllocator: not enough memory` / 프로세스 `Killed` (dmesg에 oom-killer) | **호스트 RAM** | 본 가이드 대상. `rss_gb` 기울기 확인 → §7 |

> `dmesg -T | grep -i oom` 로 OOM-killer가 죽였는지 확인 가능.

---

## 6. 합격 기준 (이러면 batch 키워도 OK)

- [ ] smoke 3 step 크래시 없음 + 수치 동일
- [ ] `cache_hit_rate ≈ 0.5` (μ=2)
- [ ] `rss_gb` 평탄, 또는 기울기가 노드 RAM 천장 안에서 완주 가능
- [ ] Stage B(grad_accum=2)에서 OOM 없이 진행

---

## 7. RSS가 계속 오르면 — Tier2 (누수 천장 올리기)

`dataset.py`의 디코드 경로 churn 제거 (효과 큰 순):

1. `process_video_frames`의 **매 호출 `copy.deepcopy(self.data_args.image_processor)` 제거**
   (프로세서를 1회 만들어 재사용/복원). 매 비디오마다 deepcopy하던 churn 제거.
2. **주기적 `gc.collect()`** (예: 콜백에서 50 step마다) + Linux `ctypes.CDLL("libc.so.6").malloc_trim(0)`.
3. `video_max_frames` 128 → 64 (디코드 1회당 churn 절반, temporal 해상도 trade-off).

> Tier2는 `dataset.py`(공유 파일) 수정이라 적용 시 gate 또는 별도 검증 필요.

---

## 8. 롤백 / 원복

- 계측·캐시는 **μ>1일 때만** 동작. μ=1 또는 SFT는 자동 OFF=기존 동작.
- 완전 비활성: config에서 `clip.num_iterations: 1` (캐시·계측 모두 미진입).
- `gdpo_trainer_batch.py`만 안 쓰면 됨. `dataset.py`는 OFF 기본이라 되돌릴 필요 없음.

---

## 부록: 계측 위치 (코드)

- RSS 헬퍼: `gdpo_trainer_batch.py` `_read_rss_gb()` (`/proc/self/status` VmRSS, Linux 전용)
- 캐시 카운터: `main()` 데이터셋 생성 직후 `process_video` monkeypatch (dataset.py 무수정)
- 로깅: `GDPOTrainer.compute_loss` 끝 (rank0, μ>1) → `_metrics["cache_hit_rate"|"rss_gb"]` + 콘솔 20 step마다
