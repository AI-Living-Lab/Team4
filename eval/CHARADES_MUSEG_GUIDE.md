# MUSEG × Charades-STA 평가 가이드

생성일 2026-08-30. 데이터 빌더: `Team4/eval/build_charades_museg.py`.

## 1. 데이터 (생성 완료)

| 파일 | n | 용도 |
|---|---|---|
| `data/test/charades_sta_museg.json` | 3720 | flat 테스트셋 (채점 시 `--test_json`) |
| `data/test/charades_sta_museg_sanity30.json` | 30 | 스모크 |
| `data/test/charades_sta_museg/chunk_0000..0007.json` | 500×7+220 | 추론 입력 청크 |

출처: `datasets/charades_sta/annotations/charades_sta_test.txt` (공식 Charades-STA test, 3720 문장-구간 쌍 / 1334 영상).
영상 경로는 이 서버 실제 경로(`datasets/charades_sta/Charades_v1/*.mp4`)로 이미 기입 — **경로 rewrite 불필요**.
다른 서버로 옮길 땐 `build_charades_museg.py --video_root ... --audio_root ...` 로 재생성.

### 스키마 (`thumos_tail_museg.json` + MUSEG 파이프라인 필드)
```json
{
  "id": "3MSZA_00000",            // {vid}_{글로벌index:05d} — eval_museg.py 의 GT 매칭 키
  "vid": "3MSZA",
  "video": "/home/team404/workspace/datasets/charades_sta/Charades_v1/3MSZA.mp4",
  "audio": "/home/team404/workspace/datasets/charades_sta/audio/3MSZA.wav",
  "use_audio": false,             // MUSEG = 비디오 전용
  "question": "You are given a video about human daily activities. Watch the video carefully and find the visual event described by the sentence: 'person turn a light on'. Output in the format of \"X.XX-X.XX\".",
  "conversations": [{"from": "human", "value": "<video>\n<question>"}, {"from": "gpt", "value": "24.30-30.40"}],
  "gt_label": "person turn a light on",
  "gt_segments": [[24.3, 30.4]],
  "groundtruth": [[24.3, 30.4]]   // eval_grounding.py 요구 필드
}
```
- 프롬프트는 `thumos_tail_museg.json` 의 MUSEG 스타일을 문장 grounding 용으로 맞춘 것
  (THUMOS 는 action category, Charades 는 문장 쿼리). **format 블록 없음** — MUSEG 네이티브
  TEMPLATE 이 `<think>/<answer>` 를 이미 강제하므로 중복 방지(`build_museg_inputs.py` 와 동일 방침).
- Charades-STA 는 문장당 세그먼트 1개 → 멀티세그 아님. pairwise/union/sample mIoU 가 모두 동일값.

### 주의: 원본 annotation 1줄 손상
`charades_sta_test.txt` 286번째 줄의 vid 토큰이 `datasets/thumos/LongVALE5B9XE` 로 오염돼 있음
(과거 일괄 치환 사고로 보임). 빌더가 뒤 5자(`5B9XE`)로 자동 복구하고 로그를 남김.
원본 파일은 건드리지 않았으므로, 다른 스크립트가 이 파일을 읽는다면 같은 문제가 재현됨.

## 2. 추론

이 서버(team404)에는 **MUSEG repo(`github/MUSEG`)도 `MUSEG-7B` 체크포인트도 없음**.
둘 다 있는 서버(aix23102)에서 아래처럼 실행 — `run_museg_multigpu.sh` 의 경로만 바꾼 형태:

```bash
WS=<서버 workspace>
IND=$WS/data/test/charades_sta_museg          # 청크 그대로 입력으로 사용
RESD=$WS/outputs/base/MUSEG/charades_sta/results
mkdir -p "$RESD"
conda activate museg

for sid in 0 1 2 3 4 5; do
  gpu=$((2 + sid))
  CUDA_VISIBLE_DEVICES=$gpu nohup python $WS/github/MUSEG/src/infer_unav_chunks.py \
    --ckpt_path $WS/checkpoints/base/MUSEG-7B \
    --inputs_dir "$IND" --results_dir "$RESD" \
    --batch_num 8 --max_model_len 32768 \
    --num_shards 6 --shard_id $sid \
    > $WS/outputs/base/MUSEG/charades_sta/run_shard${sid}.log 2>&1 &
  sleep 25   # 모델 로드 디스크 경합 완화
done
```
- 샤드별로 서로소 청크를 맡고 `results/chunk_XXXX.json` 을 개별 저장 → **resume 안전**(있는 청크 skip).
- 영상 경로가 다른 서버면 `build_museg_inputs.py --src data/test/charades_sta_museg.json
  --out_dir <inputs> --old /home/team404/workspace/datasets/charades_sta/
  --new <그 서버 경로>` 로 청크를 다시 생성.
- 결과 스키마는 `{"id", "question", "output"}` 이어야 함(eval_museg.py 가 그걸 기대).

## 3. 채점

```bash
python3 Team4/eval/eval_museg.py \
  --results_dir  <WS>/outputs/base/MUSEG/charades_sta/results \
  --test_json    <WS>/data/test/charades_sta_museg.json \
  --eval_dir     <WS>/outputs/base/MUSEG/charades_sta/eval \
  --label MUSEG-7B --testset charades_sta
```
내부 동작: chunk merge → `output`→`pred`, `id`로 GT embed → `eval/test_results_rank0.json`
→ `eval_miou.py --natural` (X.XX-X.XX 소수초 관대 파싱, `<answer>...</answer>` 자동 추출)
→ `maketable.py` → `eval/table.txt`.

산출물: `pairwise/union/sample_miou_summary.json`, `table.txt`.
**Charades-STA 표준 지표(R@1 IoU=0.3/0.5/0.7, mIoU)는 `sample_*` 행의 `R@0.3/R@0.5/R@0.7`, `sample_mIoU`** 를 그대로 쓰면 됨.

### 검증 완료 (2026-08-30)
sanity30 + 가짜 예측(`<think>..</think><answer>X.XX-X.XX</answer>`)으로 파이프라인 dry-run:
id 매칭 30/30, parse_ok 30/0, 3종 summary 정상 생성.
단 `maketable.py` 는 **n_samples<500 행을 버리므로** sanity30 만으로는 `table.txt` 가 빈 표로 나옴(정상).
전체 3720 런에서는 정상 출력.

## 4. 비교 기준선

같은 Charades-STA 테스트셋으로 돌린 기존 결과:
`outputs/base/ChronusOmni/charades_rlp_noaudio/`,
`outputs/gdpo/sft_7b_*/checkpoint-*/fps5_tti*/charades_rlp_{audio,noaudio}` (titok=ours).
다만 그쪽은 프롬프트가 모델별 format 블록 포함이라 **프롬프트가 동일하지 않음** — mIoU 절대비교 시 감안.
