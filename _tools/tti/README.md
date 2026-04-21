# TTI (Time-Token Interleaving) 검증 모음

Step 1~4 로 구현한 TTI 경로가 실제로 설계대로 작동하는지 단계별로 확인하는
스크립트 모음입니다. 체크포인트 로딩 없이 (또는 토크나이저만 로드해서)
빠르게 돌도록 구성되어 있어, 코드 변경 후 regression 확인용으로 적합합니다.

## 실행 순서

`./run_all.sh <model_path>` 로 한 번에 돌리거나, 개별 파일을 순서대로 실행합니다.
어느 하나가 FAIL 하면 그 단계의 설계/구현을 먼저 고친 뒤 다음으로 진행하세요.

| # | 스크립트 | 무엇을 검증하나 | 필요 인자 |
|---|---|---|---|
| 01 | `01_check_tokenizer_ids.py` | Pre-baked 체크포인트의 타임토큰 11개가 단일 ID로 등록되어 있고 ID 구간이 연속이며 config.time_token_id_range 값과 일치하는지 | `--model_path` |
| 02 | `02_check_dataset_interleave.py` | `sec_to_time_token_str` 변환과 dataset.py 인터리빙 삽입 문자열이 `<t*>×6 → video_pad → audio_pad` 순서로 나오는지 | (없음) |
| 03 | `03_check_rope_with_time_tokens.py` | 타임토큰 활성화 시 `get_rope_index_25` Case 1이 3D position을 설계대로 채우는지 | (없음) |
| 04 | `04_check_rope_regression.py` | 타임토큰 미전달 시 기존 비디오+오디오 인터리빙 position 결과가 바뀌지 않았는지 | (없음) |
| 05 | `05_check_modeling_delegate.py` | modeling_qwen2_5_vl.py 의 `get_rope_index` shim 이 rope2d 로 올바르게 위임하고 `config.time_token_id_range` 를 자동으로 꺼내 쓰는지 | (없음) |

## 사용 예시

```bash
# conda env 활성화 (salmonn2plus)
source /workspace/setup.sh && conda activate salmonn2plus

cd /workspace/tti/Team4

# 전체 실행
bash _tools/tti/run_all.sh /workspace/checkpoints/base/video_salmonn2_plus_7B_time_tokens

# 개별 실행 (체크포인트 검증만)
python _tools/tti/01_check_tokenizer_ids.py \
  --model_path /workspace/checkpoints/base/video_salmonn2_plus_7B_time_tokens
```

## 참고

- `_tools/sft/verify_time_tokens.py` 는 **체크포인트 굽기 직후** 돌리는
  원본 검증입니다. 본 폴더의 `01` 은 동일 검증을 반복하면서 추가로
  config 저장 필드(`time_token_id_range`)까지 확인합니다.
- 03, 04, 05 는 체크포인트 없이도 돌아갑니다 (rope 로직만 검증).
- 각 스크립트는 종료 코드 0=PASS, 1=FAIL 이라 CI 파이프라인에 그대로 연결
  가능합니다.
