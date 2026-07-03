# 비교모델 추론·채점 핸드오프 (museg / avicuna 작업용)

멀티세그 힌트 프롬프트로 비교모델(avicuna·museg·chronus·arc-hunyuan·titok)을 UnAV-100에서
추론→채점하는 작업. titok=ours. chronus·arc는 진행/완료, **museg·avicuna 남음**.

## 1. 테스트셋 & 프롬프트 (이미 생성됨)

빌드 스크립트: `data/test/build_multiseg_prompts.py` (소스 5개 모델 json → 프롬프트만 멀티세그 힌트로 교체).
공통 stem: `At what point in the video does {label} occur in terms of both video and audio?` + 모델별 format 블록.

| 모델 | flat json | format | 세그내 | 멀티세그 구분 | 프롬프트 필드 |
|---|---|---|---|---|---|
| avicuna | `data/test/unav100_avicuna.json` | `from XX to YY` | to | `. ` | `question` (no `<video>`) |
| museg | `data/test/unav100_museg.json` | `X.XX-X.XX` | `-` | **공백** | `question` (no `<video>`) |
| chronus | unav100_chronus.json | `second{start}-second{end}` | `-` | `. ` | question |
| arc | unav100_arc.json | `00:00:00 - 00:00:00` | ` - ` | `. ` | conversations(`<video>\n`) |
| titok | unav100_titok.json | `From <tX>..<tdot><tX>` 토큰 | to | `. ` | conversations |

- 각 3455 샘플. **avicuna는 `audio_clap`(npy) 필드 보존**, **museg는 `groundtruth` 필드 보존**.
- 청크 폴더도 생성됨: `data/test/unav100_{model}/chunk_0000..0006.json` (500씩, 7청크).
  재생성: `python3 master/Team4/eval/_chunk_helpers.py split --test_json <flat.json> --chunks_dir <dir>/ --chunk_size 500`

## 2. ★경로 rewrite 필수 (json 경로가 이 서버엔 없음)

실제 데이터 루트: **`/home/aix23102/audiolm/workspace/datasets/unav_100/`** (`videos/*.mp4`, `audio/*.wav`, 10790개)

json 안의 경로는 옛 서버 것이라 반드시 치환:
- `/workspace/datasets/unav_100/` → `/home/aix23102/audiolm/workspace/datasets/unav_100/` (avicuna·museg·chronus)
- `/data0/aix23102/unav_100/` → `/home/aix23102/audiolm/workspace/datasets/unav_100/` (arc·titok)

## 3. 추론 패턴 (chronus 사례 = 참고 템플릿)

`outputs/base/<Model>/unav100_multiseg/` 아래에:
- `inputs/chunk_XXXX.json` (경로 rewrite된 입력)
- `results/chunk_XXXX.json` (모델 출력)
- `run_all.log`

nohup 으로 **청크별 순차** 실행(resume 안전 — 이미 있는 청크 skip). GPU는 `CUDA_VISIBLE_DEVICES=N`.

### avicuna/museg 추론 코드 (다른 챗에서 확보 필요)
- 자산: env `avicuna` 존재 / 체크포인트 `checkpoints/base/AVicuna`, `checkpoints/base/MUSEG-7B` 존재.
- github repo: 현재 `github/` 엔 ARC-Hunyuan-Video-7B, Chronus, **MMN** 만 있음 (museg=MMN일 가능성 확인 요, avicuna repo 미클론).
- museg 전용 conda env는 목록에 안 보임(있는지 확인 필요; `avf` 는 별개).
- **가속**: chronus는 eval.py의 `use_flash_attn=False`를 True로 켜서 가속함. arc는 vLLM(배치) 사용. museg/avicuna도 repo readme에서 flash-attn/배치/vLLM 지원 여부 먼저 확인할 것.
- **오디오 주의**(arc 교훈): 일부 추론코드는 비디오에서 오디오를 추출해 `audio_path`에 덮어씀 → 데이터셋 .wav 덮어쓰지 말고 temp 경로 쓸 것.

## 4. 채점 → table.txt (핵심)

평가기 `master/Team4/eval/eval_miou.py` 는 예측을 **`pred`** 필드에서 읽고, GT는
**embedded `gt_segments` 우선**(그다음 ref, 그다음 test_json 매칭). 3종 summary(pairwise/union/sample) 생성.
집계표는 `maketable.py <dir>` → `<dir>/table.txt` (**n_samples<500 행 제외**, sample_mIoU 내림차순).

### chronus용 래퍼(참고): `master/Team4/eval/eval_chronus.py`
결과가 `{id,question,output}` 뿐이라 → chunk merge + `output→pred` + `id`로 gt embed →
`test_results_rank0.json` → `eval_miou.py --natural` → `maketable.py`. **avicuna/museg도 이 패턴 복제** 권장
(`eval_avicuna.py` / `eval_museg.py`), 각 결과 스키마에 맞춰 pred 추출만 바꾸면 됨.

### 모델별 파싱 플래그 (중요)
- **avicuna**: pred가 영상길이 대비 **%(0~100)** → `eval_miou.py --pred_percent --duration_key <필드>`
  (결과 항목에 duration 필드 필요). 그냥 초로 파싱하면 안 됨.
- **museg**: `X.XX-X.XX` 소수초 → `--natural` 로 파싱.
- **chronus/arc**: `--natural` (second{}/HH:MM:SS 관대 파싱). arc는 `<answer>..</answer>` 자동 추출됨.
- 공통: `--natural` 은 HH:MM:SS·M:SS·소수초·second{}·from X to Y 등 다 회수. finditer라 멀티세그 구분자 무관.

### GT 매칭 주의
결과의 `id`(예 `--Bu2xe4OSo_0000`)로 test_json과 매칭. test_json은 video basename(`--Bu2xe4OSo`)+gt_label로
키를 잡는데 한 영상에 여러 이벤트라 video 단독매칭이 ambiguous가 됨 → **id로 gt를 직접 embed하는 방식(eval_chronus.py)이 안전**.

## 5. 현재까지 결과 (동일 unav100 멀티세그 프롬프트, sample_mIoU)

| 모델 | sample_mIoU | n | 상태 |
|---|---|---|---|
| titok(ours) ckpt-1400 | 61.22 | 3455 | 완료 (`outputs/gdpo/.../unav100_titok/table.txt`) |
| ChronusOmni | 63.25 | 2500 | 진행중 (`outputs/base/ChronusOmni/unav100_multiseg/eval/table.txt`) |
| arc-hunyuan | — | — | vLLM 세팅 완료, 스모크 중 |
| museg / avicuna | — | — | **미시작 (이 작업)** |

집계표: `outputs/gdpo/table.txt` (titok/charades), 각 모델 eval 폴더의 table.txt.

## 6. 통합 eval 문서
`master/Team4/eval/통합평가방법총정리(26-06-08).md` — eval.sh/eval_miou.py/maketable.py 전체 사용법.
(단, eval.sh 런처는 salmonn2plus 전용. 비교모델은 각 repo로 추론 후 eval_miou.py로 채점.)
