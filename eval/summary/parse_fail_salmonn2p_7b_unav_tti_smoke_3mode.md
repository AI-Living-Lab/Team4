# parse_fail stats — salmonn2p_7b_unav_tti_smoke_3mode

기존 lenient parser 가 통과시키는 비정상 pred 들을, GT 형식 (`<tN>×4 <tdot> <tN>` 반복) 기준으로 엄격하게 재분류한 카운트.
각 카테고리는 **샘플 단위**로 카운트되며 서로 겹칠 수 있음 (한 샘플이 여러 카테고리에 동시 해당 가능).
마지막 행 `strict_parse_fail (union)` 은 위 카테고리 중 하나라도 해당하는 샘플의 dedup 합집합.

## Row descriptions

- **`parse_fail`** — 기존 lenient parser (eval_miou_multiseg.py) 가 단 한 개의 segment 도 추출 못한 샘플.
- **`no <tdot>`** — `<tdot>` 자체가 없는 숫자가 1개 이상 포함 (예: `<t0><t0><t0><t5>` 그대로 끝남).
- **`literal `.` sep`** — `<tdot>` 대신 평문 `.` 를 소수점으로 사용 (예: `<t0><t0><t0><t7>.<t1>`).
- **`bad digit count`** — 숫자 자릿수 비정상: 정수부 ≠ 4, 소수부 ≠ 1, 또는 한 숫자에 `<tdot>` 가 2회 이상.
- **`malformed / stray`** — 전체 구조 깨짐 또는 허용 외 문자 (`From..to..` 골격 깨짐, time 토큰 자체 없음, `:` 등 잡문자).
- **`end <= start`** — lenient parse 결과 중 end ≤ start 인 퇴화 구간이 1개 이상.
- **`strict_parse_fail (union)`** — 위 카테고리 중 하나라도 해당하는 샘플의 dedup 합집합 (= 엄격 기준 총 실패).

## Counts

| 항목 | off | natural | tti |
|---|---|---|---|
| n_samples | 3455 | 3455 | 3455 |
| `parse_fail` | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| `no <tdot>` | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| `literal `.` sep` | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| `bad digit count` | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| `malformed / stray` | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| `end <= start` | 0 (0.00%) | 2 (0.06%) | 27 (0.78%) |
| **`strict_parse_fail (union)`** | **0 (0.00%)** | **2 (0.06%)** | **27 (0.78%)** |

## off weird prediction examples

### `parse_fail`

_(none)_

### `no <tdot>`

_(none)_

### `literal `.` sep`

_(none)_

### `bad digit count`

_(none)_

### `malformed / stray`

_(none)_

### `end <= start`

_(none)_

## natural weird prediction examples

### `parse_fail`

_(none)_

### `no <tdot>`

_(none)_

### `literal `.` sep`

_(none)_

### `bad digit count`

_(none)_

### `malformed / stray`

_(none)_

### `end <= start`

```json
{
  "video": "/workspace/datasets/unav_100/videos/2mfuHazBzsw.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase people sneezing in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t0><tdot><t2> to <t0><t0><t1><t1><tdot><t8>. From <t0><t0><t1><t5><tdot><t5> to <t0><t0><t5><t3><tdot><t4>.",
  "audio": "/workspace/datasets/unav_100/audio/2mfuHazBzsw.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t3><tdot><t1> to <t0><t0><t0><t4><tdot><t1>. From <t0><t0><t0><t6><tdot><t7> to <t0><t0><t0><t7><tdot><t7>. From <t0><t0><t1><t0><tdot><t0> to <t0><t0><t1><t0><tdot><t9>. From <t0><t0><t1><t2><tdot><t8> to <t0><t0><t1><t3><tdot><t8>. From <t0><t0><t1><t5><tdot><t9> to <t0><t0><t1><t6><tdot><t9>. From <t0><t0><t1><t9><tdot><t0> to <t0><t0><t1><t9><tdot><t0>. From <t0><t0><t2><t0><tdot><t0> to <t0><t0><t2><t0><tdot><t0>. From <t0><t0><t2><t1><tdot><t0> to <t0><t0><t2><t1><tdot><t0>. From <t0><t0><t2><t3><tdot><t0> to <t0><t0><t2><t3><tdot><t0>. From <t0><t0><t2><t5><tdot><t0> to <t0><t0><t2><t5><tdot><t0>. From <t0><t0><t2><t7><tdot><t0> to <t0><t0><t2><t7><tdot><t0>. From <t0><t0><t2><t8><tdot><t0> to <t0><t0><t2><t8><tdot><t0>. From <t0><t0><t3><t0><tdot><t0> to <t0><t0><t3><t0><tdot><t0>. From <t0><t0><t3><t2><tdot><t0> to <t0><t0><t3><t2><tdot><t0>. From <t0><t0><t3><t3><tdot><t0> to <t0><t0><t3><t3><tdot><t0>. From <t0><t0><t3><t5><tdot><t0> to <t0><t0><t3><t5><tdot><t0>. From <t0><t0><t3><t7><tdot><t0> to <t0><t0><t3><t7><tdot><t0>. From <t0><t0><t3><t9><tdot><t0> to <t0><t0><t3><t9><tdot><t0>. From <t0><t0><t4><t1><tdot><t0> to <t0><t0><t4><t1><tdot><t0>. From <t0><t0><t4><t3><tdot><t0> to <t0><t0><t4><t3><tdot><t0>. From <t0><t0><t4><t5><tdot><t0> to <t0><t0><t4><t5><tdot><t0>. From <t0><t0><t4><t7><tdot><t0> to <t0><t0><t4><t7><tdot><t0>. From <t0><t0><t4><t9><tdot><t0> to <t0><t0><t4><t9><tdot><t0>."
}
```

```json
{
  "video": "/workspace/datasets/unav_100/videos/CIxMFQX2LgU.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase people coughing in both imagery and sound?"
  },
  "ref": "From <t0><t0><t1><t8><tdot><t2> to <t0><t0><t1><t9><tdot><t2>.",
  "audio": "/workspace/datasets/unav_100/audio/CIxMFQX2LgU.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t4><tdot><t6> to <t0><t0><t0><t5><tdot><t6>. From <t0><t0><t1><t0><tdot><t0> to <t0><t0><t1><t0><tdot><t9>. From <t0><t0><t1><t7><tdot><t2> to <t0><t0><t1><t8><tdot><t2>. From <t0><t0><t2><t1><tdot><t3> to <t0><t0><t2><t2><tdot><t3>. From <t0><t0><t2><t6><tdot><t0> to <t0><t0><t2><t6><tdot><t0>."
}
```

## tti weird prediction examples

### `parse_fail`

_(none)_

### `no <tdot>`

_(none)_

### `literal `.` sep`

_(none)_

### `bad digit count`

_(none)_

### `malformed / stray`

_(none)_

### `end <= start`

```json
{
  "video": "/workspace/datasets/unav_100/videos/-QYcCxdc5gM.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase woman speaking in both imagery and sound?"
  },
  "ref": "From <t0><t0><t3><t6><tdot><t2> to <t0><t0><t3><t8><tdot><t4>.",
  "audio": "/workspace/datasets/unav_100/audio/-QYcCxdc5gM.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t0><tdot><t0> to <t0><t0><t0><t0><tdot><t0>. From <t0><t0><t0><t6><tdot><t9> to <t0><t0><t0><t7><tdot><t9>. From <t0><t0><t1><t0><tdot><t3> to <t0><t0><t1><t1><tdot><t3>. From <t0><t0><t1><t6><tdot><t0> to <t0><t0><t1><t7><tdot><t0>. From <t0><t0><t2><t0><tdot><t0> to <t0><t0><t2><t1><tdot><t0>. From <t0><t0><t2><t6><tdot><t0> to <t0><t0><t2><t7><tdot><t0>. From <t0><t0><t3><t4><tdot><t4> to <t0><t0><t3><t5><tdot><t4>."
}
```

```json
{
  "video": "/workspace/datasets/unav_100/videos/gBTVT4OzDWA.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase people coughing in both imagery and sound?"
  },
  "ref": "From <t0><t0><t3><t1><tdot><t3> to <t0><t0><t3><t2><tdot><t3>. From <t0><t0><t3><t5><tdot><t6> to <t0><t0><t3><t6><tdot><t6>.",
  "audio": "/workspace/datasets/unav_100/audio/gBTVT4OzDWA.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t0><tdot><t0> to <t0><t0><t0><t0><tdot><t0>."
}
```
