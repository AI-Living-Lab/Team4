[SAVED] /workspace/rl_tti/Team4/eval/summary/parse_fail_gdpo_run3_temp1.5_3mode.md
키는 비정상 pred 들을, GT 형식 (`<tN>×4 <tdot> <tN>` 반복) 기준으로 엄격하게 재분류한 카운트.
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

| 항목 | off | tti | fromto |
|---|---|---|---|
| n_samples | 3455 | 3455 | 1000 |
| `parse_fail` | 77 (2.23%) | 5 (0.14%) | 246 (24.60%) |
| `no <tdot>` | 2458 (71.14%) | 3028 (87.64%) | 599 (59.90%) |
| `literal `.` sep` | 95 (2.75%) | 371 (10.74%) | 299 (29.90%) |
| `bad digit count` | 676 (19.57%) | 4 (0.12%) | 0 (0.00%) |
| `malformed / stray` | 7 (0.20%) | 0 (0.00%) | 0 (0.00%) |
| `end <= start` | 33 (0.96%) | 409 (11.84%) | 107 (10.70%) |
| **`strict_parse_fail (union)`** | **3062 (88.63%)** | **3184 (92.16%)** | **602 (60.20%)** |

## off weird prediction examples

### `parse_fail`

```json
{
  "video": "/workspace/datasets/unav_100/videos/etcezCPWyqw.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase wind noise in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t2><tdot><t2> to <t0><t0><t3><t2><tdot><t2>.",
  "audio": "/workspace/datasets/unav_100/audio/etcezCPWyqw.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0>:<t0><t0> to <t0><t0><t1><t2>."
}
```

```json
{
  "video": "/workspace/datasets/unav_100/videos/2opu0xJeVYo.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase sea waves in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t0><tdot><t0> to <t0><t0><t3><t8><tdot><t0>.",
  "audio": "/workspace/datasets/unav_100/audio/2opu0xJeVYo.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0>.<t0> to <t0><t1><tdot><t9>."
}
```

### `no <tdot>`

```json
{
  "video": "/workspace/datasets/unav_100/videos/-UBVqmhbT50.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase dog howling in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t7><tdot><t8> to <t0><t0><t2><t0><tdot><t1>. From <t0><t0><t2><t7><tdot><t6> to <t0><t0><t3><t8><tdot><t7>.",
  "audio": "/workspace/datasets/unav_100/audio/-UBVqmhbT50.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t0> to <t0><t0><t2><t9>"
}
```

```json
{
  "video": "/workspace/datasets/unav_100/videos/ekCzMX9vEyk.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase woman speaking in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t0><tdot><t0> to <t0><t0><t1><t0><tdot><t9>. From <t0><t0><t2><t0><tdot><t5> to <t0><t0><t2><t5><tdot><t2>. From <t0><t0><t3><t0><tdot><t4> to <t0><t0><t3><t4><tdot><t5>.",
  "audio": "/workspace/datasets/unav_100/audio/ekCzMX9vEyk.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t0> to <t0><t0><t1><t5>."
}
```

### `literal `.` sep`

```json
{
  "video": "/workspace/datasets/unav_100/videos/-VvCOGy5oTc.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase man speaking in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t1><tdot><t5> to <t0><t0><t0><t3><tdot><t6>.",
  "audio": "/workspace/datasets/unav_100/audio/-VvCOGy5oTc.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><tdot><t0> to <t1><t2><t5>.<t9>."
}
```

```json
{
  "video": "/workspace/datasets/unav_100/videos/2opu0xJeVYo.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase sea waves in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t0><tdot><t0> to <t0><t0><t3><t8><tdot><t0>.",
  "audio": "/workspace/datasets/unav_100/audio/2opu0xJeVYo.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0>.<t0> to <t0><t1><tdot><t9>."
}
```

### `bad digit count`

```json
{
  "video": "/workspace/datasets/unav_100/videos/6RBc4E7Fgf0.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase male singing in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t4><tdot><t5> to <t0><t0><t0><t6><tdot><t8>.",
  "audio": "/workspace/datasets/unav_100/audio/6RBc4E7Fgf0.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><tdot><t0> to <t1><t5><tdot><t2>."
}
```

```json
{
  "video": "/workspace/datasets/unav_100/videos/9ezCDjjADL8.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase man speaking in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t5><tdot><t1> to <t0><t0><t0><t7><tdot><t9>.",
  "audio": "/workspace/datasets/unav_100/audio/9ezCDjjADL8.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><tdot><t0> to <t2><t1><tdot><t9>."
}
```

### `malformed / stray`

```json
{
  "video": "/workspace/datasets/unav_100/videos/etcezCPWyqw.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase wind noise in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t2><tdot><t2> to <t0><t0><t3><t2><tdot><t2>.",
  "audio": "/workspace/datasets/unav_100/audio/etcezCPWyqw.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0>:<t0><t0> to <t0><t0><t1><t2>."
}
```

```json
{
  "video": "/workspace/datasets/unav_100/videos/7zPl1hU4BZ0.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase people eating in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t8><tdot><t3> to <t0><t0><t1><t1><tdot><t3>. From <t0><t0><t3><t0><tdot><t2> to <t0><t0><t3><t5><tdot><t0>.",
  "audio": "/workspace/datasets/unav_100/audio/7zPl1hU4BZ0.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0>:<t0><t0><tdot><t0> to <t0><t0><t0><t0>.<t0>."
}
```

### `end <= start`

```json
{
  "video": "/workspace/datasets/unav_100/videos/WGn2wGR20BQ.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase people sobbing in both imagery and sound?"
  },
  "ref": "From <t0><t0><t1><t9><tdot><t2> to <t0><t0><t2><t4><tdot><t9>.",
  "audio": "/workspace/datasets/unav_100/audio/WGn2wGR20BQ.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t0><tdot><t0> to <t0><t0><t0><t0><tdot><t0>."
}
```

```json
{
  "video": "/workspace/datasets/unav_100/videos/-otKTQ80mMw.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase telephone bell ringing in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t7><tdot><t3> to <t0><t0><t0><t9><tdot><t6>. From <t0><t0><t1><t3><tdot><t7> to <t0><t0><t1><t5><tdot><t7>. From <t0><t0><t1><t9><tdot><t6> to <t0><t0><t2><t1><tdot><t7>. From <t0><t0><t2><t5><tdot><t8> to <t0><t0><t2><t7><tdot><t6>. From <t0><t0><t3><t1><tdot><t6> to <t0><t0><t3><t3><tdot><t8>.",
  "audio": "/workspace/datasets/unav_100/audio/-otKTQ80mMw.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><tdot><t0> to <t0><t0><tdot><t0>."
}
```

## tti weird prediction examples

### `parse_fail`

```json
{
  "video": "/workspace/datasets/unav_100/videos/8jZwSrV2LFs.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase bull bellowing in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t6><tdot><t7> to <t0><t0><t0><t9><tdot><t7>.",
  "audio": "/workspace/datasets/unav_100/audio/8jZwSrV2LFs.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t7>.<t2> to <t0><t0><t1><t0><tdot><t8>."
}
```

```json
{
  "video": "/workspace/datasets/unav_100/videos/A6_-Ji7bq5E.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase people eating in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t0><tdot><t0> to <t0><t0><t4><t7><tdot><t0>.",
  "audio": "/workspace/datasets/unav_100/audio/A6_-Ji7bq5E.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0>.<t0> to <t0><t0><t0><t0>.<t7>."
}
```

### `no <tdot>`

```json
{
  "video": "/workspace/datasets/unav_100/videos/-UBVqmhbT50.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase dog howling in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t7><tdot><t8> to <t0><t0><t2><t0><tdot><t1>. From <t0><t0><t2><t7><tdot><t6> to <t0><t0><t3><t8><tdot><t7>.",
  "audio": "/workspace/datasets/unav_100/audio/-UBVqmhbT50.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t0> to <t0><t0><t4><t4>."
}
```

```json
{
  "video": "/workspace/datasets/unav_100/videos/ekCzMX9vEyk.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase woman speaking in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t0><tdot><t0> to <t0><t0><t1><t0><tdot><t9>. From <t0><t0><t2><t0><tdot><t5> to <t0><t0><t2><t5><tdot><t2>. From <t0><t0><t3><t0><tdot><t4> to <t0><t0><t3><t4><tdot><t5>.",
  "audio": "/workspace/datasets/unav_100/audio/ekCzMX9vEyk.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t0> to <t0><t0><t0><t1>.<t3>."
}
```

### `literal `.` sep`

```json
{
  "video": "/workspace/datasets/unav_100/videos/ekCzMX9vEyk.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase woman speaking in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t0><tdot><t0> to <t0><t0><t1><t0><tdot><t9>. From <t0><t0><t2><t0><tdot><t5> to <t0><t0><t2><t5><tdot><t2>. From <t0><t0><t3><t0><tdot><t4> to <t0><t0><t3><t4><tdot><t5>.",
  "audio": "/workspace/datasets/unav_100/audio/ekCzMX9vEyk.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t0> to <t0><t0><t0><t1>.<t3>."
}
```

```json
{
  "video": "/workspace/datasets/unav_100/videos/0DLQxoJg0UM.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase man speaking in both imagery and sound?"
  },
  "ref": "From <t0><t0><t3><t8><tdot><t7> to <t0><t0><t4><t0><tdot><t8>.",
  "audio": "/workspace/datasets/unav_100/audio/0DLQxoJg0UM.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t0> to <t0><t0><t0><t1>.<t9>."
}
```

### `bad digit count`

```json
{
  "video": "/workspace/datasets/unav_100/videos/720TodNLrTg.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase playing volleyball in both imagery and sound?"
  },
  "ref": "From <t0><t0><t2><t7><tdot><t9> to <t0><t0><t4><t1><tdot><t2>.",
  "audio": "/workspace/datasets/unav_100/audio/720TodNLrTg.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t2><tdot><t6> to <t0><t0><t0><t2><t9><tdot><t1>."
}
```

```json
{
  "video": "/workspace/datasets/unav_100/videos/4go6CQ_RhYQ.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase people shouting in both imagery and sound?"
  },
  "ref": "From <t0><t0><t1><t1><tdot><t2> to <t0><t0><t1><t3><tdot><t9>.",
  "audio": "/workspace/datasets/unav_100/audio/4go6CQ_RhYQ.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t2><t1><tdot><t0> to <t0><t0><t0><t2><t1><tdot><t7>."
}
```

### `malformed / stray`

_(none)_

### `end <= start`

```json
{
  "video": "/workspace/datasets/unav_100/videos/9ezCDjjADL8.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase man speaking in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t5><tdot><t1> to <t0><t0><t0><t7><tdot><t9>.",
  "audio": "/workspace/datasets/unav_100/audio/9ezCDjjADL8.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t0> to <t0><t0><t0><t0>."
}
```

```json
{
  "video": "/workspace/datasets/unav_100/videos/6kXUG1Zo6VA.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase people shouting in both imagery and sound?"
  },
  "ref": "From <t0><t0><t1><t0><tdot><t6> to <t0><t0><t1><t1><tdot><t3>.",
  "audio": "/workspace/datasets/unav_100/audio/6kXUG1Zo6VA.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t0> to <t0><t0><t0><t0>."
}
```

## fromto weird prediction examples

### `parse_fail`

```json
{
  "video": "/workspace/datasets/unav_100/videos/-UBVqmhbT50.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase dog howling in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t7><tdot><t8> to <t0><t0><t2><t0><tdot><t1>. From <t0><t0><t2><t7><tdot><t6> to <t0><t0><t3><t8><tdot><t7>.",
  "audio": "/workspace/datasets/unav_100/audio/-UBVqmhbT50.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t0>.<t0> to <t0><t0><t0><t0><tdot><t2>."
}
```

```json
{
  "video": "/workspace/datasets/unav_100/videos/ekCzMX9vEyk.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase woman speaking in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t0><tdot><t0> to <t0><t0><t1><t0><tdot><t9>. From <t0><t0><t2><t0><tdot><t5> to <t0><t0><t2><t5><tdot><t2>. From <t0><t0><t3><t0><tdot><t4> to <t0><t0><t3><t4><tdot><t5>.",
  "audio": "/workspace/datasets/unav_100/audio/ekCzMX9vEyk.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t0>.<t0> to <t0><t0><t0><t0><tdot><t5>."
}
```

### `no <tdot>`

```json
{
  "video": "/workspace/datasets/unav_100/videos/-UBVqmhbT50.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase dog howling in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t7><tdot><t8> to <t0><t0><t2><t0><tdot><t1>. From <t0><t0><t2><t7><tdot><t6> to <t0><t0><t3><t8><tdot><t7>.",
  "audio": "/workspace/datasets/unav_100/audio/-UBVqmhbT50.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t0>.<t0> to <t0><t0><t0><t0><tdot><t2>."
}
```

```json
{
  "video": "/workspace/datasets/unav_100/videos/ekCzMX9vEyk.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase woman speaking in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t0><tdot><t0> to <t0><t0><t1><t0><tdot><t9>. From <t0><t0><t2><t0><tdot><t5> to <t0><t0><t2><t5><tdot><t2>. From <t0><t0><t3><t0><tdot><t4> to <t0><t0><t3><t4><tdot><t5>.",
  "audio": "/workspace/datasets/unav_100/audio/ekCzMX9vEyk.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t0>.<t0> to <t0><t0><t0><t0><tdot><t5>."
}
```

### `literal `.` sep`

```json
{
  "video": "/workspace/datasets/unav_100/videos/-UBVqmhbT50.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase dog howling in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t7><tdot><t8> to <t0><t0><t2><t0><tdot><t1>. From <t0><t0><t2><t7><tdot><t6> to <t0><t0><t3><t8><tdot><t7>.",
  "audio": "/workspace/datasets/unav_100/audio/-UBVqmhbT50.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t0>.<t0> to <t0><t0><t0><t0><tdot><t2>."
}
```

```json
{
  "video": "/workspace/datasets/unav_100/videos/ekCzMX9vEyk.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase woman speaking in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t0><tdot><t0> to <t0><t0><t1><t0><tdot><t9>. From <t0><t0><t2><t0><tdot><t5> to <t0><t0><t2><t5><tdot><t2>. From <t0><t0><t3><t0><tdot><t4> to <t0><t0><t3><t4><tdot><t5>.",
  "audio": "/workspace/datasets/unav_100/audio/ekCzMX9vEyk.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t0>.<t0> to <t0><t0><t0><t0><tdot><t5>."
}
```

### `bad digit count`

_(none)_

### `malformed / stray`

_(none)_

### `end <= start`

```json
{
  "video": "/workspace/datasets/unav_100/videos/etcezCPWyqw.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase wind noise in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t2><tdot><t2> to <t0><t0><t3><t2><tdot><t2>.",
  "audio": "/workspace/datasets/unav_100/audio/etcezCPWyqw.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t0> to <t0><t0><t0><t0>.<t9>."
}
```

```json
{
  "video": "/workspace/datasets/unav_100/videos/r-5NCjm4GlQ.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase tap dancing in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t0><tdot><t0> to <t0><t0><t1><t7><tdot><t3>. From <t0><t0><t2><t6><tdot><t2> to <t0><t0><t2><t9><tdot><t5>. From <t0><t0><t3><t4><tdot><t7> to <t0><t0><t3><t8><tdot><t0>.",
  "audio": "/workspace/datasets/unav_100/audio/r-5NCjm4GlQ.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t0> to <t0><t0><t0><t0>.<t9>."
}
```
