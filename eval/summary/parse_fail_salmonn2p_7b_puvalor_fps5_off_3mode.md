# parse_fail stats — salmonn2p_7b_puvalor_fps5_off_3mode

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
| `parse_fail` | 51 (1.48%) | 34 (0.98%) | 5 (0.14%) |
| `no <tdot>` | 2548 (73.75%) | 3055 (88.42%) | 2976 (86.14%) |
| `literal `.` sep` | 69 (2.00%) | 66 (1.91%) | 369 (10.68%) |
| `bad digit count` | 526 (15.22%) | 5 (0.14%) | 5 (0.14%) |
| `malformed / stray` | 5 (0.14%) | 18 (0.52%) | 0 (0.00%) |
| `end <= start` | 38 (1.10%) | 5 (0.14%) | 365 (10.56%) |
| **`strict_parse_fail (union)`** | **3036 (87.87%)** | **3081 (89.18%)** | **3133 (90.68%)** |

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
  "video": "/workspace/datasets/unav_100/videos/2VuKP6zZDDc.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase baby laughter in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t0><tdot><t0> to <t0><t0><t0><t4><tdot><t6>.",
  "audio": "/workspace/datasets/unav_100/audio/2VuKP6zZDDc.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0>.<t0> to <t2><tdot><t9>."
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
  "video": "/workspace/datasets/unav_100/videos/2VuKP6zZDDc.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase baby laughter in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t0><tdot><t0> to <t0><t0><t0><t4><tdot><t6>.",
  "audio": "/workspace/datasets/unav_100/audio/2VuKP6zZDDc.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0>.<t0> to <t2><tdot><t9>."
}
```

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
  "pred": "From <t0><tdot><t0> to <t0>.<t2>."
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
  "pred": "From <t0><tdot><t0> to <t0><tdot><t1>."
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
  "pred": "From <t0><tdot><t0> to <t1><t9><tdot><t2>."
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
  "video": "/workspace/datasets/unav_100/videos/FNYFv72QFpg.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase man speaking in both imagery and sound?"
  },
  "ref": "From <t0><t0><t5><t2><tdot><t2> to <t0><t0><t5><t8><tdot><t2>.",
  "audio": "/workspace/datasets/unav_100/audio/FNYFv72QFpg.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t0><tdot><t0> to <t0><t0><t0><t0><tdot><t0>."
}
```

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

## natural weird prediction examples

### `parse_fail`

```json
{
  "video": "/workspace/datasets/unav_100/videos/60TUirEXrow.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase people shouting in both imagery and sound?"
  },
  "ref": "From <t0><t0><t1><t0><tdot><t5> to <t0><t0><t1><t1><tdot><t7>. From <t0><t0><t3><t3><tdot><t9> to <t0><t0><t3><t4><tdot><t9>.",
  "audio": "/workspace/datasets/unav_100/audio/60TUirEXrow.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t7>.<t1> to <t0><t0><t0><t8><tdot><t9>."
}
```

```json
{
  "video": "/workspace/datasets/unav_100/videos/60TUirEXrow.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase woman speaking in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t6><tdot><t3> to <t0><t0><t0><t8><tdot><t4>. From <t0><t0><t2><t5><tdot><t0> to <t0><t0><t2><t7><tdot><t4>.",
  "audio": "/workspace/datasets/unav_100/audio/60TUirEXrow.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t7>.<t1> to <t0><t0><t0><t8><tdot><t9>."
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
  "pred": "From <t0><t0><t0><t7> to <t0><t0><t1><t9>."
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
  "pred": "From <t0><t0><t0><t0> to <t0><t0><t2><t5>."
}
```

### `literal `.` sep`

```json
{
  "video": "/workspace/datasets/unav_100/videos/2YuwwJbBL3w.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase beat boxing in both imagery and sound?"
  },
  "ref": "From <t0><t0><t1><t2><tdot><t6> to <t0><t0><t4><t2><tdot><t0>.",
  "audio": "/workspace/datasets/unav_100/audio/2YuwwJbBL3w.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t0> to <t0><t0><t1><t2>.<t9>."
}
```

```json
{
  "video": "/workspace/datasets/unav_100/videos/1g-1Gpwx9EE.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase playing cornet in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t0><tdot><t0> to <t0><t0><t5><t4><tdot><t0>.",
  "audio": "/workspace/datasets/unav_100/audio/1g-1Gpwx9EE.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t0> to <t0><t0><t5><t4>.<t9>."
}
```

### `bad digit count`

```json
{
  "video": "/workspace/datasets/unav_100/videos/0U7PeaSWk14.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase people eating in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t2><tdot><t7> to <t0><t0><t0><t8><tdot><t8>. From <t0><t0><t1><t3><tdot><t9> to <t0><t0><t1><t9><tdot><t8>. From <t0><t0><t2><t6><tdot><t6> to <t0><t0><t3><t8><tdot><t3>.",
  "audio": "/workspace/datasets/unav_100/audio/0U7PeaSWk14.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t0><tdot><t0> to <t0><t0><t0><t1><t5><tdot><t9>."
}
```

```json
{
  "video": "/workspace/datasets/unav_100/videos/VUbO1dwu1oc.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase kid speaking in both imagery and sound?"
  },
  "ref": "From <t0><t0><t3><t4><tdot><t6> to <t0><t0><t3><t5><tdot><t5>.",
  "audio": "/workspace/datasets/unav_100/audio/VUbO1dwu1oc.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t1><t2>.<t9> to <t0><t0><t0><t1><t3><tdot><t5>."
}
```

### `malformed / stray`

```json
{
  "video": "/workspace/datasets/unav_100/videos/-cRBIvS8fUU.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase man speaking in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t0><tdot><t0> to <t0><t0><t1><t6><tdot><t0>.",
  "audio": "/workspace/datasets/unav_100/audio/-cRBIvS8fUU.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "<t0><t0><t0><t0><tdot><t0> to <t0><t0><t1><t7><tdot><t4>."
}
```

```json
{
  "video": "/workspace/datasets/unav_100/videos/8Uip3YiRu2g.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase machine gun shooting in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t5><tdot><t9> to <t0><t0><t0><t8><tdot><t0>. From <t0><t0><t0><t9><tdot><t9> to <t0><t0><t1><t1><tdot><t4>. From <t0><t0><t1><t8><tdot><t6> to <t0><t0><t2><t7><tdot><t6>.",
  "audio": "/workspace/datasets/unav_100/audio/8Uip3YiRu2g.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "As the man speaks, gunfire sounds."
}
```

### `end <= start`

```json
{
  "video": "/workspace/datasets/unav_100/videos/Y4zmftgyxkQ.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase people eating in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t4><tdot><t8> to <t0><t0><t0><t6><tdot><t5>.",
  "audio": "/workspace/datasets/unav_100/audio/Y4zmftgyxkQ.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t0><tdot><t0> to <t0><t0><t0><t0><tdot><t0>."
}
```

```json
{
  "video": "/workspace/datasets/unav_100/videos/10RyykoW3Mw.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase people clapping in both imagery and sound?"
  },
  "ref": "From <t0><t0><t2><t0><tdot><t6> to <t0><t0><t2><t4><tdot><t1>.",
  "audio": "/workspace/datasets/unav_100/audio/10RyykoW3Mw.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t0><tdot><t0> to <t0><t0><t0><t0><tdot><t0>."
}
```

## tti weird prediction examples

### `parse_fail`

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

```json
{
  "video": "/workspace/datasets/unav_100/videos/AA_XVhso6IU.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase people clapping in both imagery and sound?"
  },
  "ref": "From <t0><t0><t2><t8><tdot><t8> to <t0><t0><t3><t2><tdot><t2>.",
  "audio": "/workspace/datasets/unav_100/audio/AA_XVhso6IU.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t2><t7>.<t3> to <t0><t0><t2><t7><tdot><t9>."
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
  "pred": "From <t0><t0><t0><t0> to <t0><t0><t4><t0>."
}
```

### `literal `.` sep`

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
  "pred": "From <t0><t0><t0><t0> to <t0><t0><t0><t1>.<t4>."
}
```

```json
{
  "video": "/workspace/datasets/unav_100/videos/63OzQd1VroA.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase people cheering in both imagery and sound?"
  },
  "ref": "From <t0><t0><t2><t2><tdot><t4> to <t0><t0><t2><t5><tdot><t9>. From <t0><t0><t3><t1><tdot><t6> to <t0><t0><t3><t5><tdot><t5>.",
  "audio": "/workspace/datasets/unav_100/audio/63OzQd1VroA.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t0> to <t0><t0><t0><t1>.<t5>."
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
  "video": "/workspace/datasets/unav_100/videos/vyYpxGJr7zI.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase dog barking in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t1><tdot><t7> to <t0><t0><t0><t5><tdot><t2>. From <t0><t0><t2><t2><tdot><t8> to <t0><t0><t4><t8><tdot><t9>.",
  "audio": "/workspace/datasets/unav_100/audio/vyYpxGJr7zI.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t0><tdot><t0> to <t0><t0><t0><t0><tdot><t0>."
}
```

```json
{
  "video": "/workspace/datasets/unav_100/videos/0JZetKN2fMw.mp4",
  "image": null,
  "prompt": {
    "from": "human",
    "value": "<video>\nWhich parts of the video showcase thunder in both imagery and sound?"
  },
  "ref": "From <t0><t0><t0><t9><tdot><t2> to <t0><t0><t1><t1><tdot><t9>. From <t0><t0><t2><t0><tdot><t2> to <t0><t0><t2><t5><tdot><t3>. From <t0><t0><t4><t1><tdot><t5> to <t0><t0><t4><t3><tdot><t9>.",
  "audio": "/workspace/datasets/unav_100/audio/0JZetKN2fMw.wav",
  "use_audio": true,
  "should_use": true,
  "pred": "From <t0><t0><t0><t0> to <t0><t0><t0><t0>."
}
```
