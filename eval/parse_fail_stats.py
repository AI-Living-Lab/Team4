#!/usr/bin/env python3
"""
parse_fail_stats.py
  - test_results_rank0.json 의 "pred" 값을 읽어
    eval_miou_multiseg.py 의 lenient parser 가 놓치는 비정상 케이스를
    엄격한 정규식 기준으로 분류·카운트해 markdown 표로 출력.
  - 각 실패 카테고리별 pred 예시 샘플도 함께 첨부.

  - GT 형식 가정: "From <tN><tN><tN><tN><tdot><tN> to <tN><tN><tN><tN><tdot><tN>." 의 반복.

Usage
# 디폴트 (alias / table-name 자동 추출)
python3 eval/parse_fail_stats.py \
  --results .../fps5_natural/unav100/test_results_rank0.json \
            .../fps5_off/unav100/test_results_rank0.json

# 옵션 명시 
python3 eval/parse_fail_stats.py \
  --results outputs/sft/salmonn2p_7b_unav_tti_smoke/checkpoint-1500/fps5_off/unav100/test_results_rank0.json \
            outputs/sft/salmonn2p_7b_unav_tti_smoke/checkpoint-1500/fps5_natural/unav100/test_results_rank0.json \
            outputs/sft/salmonn2p_7b_unav_tti_smoke/checkpoint-1500/fps5_tti/unav100/test_results_rank0.json \
  --aliases off natural tti \
  --table-name salmonn2p_7b_unav_tti_smoke_3mode \
  --samples-per-cat 2 
"""
import argparse
import json
import os
import re
import sys


# ---------- lenient (eval_miou_multiseg.py 의 동작) ----------
LENIENT_SEG_RE = re.compile(
    r"[Ff]rom\s+((?:<t\d>)+(?:<tdot>(?:<t\d>)+)?)\s+to\s+((?:<t\d>)+(?:<tdot>(?:<t\d>)+)?)"
)


def decode_vtg_time(token_str, max_time=9999.9):
    has_dot = "<tdot>" in token_str
    if has_dot:
        parts = token_str.split("<tdot>")
        int_part = re.findall(r"<t(\d)>", parts[0])
        dec_part = re.findall(r"<t(\d)>", parts[1]) if len(parts) > 1 else []
    else:
        int_part = re.findall(r"<t(\d)>", token_str)
        dec_part = []
    if not int_part:
        return None
    integer_part = int("".join(int_part))
    decimal_part = int(dec_part[0]) if dec_part else 0
    return min(integer_part + decimal_part / 10.0, max_time)


def parse_lenient(pred, max_time=9999.9):
    segments = []
    for m in LENIENT_SEG_RE.finditer(pred):
        s = decode_vtg_time(m.group(1), max_time)
        e = decode_vtg_time(m.group(2), max_time)
        if s is not None and e is not None:
            segments.append((s, e))
    return segments


# ---------- strict number checker ----------
def split_segments_loose(pred):
    """'From X to Y' 블록을 느슨하게 분리해 (x_str, y_str) 리스트로."""
    pairs = []
    parts = re.split(r"[Ff]rom\s+", pred)
    for chunk in parts[1:]:
        m = re.match(r"(.*?)\s+to\s+(.+?)(?:\.\s*(?=[Ff]rom)|\.\s*$|\s*(?=[Ff]rom)|$)", chunk, re.DOTALL)
        if m:
            pairs.append((m.group(1).strip(), m.group(2).strip()))
    return pairs


def check_number(num_str):
    """단일 숫자 토큰열에서 발견된 이슈 set 반환."""
    issues = set()
    if not re.search(r"<t\d>", num_str):
        issues.add("no_time_tokens")
        return issues

    if re.search(r"<t\d>\s*\.\s*<t\d>", num_str):
        issues.add("literal_dot_separator")

    cleaned = re.sub(r"<t\d>|<tdot>|\s|\.", "", num_str)
    if cleaned:
        issues.add("stray_chars")

    n_tdot = num_str.count("<tdot>")
    if n_tdot == 0:
        issues.add("no_decimal")
    elif n_tdot > 1:
        issues.add("multiple_tdot")
    else:
        left, right = num_str.split("<tdot>", 1)
        int_digits = len(re.findall(r"<t(\d)>", left))
        dec_digits = len(re.findall(r"<t(\d)>", right))
        if int_digits != 4:
            issues.add("bad_int_digit_count")
        if dec_digits != 1:
            issues.add("bad_dec_digit_count")
    return issues


def classify_pred(pred, max_time=9999.9):
    """단일 pred 의 모든 실패 카테고리 set 반환."""
    p = pred.strip()
    fails = set()

    lenient_segs = parse_lenient(p, max_time=max_time)
    if not lenient_segs:
        fails.add("parse_fail")

    if not re.search(r"<t\d>", p):
        fails.add("no_time_tokens")
        return fails

    pairs = split_segments_loose(p)
    if not pairs:
        fails.add("malformed_structure")
    else:
        for x, y in pairs:
            fails |= check_number(x)
            fails |= check_number(y)

    for s, e in lenient_segs:
        if e <= s:
            fails.add("degenerate_interval")
            break

    return fails


# ---------- 카테고리: (group_key, 짧은 영어 라벨, 한국어 설명, [underlying classify_pred 키들]) ----------
# 비슷한 실패는 한 그룹으로 묶어 row 개수를 줄임. 한 샘플이 그룹 내 underlying 키 중
# 하나라도 hit 하면 해당 그룹에 카운트.
CATEGORIES = [
    ("parse_fail",       "parse_fail",       "기존 lenient parser (eval_miou_multiseg.py) 가 단 한 개의 segment 도 추출 못한 샘플.",
        ["parse_fail"]),
    ("no_decimal",       "no <tdot>",        "`<tdot>` 자체가 없는 숫자가 1개 이상 포함 (예: `<t0><t0><t0><t5>` 그대로 끝남).",
        ["no_decimal"]),
    ("literal_dot",      "literal `.` sep",  "`<tdot>` 대신 평문 `.` 를 소수점으로 사용 (예: `<t0><t0><t0><t7>.<t1>`).",
        ["literal_dot_separator"]),
    ("bad_digit_count",  "bad digit count",  "숫자 자릿수 비정상: 정수부 ≠ 4, 소수부 ≠ 1, 또는 한 숫자에 `<tdot>` 가 2회 이상.",
        ["bad_int_digit_count", "bad_dec_digit_count", "multiple_tdot"]),
    ("malformed_or_stray", "malformed / stray", "전체 구조 깨짐 또는 허용 외 문자 (`From..to..` 골격 깨짐, time 토큰 자체 없음, `:` 등 잡문자).",
        ["malformed_structure", "no_time_tokens", "stray_chars"]),
    ("end_le_start",     "end <= start",     "lenient parse 결과 중 end ≤ start 인 퇴화 구간이 1개 이상.",
        ["degenerate_interval"]),
]


def aggregate(results_path, max_time=9999.9, n_samples_per_cat=1):
    with open(results_path, "r") as f:
        data = json.load(f)
    counts = {g: 0 for g, _, _, _ in CATEGORIES}
    samples = {g: [] for g, _, _, _ in CATEGORIES}
    strict_fail = 0
    for d in data:
        pred = d.get("pred", "")
        fails = classify_pred(pred, max_time=max_time)
        for g, _, _, underlying in CATEGORIES:
            if any(k in fails for k in underlying):
                counts[g] += 1
                if len(samples[g]) < n_samples_per_cat:
                    samples[g].append(d)
        if fails:
            strict_fail += 1
    return len(data), counts, strict_fail, samples


def derive_alias(path):
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) >= 3:
        return parts[-3]
    return os.path.splitext(os.path.basename(path))[0]


def derive_table_name(path):
    parts = os.path.normpath(path).split(os.sep)
    for i, p in enumerate(parts):
        if p.startswith("checkpoint-") and i > 0:
            return parts[i - 1]
    return "parse_fail_report"


def fmt_cell(cnt, n):
    return f"{cnt} ({cnt * 100 / n:.2f}%)" if n > 0 else f"{cnt} (-)"


def render_md(table_name, aliases, n_list, counts_list, strict_list, samples_list):
    lines = [
        f"# parse_fail stats — {table_name}",
        "",
        "기존 lenient parser 가 통과시키는 비정상 pred 들을, GT 형식 (`<tN>×4 <tdot> <tN>` 반복) 기준으로 엄격하게 재분류한 카운트.",
        "각 카테고리는 **샘플 단위**로 카운트되며 서로 겹칠 수 있음 (한 샘플이 여러 카테고리에 동시 해당 가능).",
        "마지막 행 `strict_parse_fail (union)` 은 위 카테고리 중 하나라도 해당하는 샘플의 dedup 합집합.",
        "",
        "## Row descriptions",
        "",
    ]
    for _, label, desc, _ in CATEGORIES:
        lines.append(f"- **`{label}`** — {desc}")
    lines.append("- **`strict_parse_fail (union)`** — 위 카테고리 중 하나라도 해당하는 샘플의 dedup 합집합 (= 엄격 기준 총 실패).")
    lines.append("")

    # ----- table -----
    lines.append("## Counts")
    lines.append("")
    header = "| 항목 | " + " | ".join(aliases) + " |"
    sep = "|" + "---|" * (1 + len(aliases))
    lines += [header, sep,
              "| n_samples | " + " | ".join(str(n) for n in n_list) + " |"]

    for key, label, _, _ in CATEGORIES:
        row = "| `" + label + "` | " + " | ".join(
            fmt_cell(c[key], n) for c, n in zip(counts_list, n_list)
        ) + " |"
        lines.append(row)

    lines.append(
        "| **`strict_parse_fail (union)`** | "
        + " | ".join(f"**{fmt_cell(s, n)}**" for s, n in zip(strict_list, n_list))
        + " |"
    )
    lines.append("")

    # ----- per-file samples -----
    for alias, samples in zip(aliases, samples_list):
        lines.append(f"## {alias} weird prediction examples")
        lines.append("")
        for key, label, _, _ in CATEGORIES:
            lines.append(f"### `{label}`")
            lines.append("")
            picks = samples[key]
            if not picks:
                lines.append("_(none)_")
                lines.append("")
                continue
            for s in picks:
                lines.append("```json")
                lines.append(json.dumps(s, indent=2, ensure_ascii=False))
                lines.append("```")
                lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True,
                    help="test_results_rank0.json 파일 경로들 (여러 개 가능)")
    ap.add_argument("--aliases", nargs="+", default=None,
                    help="각 결과 파일의 컬럼명 alias. 미지정 시 경로에서 grandparent 폴더명 사용.")
    ap.add_argument("--table-name", default=None,
                    help="출력 테이블 이름. 미지정 시 첫 결과 파일의 model 폴더명 사용.")
    ap.add_argument("--out-dir",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "summary"),
                    help="출력 .md 가 저장될 폴더.")
    ap.add_argument("--samples-per-cat", type=int, default=1,
                    help="각 실패 카테고리에서 출력할 샘플 개수 (default: 1).")
    ap.add_argument("--max-time", type=float, default=9999.9)
    args = ap.parse_args()

    aliases = args.aliases or [derive_alias(p) for p in args.results]
    if len(aliases) != len(args.results):
        print(f"[ERR] aliases ({len(aliases)}) ≠ results ({len(args.results)})", file=sys.stderr)
        sys.exit(2)

    table_name = args.table_name or derive_table_name(args.results[0])

    n_list, counts_list, strict_list, samples_list = [], [], [], []
    for path in args.results:
        if not os.path.isfile(path):
            print(f"[ERR] not found: {path}", file=sys.stderr)
            sys.exit(2)
        n, counts, strict_fail, samples = aggregate(
            path, max_time=args.max_time, n_samples_per_cat=args.samples_per_cat
        )
        n_list.append(n)
        counts_list.append(counts)
        strict_list.append(strict_fail)
        samples_list.append(samples)

    md = render_md(table_name, aliases, n_list, counts_list, strict_list, samples_list)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"parse_fail_{table_name}.md")
    with open(out_path, "w") as f:
        f.write(md)
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
