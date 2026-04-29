#!/usr/bin/env python3
"""
TTI debug_interleave — cross-config 비교 요약.

sweep_dump.sh 결과 디렉토리(<base>/<config_tag>/<NNN>_<sample>.json)를 스캔해서
샘플별 또는 설정별로 핵심 메트릭을 한 테이블로 출력.

사용법:
  python _tools/debug/compare.py --in_dir _debug_out/sweep
  python _tools/debug/compare.py --in_dir _debug_out/sweep --by config
  python _tools/debug/compare.py --in_dir _debug_out/sweep --format md --out compare.md
  python _tools/debug/compare.py --in_dir _debug_out/sweep --format csv --out compare.csv
  python _tools/debug/compare.py --in_dir _debug_out/sweep --samples short,longest

그룹핑:
  --by sample  : 샘플 하나당 한 테이블, 각 행이 설정
  --by config  : 설정 하나당 한 테이블, 각 행이 샘플
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# 출력 컬럼 (행 공통)
CONFIG_COL = "config"
SAMPLE_COL = "sample"
METRIC_COLS = [
    "mode", "seq_len",
    "time", "video", "audio",
    "chunks", "vid/ch", "aud/ch(avg)", "val",
]


def _parse_config_tag(tag: str) -> Tuple[float, int]:
    m = re.match(r"interval([\d.]+)_maxf(\d+)", tag)
    if not m:
        return (float("inf"), -1)
    return (float(m.group(1)), int(m.group(2)))


def _validation_mark(val: Dict[str, Any]) -> str:
    """핵심 bool 검사만 집계. 전부 통과면 'ok', 실패 있으면 'FAIL(...)', N/A 제외."""
    fails = []
    for k, v in val.items():
        if isinstance(v, bool) and not v:
            fails.append(k)
    if fails:
        return "FAIL(" + ",".join(fails) + ")"
    return "ok"


def _extract(record: Dict[str, Any]) -> Dict[str, Any]:
    tc = record["token_counts"]
    il = record["interleaving"]
    val = record["validation"]
    return {
        "mode": val.get("mode", "?"),
        "seq_len": tc.get("total", 0),
        "time": tc.get("time_token", 0),
        "video": tc.get("video_pad", 0),
        "audio": tc.get("audio_pad", 0),
        "chunks": il.get("num_chunks", 0),
        "vid/ch": il.get("per_chunk_video_pad", 0),
        "aud/ch(avg)": round(il.get("per_chunk_audio_pad_avg", 0), 2),
        "val": _validation_mark(val),
    }


def _collect(in_dir: Path) -> List[Dict[str, Any]]:
    """[{config_tag, sample_tag, metrics...}] 반환."""
    rows = []
    for cfg_dir in sorted(in_dir.iterdir()):
        if not cfg_dir.is_dir():
            continue
        for jf in sorted(cfg_dir.glob("*.json")):
            try:
                rec = json.load(open(jf))
            except Exception:
                continue
            sample = rec.get("video_meta", {}).get("debug_tag") or rec.get("sample_tag")
            metrics = _extract(rec)
            rows.append({
                "config": cfg_dir.name,
                "sample": sample,
                **metrics,
            })
    return rows


# ---------------- 렌더러 ----------------

def _render_table_plain(headers: List[str], rows: List[List[str]]) -> str:
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    lines = []
    lines.append("  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    lines.append("  ".join("-" * widths[i] for i in range(len(headers))))
    for r in rows:
        lines.append("  ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)


def _render_table_md(headers: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _render(by: str, rows: List[Dict[str, Any]], fmt: str) -> str:
    headers = [CONFIG_COL if by == "sample" else SAMPLE_COL] + METRIC_COLS
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    key_col, var_col = ("sample", "config") if by == "sample" else ("config", "sample")
    for r in rows:
        groups[r[key_col]].append(r)

    # 그룹 키 정렬: sample 은 이름순, config 은 (interval, maxf) 순
    if by == "sample":
        sorted_keys = sorted(groups.keys())
    else:
        sorted_keys = sorted(groups.keys(), key=_parse_config_tag)

    chunks: List[str] = []
    for k in sorted_keys:
        group = groups[k]
        # 그룹 내 행 정렬
        if by == "sample":
            group.sort(key=lambda r: _parse_config_tag(r["config"]))
        else:
            group.sort(key=lambda r: r["sample"] or "")
        table_rows = []
        for r in group:
            first = r[var_col]
            table_rows.append([first] + [r[c] for c in METRIC_COLS])
        title = f"[{'sample' if by == 'sample' else 'config'}] {k}  (n={len(group)})"
        if fmt == "md":
            chunks.append(f"### {title}\n")
            chunks.append(_render_table_md(headers, table_rows))
            chunks.append("")
        else:
            chunks.append(title)
            chunks.append(_render_table_plain(headers, table_rows))
            chunks.append("")
    return "\n".join(chunks)


def _render_csv(rows: List[Dict[str, Any]]) -> str:
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([CONFIG_COL, SAMPLE_COL] + METRIC_COLS)
    for r in sorted(rows, key=lambda x: (_parse_config_tag(x["config"]), x["sample"])):
        w.writerow([r["config"], r["sample"]] + [r[c] for c in METRIC_COLS])
    return buf.getvalue()


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True,
                    help="sweep 결과 루트 (하위에 <config_tag>/*.json)")
    ap.add_argument("--by", choices=["sample", "config"], default="sample",
                    help="그룹핑 축 (기본 sample)")
    ap.add_argument("--format", choices=["plain", "md", "csv"], default="plain")
    ap.add_argument("--samples", default="",
                    help="쉼표 구분 sample 태그 필터 (예: short,longest)")
    ap.add_argument("--configs", default="",
                    help="쉼표 구분 config 태그 부분 일치 필터 (예: interval0.2,interval0.5)")
    ap.add_argument("--out", default="",
                    help="파일 저장 (비우면 stdout)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).resolve()
    if not in_dir.is_dir():
        raise SystemExit(f"[compare] in_dir 존재하지 않음: {in_dir}")

    rows = _collect(in_dir)
    if args.samples:
        wanted = set(s.strip() for s in args.samples.split(","))
        rows = [r for r in rows if r["sample"] in wanted]
    if args.configs:
        filters = [c.strip() for c in args.configs.split(",")]
        rows = [r for r in rows if any(f in r["config"] for f in filters)]
    if not rows:
        raise SystemExit("[compare] 조건에 맞는 행 없음")

    n_cfg = len({r["config"] for r in rows})
    n_sam = len({r["sample"] for r in rows})
    fails = [r for r in rows if str(r["val"]).startswith("FAIL")]

    if args.format == "csv":
        # CSV 는 헤더/푸터 없이 순수 표만.
        full = _render_csv(rows)
    else:
        text = _render(args.by, rows, args.format)
        header = (f"compare: in_dir={in_dir}  configs={n_cfg}  samples={n_sam}  "
                  f"rows={len(rows)}  fails={len(fails)}")
        if args.format == "md":
            header = "# " + header
        full = header + "\n\n" + text + "\n"
        if fails:
            full += "\n" + ("---- VALIDATION FAILURES ----\n"
                            if args.format == "plain"
                            else "## Validation failures\n\n")
            for f in fails:
                full += f"  {f['config']}  /  {f['sample']}  /  {f['val']}\n"

    if args.out:
        Path(args.out).write_text(full)
        print(f"[compare] wrote {args.out}  (rows={len(rows)})"
              + (f"  FAILS={len(fails)}" if fails else ""))
    else:
        print(full)


if __name__ == "__main__":
    main()
