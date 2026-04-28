"""Avicuna JSON hybrid × UnAV-100 eval — Union-IoU + FP_rate + FN_rate.

Parser: JSON list output 에서 `event` fuzzy-match gt_label, `timestamps` "from X to Y" pct 추출.
ex output: [{"event": "playing ukulele", "timestamps": "from 0 to 95"}, {"event": "...", "timestamps": "from XX to YY"}]
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_utils import score_sample, summarize, print_report

# JSON list 추출 (가장 바깥의 [] 블록)
LIST_RE = re.compile(r"\[.*?\]", re.DOTALL)
# 각 dict 안의 "event":"..." 과 "timestamps":"..."
EVENT_RE = re.compile(r'"event"\s*:\s*"([^"]+)"', re.IGNORECASE)
TS_RE = re.compile(r'"timestamps"\s*:\s*"([^"]+)"', re.IGNORECASE)
FROM_TO_RE = re.compile(r"from\s+(\d{1,3})\s+to\s+(\d{1,3})", re.IGNORECASE)


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).replace("-", " ").replace("_", " ")


def fuzzy_match(pred_label: str, query_label: str) -> bool:
    p, q = normalize(pred_label), normalize(query_label)
    if p == q or p in q or q in p:
        return True
    p_toks, q_toks = set(p.split()), set(q.split())
    if p_toks & q_toks:
        return len(p_toks & q_toks) / max(min(len(p_toks), len(q_toks)), 1) >= 0.5
    return False


def parse_json_preds(raw: str, query_label: str, duration: float, max_time: float = 60.0):
    """Return list of [s,e] in seconds, matching query_label via fuzzy."""
    # JSON list 블록 탐색 (여러 개면 각각 시도)
    out = []
    # 대괄호 블록 찾기 (non-greedy)
    for list_match in LIST_RE.finditer(raw):
        block = list_match.group(0)
        # 각 element (dict) 별로 event/timestamps 짝 추출
        # 간단히: 모든 event 와 그 직후 timestamps 쌍 매칭
        events = EVENT_RE.findall(block)
        timestamps = TS_RE.findall(block)
        # 짝이 안 맞으면 순서대로 zip
        for ev, ts in zip(events, timestamps):
            if not fuzzy_match(ev, query_label):
                continue
            m = FROM_TO_RE.search(ts)
            if not m:
                continue
            s_pct, e_pct = int(m.group(1)), int(m.group(2))
            s_pct = min(s_pct, 99); e_pct = min(e_pct, 99)
            if s_pct < e_pct:
                s = s_pct * duration / 100.0
                e = e_pct * duration / 100.0
                s = max(0.0, min(s, max_time))
                e = max(0.0, min(e, max_time))
                if e > s:
                    out.append([s, e])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True,
                    help="Avicuna predictions.jsonl (each line: {raw, gt_label, gt_segments, duration, ...})")
    ap.add_argument("--max_time", type=float, default=60.0)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.predictions))
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    all_ious = []
    total_preds = fp_preds = total_gts = fn_gts = 0
    parse_ok = parse_fail = 0
    n_samples = 0

    with open(args.predictions) as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            n_samples += 1
            gt_segments = r.get("gt_segments", [])
            gt_label = r.get("gt_label", "")
            if not gt_segments: continue

            if r.get("error", "").startswith("feature missing") or r.get("error") == "no duration" \
               or r.get("error", "").startswith("gen_error"):
                parse_fail += 1
                for _ in gt_segments:
                    all_ious.append(0.0)
                total_gts += len(gt_segments)
                fn_gts += len(gt_segments)
                continue

            dur = float(r["duration"])
            pred_segs = parse_json_preds(r.get("raw", ""), gt_label, dur, args.max_time)
            if not pred_segs:
                parse_fail += 1
                for _ in gt_segments:
                    all_ious.append(0.0)
                total_gts += len(gt_segments)
                fn_gts += len(gt_segments)
                continue

            parse_ok += 1
            per_gt, n_fp, n_pred, n_fn, n_gt = score_sample(gt_segments, pred_segs)
            all_ious.extend(per_gt)
            total_preds += n_pred; fp_preds += n_fp
            total_gts += n_gt; fn_gts += n_fn

    summary = summarize(all_ious, total_preds, fp_preds, total_gts, fn_gts,
                        parse_ok, parse_fail, n_samples)
    summary["parser"] = 'avicuna_json ([{"event":..., "timestamps":"from X to Y"}], fuzzy label)'
    print_report("Avicuna JSON hybrid × UnAV-100 — Union-IoU", summary)

    out_path = os.path.join(out_dir, "eval_miou_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
