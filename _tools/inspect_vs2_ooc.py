#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import string
from collections import defaultdict, Counter
from difflib import get_close_matches
from typing import Any, Dict, List, Tuple, Optional

# ----------------------------
# Basic utils
# ----------------------------

def vid_from_id_field(id_field) -> str:
    if isinstance(id_field, list) and len(id_field) > 0:
        video_path = id_field[0]
        return os.path.splitext(os.path.basename(video_path))[0]
    if isinstance(id_field, str):
        return os.path.splitext(os.path.basename(id_field))[0]
    return ""

_SEP_RE = re.compile(r"[_\-/]+")   # _, -, / -> space
_WS_RE = re.compile(r"\s+")

def norm_label(s: str) -> str:
    """
    Same normalization policy as eval_mAP_multi.py (format-only).
    """
    if not isinstance(s, str):
        return ""
    x = s.strip()
    if not x:
        return ""
    x = x.lower()
    x = _SEP_RE.sub(" ", x)
    x = _WS_RE.sub(" ", x).strip()

    # strip punctuation only at ends
    punct = set(string.punctuation)
    while x and x[0] in punct:
        x = x[1:].lstrip()
    while x and x[-1] in punct:
        x = x[:-1].rstrip()

    x = _WS_RE.sub(" ", x).strip()
    return x

def basic_norm_label(s: str) -> str:
    """
    Old/simple normalization: strip + lowercase + collapse whitespace only.
    Used to count how many labels are rescued by the stronger normalization.
    """
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def extract_pred_events(text: str) -> Optional[List[Dict[str, Any]]]:
    """
    Robustly extract events from VS2 pred string.
    Supports:
      - JSON array
      - single JSON object
      - stream of JSON objects separated by commas/newlines
    Stops when it hits 'Note:' etc.
    """
    if not isinstance(text, str):
        return None
    s = text.replace("<|im_end|>", "").strip()

    # 1) prefer JSON array if present
    m = re.search(r"\[[\s\S]*\]", s)
    if m:
        cand = m.group(0).strip()
        try:
            arr = json.loads(cand)
            if isinstance(arr, list):
                return [x for x in arr if isinstance(x, dict)]
        except Exception:
            pass

    # 2) decode stream of JSON values
    dec = json.JSONDecoder()
    i, n = 0, len(s)
    out = []

    stop_markers = ["Note:", "NOTE:", "note:"]

    while i < n:
        # stop marker
        if any(s.startswith(mk, i) for mk in stop_markers):
            break

        # skip separators
        while i < n and s[i] in " \t\r\n,":
            i += 1
        if i >= n:
            break

        if s[i] not in "[{":
            break

        try:
            obj, j = dec.raw_decode(s, i)
            i = j
            if isinstance(obj, list):
                out.extend([x for x in obj if isinstance(x, dict)])
            elif isinstance(obj, dict):
                out.append(obj)
        except Exception:
            break

    return out if out else None


# ----------------------------
# Load GT labels from UnAV annotations
# ----------------------------

def load_unav_labels(unav_json: str, split: str) -> Tuple[List[str], Dict[str, Any]]:
    with open(unav_json, "r", encoding="utf-8") as f:
        db = json.load(f)["database"]

    labels = set()
    vids = 0
    instances = 0
    for vid, v in db.items():
        if v.get("subset", "").lower() != split.lower():
            continue
        vids += 1
        for ann in v.get("annotations", []):
            lab = ann.get("label", "")
            if isinstance(lab, str) and lab.strip():
                labels.add(norm_label(lab))
            instances += 1

    labels = sorted(labels)
    meta = {"gt_videos": vids, "gt_instances": instances, "gt_num_labels": len(labels)}
    return labels, meta


# ----------------------------
# Main inspection
# ----------------------------

def main():
    # ✅ 여기만 수정해서 쓰시면 됩니다.
    UNAV_JSON = os.environ.get('UNAV_ANNO', 'data/unav100_annotations.json')
    VS2_RESULTS = os.environ.get('VS2_RESULTS', 'output/test/1_multi_ft/test_results.json')
    SPLIT = "test"

    # how many samples to print
    PRINT_EXAMPLES_OOC = 30
    TOPK_SIMILAR = 5

    gt_labels, gt_meta = load_unav_labels(UNAV_JSON, SPLIT)
    gt_set = set(gt_labels)

    with open(VS2_RESULTS, "r", encoding="utf-8") as f:
        res = json.load(f)

    # Stats
    parse_fail = 0
    bad_fields = 0
    total_items = 0
    total_events = 0

    inclass_events = 0
    ooc_events = 0

    # OOC analysis
    ooc_counter = Counter()
    rescued_by_normalization = Counter()   # normalized label hits GT? (case/space only)
    ooc_samples = []  # for qualitative prints: (vid, raw_label, norm_label, suggested)

    # For: per-sample report
    sample_ooc_events = defaultdict(list)   # vid -> list of raw labels
    sample_in_events = defaultdict(list)    # vid -> list of labels

    # categorize OOC types
    cat_counter = Counter()

    def categorize(raw: str) -> str:
        if not raw or not isinstance(raw, str):
            return "non_string_or_empty"
        if raw.strip() != raw:
            return "leading_or_trailing_space"
        if any(c.isupper() for c in raw):
            return "has_uppercase"
        if any(c in "_-/" for c in raw):
            return "has_separator_char"
        if any(c in string.punctuation for c in raw):
            return "has_punct"
        # sentence-like heuristic (not just length)
        if len(raw.split()) >= 4 or "note:" in raw.lower():
            return "sentence_like_label"
        return "other"

    # Iterate items
    for item in res:
        total_items += 1
        vid = vid_from_id_field(item.get("id"))
        arr = extract_pred_events(item.get("pred", ""))
        if arr is None:
            parse_fail += 1
            continue

        for obj in arr:
            total_events += 1
            lab = obj.get("label", None)
            s = obj.get("start", None)
            e = obj.get("end", None)
            sc = obj.get("score", None)

            # minimal validity check
            try:
                if not isinstance(lab, str) or not lab.strip():
                    raise ValueError("bad label")
                s = float(s); e = float(e)
                if e <= s:
                    raise ValueError("end<=start")
                if sc is None:
                    sc = 1.0
                sc = float(sc)
            except Exception:
                bad_fields += 1
                continue

            raw_lab = lab
            nlab = norm_label(raw_lab)              # strong normalization (same as eval)
            blab = basic_norm_label(raw_lab)        # old/simple normalization (for rescue stats)

            if nlab in gt_set:
                inclass_events += 1
                sample_in_events[vid].append(raw_lab)

                # Count "rescued": basic norm would NOT match, but strong norm matches
                if blab not in gt_set and blab != nlab:
                    rescued_by_normalization[raw_lab] += 1
            else:
                ooc_events += 1
                ooc_counter[raw_lab] += 1
                sample_ooc_events[vid].append(raw_lab)

                cat = categorize(raw_lab)
                cat_counter[cat] += 1

                # Suggest closest GT labels (string similarity on normalized space)
                sugg = get_close_matches(nlab, gt_labels, n=TOPK_SIMILAR, cutoff=0.0)
                ooc_samples.append((vid, raw_lab, nlab, sugg))


    # Print summary
    print("========== VS2 OOC Inspection ==========")
    print(f"GT meta: videos={gt_meta['gt_videos']}, instances={gt_meta['gt_instances']}, labels={gt_meta['gt_num_labels']}")
    print(f"Total items: {total_items}")
    print(f"Parse fails (no usable JSON): {parse_fail}")
    print(f"Bad fields (missing label/start/end/score): {bad_fields}")
    print(f"Total parsed events: {total_events}")
    print(f"In-class events: {inclass_events}")
    print(f"Out-of-class events: {ooc_events}")
    if total_events > 0:
        print(f"OOC ratio: {ooc_events/total_events*100:.2f}%")
    print()

    print("[OOC type breakdown]")
    for k, v in cat_counter.most_common():
        print(f"- {k}: {v}")
    print()

    print("[Rescued by normalization (basic failed but strong matched GT)]")
    total_rescued = sum(rescued_by_normalization.values())
    print(f"Total rescued events: {total_rescued}")
    if total_rescued > 0:
        print("Top-20 rescued raw labels:")
        for lab, cnt in rescued_by_normalization.most_common(20):
            print(f"{cnt:5d}  {lab}")
    print()

    print("[Top-30 OOC labels by frequency]")
    for lab, cnt in ooc_counter.most_common(30):
        print(f"{cnt:5d}  {lab}")
    print()

    # Qualitative: show samples with OOC labels + suggestions
    print(f"[Qualitative examples: first {PRINT_EXAMPLES_OOC} OOC events with suggestions]")
    shown = 0
    for (vid, raw_lab, nlab, sugg) in ooc_samples:
        print("-" * 80)
        print(f"vid: {vid}")
        print(f"raw label: {raw_lab}")
        print(f"norm label: {nlab}")
        print(f"suggested GT labels: {sugg[:TOPK_SIMILAR]}")
        shown += 1
        if shown >= PRINT_EXAMPLES_OOC:
            break

    # Optional: per-video quick glance (videos with many OOC predictions)
    print()
    print("[Videos with most OOC labels predicted]")
    vid_ooc_counts = sorted([(vid, len(labs)) for vid, labs in sample_ooc_events.items()],
                            key=lambda x: x[1], reverse=True)[:20]
    for vid, cnt in vid_ooc_counts:
        print(f"{cnt:3d}  {vid}")

    # Save a report json next to results
    out_path = os.path.join(os.path.dirname(VS2_RESULTS), "ooc_report.json")
    report = {
        "gt_meta": gt_meta,
        "summary": {
            "total_items": total_items,
            "parse_fail": parse_fail,
            "bad_fields": bad_fields,
            "total_events": total_events,
            "inclass_events": inclass_events,
            "ooc_events": ooc_events,
            "ooc_ratio": (ooc_events / total_events) if total_events else None,
            "rescued_by_normalization_total": total_rescued,
            "rescued_by_normalization_top": rescued_by_normalization.most_common(200),
        },
        "ooc_type_breakdown": dict(cat_counter),
        "top_ooc_labels": ooc_counter.most_common(200),
        "qualitative_samples": [
            {"vid": vid, "raw_label": raw, "norm_label": nlab, "suggested": sugg}
            for (vid, raw, nlab, sugg) in ooc_samples[:500]
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print()
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
