#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parse_and_eval.py
  - vS2 inference 결과(test_results.json)를 파싱해서
  - UnAV-100 기준 mAP를 계산한다 (AVicuna 방식: score=1.0 고정)

Usage:
  python eval/parse_and_eval.py \
    --results   eval/results/unav100_test/test_results.json \
    --unav_json /home/aix23102/audiolm/CCNet/data/unav_100/annotations/unav100_annotations.json \
    --split     test \
    --max_time  60.0 \
    --out_dir   eval/results/unav100_test
"""
import os
import sys
sys.path.insert(0, os.environ.get('CLAP_SRC', 'CLAP/src'))
import laion_clap

import argparse
import ast
import json
import os
import re
import difflib
import numpy as np
from collections import defaultdict
import numpy as np

# 전역 캐시
_clap_model = None
_cached_valid_labels = None
_cached_label_embeddings = None

# ──────────────────────────────────────────────────────────────
# 1. video_id 추출
# ──────────────────────────────────────────────────────────────

def extract_video_id(item: dict) -> str | None:
    """
    test_results.json의 id 필드 형식:
      ["['/path/to/video.mp4', '/path/to/audio.wav']"]
    → "8WSesSOfBrE"
    """
    raw_id = item.get("id", None)

    # 리스트면 첫 번째 원소 꺼내기
    if isinstance(raw_id, list) and len(raw_id) > 0:
        raw_id = raw_id[0]

    if not isinstance(raw_id, str):
        return None

    # 문자열화된 리스트 "['/path/a.mp4', '/path/b.wav']" 파싱
    raw_id = raw_id.strip()
    if raw_id.startswith("["):
        try:
            parsed = ast.literal_eval(raw_id)
            if isinstance(parsed, list) and len(parsed) > 0:
                raw_id = parsed[0]   # 첫 번째 경로 = 비디오 경로
        except Exception:
            pass

    # 경로에서 stem 추출
    m = re.search(r"([^/\\]+)\.(mp4|avi|mkv|mov|wav|flac)", raw_id, re.IGNORECASE)
    if m:
        return m.group(1)

    return None


# ──────────────────────────────────────────────────────────────
# 2. VTG-LLM time token 역변환
# ──────────────────────────────────────────────────────────────

def decode_vtg_time(token_str: str, max_time: float = 60.0) -> float | None:
    """
    "<t0><t0><t3><t9><tdot><t0>" → 39.0
    형식: 정수부 4자리 digit + <tdot> + 소수부 1자리 digit → 6토큰 고정
    <tdot> 없는 경우(불완전 출력) → 가능한 범위까지만 파싱
    """
    has_dot = "<tdot>" in token_str

    if has_dot:
        parts     = token_str.split("<tdot>")
        int_part  = re.findall(r"<t(\d)>", parts[0])
        dec_part  = re.findall(r"<t(\d)>", parts[1]) if len(parts) > 1 else []
    else:
        # <tdot> 없으면 전체를 정수부로 취급
        int_part  = re.findall(r"<t(\d)>", token_str)
        dec_part  = []

    if not int_part:
        return None

    integer_part = int("".join(int_part))
    decimal_part = int(dec_part[0]) if dec_part else 0
    t = integer_part + decimal_part / 10.0
    return min(t, max_time)


# ──────────────────────────────────────────────────────────────
# 3. pred 텍스트 파싱
# ──────────────────────────────────────────────────────────────

def normalize_label(s: str) -> str:
    return re.sub(r"[\s_\-]+", " ", s.strip().lower())


def _get_clap_model():
    global _clap_model
    if _clap_model is None:
        print("[INFO] Loading CLAP model...")
        _clap_model = laion_clap.CLAP_Module(enable_fusion=True)  # False → True로 변경
        _clap_model.load_ckpt(os.environ.get('CLAP_CKPT', 'CLAP/630k-audioset-fusion-best.pt'))
        _clap_model.eval()
        print("[INFO] CLAP model loaded.")
    return _clap_model

def _get_label_embeddings(valid_labels: list[str]):
    global _cached_valid_labels, _cached_label_embeddings
    if _cached_valid_labels != valid_labels:
        model = _get_clap_model()
        _cached_label_embeddings = model.get_text_embedding(
            valid_labels, use_tensor=False
        )  # shape: (N, D), numpy array
        _cached_valid_labels = valid_labels
    return _cached_label_embeddings


def fuzzy_match_label(pred_label: str, valid_labels: list[str],
                      cutoff: float = 0.6) -> str:
    pred_norm = normalize_label(pred_label)
    norm_map  = {normalize_label(v): v for v in valid_labels}

    # 1. exact match
    if pred_norm in norm_map:
        return norm_map[pred_norm]

    # 2. CLAP text embedding cosine similarity NN
    model = _get_clap_model()
    pred_emb = model.get_text_embedding([pred_label], use_tensor=False)  # (1, D)
    label_embs = _get_label_embeddings(valid_labels)                      # (N, D)

    # L2 normalize
    pred_emb  = pred_emb  / (np.linalg.norm(pred_emb,  axis=1, keepdims=True) + 1e-8)
    label_embs_norm = label_embs / (np.linalg.norm(label_embs, axis=1, keepdims=True) + 1e-8)

    sims = (label_embs_norm @ pred_emb.T).squeeze()  # (N,)
    best_idx = int(np.argmax(sims))
    return valid_labels[best_idx]


def _extract_event_dicts(raw: str) -> list[dict]:
    # ── 전처리 ───────────────────────────────────────────────
    raw = re.sub(r"[a-zA-Z가-힣]{4,}(\{)", r"\1", raw)
    raw = re.sub(r"(\})[a-zA-Z가-힣]{4,}", r"\1", raw)

    # ── 작은따옴표 → 큰따옴표 변환 ───────────────────────────
    def to_double_quote(s: str) -> str:
        s = re.sub(r"'([^']*)'(\s*:)", r'"\1"\2', s)  # key
        s = re.sub(r":\s*'([^']*)'",   r': "\1"',  s)  # value
        return s

    raw_dq = to_double_quote(raw)

    # ── 전략 1: 정상 JSON 배열 (큰따옴표 변환 후) ────────────
    m = re.search(r"(\[.*?\])", raw_dq, re.DOTALL)
    if m:
        try:
            items = json.loads(m.group(1))
            if isinstance(items, list):
                return [x for x in items if isinstance(x, dict)]
        except json.JSONDecodeError:
            pass

    # ── 전략 1b: 원본으로 한 번 더 시도 ─────────────────────
    m = re.search(r"(\[.*?\])", raw, re.DOTALL)
    if m:
        try:
            items = json.loads(m.group(1))
            if isinstance(items, list):
                return [x for x in items if isinstance(x, dict)]
        except json.JSONDecodeError:
            pass

    # ── 전략 2: 단일 객체 { ... } ────────────────────────────
    m = re.search(r"(\{[^{}]*\})", raw_dq, re.DOTALL)
    if m:
        try:
            item = json.loads(m.group(1))
            if isinstance(item, dict):
                return [item]
        except json.JSONDecodeError:
            pass

    # ── 전략 3: 정규식으로 개별 필드 추출 ────────────────────
    items = []
    blocks = re.split(r"(?=\{[^{}]*['\"]event['\"])", raw)
    for block in blocks:
        event_m = re.search(r'["\']event["\']\s*["\']?\s*:\s*["\']([^"\']+)["\']', block)
        start_m = re.search(r'["\']start["\']\s*:\s*["\']([^"\']*)["\']', block)
        end_m   = re.search(r'["\']end["\']\s*:\s*["\']([^"\']*)["\']',   block)
        if event_m and (start_m or end_m):
            items.append({
                "event": event_m.group(1),
                "start": start_m.group(1) if start_m else "",
                "end":   end_m.group(1)   if end_m   else "",
            })
    if items:
        return items

    return []


def parse_single_output(raw: str, valid_labels: list[str],
                        max_time: float = 60.0,
                        fuzzy_cutoff: float = 0.6) -> list[dict]:
    """
    vS2 출력 텍스트 → [{"label": str, "score": float, "segment": [s, e]}, ...]
    """
    event_dicts = _extract_event_dicts(raw)
    predictions = []

    for item in event_dicts:
        event_raw = item.get("event", item.get("label", "")).strip()
        start_tok = item.get("start", "")
        end_tok   = item.get("end",   "")

        start = decode_vtg_time(start_tok, max_time)
        end   = decode_vtg_time(end_tok,   max_time)

        if start is None and end is None:
            continue
        # 한쪽만 파싱된 경우도 최대한 살리기
        if start is None:
            start = 0.0
        if end is None:
            end = max_time

        if end <= start:
            end = min(start + 1.0, max_time)

        label = fuzzy_match_label(event_raw, valid_labels, cutoff=fuzzy_cutoff)

        predictions.append({
            "label":   label,
            "score":   1.0,    # AVicuna 방식: confidence 고정
            "segment": [start, end],
        })

    return predictions


# ──────────────────────────────────────────────────────────────
# 4. GT annotation 로드
# ──────────────────────────────────────────────────────────────

def load_gt(unav_json: str, split: str) -> tuple[dict, list[str]]:
    """
    Returns:
      gt_dict   : {video_id: [{"label": str, "segment": [s, e]}, ...]}
      all_labels: UnAV-100 전체 레이블 목록 (sorted)
    """
    with open(unav_json, "r") as f:
        db = json.load(f)["database"]

    gt_dict   = {}
    label_set = set()

    for vid, item in db.items():
        if item.get("subset", "").lower() != split.lower():
            continue
        anns = []
        for ann in item.get("annotations", []):
            lbl = ann["label"]
            seg = [float(ann["segment"][0]), float(ann["segment"][1])]
            anns.append({"label": lbl, "segment": seg})
            label_set.add(lbl)
        if anns:
            gt_dict[vid] = anns

    return gt_dict, sorted(label_set)


# ──────────────────────────────────────────────────────────────
# 5. ANETdetection (UnAV-100 방식)
# ──────────────────────────────────────────────────────────────

class ANETdetection:
    """
    UnAV-100 원 repo(ttgeng233/UnAV)의 metrics.py를 인라인 이식.
    외부 의존성 없이 numpy만 사용.
    """

    def __init__(self, tiou_thresholds=None):
        if tiou_thresholds is None:
            tiou_thresholds = np.linspace(0.1, 0.9, 9)
        self.tiou_thresholds = np.array(tiou_thresholds)

    def evaluate(self, gt_dict: dict, pred_dict: dict) -> dict:
        all_gt_labels = set()
        for anns in gt_dict.values():
            for a in anns:
                all_gt_labels.add(a["label"])
        all_labels = sorted(all_gt_labels)

        per_tiou_map = {}
        ap_list      = []

        for tiou in self.tiou_thresholds:
            ap_per_class = self._eval_at_tiou(gt_dict, pred_dict, all_labels, tiou)
            mean_ap = float(np.mean(ap_per_class)) if ap_per_class else 0.0
            per_tiou_map[round(float(tiou), 2)] = mean_ap
            ap_list.append(mean_ap)

        return {
            "per_tiou_mAP": per_tiou_map,
            "average_mAP":  float(np.mean(ap_list)),
        }

    def _eval_at_tiou(self, gt_dict, pred_dict, all_labels, tiou):
        ap_per_class = []

        for label in all_labels:
            # GT
            gt_segs = {}
            n_gt    = 0
            for vid, anns in gt_dict.items():
                segs = [a["segment"] for a in anns if a["label"] == label]
                if segs:
                    gt_segs[vid] = segs
                    n_gt += len(segs)
            if n_gt == 0:
                continue

            # Pred
            preds = []
            for vid, dets in pred_dict.items():
                for d in dets:
                    if d["label"] == label:
                        preds.append({
                            "video_id": vid,
                            "score":    d["score"],
                            "segment":  d["segment"],
                        })

            if not preds:
                ap_per_class.append(0.0)
                continue

            preds.sort(key=lambda x: -x["score"])

            tp      = np.zeros(len(preds))
            fp      = np.zeros(len(preds))
            matched = defaultdict(set)

            for i, pred in enumerate(preds):
                vid  = pred["video_id"]
                pseg = pred["segment"]

                if vid not in gt_segs:
                    fp[i] = 1
                    continue

                ious   = [self._tiou(pseg, gseg) for gseg in gt_segs[vid]]
                best_j = int(np.argmax(ious))

                if ious[best_j] >= tiou and best_j not in matched[vid]:
                    tp[i] = 1
                    matched[vid].add(best_j)
                else:
                    fp[i] = 1

            cum_tp    = np.cumsum(tp)
            cum_fp    = np.cumsum(fp)
            recall    = cum_tp / n_gt
            precision = cum_tp / (cum_tp + cum_fp + 1e-8)

            ap_per_class.append(self._voc_ap(recall, precision))

        return ap_per_class

    @staticmethod
    def _tiou(seg1, seg2):
        inter_s = max(seg1[0], seg2[0])
        inter_e = min(seg1[1], seg2[1])
        inter   = max(0.0, inter_e - inter_s)
        union   = (seg1[1] - seg1[0]) + (seg2[1] - seg2[0]) - inter
        return inter / (union + 1e-8) if union > 0 else 0.0

    @staticmethod
    def _voc_ap(recall, precision):
        """VOC 11-point interpolation AP."""
        ap = 0.0
        for thr in np.linspace(0, 1, 11):
            prec_at_thr = precision[recall >= thr]
            p = float(prec_at_thr.max()) if prec_at_thr.size > 0 else 0.0
            ap += p / 11.0
        return ap


# ──────────────────────────────────────────────────────────────
# 6. 메인
# ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results",      required=True,
                    help="train.py --do_test 가 출력한 test_results.json 경로")
    ap.add_argument("--unav_json",    required=True,
                    help="원본 UnAV-100 annotation JSON (database 키 포함)")
    ap.add_argument("--split",        default="test",
                    choices=["train", "val", "test"])
    ap.add_argument("--max_time",     type=float, default=60.0,
                    help="학습 시 max_time 값 (time token 역변환 클램핑용)")
    ap.add_argument("--fuzzy_cutoff", type=float, default=0.6,
                    help="레이블 fuzzy matching 임계값 (0~1)")
    ap.add_argument("--out_dir",      default=None,
                    help="결과 저장 디렉토리 (생략 시 results와 같은 폴더)")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.results))
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. GT 로드 ────────────────────────────────────────────
    print("[1/4] Loading GT annotations ...")
    gt_dict, all_labels = load_gt(args.unav_json, args.split)
    print(f"      {len(gt_dict)} videos, {len(all_labels)} classes")

    # ── 2. Inference 결과 로드 ────────────────────────────────
    print("[2/4] Loading inference results ...")
    with open(args.results, "r") as f:
        raw_results = json.load(f)
    print(f"      {len(raw_results)} items")

    # ── 3. 파싱 ──────────────────────────────────────────────
    print("[3/4] Parsing predictions ...")
    pred_dict   = {}
    parse_stats = {"ok": 0, "empty": 0, "no_vid": 0}
    failed_preds = []   # 파싱 실패 케이스 저장 (디버그용)

    for item in raw_results:
        vid = extract_video_id(item)
        if vid is None:
            parse_stats["no_vid"] += 1
            continue

        raw_pred = item.get("pred", "")
        preds    = parse_single_output(
            raw_pred, all_labels,
            max_time=args.max_time,
            fuzzy_cutoff=args.fuzzy_cutoff,
        )

        if not preds:
            parse_stats["empty"] += 1
            failed_preds.append({"video_id": vid, "pred": raw_pred[:200]})
        else:
            parse_stats["ok"] += 1

        # 동일 video_id 누적 (multi-gpu dedup 후에도 혹시 몰라서)
        if vid not in pred_dict:
            pred_dict[vid] = []
        pred_dict[vid].extend(preds)

    total = parse_stats["ok"] + parse_stats["empty"] + parse_stats["no_vid"]
    print(f"      total={total}  ok={parse_stats['ok']}  "
          f"empty={parse_stats['empty']}  no_vid={parse_stats['no_vid']}")

    # 파싱 샘플 출력
    print("\n  [SAMPLE PREDICTIONS]")
    for i, (vid, dets) in enumerate(list(pred_dict.items())[:5]):
        print(f"  {vid}: {dets}")

    if failed_preds:
        print(f"\n  [FAILED PARSE SAMPLES] (first 3)")
        for fp in failed_preds[:3]:
            print(f"  {fp['video_id']}: {repr(fp['pred'])}")

    # ── 4. mAP 계산 ──────────────────────────────────────────
    print("\n[4/4] Computing mAP ...")

    # AVicuna 방식: tIoU [0.5:0.1:0.9] (5개)
    tiou_avicuna = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    # UnAV-100 full: tIoU [0.1:0.1:0.9] (9개)
    tiou_full    = np.linspace(0.1, 0.9, 9)

    res_avicuna = ANETdetection(tiou_avicuna).evaluate(gt_dict, pred_dict)
    res_full    = ANETdetection(tiou_full).evaluate(gt_dict, pred_dict)

    # ── 결과 출력 ─────────────────────────────────────────────
    SEP = "=" * 62
    print(f"\n{SEP}")
    print("  UnAV-100 Evaluation Results  |  vS2 fine-tuned")
    print(SEP)

    print("\n  [AVicuna 방식]  tIoU @ [0.5 : 0.1 : 0.9]")
    for tiou, val in sorted(res_avicuna["per_tiou_mAP"].items()):
        print(f"    mAP @ {tiou:.1f} = {val * 100:6.2f}")
    print(f"    ─────────────────────────────────────")
    print(f"    Average mAP  = {res_avicuna['average_mAP'] * 100:6.2f}")

    print("\n  [Full]  tIoU @ [0.1 : 0.1 : 0.9]")
    for tiou, val in sorted(res_full["per_tiou_mAP"].items()):
        print(f"    mAP @ {tiou:.1f} = {val * 100:6.2f}")
    print(f"    ─────────────────────────────────────")
    print(f"    Average mAP  = {res_full['average_mAP'] * 100:6.2f}")

    print(f"\n{SEP}\n")

    # ── 저장 ─────────────────────────────────────────────────
    summary = {
        "avicuna_style": {
            "tiou_range":    "0.5:0.1:0.9",
            "per_tiou_mAP_%": {
                str(k): round(v * 100, 4)
                for k, v in res_avicuna["per_tiou_mAP"].items()
            },
            "average_mAP_%": round(res_avicuna["average_mAP"] * 100, 4),
        },
        "full": {
            "tiou_range":    "0.1:0.1:0.9",
            "per_tiou_mAP_%": {
                str(k): round(v * 100, 4)
                for k, v in res_full["per_tiou_mAP"].items()
            },
            "average_mAP_%": round(res_full["average_mAP"] * 100, 4),
        },
        "parse_stats": parse_stats,
    }

    summary_path = os.path.join(out_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    pred_path = os.path.join(out_dir, "predictions.json")
    with open(pred_path, "w") as f:
        json.dump(pred_dict, f, indent=2, ensure_ascii=False)

    failed_path = os.path.join(out_dir, "failed_parses.json")
    with open(failed_path, "w") as f:
        json.dump(failed_preds, f, indent=2, ensure_ascii=False)

    print(f"[SAVED] eval_summary.json  → {summary_path}")
    print(f"[SAVED] predictions.json   → {pred_path}")
    print(f"[SAVED] failed_parses.json → {failed_path}  ({len(failed_preds)} items)")


if __name__ == "__main__":
    main()
