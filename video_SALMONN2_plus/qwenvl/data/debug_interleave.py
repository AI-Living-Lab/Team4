"""
TTI 디버그 모드 — dataset collate 직후 한 샘플의 인터리빙 구조를 덤프.

세 가지 입력 측 마커 모드를 지원:
  - off           : 마커 미삽입 (Qwen2.5-VL 베이스라인)
  - special_token : <t0>..<tdot>..<t*> = 6 special tokens / chunk
  - natural_text  : 'second{XXXX.Y}' = 9 plain text tokens / chunk

단일 진입점: dump_sample(...)
  <out_dir>/<NNN>_<tag>.json  — 구조화된 덤프 (config, counts, chunks, time_markers, rope, validation)
  <out_dir>/<NNN>_<tag>.txt   — 사람이 읽는 요약 (layout, time marker strip, RoPE samples)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


# ---------------- Token 분류 ----------------

def _special_ids(tokenizer) -> Dict[str, Optional[int]]:
    unk = tokenizer.unk_token_id
    names = {
        "vision_start": "<|vision_start|>",
        "vision_end":   "<|vision_end|>",
        "video_pad":    "<|video_pad|>",
        "audio_pad":    "<|audio_pad|>",
        "image_pad":    "<|image_pad|>",
        "tdot":         "<tdot>",
    }
    out: Dict[str, Optional[int]] = {}
    for k, tok in names.items():
        tid = tokenizer.convert_tokens_to_ids(tok)
        out[k] = tid if (tid is not None and tid != unk) else None
    return out


def _classify(tid: int, spec: Dict[str, Optional[int]],
              tt_range: Optional[Tuple[int, int]]) -> str:
    if tid == spec["vision_start"]: return "vision_start"
    if tid == spec["vision_end"]:   return "vision_end"
    if tid == spec["video_pad"]:    return "video_pad"
    if tid == spec["audio_pad"]:    return "audio_pad"
    if tid == spec["image_pad"]:    return "image_pad"
    if tt_range is not None:
        lo, hi = tt_range
        if lo <= tid <= hi:
            return "time_token"
    return "text"


# ---------------- Vision span ----------------

def _find_vision_span(categories: List[str]) -> Optional[Tuple[int, int]]:
    try:
        s = categories.index("vision_start")
        e = categories.index("vision_end", s)
        return s, e
    except ValueError:
        return None


# ---------------- Time marker 검출 (모드별) ----------------

_NATURAL_RE = re.compile(r"second\{(\d+\.\d+)\}")


def _find_markers_special_token(
    input_ids: List[int],
    tt_range: Tuple[int, int],
    tdot_id: Optional[int],
    marker_len: int,
    tokenizer,
) -> List[Dict[str, Any]]:
    """special_token 모드: 연속된 time-token run 을 marker_len 단위로 분할."""
    lo, hi = tt_range
    markers: List[Dict[str, Any]] = []
    n = len(input_ids)
    i = 0
    while i < n:
        if lo <= input_ids[i] <= hi:
            j = i
            while j < n and lo <= input_ids[j] <= hi:
                j += 1
            for k in range(i, j, marker_len):
                seg = input_ids[k:k + marker_len]
                m: Dict[str, Any] = {
                    "pos": k,
                    "end": k + len(seg),
                    "tokens": tokenizer.convert_ids_to_tokens(seg),
                }
                # XXXX.Y 6토큰 가정으로 seconds 복원 시도 (marker_len=6 때만)
                ok = (
                    len(seg) == marker_len
                    and tdot_id is not None
                    and marker_len == 6
                    and seg[4] == tdot_id
                )
                if ok:
                    d = [seg[x] - lo for x in (0, 1, 2, 3)]
                    f = seg[5] - lo
                    if all(0 <= x <= 9 for x in d) and 0 <= f <= 9:
                        m["seconds"] = (d[0] * 1000 + d[1] * 100 + d[2] * 10 + d[3]) + f / 10.0
                    else:
                        m["malformed"] = True
                else:
                    m["malformed"] = True
                markers.append(m)
            i = j
        else:
            i += 1
    return markers


def _find_markers_natural_text(
    input_ids: List[int],
    vision_span: Optional[Tuple[int, int]],
    spec: Dict[str, Optional[int]],
    marker_len: int,
    tokenizer,
) -> List[Dict[str, Any]]:
    """natural_text 모드: vision span 내에서 video_pad/audio_pad 가 아닌 토큰의
    marker_len-단위 run 을 marker 로 식별, 'second{XXXX.Y}' 텍스트에서 seconds 복원."""
    if vision_span is None:
        return []
    s, e = vision_span
    vid = spec.get("video_pad")
    aud = spec.get("audio_pad")
    markers: List[Dict[str, Any]] = []
    i = s + 1                                # vision_start 뒤부터 스캔
    while i < e:
        tid = input_ids[i]
        if tid == vid or tid == aud:
            i += 1
            continue
        # 비-video, 비-audio 토큰이 marker run 의 시작
        seg = input_ids[i:i + marker_len]
        text = tokenizer.decode(seg)
        m: Dict[str, Any] = {
            "pos": i,
            "end": i + len(seg),
            "tokens": tokenizer.convert_ids_to_tokens(seg),
            "text": text,
        }
        match = _NATURAL_RE.search(text)
        if match:
            try:
                m["seconds"] = float(match.group(1))
            except ValueError:
                m["malformed"] = True
        else:
            m["malformed"] = True
        markers.append(m)
        i += marker_len                      # 다음 marker 후보로 이동


    return markers


def _find_markers(
    input_ids: List[int],
    vision_span: Optional[Tuple[int, int]],
    spec: Dict[str, Optional[int]],
    tt_range: Optional[Tuple[int, int]],
    tti_time_format: str,
    marker_len: Optional[int],
    tokenizer,
) -> List[Dict[str, Any]]:
    if tti_time_format == "special_token" and tt_range is not None:
        return _find_markers_special_token(
            input_ids, tt_range, spec.get("tdot"), marker_len or 6, tokenizer
        )
    if tti_time_format == "natural_text" and marker_len:
        return _find_markers_natural_text(
            input_ids, vision_span, spec, marker_len, tokenizer
        )
    return []


# ---------------- Chunk layout ----------------

def _chunk_layout(
    categories: List[str],
    vision_span: Tuple[int, int],
    time_markers: List[Dict[str, Any]],
    marker_len: int,
) -> List[Dict[str, Any]]:
    """vision span 내부의 marker 위치를 chunk 시작으로 삼아 집계."""
    s, e = vision_span
    in_span = [m for m in time_markers if s <= m["pos"] < e]
    chunks: List[Dict[str, Any]] = []
    for idx, m in enumerate(in_span):
        start = m["pos"]
        end = in_span[idx + 1]["pos"] if idx + 1 < len(in_span) else e
        body_start = m["end"]
        n_vid = n_aud = n_time = n_other = 0
        for p in range(body_start, end):
            c = categories[p]
            if c == "video_pad": n_vid += 1
            elif c == "audio_pad": n_aud += 1
            elif c == "time_token": n_time += 1
            else: n_other += 1
        chunks.append({
            "idx": idx,
            "start_pos": start,
            "end_pos": end,
            "marker_len": marker_len,
            "time_marker_seconds": m.get("seconds"),
            "time_marker_tokens": m["tokens"],
            "time_marker_text": m.get("text"),
            "n_video_pad": n_vid,
            "n_audio_pad": n_aud,
            "n_stray_time": n_time,
            "n_stray_other": n_other,
        })
    return chunks


# ---------------- RoPE sample ----------------

def _rope_samples(
    position_ids_2d: torch.Tensor,   # [3, N]
    categories: List[str],
    vision_span: Optional[Tuple[int, int]],
    chunks: List[Dict[str, Any]],
    marker_len: int,
) -> Dict[str, Any]:
    t_axis = position_ids_2d[0].tolist()
    h_axis = position_ids_2d[1].tolist()
    w_axis = position_ids_2d[2].tolist()

    def pick(pos: int, kind: str) -> Dict[str, Any]:
        return {"pos": pos, "kind": kind,
                "t": t_axis[pos], "h": h_axis[pos], "w": w_axis[pos]}

    samples: List[Dict[str, Any]] = []
    if vision_span is None:
        return {"samples": samples, "t_axis_range_vision": None,
                "t_axis_monotonic_vision": None}

    s, e = vision_span
    samples.append(pick(s, "vision_start"))
    if chunks:
        for ch in chunks:
            sec = ch["time_marker_seconds"]
            tag = f"time_marker[{sec}]" if sec is not None else "time_marker[?]"
            m_s = ch["start_pos"]
            m_last = m_s + marker_len - 1
            samples.append(pick(m_s, f"{tag}_first"))
            if m_last != m_s and m_last < len(t_axis):
                samples.append(pick(m_last, f"{tag}_last"))
            vid_pos = [p for p in range(m_s + marker_len, ch["end_pos"])
                       if categories[p] == "video_pad"]
            if vid_pos:
                samples.append(pick(vid_pos[0],  f"video_pad[ch={ch['idx']}]_first"))
                if len(vid_pos) > 1:
                    samples.append(pick(vid_pos[-1], f"video_pad[ch={ch['idx']}]_last"))
            aud_pos = [p for p in range(m_s + marker_len, ch["end_pos"])
                       if categories[p] == "audio_pad"]
            if aud_pos:
                samples.append(pick(aud_pos[0],  f"audio_pad[ch={ch['idx']}]_first"))
                if len(aud_pos) > 1:
                    samples.append(pick(aud_pos[-1], f"audio_pad[ch={ch['idx']}]_last"))
    else:
        # off 모드: 청크 분리 없음 — vision span 의 video_pad/audio_pad 를 통째로
        vid_pos = [p for p in range(s + 1, e) if categories[p] == "video_pad"]
        aud_pos = [p for p in range(s + 1, e) if categories[p] == "audio_pad"]
        if vid_pos:
            samples.append(pick(vid_pos[0],  "video_pad_first"))
            if len(vid_pos) > 1:
                samples.append(pick(vid_pos[-1], "video_pad_last"))
        if aud_pos:
            samples.append(pick(aud_pos[0],  "audio_pad_first"))
            if len(aud_pos) > 1:
                samples.append(pick(aud_pos[-1], "audio_pad_last"))
    samples.append(pick(e, "vision_end"))

    t_span = t_axis[s:e + 1]
    mono = all(t_span[i] <= t_span[i + 1] for i in range(len(t_span) - 1))
    return {
        "samples": samples,
        "t_axis_range_vision": [min(t_span), max(t_span)],
        "t_axis_monotonic_vision": mono,
    }


# ---------------- Validation ----------------

def _detect_mode(token_counts: Dict[str, int],
                 grid_thw_video: Optional[List[List[int]]],
                 tti_time_format: str) -> str:
    has_vid = token_counts.get("video_pad", 0) > 0
    has_aud_seq = token_counts.get("audio_pad", 0) > 0
    if has_vid and has_aud_seq and tti_time_format != "off":
        return f"tti_{tti_time_format}"   # tti_special_token | tti_natural_text
    if has_vid and has_aud_seq:
        return "av_baseline"               # off 모드 + video+audio
    if has_vid:
        return "video_only"
    if has_aud_seq:
        return "audio_only"
    if grid_thw_video is None and not has_vid and not has_aud_seq:
        return "text_only"
    return "mixed"


def _validate(
    token_counts: Dict[str, int],
    chunks: List[Dict[str, Any]],
    grid_thw_video: Optional[List[List[int]]],
    merge_size: int,
    audio_lengths: Optional[List[int]],
    rope_info: Dict[str, Any],
    mode: str,
    marker_len: int,
) -> Dict[str, Any]:
    """mode 에 해당하는 불변식만 체크. 부적용 필드는 'N/A (reason)' 문자열."""
    v: Dict[str, Any] = {"mode": mode}

    # ---- TTI 전용 (special_token / natural_text) ----
    if mode.startswith("tti_") and grid_thw_video:
        T = sum(g[0] for g in grid_thw_video)
        v["num_chunks_eq_T"] = len(chunks) == T
        if mode == "tti_special_token":
            v["time_tokens_eq_M_times_T"] = (
                token_counts.get("time_token", 0) == marker_len * T
            )
        else:
            # natural_text: 마커 토큰은 'text' 로 분류되므로 ID 카운트 대신 청크별 길이로 검증
            v["natural_marker_chunks_consistent"] = all(
                ch.get("marker_len") == marker_len for ch in chunks
            )
    else:
        v["num_chunks_eq_T"] = "N/A (not tti mode)"

    # ---- Video 있는 모든 모드 ----
    if grid_thw_video:
        per_chunk_vid = (grid_thw_video[0][1] * grid_thw_video[0][2]) // (merge_size ** 2)
        v["expected_per_chunk_video_pad"] = per_chunk_vid
        T = sum(g[0] for g in grid_thw_video)
        if chunks:
            v["video_pad_per_chunk_consistent"] = all(
                ch["n_video_pad"] == per_chunk_vid for ch in chunks
            )
        else:
            v["video_pad_total_eq_T_times_per_chunk"] = (
                token_counts.get("video_pad", 0) == T * per_chunk_vid
            )

    # ---- Audio (sequence 안에 audio_pad 가 실제로 있는 경우만) ----
    if audio_lengths is not None and token_counts.get("audio_pad", 0) > 0:
        v["audio_pad_sum_matches"] = (
            token_counts.get("audio_pad", 0) == sum(audio_lengths)
        )
    elif audio_lengths is not None:
        v["audio_pad_sum_matches"] = (
            f"N/A (audio loaded len={sum(audio_lengths)} but 0 audio_pad in sequence)"
        )

    # ---- RoPE (vision span 이 있는 경우만) ----
    if rope_info.get("t_axis_monotonic_vision") is not None:
        v["rope_t_monotonic_vision"] = rope_info["t_axis_monotonic_vision"]
    return v


# ---------------- TXT 렌더 ----------------

def _render_txt(rec: Dict[str, Any]) -> str:
    L: List[str] = []
    meta = rec["video_meta"]
    cfg = rec["config"]
    L.append(f"=== {rec['sample_tag']}  (duration={meta.get('duration_sec')}s) ===")
    L.append(f"video: {meta.get('video_path')}")
    L.append(f"audio: {meta.get('audio_path')}  use_audio={meta.get('use_audio')}")
    L.append(f"tti_time_format: {cfg.get('tti_time_format')}  "
             f"marker_len={cfg.get('time_marker_token_len')}")
    L.append(f"config: base_interval={cfg['base_interval']}  "
             f"video_frames=[{cfg['video_min_frames']},{cfg['video_max_frames']}]  "
             f"pixels=[{cfg['min_pixels']}..{cfg['max_pixels']}]  "
             f"merge={cfg['merge_size']}")
    tc = rec["token_counts"]
    L.append(f"seq_len={tc['total']}  text={tc.get('text',0)}  "
             f"time={tc.get('time_token',0)}  "
             f"video_pad={tc.get('video_pad',0)}  "
             f"audio_pad={tc.get('audio_pad',0)}  "
             f"vis_start/end={tc.get('vision_start',0)}/{tc.get('vision_end',0)}")
    L.append("")
    L.append("layout:")
    chunks = rec["chunks"]
    if chunks:
        shown = chunks[:20]
        for ch in shown:
            sec = ch["time_marker_seconds"]
            tag = f"t={sec:>6}s" if sec is not None else "t=  ??  "
            L.append(f"  chunk {ch['idx']:>3}  {tag}  "
                     f"VID×{ch['n_video_pad']:<4} AUD×{ch['n_audio_pad']:<4} "
                     f"[{ch['start_pos']}..{ch['end_pos']})  "
                     f"marker={ch.get('time_marker_text') or ''}")
        if len(chunks) > 20:
            L.append(f"  ... ({len(chunks) - 20} more chunks)")
    else:
        if rec.get("validation", {}).get("mode") == "av_baseline":
            L.append("  (off 모드: 청크 마커 없음 — vision span 내 video_pad+audio_pad 만)")
        else:
            L.append("  (no vision span — pure text/audio path)")
    L.append("")
    secs = [ch["time_marker_seconds"] for ch in chunks]
    L.append(f"time markers ({len(secs)}): {secs[:30]}"
             + (" ..." if len(secs) > 30 else ""))
    L.append("")
    L.append("RoPE samples (t, h, w):")
    rope = rec["rope"]
    for r in rope["samples"][:40]:
        L.append(f"  pos={r['pos']:>5}  {r['kind']:<36}  "
                 f"t={r['t']:>5} h={r['h']:>4} w={r['w']:>4}")
    if len(rope["samples"]) > 40:
        L.append(f"  ... ({len(rope['samples']) - 40} more)")
    L.append(f"t_axis (vision): range={rope['t_axis_range_vision']}  "
             f"monotonic={rope['t_axis_monotonic_vision']}")
    L.append("")
    L.append("validation:")
    for k, val in rec["validation"].items():
        if isinstance(val, bool):
            mark = "OK" if val else "FAIL"
        elif isinstance(val, str) and val.startswith("N/A"):
            mark = " - "
        else:
            mark = "   "
        L.append(f"  {mark}  {k}: {val}")
    return "\n".join(L) + "\n"


# ---------------- 진입점 ----------------

def dump_sample(
    out_dir: str,
    sample_idx: int,
    sample_tag: str,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    tokenizer,
    time_token_id_range: Optional[Tuple[int, int]],
    video_grid_thw: Optional[torch.Tensor],
    second_per_grid_ts: Optional[List],
    audio_lengths: Optional[List[int]],
    merge_size: int,
    data_args,
    source: Dict[str, Any],
    tti_time_format: str = "off",
    time_marker_token_len: Optional[int] = None,
) -> str:
    """단일 샘플 덤프. 출력 파일 base path 반환."""
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    # shape 정규화
    if input_ids.dim() == 2:
        input_ids = input_ids[0]
    if position_ids.dim() == 3:
        position_ids = position_ids[:, 0, :]
    ids_list: List[int] = input_ids.tolist()

    spec = _special_ids(tokenizer)
    marker_len = time_marker_token_len or 0

    # ---- categories ----
    categories = [_classify(t, spec, time_token_id_range) for t in ids_list]
    token_counts: Dict[str, int] = {"total": len(ids_list)}
    for c in categories:
        token_counts[c] = token_counts.get(c, 0) + 1

    # ---- vision span ----
    vision_span = _find_vision_span(categories)

    # ---- time markers (mode 별) ----
    time_markers = _find_markers(
        ids_list, vision_span, spec, time_token_id_range,
        tti_time_format, marker_len, tokenizer,
    )

    # ---- chunk layout ----
    chunks = (_chunk_layout(categories, vision_span, time_markers, marker_len)
              if vision_span and time_markers else [])

    rope_info = _rope_samples(position_ids, categories, vision_span, chunks, marker_len)

    grid_list: Optional[List[List[int]]] = None
    if video_grid_thw is not None and hasattr(video_grid_thw, "tolist"):
        grid_list = video_grid_thw.tolist()
    spgt_list = None
    if second_per_grid_ts is not None:
        spgt_list = [list(x) if hasattr(x, "__iter__") else x
                     for x in second_per_grid_ts]

    mode = _detect_mode(token_counts, grid_list, tti_time_format)
    validation = _validate(
        token_counts, chunks, grid_list, merge_size,
        list(audio_lengths) if audio_lengths is not None else None,
        rope_info, mode, marker_len,
    )

    record = {
        "sample_idx": sample_idx,
        "sample_tag": sample_tag,
        "config": {
            "tti_time_format": tti_time_format,
            "time_marker_token_len": time_marker_token_len,
            "base_interval": float(getattr(data_args, "base_interval", 0) or 0),
            "video_min_frames": getattr(data_args, "video_min_frames", None),
            "video_max_frames": getattr(data_args, "video_max_frames", None),
            "max_pixels": getattr(data_args, "max_pixels", None),
            "min_pixels": getattr(data_args, "min_pixels", None),
            "video_max_frame_pixels": getattr(data_args, "video_max_frame_pixels", None),
            "video_min_frame_pixels": getattr(data_args, "video_min_frame_pixels", None),
            "merge_size": merge_size,
            "time_token_id_range": list(time_token_id_range) if time_token_id_range else None,
        },
        "video_meta": {
            "video_path": source.get("video"),
            "audio_path": source.get("audio"),
            "use_audio": source.get("use_audio", False),
            "duration_sec": source.get("_duration_sec"),
            "debug_tag": source.get("_debug_tag"),
            "grid_thw_video": grid_list,
            "second_per_grid_ts": spgt_list,
            "audio_lengths": list(audio_lengths) if audio_lengths is not None else None,
        },
        "token_counts": token_counts,
        "interleaving": {
            "num_chunks": len(chunks),
            "per_chunk_marker_tokens": marker_len if chunks else 0,
            "per_chunk_video_pad": chunks[0]["n_video_pad"] if chunks else 0,
            "per_chunk_audio_pad_avg": (
                sum(c["n_audio_pad"] for c in chunks) / len(chunks) if chunks else 0
            ),
            "per_chunk_audio_pad_min": min((c["n_audio_pad"] for c in chunks), default=0),
            "per_chunk_audio_pad_max": max((c["n_audio_pad"] for c in chunks), default=0),
            "chunk_offsets": [c["start_pos"] for c in chunks],
        },
        "chunks": chunks,
        "time_markers": time_markers,
        "rope": rope_info,
        "validation": validation,
    }

    tag = sample_tag or f"sample{sample_idx:03d}"
    safe_tag = "".join(c if (c.isalnum() or c in "._-") else "_" for c in tag)
    base = out_p / f"{sample_idx:03d}_{safe_tag}"
    with open(base.with_suffix(".json"), "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    with open(base.with_suffix(".txt"), "w") as f:
        f.write(_render_txt(record))
    return str(base)
