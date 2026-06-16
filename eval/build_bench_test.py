#!/usr/bin/env python3
"""Build SALMONN2+ tail-format test JSON for external benchmarks
(LongVALE / Charades-STA), reusing the exact unav100_tail item schema so the
unified eval.sh / eval_miou.py pipeline works unchanged.

Each item:
  {
    "video": <abs mp4>, "audio": <abs wav>, "use_audio": true,
    "conversations": [
       {"from":"human","value":"<video>\\n<official query> <token tail>"},
       {"from":"gpt","value":"From <tok> to <tok>."}   # per-sample GT (ref)
    ],
    "gt_label": <query/sentence>, "gt_segments": [[s,e], ...]
  }

GT is encoded into the gpt value (token format). eval reads it back as the
per-sample `ref` (priority over positional test_json matching) -> no GT
mis-alignment even when one video has many queries (Charades).

Prompts (query stem = each benchmark's official; output tail = our native
time-token format, identical to the UnAV-100 tail run):
  LongVALE  (paper 2411.19772v3 §9.2, instruction-tuned grounding):
      "During which frames does {event} occur in the video?"
  Charades  (no dataset-canonical prompt; MUSEG in-repo comparison query):
      "Please find the visual event described by a sentence in the video,
       determining its starting and ending times. Now I will give you the
       textual sentence: {sentence}"
"""
import argparse
import json
import os

TAIL = ' Output in the format of "From <tX><tX><tX><tdot><tX> to <tX><tX><tX><tdot><tX>."'

LONGVALE_Q = "During which frames does {q} occur in the video?"
CHARADES_Q = ("Please find the visual event described by a sentence in the video, "
              "determining its starting and ending times. Now I will give you the "
              "textual sentence: {q}")

# Charades-STA: ask with the EXACT prompt our model saw during RL training
# (data/train/unav100_v2.json template). {q} = the Charades sentence (event slot).
# This template already carries the answer-format, so NO extra TAIL is appended.
# Verified byte-identical to the training prompt via --verify_rl_from at build time.
RL_PROMPT = '''At what point in the video does {q} occur in terms of both video and audio?

Answer format:
"From <tX><tX><tX><tdot><tX> to <tX><tX><tX><tdot><tX>."

For multiple segments, Separate multiple segments with a period and space, like:
"From <tX><tX><tX><tdot><tX> to <tX><tX><tX><tdot><tX>. From <tX><tX><tX><tdot><tX> to <tX><tX><tX><tdot><tX>. ..."'''


def tok(t):
    """seconds -> '<tA><tB><tC><tdot><tD>'  (3 int digits + 1 decimal, 0.1 prec)."""
    t = float(t)
    if t < 0:
        t = 0.0
    if t > 999.9:
        t = 999.9
    tenths = int(round(t * 10))
    w = tenths // 10
    d = tenths % 10
    return f"<t{w // 100 % 10}><t{w // 10 % 10}><t{w % 10}><tdot><t{d}>"


def gpt_val(segs):
    parts = [f"From {tok(s)} to {tok(e)}." for s, e in segs]
    return " ".join(parts)


def make_item(video, audio, query_q, qtmpl, segs, tail=TAIL, use_audio=True):
    human = "<video>\n" + qtmpl.format(q=query_q) + tail
    return {
        "video": video,
        "audio": audio,
        "use_audio": bool(use_audio),
        "conversations": [
            {"from": "human", "value": human},
            {"from": "gpt", "value": gpt_val(segs)},
        ],
        "gt_label": query_q,
        "gt_segments": [[round(float(s), 1), round(float(e), 1)] for s, e in segs],
    }


def build_longvale(args):
    ann = json.load(open(args.ann))
    vid_dir = os.path.join(args.root, "videos")
    aud_dir = os.path.join(args.root, "audios")
    out = []
    skipped = 0
    for vid, rec in ann.items():
        vpath = os.path.join(vid_dir, vid + ".mp4")
        apath = os.path.join(aud_dir, vid + ".wav")
        if not (os.path.exists(vpath) and os.path.exists(apath)):
            skipped += 1
            continue
        sents = rec["sentences"]
        ts = rec["timestamps"]
        for s, seg in zip(sents, ts):
            out.append(make_item(vpath, apath, s.strip(), LONGVALE_Q, [seg]))
    return out, skipped


def build_charades(args):
    vid_dir = os.path.join(args.root, "Charades_v1")
    aud_dir = os.path.join(args.root, "audio")
    # prompt: RL training template (default) vs MUSEG comparison query
    if args.prompt == "rl":
        qtmpl, tail = RL_PROMPT, ""   # RL template already has the answer-format
    else:
        qtmpl, tail = CHARADES_Q, TAIL
    out = []
    skipped = 0
    for line in open(args.ann):
        line = line.strip()
        if not line:
            continue
        meta, sent = line.split("##", 1)
        vid, st, en = meta.split()
        vpath = os.path.join(vid_dir, vid + ".mp4")
        apath = os.path.join(aud_dir, vid + ".wav")
        if not (os.path.exists(vpath) and os.path.exists(apath)):
            skipped += 1
            continue
        # event slot: drop trailing period so the sentence reads like an event
        # phrase inside "...does <X> occur..." (training events have no period)
        q = sent.strip().rstrip(".").strip()
        out.append(make_item(vpath, apath, q, qtmpl,
                             [[float(st), float(en)]],
                             tail=tail, use_audio=args.use_audio))
    return out, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", choices=["longvale", "charades"])
    ap.add_argument("--root", required=True, help="dataset root dir")
    ap.add_argument("--ann", required=True, help="annotation file")
    ap.add_argument("--out", required=True, help="output json path")
    ap.add_argument("--limit", type=int, default=0, help="cap #items (sanity)")
    ap.add_argument("--prompt", choices=["rl", "museg"], default="rl",
                    help="charades prompt: rl=RL training template, museg=MUSEG query")
    ap.add_argument("--use_audio", type=lambda s: s.lower() not in ("0", "false", "no"),
                    default=True, help="per-item use_audio (true/false)")
    ap.add_argument("--verify_rl_from", default="",
                    help="path to RL train json; assert RL_PROMPT matches its template")
    args = ap.parse_args()

    if args.verify_rl_from:
        tr = json.load(open(args.verify_rl_from))[0]
        h = tr["conversations"][0]["value"]
        ev = tr["event"]
        derived = h.replace("<video>\n", "").replace(ev, "{q}")
        assert derived == RL_PROMPT, (
            "RL_PROMPT MISMATCH with training file!\n--- training ---\n"
            + repr(derived) + "\n--- RL_PROMPT ---\n" + repr(RL_PROMPT))
        print(f"[verify] RL_PROMPT == training template ✓ (event='{ev}')")

    if args.bench == "longvale":
        items, skipped = build_longvale(args)
    else:
        items, skipped = build_charades(args)

    if args.limit:
        items = items[: args.limit]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(items, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"[{args.bench}] wrote {len(items)} items -> {args.out}  (skipped {skipped} missing files)")
    if items:
        print("=== sample item ===")
        print(json.dumps(items[0], ensure_ascii=False, indent=2)[:1200])


if __name__ == "__main__":
    main()
