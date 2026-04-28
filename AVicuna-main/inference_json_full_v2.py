"""
Experiment F: JSON prompt inference (full 2167 videos)
Prompt from AVicuna paper Figure 7 qualitative results.
"""
import os, json, torch, numpy as np, re, sys, time
sys.path.insert(0, '.')
from tqdm import tqdm
from avicuna.model.builder import load_pretrained_model
from avicuna.inference import inference
from easydict import EasyDict as edict


def load_feat(vid):
    v = torch.tensor(np.load(f'data/unav100/features/video_clip/{vid}.npy'), dtype=torch.bfloat16).cuda()
    a = torch.tensor(np.load(f'data/unav100/features/audio_clap/{vid}.npy'), dtype=torch.bfloat16).cuda()
    idx = torch.linspace(0, v.shape[0]-1, 75).long()
    v = v[idx]
    tmp = a.shape[0]
    rf, rm = 25 // tmp, 25 % tmp
    a = torch.cat([a[i].unsqueeze(0).repeat(rf + (1 if i < rm else 0), 1) for i in range(tmp)], dim=0)
    return [v.unsqueeze(0), a.unsqueeze(0)]


def parse_json_output(text, duration):
    """Parse JSON format output and convert 0-99 to actual seconds"""
    events = []

    json_start = text.find('[')
    if json_start == -1:
        matches = re.findall(r'[Ff]rom\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)', text)
        labels = re.findall(r'"event"\s*:\s*"([^"]+)"', text)
        for i, (s, e) in enumerate(matches):
            label = labels[i] if i < len(labels) else "unknown"
            start = float(s) / 100.0 * duration
            end = float(e) / 100.0 * duration
            if end >= start:
                events.append({"mapped_label": label, "start": start, "end": end, "score": 1.0})
        return events

    json_str = text[json_start:]

    if not json_str.rstrip().endswith(']'):
        last_brace = json_str.rfind('}')
        if last_brace != -1:
            json_str = json_str[:last_brace+1] + ']'

    try:
        data = json.loads(json_str)
        for item in data:
            event = item.get('event', '')
            ts = item.get('timestamps', '')
            matches = re.findall(r'(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)', ts)
            if matches:
                s, e = float(matches[0][0]), float(matches[0][1])
                start = s / 100.0 * duration
                end = e / 100.0 * duration
                if end >= start and event:
                    events.append({"mapped_label": event, "start": start, "end": end, "score": 1.0})
    except json.JSONDecodeError:
        event_matches = re.findall(r'"event"\s*:\s*"([^"]+)"', json_str)
        time_matches = re.findall(r'from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)', json_str, re.IGNORECASE)
        for i in range(min(len(event_matches), len(time_matches))):
            s, e = float(time_matches[i][0]), float(time_matches[i][1])
            start = s / 100.0 * duration
            end = e / 100.0 * duration
            if end >= start:
                events.append({"mapped_label": event_matches[i], "start": start, "end": end, "score": 1.0})

    return events


# ---- Main ----
args = edict({
    'clip_path': 'checkpoints/clip/ViT-L-14.pt',
    'pretrain_mm_mlp_adapter': 'checkpoints/avicuna-vicuna-v1-5-7b-stage1/mm_projector.bin',
    'pretrain_mm_mlp_adapter_a': 'checkpoints/avicuna-vicuna-v1-5-7b-stage2/mm_projector_a.bin',
    'stage3': 'checkpoints/avicuna-vicuna-v1-5-7b-stage3',
    'stage4': 'checkpoints/avicuna-vicuna-v1-5-7b-stage4',
    'model_base': 'lmsys/vicuna-7b-v1.5',
})

print("Loading model...")
tokenizer, model, _ = load_pretrained_model(args, args.stage3, args.stage4)
model = model.cuda().to(torch.bfloat16).eval()

with open('data/annotations/unav100_annotations.json') as f:
    ann = json.load(f)
test_ids = [vid for vid, info in ann['database'].items() if info.get('subset') == 'test']
durations = {vid: info['duration'] for vid, info in ann['database'].items()}

print(f"Test: {len(test_ids)} videos")

QUERY = '<video>\n Please find all audio-visual events from the video. Your answer should be in JSON format: {"event": <event>, "timestamps": "from <start_time> to <end_time>"}'

results = []
total_events = 0
parse_failures = 0
t0 = time.time()

for vi, vid in enumerate(tqdm(test_ids, desc="Videos")):
    vp = f'data/unav100/features/video_clip/{vid}.npy'
    ap = f'data/unav100/features/audio_clap/{vid}.npy'
    if not os.path.exists(vp) or not os.path.exists(ap):
        continue

    features = load_feat(vid)
    dur = durations.get(vid, 0)
    model.aud_mask = None

    out = inference(model, features, QUERY, tokenizer)
    events = parse_json_output(out, dur)

    if not events:
        parse_failures += 1

    total_events += len(events)
    results.append({
        "video_id": vid,
        "events": events,
        "raw_output": out,
    })

    if (vi + 1) % 500 == 0:
        with open('output/json_prompt_full.json', 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        elapsed = time.time() - t0
        print(f"\n  [{vi+1}/{len(test_ids)}] Saved. {elapsed/60:.1f}min, {total_events} events")

elapsed = time.time() - t0
print(f"\nDone! {len(results)} videos, {total_events} events, {elapsed/60:.1f} min")
print(f"Parse failures: {parse_failures}")
print(f"Avg events/video: {total_events/len(results):.1f}")

with open('output/json_prompt_full.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
