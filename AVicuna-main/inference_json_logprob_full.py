"""
Experiment F-logprob: JSON prompt + logprob confidence (full 2167 videos)
Same as Exp F but with model confidence instead of fixed 1.0 or CLAP sim
"""
import os, json, torch, numpy as np, re, sys, time
sys.path.insert(0, '.')
from tqdm import tqdm
from avicuna.constants import IMAGE_TOKEN_INDEX
from avicuna.conversation import conv_templates, SeparatorStyle
from avicuna.model.builder import load_pretrained_model
from avicuna.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from easydict import EasyDict as edict


def inference_with_score(model, image, query, tokenizer):
    """Inference that returns text + confidence score (mean logprob)"""
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            images=image,
            do_sample=False,
            temperature=1.0,
            num_beams=1,
            max_new_tokens=1024,
            use_cache=True,
            output_scores=True,
            return_dict_in_generate=True)

    input_token_len = input_ids.shape[1]
    generated_ids = outputs.sequences[0, input_token_len:]
    scores = outputs.scores

    log_probs = []
    for i, score in enumerate(scores):
        if i >= len(generated_ids):
            break
        token_id = generated_ids[i].item()
        probs = torch.softmax(score[0], dim=-1)
        log_prob = torch.log(probs[token_id] + 1e-10).item()
        log_probs.append(log_prob)

    confidence = np.exp(np.mean(log_probs)) if log_probs else 0.0

    text = tokenizer.batch_decode([generated_ids], skip_special_tokens=True)[0]
    text = text.strip()
    if text.endswith(stop_str):
        text = text[:-len(stop_str)]
    text = text.strip()

    return text, confidence


def load_feat(vid):
    v = torch.tensor(np.load(f'data/unav100/features/video_clip/{vid}.npy'), dtype=torch.bfloat16).cuda()
    a = torch.tensor(np.load(f'data/unav100/features/audio_clap/{vid}.npy'), dtype=torch.bfloat16).cuda()
    idx = torch.linspace(0, v.shape[0]-1, 75).long()
    v = v[idx]
    tmp = a.shape[0]
    rf, rm = 25 // tmp, 25 % tmp
    a = torch.cat([a[i].unsqueeze(0).repeat(rf + (1 if i < rm else 0), 1) for i in range(tmp)], dim=0)
    return [v.unsqueeze(0), a.unsqueeze(0)]


def parse_json_output(text, duration, confidence):
    """Parse JSON output with shared confidence score"""
    events = []

    json_start = text.find('[')
    if json_start == -1:
        event_matches = re.findall(r'"event"\s*:\s*"([^"]+)"', text)
        time_matches = re.findall(r'from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        for i in range(min(len(event_matches), len(time_matches))):
            label = event_matches[i].strip()
            s, e = float(time_matches[i][0]), float(time_matches[i][1])
            start = s / 100.0 * duration
            end = e / 100.0 * duration
            if end >= start and label:
                events.append({"mapped_label": label, "start": start, "end": end, "score": confidence})
        return events

    json_str = text[json_start:]
    if not json_str.rstrip().endswith(']'):
        last_brace = json_str.rfind('}')
        if last_brace != -1:
            json_str = json_str[:last_brace+1] + ']'

    try:
        data = json.loads(json_str)
        for item in data:
            label = item.get('event', '').strip()
            ts = item.get('timestamps', '')
            matches = re.findall(r'(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)', ts)
            if matches and label:
                s, e = float(matches[0][0]), float(matches[0][1])
                start = s / 100.0 * duration
                end = e / 100.0 * duration
                if end >= start:
                    events.append({"mapped_label": label, "start": start, "end": end, "score": confidence})
    except json.JSONDecodeError:
        event_matches = re.findall(r'"event"\s*:\s*"([^"]+)"', json_str)
        time_matches = re.findall(r'from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)', json_str, re.IGNORECASE)
        for i in range(min(len(event_matches), len(time_matches))):
            label = event_matches[i].strip()
            s, e = float(time_matches[i][0]), float(time_matches[i][1])
            start = s / 100.0 * duration
            end = e / 100.0 * duration
            if end >= start and label:
                events.append({"mapped_label": label, "start": start, "end": end, "score": confidence})

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
all_scores = []
t0 = time.time()

for vi, vid in enumerate(tqdm(test_ids, desc="Videos")):
    vp = f'data/unav100/features/video_clip/{vid}.npy'
    ap = f'data/unav100/features/audio_clap/{vid}.npy'
    if not os.path.exists(vp) or not os.path.exists(ap):
        continue

    features = load_feat(vid)
    dur = durations.get(vid, 0)
    model.aud_mask = None

    out, confidence = inference_with_score(model, features, QUERY, tokenizer)
    events = parse_json_output(out, dur, confidence)

    if not events:
        parse_failures += 1

    for ev in events:
        all_scores.append(ev['score'])

    total_events += len(events)
    results.append({
        "video_id": vid,
        "events": events,
        "raw_output": out,
    })

    if (vi + 1) % 500 == 0:
        with open('output/json_logprob_full.json', 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        elapsed = time.time() - t0
        print(f"\n  [{vi+1}/{len(test_ids)}] {elapsed/60:.1f}min, {total_events} events")

elapsed = time.time() - t0
print(f"\nDone! {len(results)} videos, {total_events} events, {elapsed/60:.1f} min")
print(f"Parse failures: {parse_failures}")
print(f"Avg events/video: {total_events/len(results):.1f}")
if all_scores:
    print(f"Confidence: min={min(all_scores):.4f}, max={max(all_scores):.4f}, mean={np.mean(all_scores):.4f}")

with open('output/json_logprob_full.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
