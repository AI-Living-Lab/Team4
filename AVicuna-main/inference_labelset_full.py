"""
Experiment G: JSON prompt with label set constraint (full 2167 videos)
- Provides UnAV-100 label set in prompt to constrain event labels
- AVicuna-style time format (0-99 relative integers)
- Confidence score: 1.0 fixed
"""
import os, json, torch, numpy as np, re, sys, time
sys.path.insert(0, '.')
from tqdm import tqdm
from avicuna.model.builder import load_pretrained_model
from avicuna.inference import inference
from easydict import EasyDict as edict

LABEL_LIST = """airplane flyby, ambulance siren, auto racing, baby babbling, baby crying, baby laughter, basketball bounce, beat boxing, bird chirping, bull bellowing, car passing by, cat meowing, chainsawing trees, child singing, church bell ringing, dog barking, dog howling, driving buses, driving motorcycle, engine knocking, female singing, fire truck siren, fireworks banging, frog croaking, hair dryer drying, hammering nails, helicopter, horse clip-clop, kid speaking, lawn mowing, lions roaring, machine gun shooting, male singing, man speaking, orchestra, people battle cry, people burping, people cheering, people clapping, people coughing, people crowd, people eating, people laughing, people nose blowing, people running, people shouting, people slapping, people slurping, people sneezing, people sobbing, people whispering, people whistling, playing accordion, playing acoustic guitar, playing badminton, playing banjo, playing base guitar, playing cello, playing clarinet, playing cornet, playing drum kit, playing electronic organ, playing erhu, playing flute, playing harmonica, playing harp, playing piano, playing saxophone, playing snare drum, playing synthesizer, playing tabla, playing table tennis, playing tennis, playing trombone, playing trumpet, playing ukulele, playing violin, playing volleyball, playing zither, police car siren, raining, rope skipping, sailing, sea waves, sheep bleating, singing choir, skateboarding, skidding, striking bowling, tap dancing, telephone bell ringing, thunder, train horning, train wheels squealing, typing on computer keyboard, vacuum cleaner cleaning floors, vehicle honking, water burbling, wind noise, woman speaking"""

QUERY = '<video>\n You are an audio-visual event localization model.\nGiven the video and audio, localize all audio-visual events.\n\nYou MUST choose event labels from the following list:\n' + LABEL_LIST + '\n\nOutput a JSON list. Each element must have:\n  "event": one label from the list above\n  "timestamps": "from <start> to <end>" where start and end are integers 0-99 representing percentage of video duration\n\nExample: [{"event": "dog barking", "timestamps": "from 10 to 45"}]'


def load_feat(vid):
    v = torch.tensor(np.load(f'data/unav100/features/video_clip/{vid}.npy'), dtype=torch.bfloat16).cuda()
    a = torch.tensor(np.load(f'data/unav100/features/audio_clap/{vid}.npy'), dtype=torch.bfloat16).cuda()
    idx = torch.linspace(0, v.shape[0]-1, 75).long()
    v = v[idx]
    tmp = a.shape[0]
    rf, rm = 25 // tmp, 25 % tmp
    a = torch.cat([a[i].unsqueeze(0).repeat(rf + (1 if i < rm else 0), 1) for i in range(tmp)], dim=0)
    return [v.unsqueeze(0), a.unsqueeze(0)]


def parse_json_output(text, duration, valid_labels):
    """Parse JSON output, validate labels against UnAV-100 set"""
    events = []

    json_start = text.find('[')
    if json_start == -1:
        # Fallback: regex
        event_matches = re.findall(r'"event"\s*:\s*"([^"]+)"', text)
        time_matches = re.findall(r'from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        for i in range(min(len(event_matches), len(time_matches))):
            label = event_matches[i].strip()
            s, e = float(time_matches[i][0]), float(time_matches[i][1])
            start = s / 100.0 * duration
            end = e / 100.0 * duration
            if end >= start and label:
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
            label = item.get('event', '').strip()
            ts = item.get('timestamps', '')
            matches = re.findall(r'(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)', ts)
            if matches and label:
                s, e = float(matches[0][0]), float(matches[0][1])
                start = s / 100.0 * duration
                end = e / 100.0 * duration
                if end >= start:
                    events.append({"mapped_label": label, "start": start, "end": end, "score": 1.0})
    except json.JSONDecodeError:
        event_matches = re.findall(r'"event"\s*:\s*"([^"]+)"', json_str)
        time_matches = re.findall(r'from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)', json_str, re.IGNORECASE)
        for i in range(min(len(event_matches), len(time_matches))):
            label = event_matches[i].strip()
            s, e = float(time_matches[i][0]), float(time_matches[i][1])
            start = s / 100.0 * duration
            end = e / 100.0 * duration
            if end >= start and label:
                events.append({"mapped_label": label, "start": start, "end": end, "score": 1.0})

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

with open('data/labels/unav100_class_labels.txt') as f:
    valid_labels = set(l.strip() for l in f if l.strip())

print(f"Test: {len(test_ids)} videos")
print(f"Query length: {len(QUERY)} chars")

results = []
total_events = 0
parse_failures = 0
label_match = 0
label_mismatch = 0
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
    events = parse_json_output(out, dur, valid_labels)

    if not events:
        parse_failures += 1

    # Check label validity
    for ev in events:
        if ev['mapped_label'] in valid_labels:
            label_match += 1
        else:
            label_mismatch += 1

    total_events += len(events)
    results.append({
        "video_id": vid,
        "events": events,
        "raw_output": out,
    })

    if (vi + 1) % 500 == 0:
        with open('output/json_labelset_full.json', 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        elapsed = time.time() - t0
        print(f"\n  [{vi+1}/{len(test_ids)}] {elapsed/60:.1f}min, {total_events} events, "
              f"label match: {label_match}, mismatch: {label_mismatch}")

elapsed = time.time() - t0
print(f"\nDone! {len(results)} videos, {total_events} events, {elapsed/60:.1f} min")
print(f"Parse failures: {parse_failures}")
print(f"Avg events/video: {total_events/len(results):.1f}")
print(f"Label match: {label_match} ({label_match/(label_match+label_mismatch)*100:.1f}%)")
print(f"Label mismatch: {label_mismatch}")

with open('output/json_labelset_full.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
