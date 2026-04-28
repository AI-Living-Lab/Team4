"""
Experiment C-v2: Label-query with NO filtering + confidence + NMS + short filter
10 video test
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
            max_new_tokens=128,
            use_cache=True,
            output_scores=True,
            return_dict_in_generate=True)

    # Get token scores for confidence
    input_token_len = input_ids.shape[1]
    generated_ids = outputs.sequences[0, input_token_len:]
    scores = outputs.scores  # tuple of (vocab_size,) per token

    # Compute mean log probability as confidence
    log_probs = []
    for i, score in enumerate(scores):
        if i >= len(generated_ids):
            break
        token_id = generated_ids[i].item()
        probs = torch.softmax(score[0], dim=-1)
        log_prob = torch.log(probs[token_id] + 1e-10).item()
        log_probs.append(log_prob)

    confidence = np.exp(np.mean(log_probs)) if log_probs else 0.0

    # Decode text
    text = tokenizer.batch_decode([generated_ids], skip_special_tokens=True)[0]
    text = text.strip()
    if text.endswith(stop_str):
        text = text[:-len(stop_str)]
    text = text.strip()

    return text, confidence


def nms(events, iou_threshold=0.3):
    """Non-Maximum Suppression for same-label overlapping events"""
    if len(events) <= 1:
        return events

    # Group by label
    from collections import defaultdict
    by_label = defaultdict(list)
    for ev in events:
        by_label[ev['mapped_label']].append(ev)

    result = []
    for label, evs in by_label.items():
        # Sort by score descending
        evs = sorted(evs, key=lambda x: x['score'], reverse=True)
        keep = []
        while evs:
            best = evs.pop(0)
            keep.append(best)
            remaining = []
            for ev in evs:
                # Compute IoU
                inter = max(0, min(best['end'], ev['end']) - max(best['start'], ev['start']))
                union = max(best['end'], ev['end']) - min(best['start'], ev['start'])
                iou = inter / union if union > 0 else 0
                if iou < iou_threshold:
                    remaining.append(ev)
            evs = remaining
        result.extend(keep)

    return result


def load_feat(vid):
    v = torch.tensor(np.load(f'data/unav100/features/video_clip/{vid}.npy'), dtype=torch.bfloat16).cuda()
    a = torch.tensor(np.load(f'data/unav100/features/audio_clap/{vid}.npy'), dtype=torch.bfloat16).cuda()
    idx = torch.linspace(0, v.shape[0]-1, 75).long()
    v = v[idx]
    tmp = a.shape[0]
    rf, rm = 25 // tmp, 25 % tmp
    a = torch.cat([a[i].unsqueeze(0).repeat(rf + (1 if i < rm else 0), 1) for i in range(tmp)], dim=0)
    return [v.unsqueeze(0), a.unsqueeze(0)]


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
test_ids = [vid for vid, info in ann['database'].items() if info.get('subset') == 'test'][:10]
durations = {vid: info['duration'] for vid, info in ann['database'].items()}

with open('data/labels/unav100_class_labels.txt') as f:
    labels = [l.strip() for l in f if l.strip()]

print(f"Test: {len(test_ids)} videos x {len(labels)} labels")

MIN_SEGMENT_SEC = 0.5  # filter segments shorter than this

results = []
total_no = 0
total_yes = 0
total_filtered = 0
t0 = time.time()

for vi, vid in enumerate(tqdm(test_ids, desc="Videos")):
    vp = f'data/unav100/features/video_clip/{vid}.npy'
    ap = f'data/unav100/features/audio_clap/{vid}.npy'
    if not os.path.exists(vp) or not os.path.exists(ap):
        continue

    features = load_feat(vid)
    dur = durations.get(vid, 0)
    events = []

    for label in labels:
        model.aud_mask = None
        q = f'<video>\n For the event "{label}", provide all time intervals where it occurs in the video. If the event is not present, answer NO.'
        out, confidence = inference_with_score(model, features, q, tokenizer)

        out_upper = out.strip().upper()
        if out_upper.startswith('NO') or 'NOT PRESENT' in out_upper or 'CANNOT' in out_upper:
            total_no += 1
            continue

        matches = re.findall(r'from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)', out, re.IGNORECASE)
        if matches:
            total_yes += 1
            for s, e in matches:
                real_start = float(s) / 100.0 * dur
                real_end = float(e) / 100.0 * dur
                seg_len = real_end - real_start

                # Filter short segments
                if seg_len < MIN_SEGMENT_SEC:
                    total_filtered += 1
                    continue

                events.append({
                    "mapped_label": label,
                    "start": real_start,
                    "end": real_end,
                    "score": confidence,
                })
        else:
            total_no += 1

    # Apply NMS
    before_nms = len(events)
    events = nms(events, iou_threshold=0.3)
    after_nms = len(events)

    results.append({"video_id": vid, "events": events})
    print(f"  {vid}: {before_nms} -> {after_nms} events (NMS removed {before_nms - after_nms})")

elapsed = time.time() - t0
total_ev = sum(len(r['events']) for r in results)
print(f"\nDone! {len(results)} videos, {total_ev} events, {elapsed/60:.1f} min")
print(f"NO: {total_no}, YES: {total_yes}, Short filtered: {total_filtered}")
if total_no + total_yes > 0:
    print(f"NO ratio: {total_no/(total_no+total_yes)*100:.1f}%")

# Save
with open('output/label_pred_v3_10.json', 'w') as f:
    json.dump(results, f, indent=2)

# Show score distribution
all_scores = [ev['score'] for r in results for ev in r['events']]
if all_scores:
    print(f"\nConfidence scores: min={min(all_scores):.4f}, max={max(all_scores):.4f}, mean={np.mean(all_scores):.4f}")
