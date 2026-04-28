"""
2-Stage Inference (Full 2167 videos):
  Stage 1: "이 비디오에 뭐가 있어?" → label 목록 추출
  Stage 2: 추출된 label만 "언제 나와? 없으면 NO" → 시간 추출
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


def find_matching_labels(text, all_labels):
    """Stage 1 출력에서 어떤 label이 언급됐는지 찾기"""
    text_lower = text.lower()
    found = []
    for label in sorted(all_labels, key=len, reverse=True):
        if label.lower() in text_lower:
            found.append(label)
            text_lower = text_lower.replace(label.lower(), '')
    return found


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
    all_labels = [l.strip() for l in f if l.strip()]

print(f"Test: {len(test_ids)} videos, {len(all_labels)} possible labels")

results = []
total_stage1_labels = 0
total_stage2_queries = 0
total_no = 0
total_yes = 0
t0 = time.time()

for vi, vid in enumerate(tqdm(test_ids, desc="Videos")):
    vp = f'data/unav100/features/video_clip/{vid}.npy'
    ap = f'data/unav100/features/audio_clap/{vid}.npy'
    if not os.path.exists(vp) or not os.path.exists(ap):
        continue

    features = load_feat(vid)
    dur = durations.get(vid, 0)

    # ---- Stage 1: 전체 describe ----
    model.aud_mask = None
    q1 = '<video>\n Describe the events in the video with timestamps.'
    stage1_out = inference(model, features, q1, tokenizer)

    # Stage 1 출력에서 label 찾기
    detected_labels = find_matching_labels(stage1_out, all_labels)
    total_stage1_labels += len(detected_labels)

    # ---- Stage 2: 감지된 label별 세부 query ----
    events = []
    for label in detected_labels:
        model.aud_mask = None
        q2 = f'<video>\n For the event "{label}", provide all time intervals where it occurs in the video. If the event is not present, answer NO.'
        out = inference(model, features, q2, tokenizer)
        total_stage2_queries += 1

        out_upper = out.strip().upper()
        if out_upper.startswith('NO') or 'NOT PRESENT' in out_upper or 'CANNOT' in out_upper:
            total_no += 1
            continue

        matches = re.findall(r'from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)', out, re.IGNORECASE)
        if matches:
            total_yes += 1
            for s, e in matches:
                events.append({
                    "mapped_label": label,
                    "start": float(s) / 100.0 * dur,
                    "end": float(e) / 100.0 * dur,
                    "score": 1.0,
                })
        else:
            total_no += 1

    results.append({"video_id": vid, "events": events})

    # 중간 저장 (500개마다)
    if (vi + 1) % 500 == 0:
        with open('output/label_pred_2stage_full.json', 'w') as f:
            json.dump(results, f, indent=2)
        elapsed = time.time() - t0
        print(f"\n  [{vi+1}/{len(test_ids)}] Saved checkpoint. "
              f"Elapsed: {elapsed/60:.1f}min, "
              f"Avg labels/video: {total_stage1_labels/(vi+1):.1f}, "
              f"Stage2 queries: {total_stage2_queries}")

elapsed = time.time() - t0
total_ev = sum(len(r['events']) for r in results)
print(f"\nDone! {len(results)} videos, {total_ev} events, {elapsed/60:.1f} min")
print(f"Stage1 avg labels: {total_stage1_labels/len(results):.1f}")
print(f"Stage2 queries: {total_stage2_queries} (vs 100-label: {len(results)*100})")
print(f"NO: {total_no}, YES: {total_yes}")
print(f"Speed: {elapsed/len(results):.1f}s per video")

with open('output/label_pred_2stage_full.json', 'w') as f:
    json.dump(results, f, indent=2)
