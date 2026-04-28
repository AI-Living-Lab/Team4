"""Quick test: 100 videos x 100 labels with short responses"""
import os, json, torch, numpy as np, re, sys
sys.path.insert(0, '.')
from tqdm import tqdm
from avicuna.model.builder import load_pretrained_model
from avicuna.inference import inference
from easydict import EasyDict as edict

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

print(f"Test: {len(test_ids)} videos x {len(labels)} labels = {len(test_ids)*len(labels)} queries")

def load_feat(vid):
    v = torch.tensor(np.load(f'data/unav100/features/video_clip/{vid}.npy'), dtype=torch.bfloat16).cuda()
    a = torch.tensor(np.load(f'data/unav100/features/audio_clap/{vid}.npy'), dtype=torch.bfloat16).cuda()
    idx = torch.linspace(0, v.shape[0]-1, 75).long()
    v = v[idx]
    tmp = a.shape[0]
    rf, rm = 25 // tmp, 25 % tmp
    a = torch.cat([a[i].unsqueeze(0).repeat(rf + (1 if i < rm else 0), 1) for i in range(tmp)], dim=0)
    return [v.unsqueeze(0), a.unsqueeze(0)]

results = []
import time
t0 = time.time()

for vi, vid in enumerate(tqdm(test_ids, desc="Videos")):
    vp = f'data/unav100/features/video_clip/{vid}.npy'
    ap = f'data/unav100/features/audio_clap/{vid}.npy'
    if not os.path.exists(vp) or not os.path.exists(ap): continue
    
    features = load_feat(vid)
    dur = durations.get(vid, 0)
    events = []
    
    for label in labels:
        model.aud_mask = None
        q = f"<video>\n For the event '{label}', provide all time intervals where it occurs in the video. If the event is not present, answer NO."
        out = inference(model, features, q, tokenizer)
        matches = re.findall(r'from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)', out, re.IGNORECASE)
        for s, e in matches:
            events.append({"mapped_label": label, "start": float(s)/100*dur, "end": float(e)/100*dur, "score": 1.0})
    
    results.append({"video_id": vid, "events": events})
    
    if vi == 0:
        elapsed = time.time() - t0
        print(f"\n1 video = {elapsed:.0f}s, estimated total: {elapsed*len(test_ids)/3600:.1f}h")

with open('output/label_pred_100.json', 'w') as f:
    json.dump(results, f, indent=2)

total_ev = sum(len(r['events']) for r in results)
elapsed = time.time() - t0
print(f"\nDone! {len(results)} videos, {total_ev} events, {elapsed/60:.1f} min")
