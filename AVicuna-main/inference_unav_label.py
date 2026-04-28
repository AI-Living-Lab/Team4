"""
Step 1 (v2): AVicuna inference - per label query
Each video x each label = separate query
"""
import os, json, argparse, torch, numpy as np, re
from tqdm import tqdm
from avicuna.model.builder import load_pretrained_model
from avicuna.inference import inference
from easydict import EasyDict as edict

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video_feat_dir", type=str, required=True)
    p.add_argument("--audio_feat_dir", type=str, required=True)
    p.add_argument("--annotation", type=str, required=True)
    p.add_argument("--labels", type=str, required=True)
    p.add_argument("--output", type=str, default="output/label_predictions.json")
    p.add_argument("--clip_path", type=str, default="checkpoints/clip/ViT-L-14.pt")
    p.add_argument("--pretrain_mm_mlp_adapter", type=str, default="checkpoints/avicuna-vicuna-v1-5-7b-stage1/mm_projector.bin")
    p.add_argument("--pretrain_mm_mlp_adapter_a", type=str, default="checkpoints/avicuna-vicuna-v1-5-7b-stage2/mm_projector_a.bin")
    p.add_argument("--stage3", type=str, default="checkpoints/avicuna-vicuna-v1-5-7b-stage3")
    p.add_argument("--stage4", type=str, default="checkpoints/avicuna-vicuna-v1-5-7b-stage4")
    p.add_argument("--model_base", type=str, default="lmsys/vicuna-7b-v1.5")
    p.add_argument("--av_ratio", type=float, default=0.25)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()

def load_features(vpath, apath, device, av_ratio=0.25):
    v = torch.tensor(np.load(vpath), dtype=torch.bfloat16).to(device)
    a = torch.tensor(np.load(apath), dtype=torch.bfloat16).to(device)
    n_video = int(100 * (1 - av_ratio))
    n_audio = int(100 * av_ratio)
    indices = torch.linspace(0, v.shape[0]-1, n_video).long()
    v = v[indices]
    tmp = a.shape[0]
    rf = n_audio // tmp
    rm = n_audio % tmp
    a = torch.cat([a[i].unsqueeze(0).repeat(rf + (1 if i < rm else 0), 1) for i in range(tmp)], dim=0)
    return [v.unsqueeze(0), a.unsqueeze(0)]

def parse_time_from_response(text):
    """Extract 'from XX to YY' from response"""
    matches = re.findall(r'from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    return [(float(s), float(e)) for s, e in matches]

def main():
    args = parse_args()
    
    # Load model
    print("Loading model...")
    tokenizer, model, _ = load_pretrained_model(args, args.stage3, args.stage4)
    model = model.to(args.device).to(torch.bfloat16).eval()
    
    # Load test video IDs
    with open(args.annotation, "r") as f:
        ann = json.load(f)
    test_ids = [vid for vid, info in ann["database"].items() if info.get("subset") == "test"]
    durations = {vid: info["duration"] for vid, info in ann["database"].items()}
    print(f"Test videos: {len(test_ids)}")
    
    # Load labels
    with open(args.labels, "r") as f:
        labels = [l.strip() for l in f if l.strip()]
    print(f"Labels: {len(labels)}")
    print(f"Total queries: {len(test_ids)} x {len(labels)} = {len(test_ids)*len(labels)}")
    
    results = []
    query_count = 0
    
    for video_id in tqdm(test_ids, desc="Videos"):
        vpath = os.path.join(args.video_feat_dir, f"{video_id}.npy")
        apath = os.path.join(args.audio_feat_dir, f"{video_id}.npy")
        if not os.path.exists(vpath) or not os.path.exists(apath):
            continue
        
        features = load_features(vpath, apath, args.device, args.av_ratio)
        duration = durations.get(video_id, 0)
        video_events = []
        
        for label in labels:
            model.aud_mask = None
            q = f"<video>\n At which intervals in the video can we identify {label}, either by watching or listening?"
            out = inference(model, features, q, tokenizer)
            query_count += 1
            
            # Parse time intervals
            times = parse_time_from_response(out)
            for start, end in times:
                # Scale 0-99 to actual duration
                real_start = start / 100.0 * duration
                real_end = end / 100.0 * duration
                video_events.append({
                    "mapped_label": label,
                    "start": real_start,
                    "end": real_end,
                    "score": 1.0,
                    "pred_text": label,
                    "raw_response": out,
                })
        
        results.append({"video_id": video_id, "events": video_events})
    
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    total_events = sum(len(r["events"]) for r in results)
    print(f"\nSaved {len(results)} videos, {total_events} events to {args.output}")
    print(f"Total queries: {query_count}")

if __name__ == "__main__":
    main()
