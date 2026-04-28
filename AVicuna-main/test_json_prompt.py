"""Test JSON format prompt on 3 videos"""
import torch, numpy as np, sys
sys.path.insert(0, '.')
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

test_vids = ['D05-MrPXqjw', '8WSesSOfBrE', '20Nlj7Cz4E0']

for vid in test_vids:
    v = torch.tensor(np.load(f'data/unav100/features/video_clip/{vid}.npy'), dtype=torch.bfloat16).cuda()
    a = torch.tensor(np.load(f'data/unav100/features/audio_clap/{vid}.npy'), dtype=torch.bfloat16).cuda()
    idx = torch.linspace(0, v.shape[0]-1, 75).long()
    v = v[idx]
    tmp = a.shape[0]
    rf, rm = 25 // tmp, 25 % tmp
    a = torch.cat([a[i].unsqueeze(0).repeat(rf + (1 if i < rm else 0), 1) for i in range(tmp)], dim=0)
    features = [v.unsqueeze(0), a.unsqueeze(0)]
    model.aud_mask = None

    q = '<video>\n Please find all audio-visual events from the video. Your answer should be in JSON format: {"event": <event>, "timestamps": "from <start_time> to <end_time>"}'
    out = inference(model, features, q, tokenizer)
    print(f'[{vid}]: {out[:500]}')
    print('---')
