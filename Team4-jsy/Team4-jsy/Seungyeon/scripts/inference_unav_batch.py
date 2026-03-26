import os
import json
import argparse
from tqdm import tqdm

from avicuna.model.builder import load_pretrained_model
from avicuna.mm_utils import get_model_name_from_path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--audio_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="output/raw_predictions.json")

    parser.add_argument("--clip_path", type=str, default="checkpoints/clip/ViT-L-14.pt")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default="checkpoints/avicuna-vicuna-v1-5-7b-stage1/mm_projector.bin")
    parser.add_argument("--pretrain_mm_mlp_adapter_a", type=str, default="checkpoints/avicuna-vicuna-v1-5-7b-stage2/mm_projector_a.bin")
    parser.add_argument("--stage3", type=str, default="checkpoints/avicuna-vicuna-v1-5-7b-stage3")
    parser.add_argument("--stage4", type=str, default="checkpoints/avicuna-vicuna-v1-5-7b-stage4")
    parser.add_argument("--model_base", type=str, default="lmsys/vicuna-7b-v1.5")

    parser.add_argument("--query", type=str, default="Describe the events in the video with timestamps.")
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def load_model(args):
    print("🔹 Loading model...")

    tokenizer, model, context_len = load_pretrained_model(
        args,
        args.stage3,
        args.stage4
    )

    if args.device == "cuda":
        model = model.cuda()
    else:
        model = model.cpu()

    model.eval()

    return tokenizer, model


def run_inference(tokenizer, model, video_path, audio_path, args):

    av_ratio = args.av_ratio
    n_audio_feats = int(100 * av_ratio)
    n_image_feats = int(100 - n_audio_feats)

    clip_model, _ = clip.load(args.clip_path)
    clip_model.eval()
    clip_model = clip_model.cuda()

    video_loader = VideoExtractor(N=n_image_feats)
    _, images = video_loader.extract({'id': None, 'video': video_path})

    transform = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])

    images = transform(images / 255.0)
    images = images.to(torch.bfloat16)

    audio = torch.tensor(np.load(audio_path), dtype=torch.bfloat16)

    with torch.no_grad():
        v_features = clip_model.encode_image(images.to('cuda'))
        a_features = audio.cuda()

        tmp_len = len(a_features)
        if tmp_len != n_audio_feats:
            repeat_factor = n_audio_feats // tmp_len
            remainder = n_audio_feats % tmp_len
            a_features = torch.cat([
                a_features[i].unsqueeze(0).repeat(
                    repeat_factor + (1 if i < remainder else 0), 1
                ) for i in range(tmp_len)
            ], dim=0)

        features = [v_features.unsqueeze(0), a_features.unsqueeze(0)]

    query = "<video>\n " + args.query

    output = inference(model, features, query, tokenizer)

    return output


def main():
    args = parse_args()

    tokenizer, model = load_model(args)

    video_files = sorted(os.listdir(args.video_dir))

    results = []

    for vid in tqdm(video_files):
        if not vid.endswith(".mp4"):
            continue

        video_path = os.path.join(args.video_dir, vid)
        audio_path = os.path.join(args.audio_dir, vid.replace(".mp4", ".npy"))

        if not os.path.exists(audio_path):
            continue

        try:
            output_text = run_inference(
            tokenizer,
            model,
            video_path,
            audio_path,
            args
            )

            results.append({
                "video_id": vid.replace(".mp4", ""),
                "raw_output": output_text
            })

        except Exception as e:
            print(f"❌ Error on {vid}: {e}")
            continue

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved: {args.output}")


if __name__ == "__main__":
    main()
