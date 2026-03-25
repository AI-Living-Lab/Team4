import argparse
import json
import torch
from tqdm import tqdm
import laion_clap


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels


def main():
    args = parse_args()

    print("🔹 Loading CLAP model...")
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("🔹 Loading labels...")
    labels = load_labels(args.labels)

    print("🔹 Encoding label texts...")
    label_embeddings = model.get_text_embedding(labels, use_tensor=True)
    label_embeddings = label_embeddings.to(device)

    print("🔹 Loading predictions...")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    for item in tqdm(data):
        video_id = item["video_id"]
        events = item["events"]

        mapped_events = []

        for ev in events:
            text = ev["text"]

            text_emb = model.get_text_embedding([text], use_tensor=True)
            text_emb = text_emb.to(device)

            sim = torch.nn.functional.cosine_similarity(
                text_emb, label_embeddings, dim=1
            )

            best_idx = torch.argmax(sim).item()
            best_label = labels[best_idx]
            best_score = sim[best_idx].item()

            mapped_events.append({
                "pred_text": text,
                "mapped_label": best_label,
                "score": best_score,
                "start": ev["start"],
                "end": ev["end"]
            })

        results.append({
            "video_id": video_id,
            "events": mapped_events
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved to {args.output}")


if __name__ == "__main__":
    main()