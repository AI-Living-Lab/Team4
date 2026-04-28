"""
Step 3: Map free-form event text → UnAV-100 class labels via CLAP embeddings.

Input:  parsed_predictions.json  (from Step 2)
Output: mapped_predictions.json

Usage:
    python map_with_clap.py \
        --input output/parsed_predictions.json \
        --labels data/unav100_class_labels.txt \
        --output output/mapped_predictions.json
"""

import argparse
import json
import torch
from tqdm import tqdm
import laion_clap


def parse_args():
    parser = argparse.ArgumentParser(
        description="Map free-form event text to UnAV-100 class labels via CLAP"
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for text embedding")
    return parser.parse_args()


def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    args = parse_args()

    # ---- Load CLAP ----
    print("Loading CLAP model...")
    model = laion_clap.CLAP_Module(enable_fusion=True)
    model.load_ckpt("checkpoints/clap/630k-audioset-fusion-best.pt")
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Encode label set (once) ----
    labels = load_labels(args.labels)
    print(f"Labels: {len(labels)}")

    label_embeddings = model.get_text_embedding(labels, use_tensor=True)
    label_embeddings = label_embeddings.to(device)            # (100, D)
    label_embeddings = torch.nn.functional.normalize(label_embeddings, dim=1)

    # ---- Load predictions ----
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ---- Collect ALL event texts for batched encoding ----
    all_texts = []
    index_map = []  # (sample_idx, event_idx)

    for si, item in enumerate(data):
        for ei, ev in enumerate(item["events"]):
            all_texts.append(ev["text"])
            index_map.append((si, ei))

    print(f"Total events to embed: {len(all_texts)}")

    # ---- Batched CLAP encoding ----
    all_mapped_labels = [None] * len(all_texts)
    all_scores = [None] * len(all_texts)

    for batch_start in tqdm(range(0, len(all_texts), args.batch_size),
                            desc="CLAP encoding"):
        batch_end = min(batch_start + args.batch_size, len(all_texts))
        batch_texts = all_texts[batch_start:batch_end]

        text_emb = model.get_text_embedding(batch_texts, use_tensor=True)
        text_emb = text_emb.to(device)
        text_emb = torch.nn.functional.normalize(text_emb, dim=1)

        # (batch, D) x (D, 100) -> (batch, 100)
        sim = torch.mm(text_emb, label_embeddings.T)

        best_indices = torch.argmax(sim, dim=1)
        best_scores = sim[torch.arange(sim.size(0)), best_indices]

        for i in range(len(batch_texts)):
            global_idx = batch_start + i
            all_mapped_labels[global_idx] = labels[best_indices[i].item()]
            all_scores[global_idx] = best_scores[i].item()

    # ---- Assemble results ----
    results = []
    for si, item in enumerate(data):
        mapped_events = []
        for ei, ev in enumerate(item["events"]):
            global_idx = index_map.index((si, ei))  # O(n) but fine for eval
            mapped_events.append({
                "pred_text": ev["text"],
                "mapped_label": all_mapped_labels[global_idx],
                "score": all_scores[global_idx],
                "start": ev["start"],
                "end": ev["end"],
            })
        results.append({
            "video_id": item["video_id"],
            "events": mapped_events,
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved to {args.output}")

    # ---- Quick sanity check ----
    if len(all_texts) > 0:
        print("\n--- Sample mappings (first 10) ---")
        for i in range(min(10, len(all_texts))):
            print(f"  '{all_texts[i]}' → '{all_mapped_labels[i]}' (score={all_scores[i]:.4f})")


if __name__ == "__main__":
    main()
