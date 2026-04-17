#!/usr/bin/env python3
"""
SALMONN2+ 모델에 VTG-LLM time token (<t0>~<t9>, <tdot>)을 추가.
- tokenizer에 special token 추가
- model embedding/lm_head resize
- 저장
"""
import argparse
import os
import torch
from transformers import AutoTokenizer

VTG_TIME_TOKENS = [f"<t{i}>" for i in range(10)] + ["<tdot>"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="SALMONN2+ model path")
    ap.add_argument("--output_path", required=True, help="Output path for model with time tokens")
    args = ap.parse_args()

    print(f"[1/4] Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print(f"  Vocab size before: {len(tokenizer)}")

    print(f"[2/4] Adding {len(VTG_TIME_TOKENS)} time tokens")
    num_added = tokenizer.add_tokens(VTG_TIME_TOKENS, special_tokens=True)
    print(f"  Added: {num_added} tokens")
    print(f"  Vocab size after: {len(tokenizer)}")

    for t in VTG_TIME_TOKENS:
        tid = tokenizer.convert_tokens_to_ids(t)
        print(f"  {t} -> {tid}")

    print(f"[3/4] Loading model and resizing embeddings")
    import sys
    sys.path.insert(0, os.path.join(os.environ["BASE_DIR"], "video_SALMONN2_plus"))
    from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus

    model = video_SALMONN2_plus.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    old_embed_size = model.get_input_embeddings().weight.shape[0]
    new_vocab_size = len(tokenizer)
    # 원본 embedding이 vocab보다 클 수 있음 (padding). 줄이지 않도록 max 사용
    target_size = max(old_embed_size, new_vocab_size)
    model.resize_token_embeddings(target_size)
    new_size = model.get_input_embeddings().weight.shape[0]
    print(f"  Embedding: {old_embed_size} -> {new_size} (vocab={new_vocab_size})")

    # time token embedding 초기화 (VTG-LLM 방식)
    # <t0>~<t9> → '0'~'9' 토큰 임베딩, <tdot> → '.' 토큰 임베딩
    init_map = {f"<t{i}>": str(i) for i in range(10)}
    init_map["<tdot>"] = "."

    in_embed = model.get_input_embeddings().weight.data
    out_embed = (model.get_output_embeddings().weight.data
                 if model.get_output_embeddings() is not None else None)

    with torch.no_grad():
        for new_tok, orig_char in init_map.items():
            new_id = tokenizer.convert_tokens_to_ids(new_tok)
            orig_ids = tokenizer.encode(orig_char, add_special_tokens=False)

            if len(orig_ids) == 0:
                print(f"  [WARN] '{orig_char}' not found. Using mean embedding for {new_tok}.")
                in_embed[new_id] = in_embed[:old_embed_size].mean(dim=0)
                if out_embed is not None:
                    out_embed[new_id] = out_embed[:old_embed_size].mean(dim=0)
            else:
                orig_id = orig_ids[0]
                in_embed[new_id] = in_embed[orig_id].clone()
                if out_embed is not None:
                    out_embed[new_id] = out_embed[orig_id].clone()
                print(f"  {new_tok} (id={new_id}) <- '{orig_char}' (id={orig_id})")

    print(f"[4/4] Saving to {args.output_path}")
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print("Done!")


if __name__ == "__main__":
    main()
