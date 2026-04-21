#!/usr/bin/env python
"""train_salmonn2plus_unav100.sh로 학습한 LoRA 체크포인트를
베이스 video_SALMONN2_plus 모델에 merge하고, 선택적으로 HuggingFace Hub에 업로드.

merge 결과물은 독립 실행 가능한 모델이며 다음을 포함:
  - q/k/v projection에 LoRA 델타가 fold된 가중치
  - modules_to_save로 학습된 lm_head / embed_tokens (학습된 사본으로 치환됨)
  - 나머지는 베이스 모델 가중치 그대로
  - 토크나이저와 이미지 프로세서 설정은 베이스 모델에서 복사

실행은 merge_and_push.sh 스크립트를 사용하거나(권장), 직접 명령줄에서 인자 지정하여 실행할 수 있습니다.
"""
import argparse
import os
import shutil

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", required=True,
                        help="베이스 모델 경로 (학습 시 MODEL_BASE와 동일해야 함)")
    parser.add_argument("--checkpoint_path", required=True,
                        help="PEFT 체크포인트 디렉터리 (예: .../checkpoint-5000)")
    parser.add_argument("--output_dir", required=True,
                        help="merge된 모델을 저장할 경로")
    parser.add_argument("--dtype", default="bfloat16", choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--repo_id", default=None,
                        help="HF 레포 id (예: 'username/model-name'); --push_to_hub 사용 시 필수")
    parser.add_argument("--private", action="store_true", help="비공개 HF 레포로 생성")
    parser.add_argument("--hf_token", default=None,
                        help="HF 토큰; 미지정 시 HF_TOKEN 환경변수 또는 huggingface-cli login 사용")
    return parser.parse_args()


def main():
    args = parse_args()

    # 전달받은 경로가 PEFT 체크포인트인지 검증
    adapter_cfg = os.path.join(args.checkpoint_path, "adapter_config.json")
    if not os.path.exists(adapter_cfg):
        raise FileNotFoundError(
            f"{adapter_cfg}가 없습니다 — {args.checkpoint_path}가 PEFT 체크포인트가 맞나요?"
        )

    dtype = DTYPE_MAP[args.dtype]

    print(f"[1/4] 베이스 모델 로드: {args.base_model_path}")
    model = video_SALMONN2_plus.from_pretrained(
        args.base_model_path,
        torch_dtype=dtype,
    )

    print(f"[2/4] LoRA 어댑터 로드: {args.checkpoint_path}")
    # audio.layers 우회 처리: PEFT가 Whisper 레이어 구조를 다루지 못하므로
    # 어댑터 로드 전에 분리했다가 다시 붙여준다 (학습 코드와 동일)
    has_audio_layers = hasattr(model, "audio") and hasattr(model.audio, "layers")
    if has_audio_layers:
        audio_layers = model.audio.layers
        del model.audio.layers
    model = PeftModel.from_pretrained(model, args.checkpoint_path)
    if has_audio_layers:
        model.model.audio.layers = audio_layers

    print("[3/4] LoRA 가중치 merge")
    # merge_and_unload: LoRA 델타를 base에 fold하고,
    # modules_to_save(lm_head/embed_tokens)는 학습된 사본으로 치환
    model = model.merge_and_unload()

    print(f"[4/4] merge된 모델 저장: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)

    # 토크나이저는 베이스에서 불러와 함께 저장 (chat_template 포함)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=False)
    tokenizer.save_pretrained(args.output_dir)

    # 모델 클래스가 직접 관리하지 않는 프로세서/채팅 템플릿 설정들을 베이스에서 복사
    for fname in [
        "preprocessor_config.json",
        "image_processor.json",
        "chat_template.json",
        "video_preprocessor_config.json",
    ]:
        src = os.path.join(args.base_model_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, args.output_dir)

    print(f"[완료] merge된 모델 저장 위치: {args.output_dir}")

    if args.push_to_hub:
        if not args.repo_id:
            raise ValueError("--push_to_hub 사용 시 --repo_id는 필수입니다")

        from huggingface_hub import HfApi, create_repo
        token = args.hf_token or os.environ.get("HF_TOKEN")

        print(f"레포 생성/확인: {args.repo_id} (private={args.private})")
        create_repo(args.repo_id, private=args.private, token=token, exist_ok=True)

        print(f"업로드 중: {args.output_dir} -> {args.repo_id}")
        HfApi(token=token).upload_folder(
            folder_path=args.output_dir,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"Upload merged model from {os.path.basename(args.checkpoint_path)}",
        )
        print(f"[완료] 업로드 성공: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
