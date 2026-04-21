#!/usr/bin/env python
"""임의의 로컬 체크포인트 디렉터리를 HuggingFace Hub 레포에 업로드.

여러 실험 버전을 한 레포에 모을 때는 --path_in_repo로 서브폴더를 구분한다.
예) sft/salmonn2p_7b_unav_baseline/checkpoint-1500
    gdpo/exp_v1/checkpoint-500

인증: --hf_token 인자 또는 HF_TOKEN 환경변수.
"""
import argparse
import os
import sys

from huggingface_hub import HfApi, create_repo


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", required=True,
                   help="업로드할 로컬 체크포인트 디렉터리")
    p.add_argument("--repo_id", required=True,
                   help="HF 레포 id (예: 'ewhaailab/salmonn2p-ckpts')")
    p.add_argument("--path_in_repo", default=None,
                   help="레포 내 업로드 경로. 미지정 시 레포 루트에 업로드")
    p.add_argument("--private", action="store_true",
                   help="레포가 없을 때 private으로 생성")
    p.add_argument("--hf_token", default=None,
                   help="HF 토큰. 미지정 시 HF_TOKEN 환경변수 사용")
    p.add_argument("--commit_message", default=None)
    p.add_argument("--ignore_patterns", nargs="*", default=None,
                   help="업로드 제외 glob (예: 'optimizer.pt' '*.bin')")
    p.add_argument("--allow_patterns", nargs="*", default=None,
                   help="업로드 포함 glob (지정 시 이 패턴만 업로드)")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.ckpt_path):
        sys.exit(f"[에러] 체크포인트 디렉터리가 없습니다: {args.ckpt_path}")

    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("[에러] HF 토큰이 없습니다. --hf_token 또는 HF_TOKEN 환경변수를 설정하세요.")

    commit_message = args.commit_message or (
        f"Upload {os.path.basename(os.path.normpath(args.ckpt_path))}"
        + (f" to {args.path_in_repo}" if args.path_in_repo else "")
    )

    print(f"레포 생성/확인: {args.repo_id} (private={args.private})")
    create_repo(args.repo_id, private=args.private, token=token, exist_ok=True,
                repo_type="model")

    print(f"업로드: {args.ckpt_path} -> {args.repo_id}"
          + (f":{args.path_in_repo}" if args.path_in_repo else ""))
    url = HfApi(token=token).upload_folder(
        folder_path=args.ckpt_path,
        repo_id=args.repo_id,
        path_in_repo=args.path_in_repo,
        repo_type="model",
        commit_message=commit_message,
        ignore_patterns=args.ignore_patterns,
        allow_patterns=args.allow_patterns,
    )
    print(f"[완료] {url}")


if __name__ == "__main__":
    main()
