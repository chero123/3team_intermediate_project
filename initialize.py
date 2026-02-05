from __future__ import annotations

"""
모델 초기 다운로드 스크립트

역할:
- HuggingFace 모델 스냅샷을 로컬에 저장
- 토크나이저도 함께 캐시
"""

from pathlib import Path

# HF 모델 다운로드
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

# 경로/모델 목록 설정
RAW_DIR = Path(__file__).resolve().parent 
# 모델 저장 디렉토리
MODEL_ASSETS_DIR = RAW_DIR / "models"

MODELS = {  # 다운로드할 모델 목록
    # "bge-m3-ko": "dragonkue/BGE-m3-ko",  # 임베딩 모델
    "ko_sroberta": "jhgan/ko-sroberta-multitask",  # 임베딩 모델
    # "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",  # 리랭커 모델
    # "bge-reranker-v2-m3-ko": "dragonkue/bge-reranker-v2-m3-ko",  # 리랭커 모델2
    # "YanoljaNEXT-EEVE-7B-v2": "YanoljaNEXT/YanoljaNEXT-EEVE-7B-v2",  # LLM
    # "A.X-4.0-Light": "skt/A.X-4.0-Light",  # LLM 대체 모델
    # "melo_yae": "ahnhs2k/yae_meloTTS",  # TTS 모델 (private repository)
    # "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct-FP8" # VLM 모델
}


def download_one(repo_id: str, local_subdir: str) -> Path:
    """
    단일 모델 다운로드
    
    Args:
        허깅페이스 리포지토리 id와 로컬 서브디렉토리 이름
    
    Returns:
        경로 반환
    """
    # 저장 경로
    local_dir = MODEL_ASSETS_DIR / local_subdir
    # 디렉토리 생성
    local_dir.mkdir(parents=True, exist_ok=True)

    # 진행 로그
    print(f"\n[INFO] snapshot_download: {repo_id}")
    # 스냅샷 다운로드
    snapshot_download(
        repo_id=repo_id,  # 모델 ID
        local_dir=str(local_dir),  # 저장 경로
        revision="main",  # 브랜치
    )

    # 토크나이저 로그
    print(f"[INFO] tokenizer save_pretrained: {repo_id}")
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
    # 로컬 저장
    tokenizer.save_pretrained(local_dir)

    print(f"[DONE] {repo_id} saved to: {local_dir}")

    return local_dir

def main() -> None:
    """
    메인 함수
    """
    MODEL_ASSETS_DIR.mkdir(parents=True, exist_ok=True) 

    # 모델 목록 순회
    for local_subdir, repo_id in MODELS.items():
        # 다운로드 실행
        download_one(repo_id=repo_id, local_subdir=local_subdir)

    print("\n[ALL DONE] models downloaded into models")

if __name__ == "__main__":
    main()
