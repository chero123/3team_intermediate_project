from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from tokenizers import Tokenizer
from transformers import AddedToken, AutoTokenizer, PreTrainedTokenizerFast
from vllm import LLM as VLLMEngine
from vllm import SamplingParams

from .config import RAGConfig
from .types import Chunk


class LLM:
    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """
        generate는 프롬프트로 텍스트를 생성

        Args:
            prompt: 프롬프트 문자열
            max_tokens: 최대 생성 토큰 수
            temperature: 샘플링 온도

        Returns:
            str: 생성된 텍스트
        """
        raise NotImplementedError


def _coerce_special_token(value: Any) -> Any:
    """
    _coerce_special_token은 special token 값을 안전한 타입으로 변환

    Args:
        value: 입력 값

    Returns:
        Any: 변환된 토큰 값
    """
    if isinstance(value, (str, AddedToken)):
        return value
    if isinstance(value, dict):
        content = value.get("content") or value.get("token") or value.get("text")
        if content is None:
            return value
        return AddedToken(content, **{k: v for k, v in value.items() if k != "content"})
    return value


def _sanitize_special_tokens_map(raw_map: Dict[str, Any]) -> Dict[str, Any]:
    """
    _sanitize_special_tokens_map은 dict 형태 토큰을 문자열로 변환

    Args:
        raw_map: 원본 special_tokens_map

    Returns:
        Dict[str, Any]: 정리된 special_tokens_map
    """
    cleaned: Dict[str, Any] = {}
    for key, value in raw_map.items():
        if isinstance(value, list):
            cleaned[key] = [
                v.get("content") if isinstance(v, dict) else v
                for v in value
            ]
        elif isinstance(value, dict):
            cleaned[key] = value.get("content")
        else:
            cleaned[key] = value
    return cleaned


def _load_tokenizer_fallback(model_path: str) -> PreTrainedTokenizerFast:
    """
    _load_tokenizer_fallback은 tokenizer.json 기반으로 토크나이저를 구성

    Args:
        model_path: 모델 경로

    Returns:
        PreTrainedTokenizerFast: 토크나이저
    """
    model_dir = Path(model_path)
    tokenizer_json = model_dir / "tokenizer.json"
    if not tokenizer_json.exists():
        raise FileNotFoundError(f"tokenizer.json not found: {tokenizer_json}")

    tokenizer_obj = Tokenizer.from_file(str(tokenizer_json))

    special_tokens_map_path = model_dir / "special_tokens_map.json"
    special_tokens: Dict[str, Any] = {}
    if special_tokens_map_path.exists():
        with open(special_tokens_map_path, "r", encoding="utf-8") as f:
            raw_map = json.load(f)
        for key in [
            "bos_token",
            "eos_token",
            "unk_token",
            "pad_token",
            "cls_token",
            "sep_token",
            "mask_token",
            "additional_special_tokens",
        ]:
            if key in raw_map:
                value = raw_map[key]
                if isinstance(value, list):
                    special_tokens[key] = [_coerce_special_token(v) for v in value]
                else:
                    special_tokens[key] = _coerce_special_token(value)

    tokenizer_config_path = model_dir / "tokenizer_config.json"
    safe_config: Dict[str, Any] = {}
    if tokenizer_config_path.exists():
        with open(tokenizer_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        for key in ["model_max_length", "padding_side", "truncation_side", "clean_up_tokenization_spaces"]:
            if key in config:
                safe_config[key] = config[key]

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        **safe_config,
        **special_tokens,
    )


def load_tokenizer(model_path: str) -> PreTrainedTokenizerFast:
    """
    load_tokenizer는 토크나이저를 로드

    Args:
        model_path: 모델 경로

    Returns:
        PreTrainedTokenizerFast: 로드된 토크나이저
    """
    try:
        return AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            local_files_only=True,
            trust_remote_code=True,
        )
    except Exception:
        try:
            return AutoTokenizer.from_pretrained(
                model_path,
                use_fast=True,
                local_files_only=True,
                trust_remote_code=True,
            )
        except Exception:
            return _load_tokenizer_fallback(model_path)


def ensure_vllm_tokenizer_dir(model_path: str, cache_dir: str) -> str:
    """
    ensure_vllm_tokenizer_dir는 vLLM이 로드 가능한 토크나이저 디렉토리 생성

    Args:
        model_path: 원본 모델 경로
        cache_dir: 캐시 디렉토리 경로

    Returns:
        str: vLLM이 사용할 토크나이저 디렉토리 경로
    """
    model_dir = Path(model_path)
    out_dir = Path(cache_dir) / model_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    for filename in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "config.json",
    ]:
        src = model_dir / filename
        if src.exists():
            shutil.copy2(src, out_dir / filename)

    tokenizer_config_path = out_dir / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        with open(tokenizer_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        config.pop("tokenizer_class", None)
        with open(out_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    special_tokens_map_path = out_dir / "special_tokens_map.json"
    if special_tokens_map_path.exists():
        with open(special_tokens_map_path, "r", encoding="utf-8") as f:
            raw_map = json.load(f)
        cleaned_map = _sanitize_special_tokens_map(raw_map)
        with open(out_dir / "special_tokens_map.json", "w", encoding="utf-8") as f:
            json.dump(cleaned_map, f, ensure_ascii=False, indent=2)

    return str(out_dir)


class VLLMLLM(LLM):
    """
    VLLMLLM은 vLLM 엔진으로 텍스트를 생성

    Args:
        model_path: 모델 경로
        device: 디바이스

    Returns:
        None
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        cache_dir: str = "data/tokenizer_cache",
        quantization: str | None = None,
    ) -> None:
        if not device.startswith("cuda"):
            raise RuntimeError("vLLM requires CUDA. Set --device cuda.")
        self.tokenizer = load_tokenizer(model_path)
        tokenizer_dir = ensure_vllm_tokenizer_dir(model_path, cache_dir)
        engine_kwargs: Dict[str, Any] = dict(
            model=model_path,
            tokenizer=tokenizer_dir,
            tokenizer_mode="auto",
        )
        if quantization:
            engine_kwargs["quantization"] = quantization
        self.engine = VLLMEngine(**engine_kwargs)

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
        )
        outputs = self.engine.generate([prompt], params)
        if not outputs:
            return ""
        return outputs[0].outputs[0].text


def build_prompt(question: str, context_chunks: List[Chunk]) -> str:
    """
    build_prompt는 RAG 프롬프트를 구성

    Args:
        question: 사용자 질문
        context_chunks: 컨텍스트 청크 리스트

    Returns:
        str: 프롬프트 문자열
    """
    context = "\n\n".join(
        f"[Source {i + 1}]\n{chunk.text}" for i, chunk in enumerate(context_chunks)
    )
    return (
        "너는 문서를 요약하는 여우 무녀다.\n"
        "살짝 건방진 말투로 간략하게 요약하라.\n"
        "설명체를 사용하지 말고 단정적인 문장으로 답하라.\n"
        "반드시 3문장으로만 답하고, 각 문장은 마침표로 끝내라.\n"
        "괄호나 특수문자를 쓰지 말고, 목록/헤더/컨텍스트 인용은 금지한다.\n"
        "컨텍스트에 없으면 모른다고만 말하라.\n\n"
        f"질문: {question}\n\n"
        f"컨텍스트:\n{context}\n"
    )


def generate_answer(llm: LLM, config: RAGConfig, question: str, context_chunks: List[Chunk]) -> str:
    """
    generate_answer는 LLM을 호출해 최종 답변을 만든다

    Args:
        llm: LLM 인스턴스
        config: 설정 객체
        question: 사용자 질문
        context_chunks: 컨텍스트 청크 리스트

    Returns:
        str: 최종 답변
    """
    prompt = build_prompt(question, context_chunks)
    return llm.generate(
        prompt=prompt,
        max_tokens=config.response_max_tokens,
        temperature=config.response_temperature,
    )
