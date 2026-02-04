from __future__ import annotations

from typing import Any, Dict, List

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


class VLLMLLM(LLM):
    """
    VLLMLLM은 vLLM 엔진으로 텍스트를 생성

    Args:
        model_path: 모델 경로
        device: 디바이스
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        cache_dir: str = "data/tokenizer_cache",
        quantization: str | None = None,
    ) -> None:
        # vLLM은 CUDA 전용이므로 CPU 장치 요청은 명시적으로 차단한다.
        if not device.startswith("cuda"):
            raise RuntimeError("vLLM requires CUDA. Set --device cuda.")
        # vLLM 엔진 생성 시 모델/토크나이저 경로를 고정한다.
        engine_kwargs: Dict[str, Any] = dict(
            model=model_path,
            tokenizer=model_path,
            tokenizer_mode="auto",
        )
        if quantization:
            # bitsandbytes 등 양자화 설정이 있을 때만 옵션을 추가한다.
            engine_kwargs["quantization"] = quantization
        self.engine = VLLMEngine(**engine_kwargs)

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        # 샘플링 파라미터는 응답 길이/온도만 노출해 단순하게 유지
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
    # 검색 결과 청크를 Source 라벨로 묶고, 메타데이터도 함께 노출한다.
    context_blocks = []
    for i, chunk in enumerate(context_chunks):
        meta = chunk.metadata or {}
        meta_lines = [f"{k}: {v}" for k, v in meta.items() if v is not None]
        meta_text = "\n".join(meta_lines)
        block = f"[Source {i + 1}]"
        if meta_text:
            block += f"\n{meta_text}"
        block += f"\n{chunk.text}"
        context_blocks.append(block)
    context = "\n\n".join(context_blocks)
    return f"""
너는 문서 기반 요약 어시스턴트다.
오직 컨텍스트에 있는 사실만 사용하고 추측하지 않는다.
질문에 필요한 정보만 골라 간결하게 답한다.
중복/군더더기 표현을 줄이고 핵심만 남긴다.
반드시 3문장으로만 답하고, 각 문장은 마침표로 끝낸다.
모든 문장은 완결된 문장으로 끝낸다. 미완성 단어로 끝내지 않는다.
요약, 결론 같은 라벨을 붙이지 않는다.
제안서 문맥이므로 과거형 서술을 피하고 현재형으로 서술한다.
괄호나 특수문자를 쓰지 말고, 목록/헤더/컨텍스트 인용은 문장으로 풀어 작성한다.
컨텍스트에 없으면 모른다고만 말하라.

질문: {question}

컨텍스트:
{context}
""".strip()


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


def rewrite_answer(llm: LLM, answer: str) -> str:
    """
    rewrite_answer는 답변을 3문장 요약 형식으로 리라이팅한다.

    Args:
        llm: LLM 인스턴스
        answer: 원본 답변

    Returns:
        str: 리라이팅된 답변
    """
    prompt = f"""
너는 원문을 구어체로 rewrite하는 AI 어시스턴트다.
자연스러운 한국어 구어체로 rewrite한다.
rewrite 시에는 반말을 사용한다.
반드시 3문장으로만 답하고, 각 문장은 마침표로 끝낸다.
괄호나 특수문자를 쓰지 말고, 목록/헤더/컨텍스트 인용은 문장으로 풀어 작성한다.
제안서 문맥에 맞추어 rewrite한다.
설명체는 사용하지 않는다.
내용을 모르면 '무슨 소리인지 모르겠네. 너 날 놀리는 거니?'라고만 말하라.

원문:
{answer}

요약:
""".strip()
    return llm.generate(prompt=prompt, max_tokens=220, temperature=0.6)


def classify_query_type(llm: LLM, question: str) -> str:
    """
    classify_query_type은 질문 유형을 single/multi/compare/followup 중 하나로 분류한다.

    Args:
        llm: LLM 인스턴스
        question: 사용자 질문

    Returns:
        str: 분류 라벨
    """
    prompt = f"""
다음 질문을 유형으로 분류하라. 출력은 라벨만 한 단어로 답한다.
라벨은 single, multi, compare, followup 중 하나다.

질문: {question}
라벨:
""".strip()
    label = llm.generate(prompt=prompt, max_tokens=8, temperature=0.0).strip().lower()
    for key in ("single", "multi", "compare", "followup"):
        if key in label:
            return key
    return ""
