from __future__ import annotations

from typing import Any, Dict, List

# 토크나이저 로딩은 vLLM 내부에 위임하고, 여기서는 최소 설정만 유지한다.
# vLLM은 로컬 LLM 실행용, OpenAI SDK는 클라우드 LLM 실행용으로 분리한다.
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


class OpenAILLM(LLM):
    """
    OpenAILLM은 OpenAI API를 사용해 텍스트를 생성한다.

    Args:
        model: OpenAI 모델 이름
        api_key: OpenAI API 키 (없으면 환경변수 사용)
    """

    def __init__(self, model: str, api_key: str | None = None) -> None:
        # OpenAI SDK는 환경변수(OPENAI_API_KEY) 또는 명시 키를 사용한다.
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - 환경 의존
            raise RuntimeError("OpenAI SDK is not installed. Install 'openai'.") from exc
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        # Responses API를 사용해 단일 텍스트 응답을 생성한다.
        resp = self.client.responses.create(
            model=self.model,
            input=prompt,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        # SDK 헬퍼가 있으면 사용하고, 없으면 출력 구조에서 텍스트를 찾는다.
        if hasattr(resp, "output_text") and resp.output_text:
            return resp.output_text
        if getattr(resp, "output", None):
            for item in resp.output:
                content = getattr(item, "content", None)
                if not content:
                    continue
                for block in content:
                    text = getattr(block, "text", None)
                    if text:
                        return text
        return ""


def build_prompt(question: str, context_chunks: List[Chunk]) -> str:
    """
    build_prompt는 RAG 프롬프트를 구성

    Args:
        question: 사용자 질문
        context_chunks: 컨텍스트 청크 리스트

    Returns:
        str: 프롬프트 문자열
    """
    # 검색 결과 청크를 Source 라벨로 묶어 모델이 인용 근거를 구분할 수 있게 한다.
    context = "\n\n".join(
        f"[Source {i + 1}]\n{chunk.text}" for i, chunk in enumerate(context_chunks)
    )
    return (
        "너는 문서를 요약기다.\n"
        "컨텍스트를 간략하게 요약한다.\n"
        "설명체를 사용하여 요약한다.\n"
        "반드시 3문장으로만 답하고, 각 문장은 마침표로 끝낸다.\n"
        "괄호나 특수문자를 쓰지 말고, 목록/헤더/컨텍스트 인용은 문장으로 풀어 작성한다.\n"
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


def rewrite_answer(llm: LLM, answer: str) -> str:
    """
    rewrite_answer는 답변을 3문장 요약 형식으로 리라이팅한다.

    Args:
        llm: LLM 인스턴스
        answer: 원본 답변

    Returns:
        str: 리라이팅된 답변
    """
    prompt = (
        "너는 문서를 요약하는 여우 요괴다.\n"
        "살짝 건방진 말투로 간략하게 요약하라.\n"
        "요약은 반말로 작성한다.\n"
        "반드시 3문장으로만 답하고, 각 문장은 마침표로 끝낸다.\n"
        "괄호나 특수문자를 쓰지 말고, 목록/헤더/컨텍스트 인용은 문장으로 풀어 작성한다.\n"
        "내용을 모르면 '무슨 소리인지 모르겠네'라고만 말하라.\n\n"
        "원문:\n"
        f"{answer}\n\n"
        "요약:\n"
    )
    return llm.generate(prompt=prompt, max_tokens=140, temperature=0.3)


def classify_query_type(llm: LLM, question: str) -> str:
    """
    classify_query_type은 질문 유형을 single/multi/compare/followup 중 하나로 분류한다.

    Args:
        llm: LLM 인스턴스
        question: 사용자 질문

    Returns:
        str: 분류 라벨
    """
    prompt = (
        "다음 질문을 유형으로 분류하라. 출력은 라벨만 한 단어로 답한다.\n"
        "라벨은 single, multi, compare, followup 중 하나다.\n\n"
        f"질문: {question}\n"
        "라벨:"
    )
    label = llm.generate(prompt=prompt, max_tokens=8, temperature=0.0).strip().lower()
    for key in ("single", "multi", "compare", "followup"):
        if key in label:
            return key
    return ""
