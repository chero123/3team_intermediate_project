from __future__ import annotations

from typing import Any, Dict, List

from vllm import LLM as VLLMEngine
from vllm import SamplingParams

from .config import RAGConfig
from .types import Chunk


class LLM:
    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        stop: List[str] | None = None,
    ) -> str:
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
            max_model_len=16000,    # 상황에 따라 조정 필요 (기본값은 8192: 너무 키우면 VRAM이 융단 폭격 당할 수 있음)
        )
        if quantization:
            # bitsandbytes 등 양자화 설정이 있을 때만 옵션을 추가한다.
            engine_kwargs["quantization"] = quantization
        self.engine = VLLMEngine(**engine_kwargs)

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        stop: List[str] | None = None,
    ) -> str:
        """
        vLLM 엔진으로 텍스트를 생성한다.

        Args:
            prompt: 입력 프롬프트
            max_tokens: 생성 토큰 상한
            temperature: 샘플링 온도

        Returns:
            str: 생성된 텍스트
        """
        # 샘플링 파라미터에 반복 억제/다양성 옵션을 포함한다.
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0 if top_p is None else top_p,
            repetition_penalty=1.0 if repetition_penalty is None else repetition_penalty,
            stop=stop,
        )
        outputs = self.engine.generate([prompt], params)
        if not outputs:
            return ""
        return outputs[0].outputs[0].text


# Prompt Builders
def build_prompt(
    question: str,
    context_chunks: List[Chunk],
    config: RAGConfig,
    previous_turn: str | None = None,
    previous_docs: list[str] | None = None,
) -> str:
    """
    build_prompt는 RAG 프롬프트를 구성

    Args:
        question: 사용자 질문
        context_chunks: 컨텍스트 청크 리스트

    Returns:
        str: 프롬프트 문자열
    """
    # 검색 결과 청크를 Source 라벨로 묶고, 메타데이터도 함께 노출한다.
    context_blocks: List[str] = []
    for i, chunk in enumerate(context_chunks):
        meta = chunk.metadata or {}
        meta_lines = [f"{k}: {v}" for k, v in meta.items() if v is not None]
        meta_text = "\n".join(meta_lines)
        block = f"[Source {i + 1}]"
        if meta_text:
            block += f"\n{meta_text}"
        chunk_text = chunk.text
        block += f"\n{chunk_text}"
        context_blocks.append(block)
    context = "\n\n".join(context_blocks)
    previous_block = ""
    if previous_turn:
        previous_block = f"\n\n[이전 턴]\n{previous_turn}\n이전 턴은 참고용이며 현재 질문을 우선한다."

    previous_docs_block = ""
    if previous_docs:
        doc_list = "\n".join(previous_docs)
        previous_docs_block = f"\n\n[이전 문서]\n{doc_list}"

    return f"""
너는 문서 기반 요약 AI 어시스턴트다.
오직 컨텍스트에 있는 사실만 사용하고 추측하지 않는다.
가정, 가능성, 예시, 추론, 일반화 표현을 쓰지 않는다.
질문에 필요한 정보만 골라 간결하게 답한다.
중복/군더더기 표현을 줄이고 핵심만 남긴다.
각 문장은 마침표로 끝내고, 미완성 단어로 끝내지 않는다.
요약, 결론 같은 라벨을 붙이지 않는다.
제안서 문맥이므로 과거형 서술을 피하고 현재형으로 서술한다.
괄호나 특수문자를 쓰지 말고, 목록/헤더/컨텍스트 인용은 문장으로 풀어 작성한다.
최대 3문장, 3줄 이내로만 답한다.
컨텍스트에 없으면 "죄송합니다. 확인할 수 없습니다"라고만 말하라.

영어 표기 규칙:
- 영어 단어는 한국어 음역으로만 표기한다.
- 예: dashboard -> 대시보드, dataset -> 데이터셋, pipeline -> 파이프라인, system -> 시스템.

숫자 표기 규칙:
- 금액은 반드시 한글 화폐식으로 쓴다.
- 예: 35,750,000원 -> 3천 5백 7십 5만원

질문: {question}
{previous_block}
{previous_docs_block}

컨텍스트:
{context}

요약:
""".strip()


# Answer Generation
def generate_answer(
    llm: LLM,
    config: RAGConfig,
    question: str,
    context_chunks: List[Chunk],
    previous_turn: str | None = None,
    previous_docs: list[str] | None = None,
) -> str:
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
    # 1) 질문 + 컨텍스트를 프롬프트로 결합
    prompt = build_prompt(
        question,
        context_chunks,
        config,
        previous_turn=previous_turn,
        previous_docs=previous_docs,
    )
    # 2) LLM 호출 (토큰 상한/온도는 config 기준)
    return llm.generate(
        prompt=prompt,
        max_tokens=config.response_max_tokens,
        temperature=config.response_temperature,
        top_p=config.response_top_p,
        repetition_penalty=config.response_repetition_penalty,
        stop=config.response_stop,
    )


# Rewrite / Classification
def _strip_rewrite_output(text: str) -> str:
    """
    리라이트 결과에서 불필요한 라벨/설명 텍스트를 제거한다.
    """
    if not text:
        return ""
    cleaned = text.strip()
    markers = ("rewrite 결과:", "rewrite 결과", "rewrite:", "결과:", "출력:", "원문")
    for marker in markers:
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[-1].strip()
    return cleaned


def rewrite_answer(llm: LLM, config: RAGConfig, answer: str) -> str:
    """
    rewrite_answer는 답변을 3문장 요약 형식으로 리라이팅한다.

    Args:
        llm: LLM 인스턴스
        answer: 원본 답변

    Returns:
        str: 리라이팅된 답변
    """
    prompt = f"""
너는 원문을 간결한 문장형 요약으로 rewrite하는 AI 어시스턴트다.
문장형으로 자연스럽게 rewrite한다.
각 문장은 마침표로 끝내고, 미완성 단어로 끝내지 않는다.
괄호나 특수문자를 쓰지 말고, 목록/헤더/컨텍스트 인용은 문장으로 풀어 작성한다.
3문장으로 rewrite한다.
중복/군더더기 표현을 줄이고 핵심만 남긴다.
내용을 모르면 "죄송합니다. 확인할 수 없습니다"라고만 말하라.
출력은 최종 문장만 작성한다. 라벨이나 설명을 출력하지 않는다.

영어 표기 규칙:
- 영어 단어는 한국어 음역으로만 표기한다.
- 예: dashboard -> 대시보드, dataset -> 데이터셋, pipeline -> 파이프라인, system -> 시스템.

숫자 표기 규칙:
- 금액은 반드시 한글 화폐식으로 쓴다.
- 예: 35,750,000원 -> 3천 5백 7십 5만원

원문:
{answer}

rewrite 결과:
""".strip()
    # 리라이트는 스타일을 살리기 위해 온도를 약간 높인다.
    output = llm.generate(
        prompt=prompt,
        max_tokens=config.rewrite_max_tokens,
        temperature=config.rewrite_temperature,
        top_p=config.rewrite_top_p,
        repetition_penalty=config.rewrite_repetition_penalty,
        stop=config.rewrite_stop,
    )
    return _strip_rewrite_output(output)


def rewrite_query(
    llm: LLM,
    config: RAGConfig,
    question: str,
    previous_question: str | None = None,
    previous_answer: str | None = None,
) -> str:
    """
    rewrite_query는 검색 성능을 높이기 위해 질문을 짧은 검색 쿼리로 재작성한다.

    Args:
        llm: LLM 인스턴스
        config: 설정 객체
        question: 현재 질문
        previous_question: 직전 질문
        previous_answer: 직전 답변

    Returns:
        str: 재작성된 검색용 쿼리
    """
    previous_block = ""
    if previous_question or previous_answer:
        prev_q = previous_question or ""
        prev_a = previous_answer or ""
        previous_block = (
            "\n[이전 턴]\n"
            f"질문: {prev_q}\n"
            f"답변: {prev_a}\n"
            "이전 턴은 참고용이며 현재 질문을 우선한다.\n"
        )

    prompt = f"""
너는 검색용 쿼리 재작성 도우미다.
현재 질문을 문서 검색에 적합한 짧은 쿼리로 바꾼다.
파일명/기관명/사업명/기간/금액 등 핵심 키워드는 반드시 유지한다.
불필요한 서술, 감탄, 질문 부호는 제거하고 한 줄로 출력한다.
답변이나 추측은 하지 말고, 재작성된 쿼리만 출력한다.
{previous_block}
[현재 질문]
{question}

출력:
""".strip()

    output = llm.generate(
        prompt=prompt,
        max_tokens=min(128, config.rewrite_max_tokens),
        temperature=min(0.2, config.rewrite_temperature),
        top_p=config.rewrite_top_p,
        repetition_penalty=config.rewrite_repetition_penalty,
        stop=config.rewrite_stop,
    )
    return _strip_rewrite_output(output).strip()


def classify_query_type(llm: LLM, question: str) -> str:
    """
    classify_query_type은 질문 유형을 single/multi/followup 중 하나로 분류한다.

    Args:
        llm: LLM 인스턴스
        question: 사용자 질문

    Returns:
        str: 분류 라벨
    """
    prompt = f"""
다음 질문을 유형으로 분류하라. 출력은 라벨만 한 단어로 답한다.
    라벨은 single, multi, followup 중 하나다.

질문: {question}
라벨:
""".strip()
    # 분류는 결정론적 출력을 위해 온도 0으로 호출한다.
    label = llm.generate(
        prompt=prompt,
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        stop=[],
    ).strip().lower()
    for key in ("single", "multi", "followup"):
        if key in label:
            return key
    return ""
