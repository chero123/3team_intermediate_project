from __future__ import annotations

from typing import List
from openai import OpenAI

from .config import RAGConfig
from .types import Chunk
from dotenv import load_dotenv
load_dotenv()

class OpenAILLM:
    """
    OpenAILLM은 OpenAI API로 텍스트를 생성한다.

    Args:
        model: OpenAI 모델 이름
        api_key: OpenAI API 키 (없으면 환경변수 사용)
    """

    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        resp = self.client.responses.create(
            model=self.model,
            input=prompt,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
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
    context = "".join(f"[Source {i + 1}] {chunk.text}" for i, chunk in enumerate(context_chunks))
    return (
        "너는 문서를 요약기다.\n"
        "컨텍스트를 간략하게 요약한다.\n"
        "설명체를 사용하여 요약한다.\n"
        "반드시 3문장으로만 답하고, 각 문장은 마침표로 끝낸다.\n"
        "괄호나 특수문자를 쓰지 말고, 목록/헤더/컨텍스트 인용은 문장으로 풀어 작성한다.\n"
        "컨텍스트에 없으면 모른다고만 말하라.\n"
        f"질문: {question}\n\n"
        f"컨텍스트:\n{context}\n"
    )


def generate_answer(llm: OpenAILLM, config: RAGConfig, question: str, context_chunks: List[Chunk]) -> str:
    prompt = build_prompt(question, context_chunks)
    return llm.generate(
        prompt=prompt,
        max_tokens=config.response_max_tokens,
        temperature=config.response_temperature,
    )


def rewrite_answer(llm: OpenAILLM, answer: str) -> str:
    prompt = (
        "너는 문서를 요약하는 여우 요괴다.\n"
        "살짝 건방진 말투로 간략하게 요약하라.\n"
        "요약은 반말로 작성한다.\n"
        "반드시 3문장으로만 답하고, 각 문장은 마침표로 끝낸다.\n"
        "괄호나 특수문자를 쓰지 말고, 목록/헤더/컨텍스트 인용은 문장으로 풀어 작성한다.\n"
        "내용을 모르면 '무슨 소리인지 모르겠네'라고만 말하라.\n"
        "원문:"
        f"{answer}\n"
        "요약:\n"
    )
    return llm.generate(prompt=prompt, max_tokens=140, temperature=0.3)


def classify_query_type(llm: OpenAILLM, question: str) -> str:
    prompt = (
        "다음 질문을 유형으로 분류하라. 출력은 라벨만 한 단어로 답한다.\n"
        "라벨은 single, multi, compare, followup 중 하나다.\n"
        f"질문: {question}\n"
        "라벨:"
    )
    label = llm.generate(prompt=prompt, max_tokens=8, temperature=0.0).strip().lower()
    for key in ("single", "multi", "compare", "followup"):
        if key in label:
            return key
    return ""
