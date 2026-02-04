from __future__ import annotations

from typing import Any, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from .config import RAGConfig
from .types import Chunk

# .env 파일이 있다면 환경 변수(OPENAI_API_KEY 등)를 로드한다.
load_dotenv()


class OpenAILLM:
    """
    OpenAILLM은 OpenAI API 호출을 감싸는 간단한 클라이언트 래퍼

    Args:
        model: 사용할 OpenAI 모델 이름
        api_key: OpenAI API 키 (None이면 환경변수 사용)
    """

    def __init__(self, model: str, api_key: Optional[str] = None) -> None:
        """
        OpenAI 클라이언트 초기화

        Args:
            model: 사용할 모델 이름
            api_key: OpenAI API 키
        """
        # OpenAI 클라이언트 생성 (키가 None이면 환경 변수에서 읽음)
        self.client = OpenAI(api_key=api_key)
        # 모델 이름 저장
        self.model = model
        # GPT-5 계열 여부를 저장 (파라미터 호환성 분기용)
        self.is_gpt5 = model.startswith("gpt-5")

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """
        프롬프트를 입력으로 텍스트를 생성한다.

        Args:
            prompt: 모델에 전달할 프롬프트 문자열
            max_tokens: 생성 토큰 상한
            temperature: 샘플링 온도

        Returns:
            str: 생성된 텍스트
        """
        # Responses API 호출 결과를 가져온다.
        resp = self._create_response(prompt, max_tokens, temperature)
        # 응답 객체에서 텍스트만 추출한다.
        text = extract_response_text(resp)
        # 텍스트가 비어 있으면 호출 자체는 성공했더라도 실패로 간주한다. (GPT-5 추론 모델 대비)
        if not text:
            raise RuntimeError(
                "No text extracted from OpenAI response.\n"
                f"Raw response: {resp}"
            )
        return text

    def _create_response(self, prompt, max_tokens, temperature):
        """
        Responses API 호출 파라미터를 구성해 요청

        Args:
            prompt: 입력 프롬프트
            max_tokens: 생성 토큰 상한
            temperature: 샘플링 온도

        Returns:
            Any: OpenAI Responses API 응답 객체
        """
        # OpenAI 최소 토큰 제한(>=16)을 보장한다.
        safe_max_tokens = max(16, max_tokens)
        # Responses API의 표준 입력 형식으로 payload를 구성한다.
        params = {
            # 사용할 모델 지정
            "model": self.model,
            # 입력은 메시지 리스트 형식으로 전달
            "input": [
                {
                    # 단일 user 메시지로 프롬프트 전달
                    "role": "user",
                    "content": [
                        # 입력 텍스트 타입은 input_text 사용
                        {"type": "input_text", "text": prompt}
                    ],
                }
            ],
            # 생성 토큰 상한
            "max_output_tokens": safe_max_tokens,
        }

        # GPT-5 계열은 temperature 대신 reasoning 옵션을 사용한다.
        if self.is_gpt5:
            # 토큰 폭주를 막기 위해 저온 추론을 강제한다.
            params["reasoning"] = {"effort": "low"}
        else:
            # GPT-5가 아니면 temperature를 그대로 전달한다.
            params["temperature"] = temperature

        # OpenAI Responses API 호출
        return self.client.responses.create(**params)


def extract_response_text(resp: Any) -> str:
    """
    Responses API 응답에서 텍스트만 추출한다.

    Args:
        resp: OpenAI Responses API 응답 객체

    Returns:
        str: 추출된 텍스트 (없으면 빈 문자열)
    """
    # output_text가 있으면 최우선으로 반환한다.
    output_text = getattr(resp, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    # output 배열을 순회하며 content.text를 찾는다.
    output = getattr(resp, "output", None)
    if not isinstance(output, list):
        return ""

    for item in output:
        # dict/객체 혼합 대응: content 필드 추출
        content = getattr(item, "content", None)
        if content is None and isinstance(item, dict):
            content = item.get("content")

        # content가 리스트가 아니면 건너뛴다.
        if not isinstance(content, list):
            continue

        for block in content:
            # block이 dict면 text 키, 아니면 text 속성으로 접근
            if isinstance(block, dict):
                text = block.get("text")
            else:
                text = getattr(block, "text", None)

            # 유효한 텍스트를 찾으면 즉시 반환
            if isinstance(text, str) and text.strip():
                return text.strip()

    return ""


def build_prompt(question: str, context_chunks: List[Chunk]) -> str:
    """
    질문과 컨텍스트 청크로 최종 프롬프트를 구성한다.

    Args:
        question: 사용자 질문
        context_chunks: 검색된 문서 청크 목록

    Returns:
        str: 모델 입력 프롬프트
    """
    # 청크를 Source 태그로 묶고 메타데이터도 함께 노출한다.
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
    # 출력 형식(3문장/마침표/특수문자 제한)을 프롬프트로 강제한다.
    return f"""
너는 문서를 요약기다.
컨텍스트를 간략하게 요약한다.
설명체를 사용하여 요약한다.
반드시 3문장으로만 답하고, 각 문장은 마침표로 끝낸다.
괄호나 특수문자를 쓰지 말고, 목록이나 헤더는 문장으로 풀어 작성한다.
컨텍스트에 없으면 모른다고만 말하라.

질문: {question}

컨텍스트:
{context}
""".strip()


def generate_answer(
    llm: OpenAILLM,
    config: RAGConfig,
    question: str,
    context_chunks: List[Chunk],
) -> str:
    """
    RAG 컨텍스트로 최종 답변을 생성한다.

    Args:
        llm: OpenAI LLM 래퍼
        config: 설정 객체
        question: 사용자 질문
        context_chunks: 검색된 문서 청크

    Returns:
        str: 생성된 답변 텍스트
    """
    # 질문 + 컨텍스트를 프롬프트로 결합한다.
    prompt = build_prompt(question, context_chunks)
    # 생성 파라미터는 config 기준으로 전달하며, gpt-5의 추론 속성을 고려해 큰 토큰 상한을 사용한다.
    return llm.generate(
        prompt=prompt,
        max_tokens=config.openai_gpt5_max_tokens,
        temperature=config.response_temperature,
    )


def rewrite_answer(llm: OpenAILLM, answer: str) -> str:
    """
    최종 답변을 스타일 규칙에 맞게 리라이트한다.

    Args:
        llm: OpenAI LLM 래퍼 (작은 모델 사용 가능)
        answer: 원본 답변

    Returns:
        str: 리라이트된 답변
    """
    # 말투/문장 수/특수문자 제한을 프롬프트로 강제한다.
    prompt = f"""
너는 문서를 요약하는 게임 캐릭터다.
살짝 건방진 말투로 간략하게 요약하라.
요약은 반말로 작성한다.
반드시 3문장으로만 답하고, 각 문장은 마침표로 끝낸다.
괄호나 특수문자를 쓰지 말고, 목록이나 헤더는 문장으로 풀어 작성한다.
내용을 모르면 '무슨 소리인지 모르겠네. 너 날 놀리는 거니?'라고만 말하라.

원문:
{answer}

요약:
""".strip()
    # 리라이트는 짧게 끝나도록 토큰 상한을 낮게 설정한다.
    return llm.generate(prompt=prompt, max_tokens=96, temperature=0.2)


def classify_query_type(llm: OpenAILLM, question: str) -> str:
    """
    질문을 분류 라벨로 변환한다.

    Args:
        llm: OpenAI LLM 래퍼 (작은 모델 사용 가능)
        question: 사용자 질문

    Returns:
        str: single | multi | compare | followup 중 하나
    """
    # 라벨만 출력하도록 프롬프트를 단순화한다.
    prompt = f"""
다음 질문을 유형으로 분류하라. 출력은 라벨만 한 단어로 답한다.
라벨은 single, multi, compare, followup 중 하나다.

질문: {question}
라벨:
""".strip()
    # 온도 0으로 결정론적 분류를 유도한다.
    label = llm.generate(prompt=prompt, max_tokens=32, temperature=0.0).lower()

    # 라벨 문자열에서 키워드를 찾아 정규화한다.
    for key in ("single", "multi", "compare", "followup"):
        if key in label:
            return key
    return ""
