"""
FastAPI + Gradio 통합 코드

실행:
  uv run python -m server.app
"""
from __future__ import annotations

import os
import uuid
from typing import Literal

import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel

from rag.pipeline import RAGPipeline
from tts_runtime.infer_onnx import infer_tts_onnx


class AskRequest(BaseModel):
    """
    API 요청 바디

    Args:
        question: 사용자 질문
        provider: local | openai (선택)
    """

    question: str
    provider: Literal["local", "openai"] | None = None
    tts: bool | None = None


def _build_pipeline(provider: str):
    if provider == "openai":
        from rag.openai_pipeline import OpenAIRAGPipeline

        return OpenAIRAGPipeline()
    return RAGPipeline()


_PIPELINE_CACHE: dict[str, object] = {}


def get_pipeline(provider: str):
    """
    프로바이더에 맞는 파이프라인을 싱글턴으로 반환한다.

    Args:
        provider: local | openai

    Returns:
        파이프라인 객체
    """
    if provider not in _PIPELINE_CACHE:
        _PIPELINE_CACHE[provider] = _build_pipeline(provider)
    return _PIPELINE_CACHE[provider]


def ask(question: str, provider: str | None = None) -> str:
    """
    질문을 파이프라인에 전달하고 답변을 반환한다.
    """
    mode = provider or os.getenv("RAG_PROVIDER", "local")
    pipeline = get_pipeline(mode)
    return pipeline.ask(question)  # type: ignore[no-any-return]


def _tts_paths():
    base = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base, "models", "melo_yae", "melo_yae.onnx")
    bert_path = os.path.join(base, "models", "melo_yae", "bert_kor.onnx")
    config_path = os.path.join(base, "models", "melo_yae", "config.json")
    out_dir = os.path.join(base, "data", "answer")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"tts_{uuid.uuid4().hex}.wav")
    return model_path, bert_path, config_path, out_path


def ask_with_tts(question: str, provider: str | None = None):
    """
    질문 → 답변 생성 → TTS wav 생성까지 수행한다.
    """
    answer = ask(question, provider)
    model_path, bert_path, config_path, out_path = _tts_paths()
    audio = infer_tts_onnx(
        onnx_path=model_path,
        bert_onnx_path=bert_path,
        config_path=config_path,
        text=answer,
        speaker_id=0,
        language="KR",
        device="cuda",
        out_path=out_path,
    )
    return answer, out_path


app = FastAPI(title="RAG API")


@app.post("/api/ask")
def ask_api(payload: AskRequest):
    """
    API 엔드포인트: 질문 → 답변
    """
    answer = ask(payload.question, payload.provider)
    return {"answer": answer}


@app.post("/api/ask_tts")
def ask_tts_api(payload: AskRequest):
    """
    API 엔드포인트: 질문 → 답변 → TTS
    """
    answer, wav_path = ask_with_tts(payload.question, payload.provider)
    return {"answer": answer, "wav_path": wav_path}


def build_gradio():
    """
    Gradio UI 구성 (API와 동일한 ask 함수 사용)
    """
    with gr.Blocks() as demo:
        gr.Markdown("# RAG Demo")
        question = gr.Textbox(label="질문")
        provider = gr.Dropdown(
            choices=["local", "openai"],
            value=os.getenv("RAG_PROVIDER", "local"),
            label="Provider",
        )
        answer = gr.Textbox(label="답변")
        audio = gr.Audio(label="음성", autoplay=True)
        run_btn = gr.Button("질문하기")
        run_btn.click(fn=ask_with_tts, inputs=[question, provider], outputs=[answer, audio])
    return demo


app = gr.mount_gradio_app(app, build_gradio(), path="/ui")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
