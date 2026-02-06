"""
FastAPI + Gradio 통합 코드

실행:
  uv run python -m server.app
"""
from __future__ import annotations

import os
import re
import uuid
from typing import Literal

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import uvicorn
from pydantic import BaseModel

from rag.pipeline import RAGPipeline
from rag.openai_pipeline import OpenAIRAGPipeline
from tts_runtime.infer_onnx import infer_tts_onnx


class AskRequest(BaseModel):
    """
    API 요청 바디

    Args:
        question: 사용자 질문
        provider: local | openai (선택)
    """

    # 사용자 질문
    question: str
    # local / openai 중 선택
    provider: Literal["local", "openai"] | None = None
    # TTS 사용 여부(확장용)
    tts: bool | None = None
    # 세션 식별자(멀티턴 메모리용)
    session_id: str | None = None


def _build_pipeline(provider: str):
    """
    프로바이더별 파이프라인을 생성한다.
    """
    if provider == "openai":
        # OpenAI 전용 파이프라인은 필요할 때만 import한다.

        return OpenAIRAGPipeline()
    # 기본은 로컬 vLLM 파이프라인
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
    # 파이프라인은 무겁기 때문에 1회 생성 후 재사용한다.
    if provider not in _PIPELINE_CACHE:
        _PIPELINE_CACHE[provider] = _build_pipeline(provider)
    return _PIPELINE_CACHE[provider]


def ask(question: str, provider: str | None = None, session_id: str | None = None) -> str:
    """
    질문을 파이프라인에 전달하고 답변을 반환한다.
    """
    # provider 지정
    mode = provider or os.getenv("RAG_PROVIDER", "local")
    pipeline = get_pipeline(mode)
    # session_id는 SQLite 세션 메모리의 키로 사용되어 멀티턴 문맥을 유지한다.
    # 파이프라인의 ask는 문자열 답변을 반환한다.
    return pipeline.ask(question, session_id=session_id)  # type: ignore[no-any-return]


def _tts_paths():
    """
    TTS 모델/출력 경로를 구성한다.
    """
    base = os.path.dirname(os.path.dirname(__file__))
    # TTS 모델/설정 파일 위치
    model_path = os.path.join(base, "models", "melo_yae", "melo_yae.onnx")
    bert_path = os.path.join(base, "models", "melo_yae", "bert_kor.onnx")
    config_path = os.path.join(base, "models", "melo_yae", "config.json")
    # 결과 wav 저장 디렉토리/파일명
    out_dir = os.path.join(base, "data", "answer")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"tts_{uuid.uuid4().hex}.wav")
    return model_path, bert_path, config_path, out_path


def _strip_reference_block(text: str) -> str:
    # TTS에서 참고문헌 블록을 읽지 않도록 제거한다.
    if not text:
        return text
    lines = text.splitlines()
    cleaned: list[str] = []
    skipping = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[참고 문헌]") or stripped.startswith("참고문헌"):
            skipping = True
        if skipping:
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def tts_only(text: str, device: str = "cuda") -> str:
    """
    텍스트만 받아 TTS를 수행하고 wav 경로를 반환한다.
    """
    model_path, bert_path, config_path, out_path = _tts_paths()
    text = _strip_reference_block(text)
    infer_tts_onnx(
        onnx_path=model_path,
        bert_onnx_path=bert_path,
        config_path=config_path,
        text=text,
        speaker_id=0,
        language="KR",
        device=device,
        out_path=out_path,
    )
    return out_path


def ask_with_tts(question: str, provider: str | None = None, session_id: str | None = None):
    """
    질문 -> 답변 생성 -> TTS wav 생성까지 수행한다.
    """
    # 1) 텍스트 답변 생성
    # session_id는 SQLite 세션 메모리에서 문맥을 복원하는 데 사용된다.
    answer = ask(question, provider, session_id=session_id)
    # 2) TTS 모델 경로 준비
    model_path, bert_path, config_path, out_path = _tts_paths()
    # 3) TTS 수행 (wav 저장 포함)
    tts_text = _strip_reference_block(answer)
    audio = infer_tts_onnx(
        onnx_path=model_path,
        bert_onnx_path=bert_path,
        config_path=config_path,
        text=tts_text,
        speaker_id=0,
        language="KR",
        device="cuda",
        out_path=out_path,
    )
    # Gradio에는 파일 경로만 넘겨도 자동 로딩된다.
    return answer, out_path

app = FastAPI(title="RAG API")


@app.get("/")
def root_redirect():
    return RedirectResponse(url="/ui")

@app.post("/api/ask")
def ask_api(payload: AskRequest):
    """
    API 엔드포인트: 질문 -> 답변
    """
    # REST API용 단순 텍스트 응답
    # session_id가 있으면 동일 세션의 검색 문서를 재활용한다.
    answer = ask(payload.question, payload.provider, payload.session_id)
    return {"answer": answer}


@app.post("/api/ask_tts")
def ask_tts_api(payload: AskRequest):
    """
    API 엔드포인트: 질문 -> 답변 -> TTS
    """
    # REST API용 TTS 응답
    # session_id가 있으면 동일 세션의 검색 문서를 재활용한다.
    answer, wav_path = ask_with_tts(payload.question, payload.provider, payload.session_id)
    return {"answer": answer, "wav_path": wav_path}


def build_gradio():
    """
    Gradio UI 구성 (API와 동일한 ask 함수 사용)
    """
    def chat_with_tts(
        message: str,
        history: list[dict[str, str]],
        provider_choice: str,
        session_id: str,
    ):
        """
        Gradio Chatbot 콜백:
        - 입력 메시지로 답변 생성
        - 히스토리 갱신
        - TTS wav 경로 반환
        """
        history = history or []
        history.append({"role": "user", "content": message})
        # 텍스트 답변을 먼저 만들고, 텍스트를 즉시 표시한다.
        # session_id는 SQLite 세션 메모리에 저장된 문서를 불러오기 위한 키다.
        answer = ask(message, provider_choice, session_id=session_id)
        history.append({"role": "assistant", "content": answer})
        # 텍스트는 먼저 출력, 오디오는 이후에 업데이트한다.
        yield history, gr.update(value=None, autoplay=True)
        # TTS는 전체 답변을 한 번만 합성한다.
        wav_path = tts_only(answer)
        yield history, gr.update(value=wav_path, autoplay=True)

    # ChatGPT 스타일에 더 근접한 레이아웃을 위한 스타일 정의
    css = """
    :root {
      --bg: #f7f7f8;
      --panel: #ffffff;
      --ink: #111827;
      --accent: #10a37f;
      --muted: #6b7280;
      --border: #e5e7eb;
      --bubble-user: #e9f6f2;
      --bubble-bot: #ffffff;
      --shadow: 0 12px 30px rgba(16, 24, 40, 0.08);
    }
    body, .gradio-container {
      font-family: "Sora", "IBM Plex Sans", "Noto Sans KR", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    .gradio-container { max-width: none !important; }
    footer { display: none !important; }
    .rag-shell {
      max-width: 1320px;
      margin: 0 auto;
      padding: 22px 16px 28px;
    }
    .rag-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }
    .rag-title {
      font-weight: 700;
      font-size: 26px;
      letter-spacing: -0.4px;
    }
    .rag-subtitle { color: var(--muted); margin-top: 4px; }
    .rag-chip {
      font-size: 12px;
      color: var(--muted);
      background: #eef2f7;
      padding: 4px 10px;
      border-radius: 999px;
    }
    .rag-card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 14px;
      box-shadow: var(--shadow);
    }
    .rag-chat {
      min-height: 620px;
      border-radius: 18px;
    }
    .rag-panel {
      background: transparent;
      border-radius: 18px;
    }
    .rag-chatbot {
      border-radius: 18px;
    }
    .rag-chatbot .message {
      border-radius: 16px;
      padding: 12px 14px;
      max-width: 82%;
      line-height: 1.55;
    }
    .rag-chatbot .message.user {
      background: var(--bubble-user);
      border: 1px solid #cbeade;
      margin-left: auto;
    }
    .rag-chatbot .message.bot {
      background: var(--bubble-bot);
      border: 1px solid var(--border);
      margin-right: auto;
    }
    .rag-input {
      border: 1px solid var(--border) !important;
      border-radius: 14px !important;
    }
    .rag-send {
      background: var(--accent) !important;
      color: #fff !important;
      border-radius: 12px !important;
      border: none !important;
    }
    .rag-clear {
      border-radius: 12px !important;
    }
    .rag-audio .wrap { background: var(--panel); }
    .rag-side-title {
      font-weight: 600;
      margin-bottom: 8px;
    }
    """

    # Gradio 레이아웃 구성
    with gr.Blocks() as demo:
        with gr.Column(elem_classes="rag-shell"):
            gr.Markdown(
                "<div class='rag-header'>"
                "<div>"
                "<div class='rag-title'>RAG Chat</div>"
                "<div class='rag-subtitle'>문서 기반 답변 · TTS 출력</div>"
                "</div>"
                "<div class='rag-chip'>FastAPI + Gradio</div>"
                "</div>"
            )

            with gr.Row():
                with gr.Column(scale=8):
                    with gr.Column(elem_classes="rag-card"):
                        chatbot = gr.Chatbot(
                            label="대화",
                            height=560,
                            elem_classes="rag-chatbot rag-chat",
                        )
                    message = gr.Textbox(
                        label="질문 입력",
                        placeholder="질문을 입력하고 Enter",
                        lines=2,
                        elem_classes="rag-input",
                    )
                    with gr.Row():
                        send_btn = gr.Button("보내기", variant="primary", elem_classes="rag-send")
                        clear_btn = gr.Button("초기화", elem_classes="rag-clear")
                with gr.Column(scale=4, elem_classes="rag-panel"):
                    with gr.Column(elem_classes="rag-card"):
                        gr.Markdown("<div class='rag-side-title'>Provider</div>")
                        provider = gr.Dropdown(
                            choices=["local", "openai"],
                            value=os.getenv("RAG_PROVIDER", "local"),
                            label="",
                        )
                    with gr.Column(elem_classes="rag-card rag-audio"):
                        gr.Markdown("<div class='rag-side-title'>음성</div>")
                        audio = gr.Audio(label="", autoplay=True)

        # 버튼 클릭 -> 메시지 처리
        session_state = gr.State(value=uuid.uuid4().hex)

        send_btn.click(
            fn=chat_with_tts,
            inputs=[message, chatbot, provider, session_state],
            outputs=[chatbot, audio],
        )
        # Enter 제출 -> 메시지 처리
        message.submit(
            fn=chat_with_tts,
            inputs=[message, chatbot, provider, session_state],
            outputs=[chatbot, audio],
        )
        # 초기화 -> 히스토리/오디오 리셋
        clear_btn.click(lambda: ([], None), outputs=[chatbot, audio])
    return demo, css


demo, demo_css = build_gradio()
app = gr.mount_gradio_app(app, demo, path="/ui")


def main() -> None:
    # Uvicorn으로 FastAPI 실행
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
