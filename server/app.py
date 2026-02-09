from __future__ import annotations

"""
Gradio 단독 UI

실행:
  uv run python -m server.app
"""

import os
import re
import uuid

import gradio as gr

from rag.openai_pipeline import OpenAIRAGPipeline
from rag.pipeline import RAGPipeline
from tts_runtime.infer_onnx import infer_tts_onnx

# 파이프라인 인스턴스를 재사용하기 위한 캐시
_PIPELINE_CACHE: dict[str, object] = {}


def _build_pipeline(provider: str):
    """
    프로바이더별 파이프라인을 생성한다.
    """
    if provider == "openai":
        # OpenAI 전용 파이프라인
        return OpenAIRAGPipeline()
    # 기본은 로컬 vLLM 파이프라인
    return RAGPipeline()


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
    # provider 지정 (미지정 시 환경변수 기본값 사용)
    mode = provider or os.getenv("RAG_PROVIDER", "local")
    # provider에 해당하는 파이프라인을 가져온다.
    pipeline = get_pipeline(mode)
    # session_id는 SQLite 세션 메모리 키다.
    # 파이프라인의 ask는 문자열 답변을 반환한다.
    return pipeline.ask(question, session_id=session_id)  # type: ignore[no-any-return]


def _tts_paths():
    """
    TTS 모델/출력 경로를 구성한다.
    """
    # 프로젝트 루트 경로
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


def _extract_last_turn(history: list[dict[str, str]]) -> tuple[str | None, str | None]:
    """
    Chatbot 히스토리에서 마지막 질문/답변을 추출한다.
    """

    def _normalize_content(value: object) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(str(item.get("text", "")))
            return " ".join(p for p in parts if p).strip()
        return ""

    last_answer: str | None = None
    last_question: str | None = None
    for item in reversed(history):
        # Gradio 버전에 따라 메시지 형식이 dict 또는 [user, assistant]일 수 있다.
        if isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg = (item[0] or "").strip() if isinstance(item[0], str) else ""
            bot_msg = (item[1] or "").strip() if isinstance(item[1], str) else ""
            if last_answer is None and bot_msg:
                last_answer = bot_msg
            if last_answer is not None and user_msg:
                last_question = user_msg
                break
            continue
        if isinstance(item, dict):
            role = item.get("role")
            if role == "assistant" and last_answer is None:
                last_answer = _normalize_content(item.get("content", ""))
                continue
            if role == "user" and last_answer is not None:
                last_question = _normalize_content(item.get("content", ""))
                break
    return last_question, last_answer


def _save_feedback(
    history: list[dict[str, str]],
    provider_choice: str,
    session_id: str,
    rating: int,
) -> str:
    """
    좋아요/싫어요 피드백을 SQLite에 저장한다.
    """
    history = history or []
    question, answer = _extract_last_turn(history)
    if not question or not answer:
        return "skip"
    pipeline = get_pipeline(provider_choice)
    memory = getattr(pipeline, "memory", None)
    if memory is None:
        return "no-memory"
    memory.save_feedback(session_id, provider_choice, question, answer, rating)
    return "ok"


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
        yield history, None
        # TTS는 전체 답변을 한 번에 합성해 품질을 보장한다.
        wav_path = tts_only(answer)
        yield history, wav_path

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
    /* 전역 배경/폰트 */
    body, .gradio-container {
      font-family: "Sora", "IBM Plex Sans", "Noto Sans KR", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    /* Gradio 기본 폭 제한 해제 */
    .gradio-container { max-width: none !important; }
    /* 기본 footer 제거 */
    footer { display: none !important; }
    /* 전체 레이아웃 래퍼 */
    .rag-shell {
      max-width: 1320px;
      margin: 0 auto;
      padding: 22px 16px 28px;
    }
    /* 상단 헤더 */
    .rag-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }
    /* 타이틀 텍스트 */
    .rag-title {
      font-weight: 700;
      font-size: 26px;
      letter-spacing: -0.4px;
    }
    /* 서브타이틀 */
    .rag-subtitle { color: var(--muted); margin-top: 4px; }
    /* 우측 칩 */
    .rag-chip {
      font-size: 12px;
      color: var(--muted);
      background: #eef2f7;
      padding: 4px 10px;
      border-radius: 999px;
    }
    /* 카드 공통 */
    .rag-card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 14px;
      box-shadow: var(--shadow);
    }
    /* 채팅 영역 최소 높이 */
    .rag-chat {
      min-height: 620px;
      border-radius: 18px;
    }
    /* 우측 패널 */
    .rag-panel {
      background: transparent;
      border-radius: 18px;
    }
    /* 챗봇 컨테이너 */
    .rag-chatbot {
      border-radius: 18px;
    }
    /* 메시지 말풍선 */
    .rag-chatbot .message {
      border-radius: 16px;
      padding: 12px 14px;
      max-width: 82%;
      line-height: 1.55;
    }
    /* 사용자 말풍선 */
    .rag-chatbot .message.user {
      background: var(--bubble-user);
      border: 1px solid #cbeade;
      margin-left: auto;
    }
    /* 어시스턴트 말풍선 */
    .rag-chatbot .message.bot {
      background: var(--bubble-bot);
      border: 1px solid var(--border);
      margin-right: auto;
    }
    /* 입력창 */
    .rag-input {
      border: 1px solid var(--border) !important;
      border-radius: 14px !important;
    }
    /* 전송 버튼 */
    .rag-send {
      background: var(--accent) !important;
      color: #fff !important;
      border-radius: 12px !important;
      border: none !important;
    }
    /* 보조 버튼(초기화/피드백) */
    .rag-clear {
      border-radius: 12px !important;
    }
    /* 오디오 카드 */
    .rag-audio .wrap { background: var(--panel); }
    /* 우측 패널 타이틀 */
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
                "<div class='rag-chip'>Gradio</div>"
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
                        send_btn = gr.Button(
                            "보내기",
                            variant="primary",
                            elem_classes="rag-send",
                            elem_id="send-btn",
                        )
                        clear_btn = gr.Button("초기화", elem_classes="rag-clear")
                        like_btn = gr.Button("👍", elem_classes="rag-clear")
                        dislike_btn = gr.Button("👎", elem_classes="rag-clear")
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
                        audio = gr.Audio(
                            label="",
                            autoplay=True,
                            interactive=False,
                            streaming=False,
                            elem_id="rag-audio",
                        )

            # 오디오 자동 재생을 위해 사용자 제스처 이후 재생 트리거를 연결한다.
            gr.HTML(
                """
                <script>
                (function() {
                  // 브라우저 정책상 사용자 제스처가 있어야 자동 재생이 허용된다.
                  let userInteracted = false;
                  const markInteracted = () => { userInteracted = true; };
                  window.addEventListener("click", markInteracted, { once: true });
                  window.addEventListener("keydown", markInteracted, { once: true });

                  function tryPlayOnce() {
                    // 제스처가 없으면 재생 시도하지 않는다.
                    if (!userInteracted) return;
                    // Gradio Audio 컴포넌트의 루트 DOM
                    const root = document.getElementById("rag-audio");
                    if (!root) return;
                    // 실제 <audio> 엘리먼트
                    const audio = root.querySelector("audio");
                    if (!audio) return;
                    // autoplay/preload 보장
                    audio.autoplay = true;
                    audio.preload = "auto";
                    // src가 있고 일시정지 상태면 재생 시도
                    if (audio.paused && audio.src) {
                      audio.load();
                      audio.play().catch(() => {});
                    }
                  }

                  function scheduleRetries(times, delayMs) {
                    // 일정 횟수 재시도로 로딩 지연/DOM 교체에 대응
                    let count = 0;
                    const id = setInterval(() => {
                      tryPlayOnce();
                      count += 1;
                      if (count >= times) clearInterval(id);
                    }, delayMs);
                  }

                  function tryAttach() {
                    // Audio DOM을 찾아 이벤트 리스너를 등록
                    const root = document.getElementById("rag-audio");
                    if (!root) return;
                    const audio = root.querySelector("audio");
                    if (!audio) return;

                    const tryPlay = () => tryPlayOnce();

                    // 로딩 단계별 이벤트에서 재생 시도
                    audio.addEventListener("loadeddata", tryPlay);
                    audio.addEventListener("canplay", tryPlay);
                    audio.addEventListener("loadedmetadata", tryPlay);
                    audio.addEventListener("durationchange", tryPlay);

                    // src가 바뀔 때마다 재생을 재시도한다.
                    const srcObserver = new MutationObserver(() => {
                      if (audio) {
                        audio.autoplay = true;
                        audio.preload = "auto";
                        audio.load();
                      }
                      tryPlay();
                      scheduleRetries(6, 250);
                    });
                    srcObserver.observe(audio, { attributes: true, attributeFilter: ["src"] });

                    // 일정 주기로도 재시도 (Gradio 내부 DOM 교체 대응)
                    if (!window._ragAutoplayTimer) {
                      window._ragAutoplayTimer = setInterval(() => {
                        tryPlayOnce();
                      }, 500);
                    }
                  }

                  // 전체 DOM 변경 시 Audio를 다시 탐색해 연결한다.
                  const observer = new MutationObserver(() => {
                    tryAttach();
                  });
                  observer.observe(document.body, { childList: true, subtree: true });
                  tryAttach();

                  // 전송 버튼 클릭 시 사용자 제스처 처리 + 재생 재시도
                  const sendBtn = document.getElementById("send-btn");
                  if (sendBtn) {
                    sendBtn.addEventListener("click", () => {
                      userInteracted = true;
                      tryPlayOnce();
                      scheduleRetries(8, 250);
                    });
                  }
                })();
                </script>
                """
            )

        # 버튼 클릭 -> 메시지 처리
        session_state = gr.State(value=uuid.uuid4().hex)
        feedback_state = gr.State(value="idle")

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
        like_btn.click(
            fn=lambda h, p, s: _save_feedback(h, p, s, 1),
            inputs=[chatbot, provider, session_state],
            outputs=[feedback_state],
        )
        dislike_btn.click(
            fn=lambda h, p, s: _save_feedback(h, p, s, -1),
            inputs=[chatbot, provider, session_state],
            outputs=[feedback_state],
        )
        # 초기화 -> 히스토리/오디오 리셋
        clear_btn.click(lambda: ([], None), outputs=[chatbot, audio])
    return demo, css


def main() -> None:
    demo, demo_css = build_gradio()
    demo.launch(server_name="0.0.0.0", server_port=8000, css=demo_css, share=True)


if __name__ == "__main__":
    main()
