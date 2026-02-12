import streamlit as st
from pathlib import Path
import re
import getpass
import shutil
import subprocess
import uuid
import sys
import os
import time
import glob
import fitz  # pymupdf
from dotenv import load_dotenv  #  추가 env 로드

load_dotenv()  # 환경변수 로드
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# tts 고유 모듈
from tts_worker import TTSWorker
from memory_store import SessionMemoryStore

# 전역 TTS 워커: 새 질문이 들어오면 이전 재생을 즉시 중단한다.
_TTS_WORKER: TTSWorker | None = None

# =========================================
# 0. 환경 설정
# =========================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
# 파일이 같은 폴더에 없을 경우를 대비해 경로 명시 지정
env_path = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(env_path)

if os.getenv("OPENAI_API_KEY"):
    print("✅ .env 파일로부터 API Key를 성공적으로 로드했습니다.")
else:
    print("⚠️ .env 파일을 찾지 못했거나 키가 설정되지 않았습니다.")

DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_db")
# 대화 이력 SQLite 경로
CHAT_DB_PATH = os.path.join(PROJECT_ROOT, "data", "chat_log.sqlite")
# TTS 입력 경로와 출력 경로를 한 곳에서 관리해 변경 지점을 단일화한다.
TTS_MODEL_PATH = Path(PROJECT_ROOT) / "models" / "melo_yae" / "melo_yae.onnx"
TTS_BERT_PATH = Path(PROJECT_ROOT) / "models" / "melo_yae" / "bert_kor.onnx"
TTS_CONFIG_PATH = Path(PROJECT_ROOT) / "models" / "melo_yae" / "config.json"

# SQLite 저장소
CHAT_STORE = SessionMemoryStore(CHAT_DB_PATH)

# ==========================================
# 1. 화면 기본 설정
# ==========================================
st.set_page_config(page_title="입찰메이트 AI (Hybrid)", page_icon="🤖", layout="wide")

st.title("입찰/공고 분석 AI: 입찰메이트 (Hybrid Edition)")

# 세션 상태 기본값을 먼저 초기화한다. (사이드바/메인 공용)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("last_answer_ready", False)
st.session_state.setdefault("last_q", None)
st.session_state.setdefault("last_a", None)
st.session_state.setdefault("last_tts_path", None)
st.session_state.setdefault("just_answered", False)

# ==========================================
# 2. 사이드바 (설정)
# ==========================================
with st.sidebar:
    st.header("⚙️ 환경 설정")

    if "OPENAI_API_KEY" not in os.environ:
        api_key = st.text_input("OpenAI API Key 입력", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("API Key 저장 완료!")

    st.subheader("모델 선택")
    model_options = ["gpt-5-mini", "gpt-5-nano", "gpt-5"]
    selected_model = st.selectbox("사용할 모델", model_options, index=0)

    st.subheader("검색 가중치 설정")
    dense_weight = st.slider(
        "Dense(의미) 비중",
        0.0,
        1.0,
        0.6,
        0.1,
        help="높을수록 문맥 위주, 낮을수록 키워드 위주",
    )
    sparse_weight = round(1.0 - dense_weight, 1)
    st.caption(f"Sparse(키워드) 비중: {sparse_weight}")

    if st.button("🗑️ 대화 내용 지우기"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("음성 재생")
    audio_placeholder = st.empty()

    if st.session_state.last_tts_path:
        audio_placeholder.empty()
        audio_placeholder.audio(
            st.session_state.last_tts_path,
            format="audio/wav",
        )
    else:
        audio_placeholder.caption("재생할 음성이 아직 없습니다.")

    st.divider()
    st.subheader("피드백")
    if (
        st.session_state.last_answer_ready
        and st.session_state.last_q
        and st.session_state.last_a
    ):
        col_like, col_dislike = st.columns(2)
        with col_like:
            if st.button("👍 좋아요"):
                ok = CHAT_STORE.update_rating(
                    st.session_state.last_q,
                    st.session_state.last_a,
                    1,
                )
                st.toast("저장 완료" if ok else "저장할 대화가 없습니다.")
        with col_dislike:
            if st.button("👎 싫어요"):
                ok = CHAT_STORE.update_rating(
                    st.session_state.last_q,
                    st.session_state.last_a,
                    -1,
                )
                st.toast("저장 완료" if ok else "저장할 대화가 없습니다.")
    else:
        st.caption("답변이 생성된 후 피드백을 남길 수 있습니다.")


# ==========================================
# 3. RAG 체인 설정 (Hybrid & LCEL Fix)
# ==========================================
@st.cache_resource(show_spinner="Hybrid 검색 엔진 가동 중...")
def load_rag_chain(model_name, dense_w, sparse_w):

    if not os.path.exists(DB_PATH):
        st.error(f"데이터베이스 없음: {DB_PATH}")
        return None

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 1. Dense Retriever (Chroma)
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name="bid_rfp_collection",
    )

    dense_retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20}
    )

    # 2. Sparse Retriever (BM25)
    try:
        raw_docs = vectorstore.get()
        docs = []
        for i in range(len(raw_docs["ids"])):
            if raw_docs["documents"][i]:
                docs.append(
                    Document(
                        page_content=raw_docs["documents"][i],
                        metadata=(
                            raw_docs["metadatas"][i] if raw_docs["metadatas"] else {}
                        ),
                    )
                )

        if not docs:
            st.error("DB에 문서가 없습니다.")
            return None

        sparse_retriever = BM25Retriever.from_documents(docs)
        sparse_retriever.k = 5

    except Exception as e:
        st.error(f"BM25 초기화 실패: {e}")
        return None

    # 3. Ensemble Retriever (Hybrid)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever], weights=[dense_w, sparse_w]
    )

    try:
        llm = ChatOpenAI(model=model_name, temperature=0)
    except Exception as e:
        st.error(f"모델 로딩 실패: {e}")
        return None

    # [프롬프트 1] 질문 재구성 (독립적 질문 생성)
    context_q_system_prompt = """
    채팅 기록과 최신 질문이 주어지면, 채팅 기록 없이도 이해할 수 있는 
    '독립적인 질문'으로 재구성하세요. 답변하지 말고 질문만 반환하세요.
    """
    context_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", context_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Chain: (Dict) -> (String)
    history_aware_chain = context_q_prompt | llm | StrOutputParser()

    # [프롬프트 2] 답변 생성 (QA)
    qa_system_prompt = """
    당신은 공공 입찰(RFP) 분석 전문가 '입찰메이트'입니다.
    아래의 [검색된 문서]를 사용하여 질문에 답변하세요.

    규칙:
    1. 문서를 기반으로 사실만 답변하고, 모르면 "문서에 해당 내용이 없습니다"라고 하세요.
    2. 예산, 기간, 날짜 등 숫자를 기재하세요. (숫자 표기 규칙 참고)
    3. 답변은 자연스러운 문장으로만 작성하세요. 목록/불릿/표는 쓰지 마세요.
    4. 답변은 존댓말로 작성하세요.
    5. 문장은 길지 않게 끊어 읽기 쉬운 길이로 유지하세요.
    6. 문단은 2~3문장마다 빈 줄(개행 2개)로 구분하세요.
    7. 괄호는 쓰지 말고, 목록/헤더/컨텍스트 인용은 문장으로 풀어 작성하세요.
    8. 특수문자(% 등)는 한국어로 풀어서 쓰세요.
    9. 출력은 10줄을 넘기지 않게 하세요.

    영어 표기 규칙:
    - 영어 단어는 한국어 음역으로만 표기하세요.
    - 예: dashboard -> 대시보드, dataset -> 데이터셋, isp -> 아이에스피, system -> 시스템.

    숫자 표기 규칙:
    - 금액은 반드시 한글 화폐식으로 작성하세요.
    - 예: 35,750,000원 -> 3천 5백 7십 5만원
    - 날짜는 'YYYY년 MM월 DD일' 형식으로 작성하세요.
    - 예: 2024-06-24 11:00:00 -> 2024년 6월 24일
    - 기간은 'N개월', 'N주', 'N일' 형식으로 작성하세요.

    [검색된 문서]:
    {context}
    """
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # 검색 쿼리 결정 함수
    def get_search_query(input_dict):
        if input_dict.get("chat_history"):
            # 대화 기록이 있으면 재구성 체인 실행 (String 반환)
            return history_aware_chain.invoke(input_dict)
        else:
            # 없으면 사용자 입력 그대로 사용 (String 반환)
            return input_dict["input"]

    # 체인 조립
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    setup_and_retrieval = RunnableParallel(
        {
            # RunnableLambda로 감싸서 문자열을 안전하게 전달
            "context": RunnableLambda(get_search_query) | ensemble_retriever,
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"],
        }
    )

    # 최종 체인
    rag_chain = setup_and_retrieval.assign(
        answer=RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ==========================================
# 4. TTS 유틸리티 함수
# ==========================================
def split_sentences_buffered(buffer: str) -> tuple[list[str], str]:
    # 소수점 보호
    protected = re.sub(r"(?<=\d)\.(?=\d)", "<DOT>", buffer)

    sentences: list[str] = []
    buf: list[str] = []
    i = 0
    while i < len(protected):
        ch = protected[i]
        buf.append(ch)

        # 문장 구두점 기준 분리
        if ch in ".!?。！？":
            sentence = "".join(buf).replace("<DOT>", ".").strip()
            if sentence:
                sentences.append(sentence)
            buf = []
            i += 1
            continue

        # 줄바꿈/문단 경계 기준 분리
        if ch == "\n":
            # 연속 개행을 하나의 경계로 처리
            while i + 1 < len(protected) and protected[i + 1] == "\n":
                i += 1
                buf.append("\n")
            sentence = "".join(buf).replace("<DOT>", ".").strip()
            if sentence:
                sentences.append(sentence)
            buf = []
        i += 1

    remainder = "".join(buf).replace("<DOT>", ".").strip()
    return [s for s in sentences if s], remainder


def _split_sentences_for_tts(text: str) -> list[str]:
    # buffered splitter를 단발 입력에 맞게 래핑한다.
    sentences, remainder = split_sentences_buffered(text)
    if remainder:
        sentences.append(remainder)
    return sentences


def _is_junk_line(line: str) -> bool:
    stripped = re.sub(r"[^0-9A-Za-z가-힣]", "", line)
    if not stripped:
        return True
    if stripped.isdigit():
        return True
    digit_ratio = sum(ch.isdigit() for ch in stripped) / max(len(stripped), 1)
    if digit_ratio > 0.4:
        return True
    if len(stripped) < 4:
        return True
    return False


def _sanitize_answer(text: str) -> str:
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("컨텍스트") or stripped.startswith("[Source"):
            continue
        if _is_junk_line(stripped):
            continue
        cleaned_lines.append(stripped)
    cleaned = " ".join(cleaned_lines)
    cleaned = re.sub(r"\([^)]*\)", "", cleaned)
    cleaned = re.sub(r"\[[^\]]*\]", "", cleaned)
    cleaned = re.sub(r"[^0-9A-Za-z가-힣\s,\.\!\?]", " ", cleaned)
    cleaned = re.sub(r"\d{20,}", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _select_audio_player(preferred: str | None = None) -> list[str] | None:
    if preferred in {"none", "off"}:
        return None
    if preferred in {"ffplay"}:
        path = shutil.which(preferred)
        if not path:
            return None
        return [path, "-autoexit", "-nodisp", "-loglevel", "error"]
    if preferred in {"mpv"}:
        path = shutil.which(preferred)
        if not path:
            return None
        return [path, "--ao=pulse", "--no-video", "--quiet", "--keep-open=no"]
    for candidate in ("ffplay",):
        path = shutil.which(candidate)
        if not path:
            continue
        return [path, "-autoexit", "-nodisp", "-loglevel", "error"]
    return None


# ==========================================
# 5. 탭 인터페이스
# ==========================================

tab1, tab2 = st.tabs(["PDF 뷰어", "채팅"])

# ------------------------------------------
# Tab 1: PDF 뷰어
# ------------------------------------------
with tab1:
    PDF_DIR = os.path.join(PROJECT_ROOT, "data", "pdf")
    pdf_files = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))

    if not pdf_files:
        st.warning(f"PDF 파일이 없습니다: {PDF_DIR}")
    else:
        for pdf_path in pdf_files:
            pdf_name = os.path.basename(pdf_path)
            st.subheader(pdf_name)

            doc = fitz.open(pdf_path)
            total_pages = len(doc)

            page_key = f"page_{pdf_name}"
            slider_key = f"slider_{pdf_name}"
            st.session_state.setdefault(page_key, 1)

            def _on_slider_change(_pk=page_key, _sk=slider_key):
                st.session_state[_pk] = st.session_state[_sk]

            # 페이지 이동
            def _go_prev(_pk=page_key, _sk=slider_key):
                if st.session_state[_pk] > 1:
                    st.session_state[_pk] -= 1
                    st.session_state[_sk] = st.session_state[_pk]

            try:
                player_cmd = _select_audio_player("mpv")
            except Exception as e:
                st.error(f"오디오 플레이어 선택 실패: {e}")

            def _go_next(_pk=page_key, _sk=slider_key, _tp=total_pages):
                if st.session_state[_pk] < _tp:
                    st.session_state[_pk] += 1
                    st.session_state[_sk] = st.session_state[_pk]

            col_left, col_info, col_right = st.columns([1, 3, 1])
            with col_left:
                st.button("◀ 이전", key=f"prev_{pdf_name}", on_click=_go_prev)
            with col_info:
                st.markdown(f"**{st.session_state[page_key]} / {total_pages} 페이지**")
            with col_right:
                st.button("다음 ▶", key=f"next_{pdf_name}", on_click=_go_next)

            st.slider(
                "페이지 이동",
                1,
                total_pages,
                st.session_state[page_key],
                key=slider_key,
                on_change=_on_slider_change,
            )

            page_num = st.session_state[page_key] - 1
            page = doc[page_num]
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")

            st.image(img_bytes, width="stretch")
            doc.close()
            st.divider()

    # 키보드 좌/우 화살표로 페이지 넘기기
    st.components.v1.html(
        """
    <script>
    const doc = window.parent.document;
    doc.addEventListener('keydown', function(e) {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        if (e.key === 'ArrowLeft') {
            const btn = doc.querySelectorAll('button');
            for (const b of btn) {
                if (b.innerText.includes('이전')) { b.click(); break; }
            }
        } else if (e.key === 'ArrowRight') {
            const btn = doc.querySelectorAll('button');
            for (const b of btn) {
                if (b.innerText.includes('다음')) { b.click(); break; }
            }
        }
    });
    </script>
    """,
        height=0,
    )

# ------------------------------------------
# Tab 2: 채팅
# ------------------------------------------
with tab2:
    st.markdown(
        """
- **Dense(의미)**: 문맥과 의미를 파악하여 검색 (Chroma)
- **Sparse(키워드)**: 공고 번호, 예산, 모델명 등 정확한 매칭 검색 (BM25)
"""
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_answer_ready" not in st.session_state:
        st.session_state.last_answer_ready = False
    if "last_q" not in st.session_state:
        st.session_state.last_q = None
    if "last_a" not in st.session_state:
        st.session_state.last_a = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("질문을 입력하세요..."):

        if "OPENAI_API_KEY" not in os.environ:
            st.warning("API Key가 필요합니다.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.last_answer_ready = False
        st.session_state.last_q = None
        st.session_state.last_a = None
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = load_rag_chain(selected_model, dense_weight, sparse_weight)

            if chain:
                history_langchain = []
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        history_langchain.append(HumanMessage(content=msg["content"]))
                    else:
                        history_langchain.append(AIMessage(content=msg["content"]))

                message_placeholder = st.empty()
                full_response = ""
                source_docs = []

                try:
                    player_cmd = _select_audio_player("ffplay")

                    full_response = ""
                    source_documents = []
                    tts_buffer = ""
                    out_dir = os.path.join(PROJECT_ROOT, "data", "answer")
                    os.makedirs(out_dir, exist_ok=True)

                    # 새 질문이 들어오면 기존 TTS 재생을 중단하고 큐를 비운다.
                    if _TTS_WORKER is not None:
                        _TTS_WORKER.cancel()

                    tts_worker = TTSWorker(
                        model_path=TTS_MODEL_PATH,
                        bert_path=TTS_BERT_PATH,
                        config_path=TTS_CONFIG_PATH,
                        out_dir=out_dir,
                        device="cpu",
                        player_cmd=player_cmd,
                        sanitize_fn=_sanitize_answer,
                        split_fn=_split_sentences_for_tts,
                    )
                    tts_worker.start()
                    _TTS_WORKER = tts_worker

                    for chunk in chain.stream(
                        {"input": query, "chat_history": history_langchain}
                    ):
                        if "answer" in chunk:
                            text = chunk["answer"]
                            full_response += text
                            tts_buffer += text

                            sentences, tts_buffer = split_sentences_buffered(tts_buffer)
                            for sent in sentences:
                                tts_worker.enqueue(sent)
                            message_placeholder.markdown(full_response + "▌")

                        if "context" in chunk:
                            source_docs = chunk["context"]

                    if tts_buffer.strip():
                        sentences, remainder = split_sentences_buffered(
                            tts_buffer.strip()
                        )
                        for sent in sentences:
                            tts_worker.enqueue(sent)
                        if remainder:
                            tts_worker.enqueue(remainder)

                    tts_worker.close()
                    last_path = tts_worker.last_path()
                    if last_path:
                        st.session_state.last_tts_path = last_path
                        audio_placeholder.empty()
                        audio_placeholder.audio(
                            st.session_state.last_tts_path,
                            format="audio/wav",
                        )

                    message_placeholder.markdown(full_response)

                    # 질문/답변 저장 (rating은 NULL)
                    CHAT_STORE.save_turn(query, full_response)
                    st.session_state.last_answer_ready = True
                    st.session_state.last_q = query
                    st.session_state.last_a = full_response
                    st.session_state.just_answered = True

                    if source_docs:
                        with st.expander("📚 참고 문서 확인하기 (Hybrid 검색)"):
                            seen = set()
                            for i, doc in enumerate(source_docs):
                                source = os.path.basename(
                                    doc.metadata.get("source", "Unknown")
                                )
                                page = doc.metadata.get("page", 0)
                                preview = doc.page_content[:40].replace("\n", " ")

                                key = f"{source}p{page}"
                                if key not in seen:
                                    st.markdown(f"**{i+1}. {source}** (Page {page+1})")
                                    st.caption(f"내용: {preview}...")
                                    seen.add(key)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response}
                    )

                except Exception as e:
                    st.error(f"에러 발생: {e}")

        if st.session_state.just_answered:
            st.session_state.just_answered = False
            st.rerun()
