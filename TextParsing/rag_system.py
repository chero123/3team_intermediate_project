import os
import re
import sys
import getpass
import shutil
import subprocess
from pathlib import Path
import uuid
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

# tts 모듈
from tts_worker import TTSWorker

# ==========================================
# 0. 환경 설정 및 초기화
# ==========================================
load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_db")
# TTS 입력 경로와 출력 경로를 한 곳에서 관리해 변경 지점을 단일화한다.
TTS_MODEL_PATH = Path(PROJECT_ROOT) / "models" / "melo_yae" / "melo_yae.onnx"
TTS_BERT_PATH = Path(PROJECT_ROOT) / "models" / "melo_yae" / "bert_kor.onnx"
TTS_CONFIG_PATH = Path(PROJECT_ROOT) / "models" / "melo_yae" / "config.json"
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key를 입력하세요: ")

SELECTED_MODEL = "gpt-5-mini" 

print(f"시스템 초기화 중... (Model: {SELECTED_MODEL})")

# ==========================================
# 1. 하이브리드 리트리버 설정
# ==========================================
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

if not os.path.exists(DB_PATH):
    print(f"오류: DB 경로({DB_PATH})가 존재하지 않습니다.")
    sys.exit()

# [1] Dense Retriever (Chroma)
vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings,
    collection_name="bid_rfp_collection"
)

dense_retriever = vectorstore.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 5, "fetch_k": 20} # 하이브리드를 위해 k값 약간 조정
)

# [2] Sparse Retriever (BM25) 
print("BM25 인덱스 생성 중... (데이터 로드)")
try:
    # DB에 저장된 모든 문서를 가져와서 BM25 인덱스를 만듭니다.
    raw_docs = vectorstore.get() 
    docs = []
    for i in range(len(raw_docs['ids'])):
        if raw_docs['documents'][i]: 
            docs.append(Document(
                page_content=raw_docs['documents'][i],
                metadata=raw_docs['metadatas'][i] if raw_docs['metadatas'] else {}
            ))
    
    if not docs:
        print("오류: DB에 문서가 비어 있습니다.")
        sys.exit()
        
    sparse_retriever = BM25Retriever.from_documents(docs)
    sparse_retriever.k = 5  # 키워드 매칭 문서 5개
    print("BM25 인덱스 생성 완료")

except Exception as e:
    print(f"BM25 초기화 실패: {e}")
    sys.exit()

# [3] Ensemble Retriever (Hybrid) - 결합
ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.6, 0.4]  # Dense(의미) 60%, Sparse(키워드) 40% 비중
)

# ==========================================
# 2. LLM & 프롬프트 설정
# ==========================================
try:
    llm = ChatOpenAI(model=SELECTED_MODEL, temperature=0)
except Exception as e:
    print(f"모델 초기화 오류: {e}")
    sys.exit()

# 2-1. 질문 재구성 (Contextualize Query)
contextualize_q_system_prompt = """
채팅 기록과 최신 사용자 질문이 주어지면, 
이 질문이 채팅 기록의 맥락을 참조하고 있을 경우 
채팅 기록 없이도 이해할 수 있는 '독립적인 질문'으로 재구성하세요.
질문에 답하지 말고, 질문을 재구성하거나 그대로 반환하기만 하세요.
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# 질문 재구성 체인
history_aware_retriever = contextualize_q_prompt | llm | StrOutputParser()

# 2-2. 답변 생성 (QA)
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

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# ==========================================
# 3. 체인 조립 (LCEL 방식)
# ==========================================

# 문서 포맷팅 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# (1) 맥락 고려하여 검색 쿼리 결정
def contextualized_question(input: dict):
    if input.get("chat_history"):
        return history_aware_retriever
    else:
        return input["input"]

# (2) 전체 RAG 체인 구성
# 변경점: retriever -> ensemble_retriever로 교체
setup_and_retrieval = RunnableParallel(
    {
        "context": contextualized_question | ensemble_retriever, 
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"],
    }
)

def format_context_for_prompt(input_dict):
    return {
        "context": format_docs(input_dict["context"]),
        "input": input_dict["input"],
        "chat_history": input_dict["chat_history"]
    }

# 최종 체인: 검색 -> 포맷팅 -> 답변생성 -> 파싱
rag_chain = setup_and_retrieval.assign(
    answer= format_context_for_prompt | qa_prompt | llm | StrOutputParser()
)

# ==========================================
# 4. TTS 보조 함수
# ==========================================

def _split_sentences(text: str) -> list[str]:
    # 문장 경계를 기준으로 자른 뒤 구두점을 유지한다.
    protected = re.sub(r"(?<=\d)\.(?=\d)", "<DOT>", text)
    parts = re.split(r"([.!?。！？]+)", protected)
    sentences: list[str] = []
    buf = ""
    for part in parts:
        if not part:
            continue
        buf += part
        if re.fullmatch(r"[.!?。！？]+", part):
            restored = buf.strip().replace("<DOT>", ".")
            if restored:
                sentences.append(restored)
            buf = ""
    if buf.strip():
        sentences.append(buf.strip().replace("<DOT>", "."))
    return sentences


def split_sentences_buffered(buffer: str) -> tuple[list[str], str]:
    # 소수점 보호
    protected = re.sub(r"(?<=\d)\.(?=\d)", "<DOT>", buffer)
    parts = re.split(r"([.!?。！？]+)", protected)

    sentences = []
    buf = ""
    for p in parts:
        if not p:
            continue
        buf += p
        if re.fullmatch(r"[.!?。！？]+", p):
            sentences.append(buf.replace("<DOT>", ".").strip())
            buf = ""
    return [s for s in sentences if s], buf.replace("<DOT>", ".")


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
    cleaned = re.sub(r"[^0-9A-Za-z가-힣\s\.\!\?]", " ", cleaned)
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
    for candidate in ("ffplay",):
        path = shutil.which(candidate)
        if not path:
            continue
        return [path, "-autoexit", "-nodisp", "-loglevel", "error"]
    return None


# ==========================================
# 5. 메인 실행 루프
# ==========================================
chat_history = [] 

print("\n" + "="*60)
print(f"입찰메이트 AI ({SELECTED_MODEL}) - Hybrid RAG Version")
print("="*60)

while True:
    query = input("\n질문 입력 (q로 종료): ")
    if query.lower() in ["q", "quit", "exit"]:
        print("종료합니다.")
        break
    
    if not query.strip():
        continue

    print("\n답변 생성 중...\n")
    
    try:
        player_cmd = _select_audio_player("ffplay")

        full_response = ""
        source_documents = []
        tts_buffer = ""
        out_dir = os.path.join(PROJECT_ROOT, "data", "answer")
        os.makedirs(out_dir, exist_ok=True)

        tts_worker = TTSWorker(
            model_path=TTS_MODEL_PATH,
            bert_path=TTS_BERT_PATH,
            config_path=TTS_CONFIG_PATH,
            out_dir=out_dir,
            device="cpu",
            player_cmd=player_cmd,
            sanitize_fn=_sanitize_answer,
            split_fn=_split_sentences,
        )
        tts_worker.start()

        for chunk in rag_chain.stream({"input": query, "chat_history": chat_history}):
            if "answer" in chunk:
                text = chunk["answer"]
                # 스트리밍 텍스트는 즉시 출력한다.
                print(text, end="", flush=True)
                full_response += text
                tts_buffer += text

                sentences, tts_buffer = split_sentences_buffered(tts_buffer)
                for sent in sentences:
                    tts_worker.enqueue(sent)

            if "context" in chunk:
                source_documents = chunk["context"]

        if tts_buffer.strip():
            sentences = _split_sentences(tts_buffer.strip())
            for sent in sentences:
                tts_worker.enqueue(sent)

        tts_worker.close()

        print("\n")

        # 출처 표시
        if source_documents:
            print("-" * 60)
            print("[참고 문서 (Hybrid 검색)]")
            seen_sources = set()
            for doc in source_documents:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", 0)
                filename = os.path.basename(source)
                
                # 문서 내용 미리보기 (앞 30자)
                preview = doc.page_content[:30].replace("\n", " ")
                
                source_key = f"{filename} (p.{page+1})"
                if source_key not in seen_sources:
                    print(f"   • {filename} [Page: {page+1}] - {preview}...")
                    seen_sources.add(source_key)
            print("-" * 60)

        # 대화 기록 업데이트
        chat_history.extend([
            HumanMessage(content=query),
            AIMessage(content=full_response)
        ])

    except Exception as e:
        print(f"\n에러 발생: {e}")
