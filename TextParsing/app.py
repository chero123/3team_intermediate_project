import streamlit as st
import os
import time
from langchain_chroma import Chroma  
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# ==========================================
# 1. 화면 기본 설정
# ==========================================
st.set_page_config(
    page_title="입찰메이트 AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 입찰/공고 분석 AI: 입찰메이트")
st.markdown("공공 입찰 공고문(RFP)에 대해 궁금한 점을 물어보세요!")

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
    # 프로젝트 가이드 기준 모델
    model_options = ["gpt-5-mini", "gpt-5-nano", "gpt-5"]
    selected_model = st.selectbox(
        "사용할 모델", 
        model_options, 
        index=0
    )
    
    # 대화 내용 초기화 버튼
    if st.button("🗑️ 대화 내용 지우기"):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# 3. RAG 체인 설정 (LCEL & Memory 적용)
# ==========================================
@st.cache_resource(show_spinner="AI 두뇌 로딩 중...")
def load_rag_chain(model_name):
    # 경로 설정
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_db")
    
    if not os.path.exists(DB_PATH):
        st.error(f"데이터베이스 없음: {DB_PATH}")
        return None

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # DB 로드
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name="bid_rfp_collection"
    )
    
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10}
    )
    
    try:
        llm = ChatOpenAI(model=model_name, temperature=0)
    except Exception as e:
        st.error(f"모델 로딩 실패: {e}")
        return None

    # [1] 질문 재구성 (Contextualize)
    context_q_system_prompt = """
    채팅 기록과 최신 질문이 주어지면, 채팅 기록 없이도 이해할 수 있는 
    '독립적인 질문'으로 재구성하세요. 답변하지 말고 질문만 반환하세요.
    """
    context_q_prompt = ChatPromptTemplate.from_messages([
        ("system", context_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = context_q_prompt | llm | StrOutputParser()

    # [2] 답변 생성 (QA)
    qa_system_prompt = """
    당신은 공공 입찰(RFP) 분석 전문가입니다.
    [검색된 문서]를 기반으로 질문에 답변하세요.
    
    규칙:
    1. 문서에 있는 사실만 답변하고, 모르면 모른다고 하세요.
    2. 답변 끝에 참고한 [문서명]을 언급하지 마세요. (별도로 표시됩니다)
    
    [검색된 문서]:
    {context}
    """
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # [3] 체인 조립 (LCEL)
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    def contextualized_question(input: dict):
        if input.get("chat_history"):
            return history_aware_retriever
        else:
            return input["input"]

    setup_and_retrieval = RunnableParallel(
        {
            "context": contextualized_question | retriever,
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"],
        }
    )
    
    # 최종 체인
    rag_chain = setup_and_retrieval.assign(
        answer=RunnablePassthrough.assign(
            context=lambda x: format_docs(x["context"])
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# ==========================================
# 4. 채팅 인터페이스
# ==========================================

# 세션 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 대화 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
if query := st.chat_input("질문을 입력하세요..."):
    
    if "OPENAI_API_KEY" not in os.environ:
        st.warning("API Key가 필요합니다.")
        st.stop()

    # 사용자 메시지 표시
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 답변 생성
    with st.chat_message("assistant"):
        chain = load_rag_chain(selected_model)
        
        if chain:
            # LangChain 포맷으로 대화 기록 변환
            history_langchain = []
            for msg in st.session_state.messages[:-1]: # 현재 질문 제외
                if msg["role"] == "user":
                    history_langchain.append(HumanMessage(content=msg["content"]))
                else:
                    history_langchain.append(AIMessage(content=msg["content"]))

            message_placeholder = st.empty()
            full_response = ""
            source_docs = []

            # 스트리밍 실행
            try:
                for chunk in chain.stream({"input": query, "chat_history": history_langchain}):
                    if "answer" in chunk:
                        full_response += chunk["answer"]
                        message_placeholder.markdown(full_response + "▌")
                    
                    if "context" in chunk:
                        source_docs = chunk["context"]

                message_placeholder.markdown(full_response)
                
                # 출처 표시 (Expander 사용)
                if source_docs:
                    with st.expander("📚 참고 문서 확인하기"):
                        seen = set()
                        for doc in source_docs:
                            source = os.path.basename(doc.metadata.get("source", "Unknown"))
                            page = doc.metadata.get("page", 0)
                            key = f"{source}p{page}"
                            if key not in seen:
                                st.markdown(f"- **{source}** (Page {page+1})")
                                seen.add(key)

                # 대화 기록 저장
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"에러 발생: {e}")