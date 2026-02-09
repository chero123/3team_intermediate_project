import streamlit as st
import os
import time
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# ==========================================
# 1. 화면 기본 설정
# ==========================================
st.set_page_config(
    page_title="입찰메이트 AI (Hybrid)",
    page_icon="🤖",
    layout="wide"
)

st.title("입찰/공고 분석 AI: 입찰메이트 (Hybrid Edition)")
st.markdown("""
- **Dense(의미)**: 문맥과 의미를 파악하여 검색 (Chroma)
- **Sparse(키워드)**: 공고 번호, 예산, 모델명 등 정확한 매칭 검색 (BM25)
""")

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
    selected_model = st.selectbox(
        "사용할 모델", 
        model_options, 
        index=0
    )

    st.subheader("검색 가중치 설정")
    dense_weight = st.slider("Dense(의미) 비중", 0.0, 1.0, 0.6, 0.1, help="높을수록 문맥 위주, 낮을수록 키워드 위주")
    sparse_weight = round(1.0 - dense_weight, 1)
    st.caption(f"Sparse(키워드) 비중: {sparse_weight}")
    
    if st.button("🗑️ 대화 내용 지우기"):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# 3. RAG 체인 설정 (Hybrid & LCEL Fix)
# ==========================================
@st.cache_resource(show_spinner="Hybrid 검색 엔진 가동 중...")
def load_rag_chain(model_name, dense_w, sparse_w):
    # 경로 설정
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_db")
    
    if not os.path.exists(DB_PATH):
        st.error(f"데이터베이스 없음: {DB_PATH}")
        return None

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 1. Dense Retriever (Chroma)
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name="bid_rfp_collection"
    )
    
    dense_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    )
    
    # 2. Sparse Retriever (BM25)
    try:
        raw_docs = vectorstore.get()
        docs = []
        for i in range(len(raw_docs['ids'])):
            if raw_docs['documents'][i]:
                docs.append(Document(
                    page_content=raw_docs['documents'][i],
                    metadata=raw_docs['metadatas'][i] if raw_docs['metadatas'] else {}
                ))
        
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
        retrievers=[dense_retriever, sparse_retriever],
        weights=[dense_w, sparse_w]
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
    context_q_prompt = ChatPromptTemplate.from_messages([
        ("system", context_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Chain: (Dict) -> (String)
    history_aware_chain = context_q_prompt | llm | StrOutputParser()

    # [프롬프트 2] 답변 생성 (QA)
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

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("질문을 입력하세요..."):
    
    if "OPENAI_API_KEY" not in os.environ:
        st.warning("API Key가 필요합니다.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})
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
                for chunk in chain.stream({"input": query, "chat_history": history_langchain}):
                    if "answer" in chunk:
                        full_response += chunk["answer"]
                        message_placeholder.markdown(full_response + "▌")
                    
                    if "context" in chunk:
                        source_docs = chunk["context"]

                message_placeholder.markdown(full_response)
                
                if source_docs:
                    with st.expander("📚 참고 문서 확인하기 (Hybrid 검색)"):
                        seen = set()
                        for i, doc in enumerate(source_docs):
                            source = os.path.basename(doc.metadata.get("source", "Unknown"))
                            page = doc.metadata.get("page", 0)
                            preview = doc.page_content[:40].replace("\n", " ")
                            
                            key = f"{source}p{page}"
                            if key not in seen:
                                st.markdown(f"**{i+1}. {source}** (Page {page+1})")
                                st.caption(f"내용: {preview}...")
                                seen.add(key)

                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"에러 발생: {e}")