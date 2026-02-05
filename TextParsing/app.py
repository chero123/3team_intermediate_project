import streamlit as st
import os
import time
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# 1. 화면 기본 설정
# ==========================================
st.set_page_config(
    page_title="입찰메이트 AI",
    page_icon="🤖",
    layout="wide"  # 넓은 화면 사용
)

st.title("🤖 입찰/공고 분석 AI: 입찰메이트")
st.markdown("공공 입찰 공고문(RFP)에 대해 궁금한 점을 물어보세요! (예산, 마감일, 자격요건 등)")

# ==========================================
# 2. 사이드바 (설정 메뉴)
# ==========================================
with st.sidebar:
    st.header("⚙️ 환경 설정")
    
    # 1) API 키 입력
    if "OPENAI_API_KEY" not in os.environ:
        api_key = st.text_input("OpenAI API Key 입력", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("API Key 저장 완료!")
    
    # 2) 모델 선택
    st.subheader("모델 선택")
    model_options = ["gpt-5-mini", "gpt-5-nano", "gpt-5"]
    selected_model = st.selectbox(
        "사용할 모델을 선택하세요", 
        model_options, 
        index=0, # 기본값: gpt-5-mini
        help="gpt-5-mini가 가성비와 성능 균형이 가장 좋습니다."
    )
    
    st.markdown("---")
    st.info("""
    **💡 사용 팁:**
    - "이 사업의 예산과 기간은?"
    - "참가 자격 요건을 요약해줘"
    - "제안서 작성 시 유의사항은?"
    """)

# ==========================================
# 3. RAG 체인 로딩 (캐싱 적용)
# ==========================================
@st.cache_resource(show_spinner="AI 두뇌를 깨우는 중...")
def load_rag_chain(model_name):
    # 경로 계산
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_db")
    
    # DB 존재 여부 확인
    if not os.path.exists(DB_PATH):
        st.error(f"데이터베이스가 없습니다. 경로를 확인하세요: {DB_PATH}")
        return None

    # 임베딩 & DB 로드
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name="bid_rfp_collection"
    )
    
    # Retriever (MMR: 다양성 확보 검색)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10}
    )
    
    # LLM 설정 (선택한 모델 적용)
    try:
        llm = ChatOpenAI(model=model_name, temperature=0)
    except Exception as e:
        st.error(f"모델 로딩 실패: {e}")
        return None
    
    # 프롬프트 (전문가 페르소나)
    template = """
    당신은 공공 입찰(RFP) 분석 수석 컨설턴트 '입찰메이트'입니다.
    아래 [검색된 문서] 내용을 기반으로 사용자의 질문에 답변하세요.
    
    [지침]
    1. 문서에 있는 사실에만 근거하여 답변하고, 없는 내용은 지어내지 마세요.
    2. 예산, 기간, 날짜 등 숫자는 정확하게 명시하세요.
    3. 답변 끝에는 반드시 참고한 [문서명]을 괄호로 표기하세요.
    
    [검색된 문서]:
    {context}
    
    질문: {question}
    답변:
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # 문서 포맷팅 (출처 포함)
    def format_docs(docs):
        return "\n\n".join([f"<출처: {d.metadata.get('source', '문서명 미상')}>\n{d.page_content}" for d in docs])
    
    # 체인 결합
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# ==========================================
# 4. 메인 채팅 인터페이스
# ==========================================

# 세션 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 내용 그리기
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if query := st.chat_input("공고 내용에 대해 질문하세요..."):
    # API 키 없으면 중단
    if "OPENAI_API_KEY" not in os.environ:
        st.warning("왼쪽 사이드바에서 API Key를 먼저 입력해주세요.")
        st.stop()

    # 1. 사용자 질문 표시
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 2. AI 답변 생성
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # 체인 로드 (선택된 모델 사용)
            chain = load_rag_chain(selected_model)
            
            if chain:
                # 스트리밍 답변 생성
                response = chain.invoke(query)
                
                # 타자 치는 효과
                for chunk in response.split(" "):
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # 대화 기록 저장
                st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
            if "404" in str(e):
                st.warning("힌트: 선택한 모델명이 올바른지, API Key 권한이 있는지 확인하세요.")