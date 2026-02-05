import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_retriever(db_path="/home/spai0630/workspace/3team_intermediate_project/db"):
    # 추가 엔진 설치 없이 동작하는 기본 임베딩 설정
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB 경로를 찾을 수 없습니다: {db_path}")
        
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    return vector_db.as_retriever(search_kwargs={"k": 10})
