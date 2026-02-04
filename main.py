import os
from dotenv import load_dotenv
from src.database import get_retriever
from src.chain import create_rag_chain

def main():
    load_dotenv() # API 키 로드
    
    try:
        print("1. 데이터베이스 로딩 중...")
        retriever = get_retriever()
        
        print("2. RAG 파이프라인 생성 중...")
        rag_chain = create_rag_chain(retriever)
        
        print("3. 질문 실행...")
        query = "이 사업의 핵심 목표가 뭐야?"
        print(f"\n질문: {query}")
        print("-" * 50)
        
        response = rag_chain.invoke(query)
        print(f"답변: {response}")
        
    except Exception as e:
        print(f"\n실행 중 에러 발생: {e}")

if __name__ == "__main__":
    main()
