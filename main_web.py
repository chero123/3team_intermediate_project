import gradio as gr
from src.database import get_retriever
from src.chain import create_rag_chain
from dotenv import load_dotenv
import os

# 환경 변수 및 RAG 체인 초기화
load_dotenv()
retriever = get_retriever()
rag_chain = create_rag_chain(retriever)

def ask_question(message, history):
    try:
        # 질문 실행 (history는 Gradio ChatInterface에서 자동 관리되므로 message만 사용)
        response = rag_chain.invoke(message)
        return response
    except Exception as e:
        return f"에러 발생: {str(e)}"

# Gradio 인터페이스 설정 (theme 인자 제거)python main_web.py
demo = gr.ChatInterface(
    fn=ask_question,
    title="🏢 3팀 프로젝트: AI 문서 비서 (RAG)",
    description="문서 내용을 바탕으로 질문에 답변해 드립니다.",
    examples=["이 사업의 핵심 목표가 뭐야?", "보안 사고 발생 시 배상 기준은?", "주요 추진 일정은 어떻게 돼?"]
)

if __name__ == "__main__":
    # 서버 환경(GCP)에서 외부 접속이 가능하도록 설정
    # share=True를 통해 외부 공유 가능한 public 링크가 생성됩니다.
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)