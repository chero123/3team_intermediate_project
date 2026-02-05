import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def create_rag_chain(retriever):

    llm = ChatOpenAI(model_name="gpt-5", temperature=0) 
    
    template = """
    당신은 공공기관 입찰 제안서 및 사업 계획서를 분석하는 전문 AI 비서입니다.
    제공된 [컨텍스트]를 바탕으로 사용자의 질문에 답변하되, 아래 규칙을 반드시 준수하세요.

    1. **정확성**: '사업비(예산)', '사업기간', '핵심 요구사항' 등 수치와 관련된 정보는 문서에서 찾아내어 정확히 기술하십시오.
    2. **가독성**: 정보를 단순히 나열하지 말고, 불렛 포인트나 표 형식을 적절히 활용하여 구조적으로 답변하십시오.
    3. **맥락 추가**: 답변 끝에 해당 정보와 연관된 유의사항이나 추가 정보를 덧붙여 사용자에게 통찰을 제공하십시오.
    4. **근거 제시**: 가능한 경우 해당 내용이 문서의 어느 부분(예: 사업 개요, 추진 일정 등)에서 추출되었는지 언급하십시오.

    질문에 대해 제공된 컨텍스트에 내용이 없다면 "해당 내용은 문서에 명시되어 있지 않습니다"라고 답변하고, 억지로 추측하지 마십시오.

    [컨텍스트]
    {context}

    질문: {question}
    답변:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain