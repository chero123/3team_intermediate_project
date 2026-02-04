import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def create_rag_chain(retriever):
    llm = ChatOpenAI(model_name="gpt-5-mini", temperature=0)
    
    template = """아래 [맥락] 정보를 활용하여 질문에 답하세요.
    [맥락]
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
