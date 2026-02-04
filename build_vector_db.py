# build_vector_db.py
import os
import glob
import json
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# =========================
# 설정
# =========================
PARSING_DIR = "data/parsing_data"
FAISS_DIR = "data/faiss_index"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# =========================
# 간단 청킹
# =========================
def simple_chunk(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


# =========================
# main
# =========================
def main():
    os.makedirs(FAISS_DIR, exist_ok=True)

    txt_files = glob.glob(os.path.join(PARSING_DIR, "*_parsed.txt"))
    print(f"로드된 파싱 파일: {len(txt_files)}개")

    documents = []

    for path in txt_files:
        filename = os.path.basename(path)
        doc_id = filename.replace("_parsed.txt", "")

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = simple_chunk(text, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}::chunk{i}"
            
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": filename,        
                        "chunk_id": chunk_id,      
                        "doc_id": doc_id,
                        "chunk_index": i,
                    },
                )
            )

    print(f"총 생성된 청크 수: {len(documents)}")

    # 임베딩
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # FAISS 생성
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(FAISS_DIR)

    print(f"FAISS 인덱스 저장 완료 → {FAISS_DIR}")


if __name__ == "__main__":
    main()
