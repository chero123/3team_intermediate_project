import os
import json
import re
import glob
from typing import List, Literal
from dataclasses import asdict

import dotenv
import pandas as pd

from chunking import Chunk, integrated_rfp_chunker
from embeddings import OpenAIEmbedding, KoSRoBERTaEmbedding
from vector_store import SearchResult, ChromaVectorDB, FAISSVectorDB

dotenv.load_dotenv()


class RAGPipeline:
    def __init__(
        self,
        embedding_model: Literal["ko-sroberta", "openai"] = "openai",
        vector_db: Literal["chroma", "faiss"] = "chroma",
        persist_dir: str = "data/vector_db/rfp_integrated",
    ):
        if embedding_model == "openai":
            self.embedding = OpenAIEmbedding()
        else:
            self.embedding = KoSRoBERTaEmbedding()

        if vector_db == "chroma":
            self.vector_db = ChromaVectorDB(
                collection_name="rfp_hybrid", persist_directory=persist_dir
            )
        else:
            self.vector_db = FAISSVectorDB(dimension=self.embedding.dimension)

    def add_document(self, text: str, doc_id: str, context_metadata: dict):
        chunks = integrated_rfp_chunker(text, doc_id, context_metadata)
        texts = [c.text for c in chunks]
        embeddings = self.embedding.embed(texts)
        self.vector_db.add(chunks, embeddings)
        print(f"  → '{doc_id}' {len(chunks)}개 청크 저장 완료")

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        query_embedding = self.embedding.embed_query(query)
        return self.vector_db.search(query_embedding, top_k)


# ============================================
# 메인 실행 - data_list.xlsx 연동 자동화
# ============================================

if __name__ == "__main__":
    # 1. 파이프라인 초기화
    pipeline = RAGPipeline(embedding_model="openai", vector_db="chroma")

    # 2. 엑셀 데이터 로드 (메타데이터 소스)
    EXCEL_PATH = "data/data_list.xlsx"
    if not os.path.exists(EXCEL_PATH):
        print(f"[오류] 엑셀 파일을 찾을 수 없습니다: {EXCEL_PATH}")
    else:
        df = pd.read_excel(EXCEL_PATH)

        # 3. MD 파일 목록 가져오기
        MD_DIR = "./data/final_docs"
        md_files = glob.glob(os.path.join(MD_DIR, "*.md"))

        print(f"\n[인덱싱 시작] 총 {len(md_files)}개 파일 처리 예정")

        for file_path in md_files:
            file_name = os.path.basename(file_path)

            # 엑셀에서 파일명 매칭 (컬럼 인덱스 5: 파일명)
            # 파일명이 완벽히 일치하지 않을 수 있으므로 포함 여부로 체크
            match = df[
                df.iloc[:, 5].astype(str).str.contains(file_name, na=False, regex=False)
            ]

            if not match.empty:
                row = match.iloc[0]
                doc_id = str(row.iloc[0])  # 공고번호
                sample_metadata = {
                    "title": str(row.iloc[1]),  # 사업명
                    "agency": str(row.iloc[3]),  # 발주기관
                    "page": "1",  # 기본값
                }
            else:
                # 매칭 실패 시 파일명에서 정보 추출 시도
                doc_id = file_name.replace(".md", "")
                sample_metadata = {"title": doc_id, "agency": "미상", "page": "1"}

            # === 파일 처리 및 저장 ===
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # [동작 1] 벡터 DB에 저장 (AI 검색용)
                pipeline.add_document(content, doc_id, sample_metadata)

                # [동작 2] JSON 파일로 따로 저장 (사람 확인용)
                chunks = integrated_rfp_chunker(content, doc_id, sample_metadata)
                debug_data = [asdict(chunk) for chunk in chunks]

                # 특수문자 제거 후 파일명 생성 (안전하게 저장하기 위함)
                safe_doc_id = re.sub(r'[\\/*?:"<>|]', "", doc_id)

                debug_dir = "data/debug"
                os.makedirs(debug_dir, exist_ok=True)
                json_path = f"data/debug/{safe_doc_id}.json"

                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(debug_data, f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"  [오류] {file_name} 처리 실패: {e}")

        # 4. 검색 테스트 (rfp_retriever.py에서 수행)
        # test_queries = [
        #     "벤처확인종합관리시스템 고도화 사업의 목적이 뭐야?",
        #     # "이 사업의 총 예산은 얼마인가요?",
        #     # "제안서 평가 기준은?",
        # ]
        #
        # for query in test_queries:
        #     print("\n" + "=" * 50)
        #     print(f"[검색 테스트] 질문: {query}")
        #
        #     results = pipeline.search(query, top_k=3)
        #
        #     for i, r in enumerate(results, 1):
        #         print(f"\n[{i}] 출처: {r.metadata.get('source_display')}")
        #         print(f"    내용: {r.text[:200]}...")
