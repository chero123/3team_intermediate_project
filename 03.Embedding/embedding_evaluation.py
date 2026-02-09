# ============================================
# 임베딩 모델 & 청킹 전략 비교 평가 시스템
# 15가지 조합 테스트 (청킹 5가지 × 임베딩 3가지)
# ============================================

import os
import json
import time
import numpy as np
import dotenv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


# ============================================
# 데이터 클래스
# ============================================


@dataclass
class EvaluationResult:
    """평가 결과"""

    chunking_method: str
    embedding_model: str
    hit_rate_at_1: float
    hit_rate_at_3: float
    hit_rate_at_5: float
    mrr: float  # Mean Reciprocal Rank
    avg_latency_ms: float
    total_chunks: int
    embedding_dim: int
    # 부분점수 (Answer 품질)
    keyword_precision: float = 0.0  # must_include 키워드 포함률
    keyword_recall: float = 0.0  # relevant_keywords 매칭률
    answer_quality_score: float = 0.0  # 종합 점수
    total_time_seconds: float = 0.0  # 추가: 테스트 총 소요시간


# ============================================
# 임베딩 모델 래퍼
# ============================================


class EmbeddingModelWrapper:
    """임베딩 모델 통합 래퍼"""

    def __init__(self, use_gpu: bool = True):
        self.models = {}
        self.openai_client = None
        self.device = "cuda" if use_gpu else "cpu"

        # GPU 사용 가능 여부 체크
        try:
            import torch

            if use_gpu and torch.cuda.is_available():
                self.device = "cuda"
                print(f"[GPU] CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                if use_gpu:
                    print("[GPU] CUDA 사용 불가, CPU 사용")
        except ImportError:
            self.device = "cpu"
            print("[GPU] PyTorch 없음, CPU 사용")

    def load_model(self, model_name: str):
        """모델 로드 (lazy loading)"""
        if model_name in self.models:
            return

        if model_name == "openai":
            try:
                from openai import OpenAI

                self.openai_client = OpenAI()
                self.models[model_name] = "openai"
                print(f"  [✓] OpenAI text-embedding-3-small 로드 완료")
            except Exception as e:
                print(f"  [✗] OpenAI 로드 실패: {e}")
                raise
        else:
            try:
                from sentence_transformers import SentenceTransformer

                model_map = {
                    "BGE-m3-ko": "dragonkue/BGE-m3-ko",
                    "MiniLM": "all-MiniLM-L6-v2",
                    "ko-sroberta": "jhgan/ko-sroberta-multitask",
                }
                if model_name not in model_map:
                    raise ValueError(f"Unknown model: {model_name}")

                print(f"  [로딩] {model_name} 모델 로딩 중... (device: {self.device})")
                self.models[model_name] = SentenceTransformer(
                    model_map[model_name], device=self.device
                )
                print(f"  [✓] {model_name} 로드 완료 ({self.device})")
            except Exception as e:
                print(f"  [✗] {model_name} 로드 실패: {e}")
                raise

    def encode(self, texts: List[str], model_name: str) -> np.ndarray:
        """텍스트를 임베딩으로 변환"""
        self.load_model(model_name)

        if model_name == "openai":
            embeddings = []
            # 배치 처리 (OpenAI는 한 번에 최대 2048개)
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small", input=batch
                )
                for item in response.data:
                    embeddings.append(item.embedding)
            return np.array(embeddings)
        else:
            model = self.models[model_name]
            # 청크가 많으면 진행률 표시
            show_progress = len(texts) > 100
            return model.encode(texts, show_progress_bar=show_progress)

    def get_embedding_dim(self, model_name: str) -> int:
        """임베딩 차원 반환"""
        dim_map = {
            "BGE-m3-ko": 1024,
            "MiniLM": 384,
            "ko-sroberta": 768,
            "openai": 1536,
        }
        return dim_map.get(model_name, 0)


# ============================================
# 평가 시스템
# ============================================


class RAGEvaluator:
    """RAG 시스템 평가기"""

    def __init__(self, base_data_dir: str = "data"):
        self.base_data_dir = base_data_dir
        self.embedding_wrapper = EmbeddingModelWrapper()

        # 청킹 방식 정의
        self.chunking_methods = {
            "chunking_data1": "안팀원-Recursive",
            "chunking_data2": "박팀원-Paragraph",
            "chunking_data3": "서팀원-Semantic",
            "chunking_data4": "김팀원-ContextEnriched",
            "chunking_data5": "장팀원-Hierarchical",
        }

        # 임베딩 모델 정의
        self.embedding_models = [
            "BGE-m3-ko",
            "MiniLM",
            "ko-sroberta",
            "openai",
        ]

    def load_chunks(self, chunking_folder: str) -> List[Dict]:
        """청킹된 데이터 로드"""
        folder_path = os.path.join(self.base_data_dir, chunking_folder)
        all_chunks = []

        if not os.path.exists(folder_path):
            print(f"  [경고] 폴더 없음: {folder_path}")
            return all_chunks

        json_files = [f for f in os.listdir(folder_path) if f.endswith("_chunked.json")]

        for json_file in json_files:
            file_path = os.path.join(folder_path, json_file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for chunk in data.get("chunks", []):
                        chunk["source_doc"] = data.get("doc_id", "")
                        all_chunks.append(chunk)
            except Exception as e:
                print(f"  [오류] {json_file} 로드 실패: {e}")

        return all_chunks

    def load_evaluation_dataset(self, eval_file: str) -> List[Dict]:
        """평가 데이터셋 로드"""
        with open(eval_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("evaluation_set", [])

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """코사인 유사도 계산"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

    def find_relevant_chunks(
        self,
        chunks: List[Dict],
        answer: str,
        keywords: List[str],
        expected_chunk_ids: List[str] = None,
    ) -> List[str]:
        """
        정답 청크 ID 찾기
        1순위: expected_chunk_ids (Ground Truth에서 지정한 청크)
        2순위: answer 텍스트 포함 여부
        3순위: 키워드 2개 이상 매칭
        """
        relevant_ids = []

        # 1순위: expected_chunk_ids가 있으면 해당 패턴과 매칭되는 청크 찾기
        if expected_chunk_ids:
            for chunk in chunks:
                # 청크의 document ID와 내부 chunk ID를 결합하여 전체 ID 생성
                full_chunk_id = (
                    f"{chunk.get('source_doc', '')}::{chunk.get('id', '')}"
                    if chunk.get("source_doc")
                    else chunk.get("id", "")
                )

                for expected_id in expected_chunk_ids:
                    # expected_id가 'doc_id::chunk_id' 형식일 경우,
                    # chunk의 full_chunk_id와 직접 비교하여 일치하는지 확인
                    if full_chunk_id == expected_id:
                        relevant_ids.append(chunk["id"])
                        break

            # 1순위에서 찾지 못했고, 그래도 expected_chunk_ids가 있다면,
            # 예외적으로 doc_id 부분만으로 매칭 시도 (이는 덜 정확할 수 있음)
            if not relevant_ids and expected_chunk_ids:
                for chunk in chunks:
                    chunk_source_doc = chunk.get("source_doc", "")
                    for expected_id in expected_chunk_ids:
                        expected_doc_part = (
                            expected_id.split("::")[0]
                            if "::" in expected_id
                            else expected_id
                        )
                        if (
                            expected_doc_part
                            and expected_doc_part.lower() in chunk_source_doc.lower()
                        ):
                            # 중복 추가 방지를 위해 이미 추가된 chunk_id는 건너뜀
                            if chunk["id"] not in relevant_ids:
                                relevant_ids.append(chunk["id"])
                            break

        # 2순위: answer 텍스트가 포함된 청크
        if not relevant_ids and answer:  # 1순위에서 찾지 못했을 때만 2순위 시도
            for chunk in chunks:
                text = chunk.get("text", "").lower()
                # answer의 핵심 부분만 체크 (첫 50자)
                answer_core = (
                    answer[:50].lower() if len(answer) > 50 else answer.lower()
                )
                if answer_core in text:
                    relevant_ids.append(chunk["id"])

        # 3순위: 키워드 매칭 (최소 2개 이상)
        if not relevant_ids:  # 1,2순위에서 찾지 못했을 때만 3순위 시도
            for chunk in chunks:
                text = chunk.get("text", "").lower()
                keyword_matches = sum(1 for kw in keywords if kw.lower() in text)
                if keyword_matches >= 2:
                    relevant_ids.append(chunk["id"])

        return relevant_ids

    def calculate_answer_quality(
        self,
        retrieved_text: str,
        must_include: List[str],
        relevant_keywords: List[str],
    ) -> Tuple[float, float, float]:
        """
        답변 품질 부분점수 계산
        Returns: (keyword_precision, keyword_recall, answer_quality_score)
        """
        retrieved_lower = retrieved_text.lower()

        # Precision: must_include 키워드 포함률
        if must_include:
            included = sum(1 for kw in must_include if kw.lower() in retrieved_lower)
            precision = included / len(must_include)
        else:
            precision = 1.0  # must_include가 없으면 만점

        # Recall: relevant_keywords 매칭률
        if relevant_keywords:
            matched = sum(
                1 for kw in relevant_keywords if kw.lower() in retrieved_lower
            )
            recall = matched / len(relevant_keywords)
        else:
            recall = 1.0

        # 종합 점수 (F1-like)
        if precision + recall > 0:
            quality = 2 * precision * recall / (precision + recall)
        else:
            quality = 0.0

        return precision, recall, quality

    def evaluate_single_combination(
        self,
        chunks: List[Dict],
        eval_data: List[Dict],
        embedding_model: str,
        chunking_method: str,
    ) -> EvaluationResult:
        """단일 조합 평가"""

        if not chunks:
            return EvaluationResult(
                chunking_method=chunking_method,
                embedding_model=embedding_model,
                hit_rate_at_1=0.0,
                hit_rate_at_3=0.0,
                hit_rate_at_5=0.0,
                mrr=0.0,
                avg_latency_ms=0.0,
                total_chunks=0,
                embedding_dim=0,
            )

        # 1. 모든 청크 임베딩 생성
        print(f"    청크 임베딩 생성 중... ({len(chunks)}개)")
        start_time = time.time()

        chunk_texts = [c["text"] for c in chunks]
        chunk_embeddings = self.embedding_wrapper.encode(chunk_texts, embedding_model)

        embed_time = time.time() - start_time

        # 2. 각 질문에 대해 평가
        hits_at_1, hits_at_3, hits_at_5 = 0, 0, 0
        mrr_sum = 0.0
        query_times = []

        # 부분점수 누적
        precision_sum, recall_sum, quality_sum = 0.0, 0.0, 0.0
        evaluated_count = 0

        for eval_item in eval_data:
            question = eval_item["question"]
            ground_truth = eval_item["ground_truth"]
            eval_criteria = eval_item.get("evaluation_criteria", {})

            # 정답 청크 찾기 (수정된 방식)
            answer = ground_truth.get("answer", "")
            keywords = ground_truth.get("relevant_keywords", [])
            expected_ids = ground_truth.get("expected_chunk_ids", [])
            relevant_chunk_ids = self.find_relevant_chunks(
                chunks, answer, keywords, expected_ids
            )

            if not relevant_chunk_ids:
                # 정답 청크를 찾지 못한 경우 스킵
                continue

            # 질문 임베딩
            q_start = time.time()
            query_embedding = self.embedding_wrapper.encode(
                [question], embedding_model
            )[0]

            # 유사도 계산
            similarities = []
            for i, chunk_emb in enumerate(chunk_embeddings):
                sim = self.cosine_similarity(query_embedding, chunk_emb)
                similarities.append((i, sim, chunks[i]["id"]))

            query_times.append((time.time() - q_start) * 1000)

            # Top-K 정렬
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Hit@K 및 MRR 계산
            top_retrieved_text = ""
            for rank, (idx, sim, chunk_id) in enumerate(similarities[:5]):
                if rank == 0:
                    top_retrieved_text = chunks[idx].get("text", "")
                if chunk_id in relevant_chunk_ids:
                    if rank < 1:
                        hits_at_1 += 1
                    if rank < 3:
                        hits_at_3 += 1
                    if rank < 5:
                        hits_at_5 += 1
                    mrr_sum += 1.0 / (rank + 1)
                    break

            # 부분점수 계산 (Top-1 청크 기준)
            must_include = eval_criteria.get("must_include", [])
            relevant_kws = ground_truth.get("relevant_keywords", [])
            prec, rec, qual = self.calculate_answer_quality(
                top_retrieved_text, must_include, relevant_kws
            )
            precision_sum += prec
            recall_sum += rec
            quality_sum += qual
            evaluated_count += 1

        num_questions = len(eval_data)

        return EvaluationResult(
            chunking_method=chunking_method,
            embedding_model=embedding_model,
            hit_rate_at_1=hits_at_1 / num_questions if num_questions > 0 else 0,
            hit_rate_at_3=hits_at_3 / num_questions if num_questions > 0 else 0,
            hit_rate_at_5=hits_at_5 / num_questions if num_questions > 0 else 0,
            mrr=mrr_sum / num_questions if num_questions > 0 else 0,
            avg_latency_ms=np.mean(query_times) if query_times else 0,
            total_chunks=len(chunks),
            embedding_dim=self.embedding_wrapper.get_embedding_dim(embedding_model),
            # 부분점수
            keyword_precision=(
                precision_sum / evaluated_count if evaluated_count > 0 else 0
            ),
            keyword_recall=recall_sum / evaluated_count if evaluated_count > 0 else 0,
            answer_quality_score=(
                quality_sum / evaluated_count if evaluated_count > 0 else 0
            ),
        )

    def run_full_evaluation(
        self,
        eval_file: str,
        output_file: str = "evaluation_results.json",
        skip_models: List[str] = None,
    ) -> List[EvaluationResult]:
        """20가지 조합 전체 평가 실행"""

        print("=" * 70)
        print("RAG 시스템 평가 시작")
        print(f"청킹 방식: {len(self.chunking_methods)}가지")
        print(f"임베딩 모델: {len(self.embedding_models)}가지")
        print(f"총 조합: {len(self.chunking_methods) * len(self.embedding_models)}가지")
        print("=" * 70)

        # 평가 데이터 로드
        print(f"\n[1] 평가 데이터셋 로드: {eval_file}")
        eval_data = self.load_evaluation_dataset(eval_file)
        print(f"    질문 수: {len(eval_data)}개")

        results = []
        skip_models = skip_models or []

        # 총 조합 수 계산 (스킵 모델 제외)
        active_models = [m for m in self.embedding_models if m not in skip_models]
        total_combos = len(self.chunking_methods) * len(active_models)
        combo_idx = 0

        for chunk_folder, chunk_name in self.chunking_methods.items():
            print(f"\n[청킹] {chunk_name}")

            # 청크 로드
            chunks = self.load_chunks(chunk_folder)
            print(f"  로드된 청크: {len(chunks)}개")

            for emb_model in active_models:
                combo_idx += 1

                if not chunks:
                    print(
                        f"  [{combo_idx}/{total_combos}] {chunk_name} + {emb_model} - 스킵 (청크 없음)"
                    )
                    continue

                print(f"\n  [{combo_idx}/{total_combos}] {chunk_name} + {emb_model}")

                try:
                    test_start_time = time.time()  # 추가: 시작 시간

                    result = self.evaluate_single_combination(
                        chunks=chunks,
                        eval_data=eval_data,
                        embedding_model=emb_model,
                        chunking_method=chunk_name,
                    )

                    result.total_time_seconds = (
                        time.time() - test_start_time
                    )  # 추가: 소요시간 기록

                    results.append(result)

                    print(f"    Hit@1: {result.hit_rate_at_1:.2%}")
                    print(f"    Hit@5: {result.hit_rate_at_5:.2%}")
                    print(f"    MRR: {result.mrr:.4f}")
                    print(f"    Latency: {result.avg_latency_ms:.1f}ms")
                    print(
                        f"    Total Time: {result.total_time_seconds:.1f}s"
                    )  # 총 소요시간 출력

                except Exception as e:
                    print(f"    [오류] {e}")

        # 결과 저장
        self._save_results(results, output_file, eval_file)

        # 요약 출력
        self._print_summary(results)

        return results

    def _save_results(
        self, results: List[EvaluationResult], output_file: str, eval_file: str
    ):
        """결과 저장"""
        output_data = {
            "evaluation_date": datetime.now().isoformat(),
            "eval_dataset": eval_file,
            "total_combinations": len(results),
            "results": [
                {
                    "chunking_method": r.chunking_method,
                    "embedding_model": r.embedding_model,
                    "hit_rate_at_1": r.hit_rate_at_1,
                    "hit_rate_at_3": r.hit_rate_at_3,
                    "hit_rate_at_5": r.hit_rate_at_5,
                    "mrr": r.mrr,
                    "keyword_precision": r.keyword_precision,
                    "keyword_recall": r.keyword_recall,
                    "answer_quality_score": r.answer_quality_score,
                    "avg_latency_ms": r.avg_latency_ms,
                    "total_chunks": r.total_chunks,
                    "embedding_dim": r.embedding_dim,
                    "total_time_seconds": r.total_time_seconds,
                }
                for r in results
            ],
        }

        output_path = os.path.join(self.base_data_dir, output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n결과 저장: {output_path}")

    def _print_summary(self, results: List[EvaluationResult]):
        """결과 요약 출력"""
        print("\n" + "=" * 70)
        print("평가 결과 요약 (MRR 기준 정렬)")
        print("=" * 70)

        # MRR 기준 정렬
        sorted_results = sorted(results, key=lambda x: x.mrr, reverse=True)

        print(
            f"{'순위':<4} {'청킹':<22} {'임베딩':<13} {'Hit@1':>7} {'Hit@5':>7} {'MRR':>7} {'품질':>7}"
        )
        print("-" * 75)

        for i, r in enumerate(sorted_results, 1):
            print(
                f"{i:<4} {r.chunking_method:<22} {r.embedding_model:<13} "
                f"{r.hit_rate_at_1:>6.1%} {r.hit_rate_at_5:>6.1%} {r.mrr:>7.4f} {r.answer_quality_score:>6.1%}"
            )

        # 최고 성능 조합
        if sorted_results:
            best = sorted_results[0]
            print("\n" + "=" * 75)
            print(f"🏆 최고 성능 조합: {best.chunking_method} + {best.embedding_model}")
            print(
                f"   Hit@1: {best.hit_rate_at_1:.1%}, Hit@5: {best.hit_rate_at_5:.1%}, MRR: {best.mrr:.4f}"
            )
            print(
                f"   키워드 정밀도: {best.keyword_precision:.1%}, 재현율: {best.keyword_recall:.1%}, 품질: {best.answer_quality_score:.1%}"
            )


# ============================================
# 메인 실행
# ============================================

if __name__ == "__main__":
    # 평가 데이터셋 파일 경로
    EVAL_DATASET = "data/evaluation_dataset2.json"

    # 스킵할 모델 (OpenAI API 키 없으면 스킵)
    SKIP_MODELS = [
        "BGE-m3-ko",
    ]  # ["openai"] 로 설정하면 OpenAI 스킵

    # API 연동
    dotenv.load_dotenv()

    # 평가기 생성 및 실행
    evaluator = RAGEvaluator(base_data_dir="data")

    # 평가 데이터셋이 있는지 확인
    if not os.path.exists(EVAL_DATASET):
        print(f"[오류] 평가 데이터셋이 없습니다: {EVAL_DATASET}")
        print("먼저 evaluation_dataset.json 파일을 생성해주세요.")
    else:
        results = evaluator.run_full_evaluation(
            eval_file=EVAL_DATASET,
            output_file="evaluation_results.json",
            skip_models=SKIP_MODELS,
        )
