"""
실행 방법 예시

uv run scripts/run_query.py --tts --device cuda
"""
# =====================================================================
# 로컬 vLLM 실행 스크립트
#
# 흐름:
# - 질문 입력
# - RAG 파이프라인 실행
# - 문장 출력
# - (옵션) TTS 합성/재생/저장
# =====================================================================
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from rag.pipeline import RAGPipeline
from tts_runtime.infer_onnx import infer_tts_onnx

# TTS 입력 경로와 출력 경로를 한 곳에서 관리해 변경 지점을 단일화한다.
_TTS_MODEL_PATH = Path("models/melo_yae/melo_yae.onnx")
_TTS_BERT_PATH = Path("models/melo_yae/bert_kor.onnx")
_TTS_CONFIG_PATH = Path("models/melo_yae/config.json")
_TTS_OUTPUT_PATH = Path("data/answer/answer.wav")


def _is_junk_line(line: str) -> bool:
    # 의미 없는 숫자/짧은 문자열을 제거해 출력 품질을 보정한다.
    stripped = re.sub(r"[^0-9A-Za-z가-힣]", "", line)
    if not stripped:
        return True
    if stripped.isdigit():
        return True
    digit_ratio = sum(ch.isdigit() for ch in stripped) / max(len(stripped), 1)
    if digit_ratio > 0.4:
        return True
    if len(stripped) < 4:
        return True
    return False


def _sanitize_answer(text: str) -> str:
    # 모델이 섞어내는 컨텍스트/잡음 라인을 제거해 최종 출력만 남긴다.
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("컨텍스트") or stripped.startswith("[Source"):
            continue
        if _is_junk_line(stripped):
            continue
        cleaned_lines.append(stripped)
    # 괄호/특수문자를 제거해 TTS 입력이 불안정해지는 것을 방지한다.
    cleaned = " ".join(cleaned_lines)
    cleaned = re.sub(r"\([^)]*\)", "", cleaned)
    cleaned = re.sub(r"\[[^\]]*\]", "", cleaned)
    cleaned = re.sub(r"[^0-9A-Za-z가-힣\s\.\!\?]", " ", cleaned)
    cleaned = re.sub(r"\d{20,}", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _split_sentences(text: str) -> list[str]:
    # 문장 경계를 기준으로 자른 뒤 마침표를 유지한다.
    parts = re.split(r"([.!?。！？]+)", text)
    # 문장 구분자를 보존하면서 문장 단위 리스트를 만든다.
    sentences: list[str] = []
    buf = ""
    for part in parts:
        if not part:
            continue
        buf += part
        if re.fullmatch(r"[.!?。！？]+", part):
            if buf.strip():
                sentences.append(buf.strip())
            buf = ""
    if buf.strip():
        sentences.append(buf.strip())
    return sentences


def _ensure_sentence(text: str) -> str:
    # 문장 끝 마침표가 없으면 추가해 출력/리라이팅 규격을 맞춘다.
    # 리라이팅 규칙(마침표 종결)을 보장한다.
    if re.search(r"[.!?。！？]\s*$", text):
        return text
    return f"{text}."


def _split_sentence_for_tts(sentence: str, max_words: int = 8, max_chars: int = 90) -> list[str]:
    # TTS는 너무 긴 문장에서 오류/왜곡이 생기므로 짧게 분할한다.
    if len(sentence) <= max_chars:
        return [sentence]
    words = sentence.split()
    if not words:
        return [sentence]
    chunks: list[str] = []
    buf: list[str] = []
    for word in words:
        candidate = " ".join(buf + [word])
        if len(candidate) > max_chars or len(buf) >= max_words:
            if buf:
                chunks.append(" ".join(buf))
            buf = [word]
        else:
            buf.append(word)
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def _apply_fade(audio: np.ndarray, sr: int, fade_ms: int = 10) -> np.ndarray:
    if audio.size == 0:
        return audio
    # 클릭 노이즈를 줄이기 위해 앞뒤에 짧은 페이드를 적용한다.
    # np.linspace는 시작~끝 값을 균등 간격으로 만드는 함수이며, 페이드 곡선을 만든다.
    # 페이드 길이는 샘플레이트 기준으로 ms -> 샘플 수로 환산한다.
    fade_len = int(sr * (fade_ms / 1000.0))
    fade_len = min(fade_len, audio.size // 2)
    if fade_len <= 1:
        return audio
    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    # astype(copy=False)는 필요할 때만 형변환하며, 불필요한 복사를 피한다.
    audio = audio.astype(np.float32, copy=False)
    # 시작/끝 구간에 곱셈으로 완만한 0->1, 1->0 변화를 만들어 클릭을 줄인다.
    audio[:fade_len] *= fade_in
    audio[-fade_len:] *= fade_out
    return audio


def _prepare_audio_for_playback(audio: np.ndarray, sr: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio
    # NaN/Inf와 과도한 레벨을 정리해 재생 안정성을 높인다.
    # np.nan_to_num은 NaN/Inf를 지정한 값(여기서는 0)으로 바꿔 파형 튐을 방지한다.
    # NaN/Inf 제거 + 클리핑 + 페이드로 CLI 재생 노이즈를 줄인다.
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    # np.clip은 값을 [-1, 1]로 제한해 과도한 진폭(클리핑 노이즈)을 줄인다.
    audio = np.clip(audio, -1.0, 1.0)
    audio = _apply_fade(audio, sr, fade_ms=12)
    return audio


def _select_audio_player(preferred: str | None = None) -> list[str] | None:
    # 플레이어가 없거나 비활성화면 재생을 건너뛴다.
    # player=none이면 재생 없이 파일 저장만 수행한다.
    if preferred in {"none", "off"}:
        return None
    if preferred in {"ffplay"}:
        path = shutil.which(preferred)
        if not path:
            return None
        if preferred == "ffplay":
            return [path, "-autoexit", "-nodisp", "-loglevel", "error"]
        return [path]
    for candidate in ("ffplay",):
        path = shutil.which(candidate)
        if not path:
            continue
        if candidate == "ffplay":
            return [path, "-autoexit", "-nodisp", "-loglevel", "error"]
        return [path]
    return None


def _run_once(pipeline: RAGPipeline, question: str, use_tts: bool, device: str, player: str) -> None:
    """
    질문을 한 번 처리하고 TTS를 실행

    Args:
        pipeline: RAG 파이프라인
        question: 사용자 질문
        use_tts: TTS 사용 여부
        device: TTS 디바이스

    Returns:
        None
    """
    # 1) RAG 질의 -> 원본 답변 생성
    answer = pipeline.ask(question)
    answer = _sanitize_answer(answer)
    sentences = [s for s in (s.strip() for s in _split_sentences(answer)) if s]
    if not sentences:
        sentences = ["답변을 생성할 수 없습니다."]

    # 2) TTS 준비
    audio_chunks: list[np.ndarray] = []
    sr = None
    player_cmd = None
    if use_tts:
        with _TTS_CONFIG_PATH.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        sr = cfg["data"]["sampling_rate"]
        player_cmd = _select_audio_player(player)

    # 3) 문장 단위로 출력/음성 합성을 수행한다.
    for sentence in sentences:
        sentence = _ensure_sentence(sentence.strip())
        if not sentence:
            continue
        print(sentence)

        # 문장 출력 직후에 음성을 생성해 실시간성을 유지한다.
        if use_tts:
            # 문장 -> 더 짧은 조각으로 분할해 안정적으로 합성한다.
            tts_segments = _split_sentence_for_tts(sentence)
            sentence_chunks = []
            for segment in tts_segments:
                # ONNX TTS를 호출해 조각 단위 오디오를 얻는다.
                audio = infer_tts_onnx(
                    onnx_path=str(_TTS_MODEL_PATH),
                    bert_onnx_path=str(_TTS_BERT_PATH),
                    config_path=str(_TTS_CONFIG_PATH),
                    text=segment,
                    speaker_id=0,
                    language="KR",
                    device=device,
                    out_path=None,
                )
                if sr is not None:
                    audio = _prepare_audio_for_playback(audio, sr)
                # 조각 오디오를 누적해 한 문장 오디오로 합친다.
                sentence_chunks.append(audio)
                if sr is not None:
                    # 문장 내 조각 사이에 짧은 무음을 넣어 끊김을 줄인다.
                    # 조각 사이에 짧은 무음을 넣어 끊김/클릭을 줄인다.
                    silence = np.zeros(int(sr * 0.12), dtype=np.float32)
                    sentence_chunks.append(silence)

            if sentence_chunks:
                # 문장 조각을 연결해 문장 단위 오디오를 만든다.
                sentence_audio = np.concatenate(sentence_chunks, axis=0)
                audio_chunks.append(sentence_audio)
                # 문장 단위로 재생해 잦은 플레이어 호출을 줄인다.
                # 문장 단위로만 플레이어를 호출해 지지직을 최소화한다.
                if player_cmd and sr is not None:
                    playback = _prepare_audio_for_playback(sentence_audio, sr)
                    tail = np.zeros(int(sr * 0.10), dtype=np.float32)
                    playback = np.concatenate([playback, tail], axis=0)
                    # ffplay는 파일 입력만 받으므로 임시 wav로 저장 후 재생한다.
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        sf.write(tmp.name, playback, sr, subtype="PCM_16")
                    subprocess.run(player_cmd + [tmp.name], check=False)

    if use_tts and audio_chunks:
        # 전체 합성 결과는 파일로 저장해 재생 없이도 확인 가능하게 한다.
        _TTS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        # 전체 문장 오디오를 합쳐 하나의 결과 wav로 저장한다.
        audio = np.concatenate(audio_chunks, axis=0)
        audio = _prepare_audio_for_playback(audio, sr)
        sf.write(_TTS_OUTPUT_PATH, audio, sr, subtype="PCM_16")


def main() -> None:
    """
    반복 및 단일 질의 실행
    """
    # CLI 인자는 최소한으로 유지해 실행 경로를 단순화한다.
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default=None)
    parser.add_argument("--tts", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--player", default="ffplay", choices=["auto", "ffplay", "none"])
    args = parser.parse_args()

    pipeline = RAGPipeline()

    if args.question:
        _run_once(pipeline, args.question, args.tts, args.device, args.player)
        return

    print("[INFO] Enter questions. Type 'exit' to quit.")
    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        _run_once(pipeline, question, args.tts, args.device, args.player)


if __name__ == "__main__":
    main()
