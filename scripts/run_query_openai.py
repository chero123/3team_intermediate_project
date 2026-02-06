"""
OpenAI LLM 버전 실행 스크립트

흐름:
- 질문 입력
- OpenAI RAG 파이프라인 실행
- 문장 출력
- (옵션) TTS 합성/재생/저장


실행 방법:
uv run scripts/run_query_openai.py --tts --device cuda
"""
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

from rag.openai_pipeline import OpenAIRAGPipeline
from tts_runtime.infer_onnx import infer_tts_onnx

# TTS ONNX 모델 경로 (Melo-TTS)
_TTS_MODEL_PATH = Path("models/melo_yae/melo_yae.onnx")
# TTS BERT 피처 모델 경로
_TTS_BERT_PATH = Path("models/melo_yae/bert_kor.onnx")
# TTS 설정 파일 경로 (샘플레이트 등 메타 포함)
_TTS_CONFIG_PATH = Path("models/melo_yae/config.json")
# 최종 합성 음성 저장 경로
_TTS_OUTPUT_PATH = Path("data/answer/answer.wav")


def _is_junk_line(line: str) -> bool:
    """
    출력에서 제외할 '노이즈 라인'인지 판별한다.

    Args:
        line: 원본 라인 문자열

    Returns:
        bool: True면 제거 대상
    """
    # 한글/영문/숫자만 남겨 최소 의미 문자열을 만든다.
    stripped = re.sub(r"[^0-9A-Za-z가-힣]", "", line)
    # 빈 문자열이면 의미 없는 라인으로 간주
    if not stripped:
        return True
    # 숫자만 있으면 의미 없는 라인으로 간주
    if stripped.isdigit():
        return True
    # 숫자 비율이 과도하면 표/번호열로 판단해 제거
    digit_ratio = sum(ch.isdigit() for ch in stripped) / max(len(stripped), 1)
    if digit_ratio > 0.4:
        return True
    # 길이가 너무 짧으면 유의미 문장으로 보기 어려움
    if len(stripped) < 4:
        return True
    return False


def _sanitize_answer(text: str) -> str:
    """
    LLM 출력에서 컨텍스트/잡문을 제거하고 문장만 남긴다.

    Args:
        text: 원본 LLM 출력

    Returns:
        str: 정제된 출력 (비어 있으면 원문 일부를 반환)
    """
    # 라인 단위 정제 결과를 누적
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        # 빈 줄은 제거
        if not stripped:
            continue
        # 컨텍스트/소스 표시 라인은 제거
        if stripped.startswith("컨텍스트") or stripped.startswith("[Source"):
            continue
        # 노이즈 라인 제거
        if _is_junk_line(stripped):
            continue
        cleaned_lines.append(stripped)
    # 라인을 합쳐서 하나의 텍스트로 만든다.
    cleaned = " ".join(cleaned_lines)
    # 괄호/대괄호 메타정보 제거
    cleaned = re.sub(r"\([^)]*\)", "", cleaned)
    cleaned = re.sub(r"\[[^\]]*\]", "", cleaned)
    # 허용 문자 외는 공백으로 치환
    cleaned = re.sub(r"[^0-9A-Za-z가-힣\s\.\!\?]", " ", cleaned)
    # 의미 없는 긴 숫자열 제거
    cleaned = re.sub(r"\d{20,}", "", cleaned)
    # 중복 공백 정리
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # 전부 지워졌다면 원문 일부를 보존한다.
    return cleaned or text.strip()


def _split_sentences(text: str) -> list[str]:
    """
    마침표/물음표/느낌표 기준으로 문장을 분리한다.

    Args:
        text: 입력 텍스트

    Returns:
        list[str]: 문장 리스트
    """
    # 구두점 토큰을 포함해 split하여 문장 경계를 유지한다.
    # 숫자 사이의 소수점(예: 5.5)은 문장 분리 대상에서 제외한다.
    protected = re.sub(r"(?<=\d)\.(?=\d)", "<DOT>", text)
    # 파일 확장자(.hwp/.pdf/.docx)는 문장 분리 대상에서 제외한다.
    protected = re.sub(r"\.(hwp|pdf|docx)\b", r"<EXTDOT>\1", protected, flags=re.IGNORECASE)
    parts = re.split(r"([.!?。！？]+)", protected)
    sentences: list[str] = []
    # 버퍼에 누적해 구두점이 나오면 문장 확정
    buf = ""
    for part in parts:
        if not part:
            continue
        buf += part
        if re.fullmatch(r"[.!?。！？]+", part):
            if buf.strip():
                restored = buf.strip().replace("<EXTDOT>", ".").replace("<DOT>", ".")
                sentences.append(restored)
            buf = ""
    # 마지막 버퍼 처리
    if buf.strip():
        restored = buf.strip().replace("<EXTDOT>", ".").replace("<DOT>", ".")
        sentences.append(restored)
    return sentences


def _ensure_sentence(text: str) -> str:
    """
    문장 끝에 마침표가 없으면 추가한다.

    Args:
        text: 문장 문자열

    Returns:
        str: 마침표로 끝나는 문장
    """
    # 이미 구두점으로 끝나면 그대로 반환
    if re.search(r"[.!?。！？]\s*$", text):
        return text
    # 끝에 마침표 추가
    return f"{text}."


def _is_reference_header(text: str) -> bool:
    """
    참고문헌 블록의 시작인지 판별해 TTS에서 제외한다.
    """
    stripped = text.strip()
    return stripped.startswith("[참고 문헌]") or stripped.startswith("참고문헌")


def _split_sentence_for_tts(sentence: str, max_words: int = 8, max_chars: int = 90) -> list[str]:
    """
    너무 긴 문장을 TTS용 짧은 구간으로 쪼갠다.

    Args:
        sentence: 입력 문장
        max_words: 세그먼트당 최대 단어 수
        max_chars: 세그먼트당 최대 문자 수

    Returns:
        list[str]: TTS 세그먼트 리스트
    """
    # 길이가 충분히 짧으면 그대로 반환
    if len(sentence) <= max_chars:
        return [sentence]
    # 공백 기준 단어 분리
    words = sentence.split()
    if not words:
        return [sentence]
    chunks: list[str] = []
    buf: list[str] = []
    for word in words:
        candidate = " ".join(buf + [word])
        # 길이/단어 수 제한을 넘으면 버퍼를 확정
        if len(candidate) > max_chars or len(buf) >= max_words:
            if buf:
                chunks.append(" ".join(buf))
            buf = [word]
        else:
            buf.append(word)
    # 마지막 버퍼 처리
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def _apply_fade(audio: np.ndarray, sr: int, fade_ms: int = 10) -> np.ndarray:
    """
    클릭 노이즈 방지를 위해 앞뒤에 짧은 페이드를 적용한다.

    Args:
        audio: 1차원 오디오 배열
        sr: 샘플레이트
        fade_ms: 페이드 길이(ms)

    Returns:
        np.ndarray: 페이드가 적용된 오디오
    """
    # 빈 오디오는 그대로 반환
    if audio.size == 0:
        return audio
    # ms 단위를 샘플 수로 변환
    fade_len = int(sr * (fade_ms / 1000.0))
    # 길이가 너무 길면 절반까지만 적용
    fade_len = min(fade_len, audio.size // 2)
    if fade_len <= 1:
        return audio
    # 선형 페이드 인/아웃 커브 생성
    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    # 곱셈을 위해 float32로 변환
    audio = audio.astype(np.float32, copy=False)
    # 앞/뒤 구간에 페이드 적용
    audio[:fade_len] *= fade_in
    audio[-fade_len:] *= fade_out
    return audio


def _prepare_audio_for_playback(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    재생 안정성을 위해 오디오를 정규화/정리한다.

    Args:
        audio: 원본 오디오
        sr: 샘플레이트

    Returns:
        np.ndarray: 정리된 오디오
    """
    # 항상 1차원 float32 배열로 변환
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    # 빈 오디오는 그대로 반환
    if audio.size == 0:
        return audio
    # NaN/Inf 제거
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    # 클리핑으로 과도한 레벨을 방지
    audio = np.clip(audio, -1.0, 1.0)
    # 클릭 노이즈 감소를 위한 페이드 적용
    audio = _apply_fade(audio, sr, fade_ms=12)
    return audio


def _select_audio_player(preferred: str | None = None) -> list[str] | None:
    """
    사용 가능한 오디오 플레이어 명령을 선택한다.

    Args:
        preferred: 사용자가 지정한 플레이어

    Returns:
        list[str] | None: 실행 커맨드(없으면 None)
    """
    # 명시적으로 끈 경우
    if preferred in {"none", "off"}:
        return None
    # 지정된 플레이어만 허용
    if preferred in {"ffplay"}:
        path = shutil.which(preferred)
        if not path:
            return None
        if preferred == "ffplay":
            return [path, "-autoexit", "-nodisp", "-loglevel", "error"]
        return [path]
    # 자동 선택(현재는 ffplay만 사용)
    for candidate in ("ffplay",):
        path = shutil.which(candidate)
        if not path:
            continue
        if candidate == "ffplay":
            return [path, "-autoexit", "-nodisp", "-loglevel", "error"]
        return [path]
    return None


def _run_once(pipeline: OpenAIRAGPipeline, question: str, use_tts: bool, device: str, player: str) -> None:
    """
    질문 하나를 처리하고 텍스트/음성을 출력한다.

    Args:
        pipeline: OpenAI RAG 파이프라인
        question: 사용자 질문
        use_tts: TTS 사용 여부
        device: TTS 실행 디바이스
        player: 재생 플레이어 이름
    """
    # RAG 파이프라인으로 답변 생성
    answer = pipeline.ask(question)
    # 후처리로 불필요한 라인을 제거
    answer = _sanitize_answer(answer)
    # 문장 단위로 분리해 그대로 출력한다.
    sentences = [s for s in (s.strip() for s in _split_sentences(answer)) if s]
    if not sentences:
        print("답변을 생성할 수 없습니다.")
        return

    # TTS 합성 결과를 문장 단위로 누적
    audio_chunks: list[np.ndarray] = []
    sr = None
    player_cmd = None
    if use_tts:
        # 샘플레이트는 TTS config에서 읽는다.
        with _TTS_CONFIG_PATH.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        sr = cfg["data"]["sampling_rate"]
        # 재생 플레이어 커맨드 결정
        player_cmd = _select_audio_player(player)

    tts_allowed = True
    for sentence in sentences:
        sentence = _ensure_sentence(sentence.strip())
        if not sentence:
            continue
        if _is_reference_header(sentence):
            tts_allowed = False
        # 문장 텍스트를 먼저 출력
        print(sentence)

        if use_tts and tts_allowed:
            # 긴 문장을 TTS 세그먼트로 분할
            tts_segments = _split_sentence_for_tts(sentence)
            sentence_chunks = []
            for segment in tts_segments:
                # ONNX TTS 모델 호출
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
                    # 재생 품질 안정화 처리
                    audio = _prepare_audio_for_playback(audio, sr)
                # 세그먼트 오디오 누적
                sentence_chunks.append(audio)
                if sr is not None:
                    # 세그먼트 사이의 짧은 무음 삽입
                    silence = np.zeros(int(sr * 0.12), dtype=np.float32)
                    sentence_chunks.append(silence)

            if sentence_chunks:
                # 문장 단위로 오디오를 연결
                sentence_audio = np.concatenate(sentence_chunks, axis=0)
                audio_chunks.append(sentence_audio)
                if player_cmd and sr is not None:
                    # 즉시 재생용 오디오를 준비
                    playback = _prepare_audio_for_playback(sentence_audio, sr)
                    # 꼬리 클릭 방지를 위한 짧은 무음
                    tail = np.zeros(int(sr * 0.10), dtype=np.float32)
                    playback = np.concatenate([playback, tail], axis=0)
                    # 임시 wav로 저장 후 재생
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        sf.write(tmp.name, playback, sr, subtype="PCM_16")
                    subprocess.run(player_cmd + [tmp.name], check=False)

    if use_tts and audio_chunks:
        # 최종 결과 wav 저장
        _TTS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        audio = np.concatenate(audio_chunks, axis=0)
        audio = _prepare_audio_for_playback(audio, sr)
        sf.write(_TTS_OUTPUT_PATH, audio, sr, subtype="PCM_16")


def main() -> None:
    """
    CLI 엔트리포인트.

    - --question: 단건 실행
    - 입력이 없으면 REPL 모드
    """
    # CLI 인자 파서 구성
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default=None)
    parser.add_argument("--tts", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--player", default="ffplay", choices=["auto", "ffplay", "none"])
    args = parser.parse_args()

    # OpenAI RAG 파이프라인 초기화
    pipeline = OpenAIRAGPipeline()

    # 단건 실행 모드
    if args.question:
        _run_once(pipeline, args.question, args.tts, args.device, args.player)
        return

    # 대화형 입력 모드
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
