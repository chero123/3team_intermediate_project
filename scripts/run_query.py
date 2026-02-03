"""
실행 방법 예시

uv run scripts/run_query.py --tts --device cuda
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

from tts_runtime.infer_onnx import infer_tts_onnx
from rag.pipeline import RAGPipeline

_REWRITE_PROMPT = (
    "너는 문서를 요약하는 여우 요괴다.\n"
    "살짝 건방진 말투로 간략하게 요약하라.\n"
    "설명체(이다/합니다/됩니다) 사용 금지, 반말로 단정적으로 말하라.\n"
    "반드시 3문장으로만 답하고, 각 문장은 마침표로 끝내라.\n"
    "괄호나 특수문자를 쓰지 말고, 목록/헤더/컨텍스트 인용은 금지한다.\n"
    "내용을 모르면 모른다고만 말하라.\n\n"
    "원문:\n{answer}\n\n"
    "요약:\n"
)


def _is_junk_line(line: str) -> bool:
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
    cleaned = " ".join(cleaned_lines)
    cleaned = re.sub(r"\([^)]*\)", "", cleaned)
    cleaned = re.sub(r"\[[^\]]*\]", "", cleaned)
    cleaned = re.sub(r"[^0-9A-Za-z가-힣\s\.\!\?]", " ", cleaned)
    cleaned = re.sub(r"\d{20,}", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"([.!?。！？]+)", text)
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


def _chunk_sentence(sentence: str, max_words: int = 12, max_chars: int = 140) -> list[str]:
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


def _ensure_sentence(text: str) -> str:
    if re.search(r"[.!?。！？]\s*$", text):
        return text
    return f"{text}."


def _target_sentence_count(question: str) -> int | None:
    if re.search(r"(세\s*문장|3\s*문장|3문장)", question):
        return 3
    if re.search(r"(두\s*문장|2\s*문장|2문장)", question):
        return 2
    if re.search(r"(한\s*문장|1\s*문장|1문장)", question):
        return 1
    return 3


def _postprocess_sentences(sentences: list[str], target_count: int | None) -> list[str]:
    sentences = [s for s in (s.strip() for s in sentences) if s]
    if target_count is None:
        return sentences
    if len(sentences) >= target_count:
        return sentences[:target_count]
    extra = []
    for sentence in list(sentences):
        if len(extra) + len(sentences) >= target_count:
            break
        parts = re.split(r"([,;:])", sentence)
        buf = ""
        for part in parts:
            buf += part
            if part in {",", ";", ":"}:
                if buf.strip():
                    extra.append(buf.strip())
                buf = ""
        if buf.strip():
            extra.append(buf.strip())
        if len(extra) + len(sentences) >= target_count:
            break
    merged = sentences + [s for s in extra if s]
    while len(merged) < target_count:
        merged.append("모른다")
    return merged[:target_count]


def _force_three_sentences(text: str) -> list[str]:
    sentences = _split_sentences(text)
    sentences = _postprocess_sentences(sentences, 3)
    if len(sentences) < 3:
        sentences += ["모른다"] * (3 - len(sentences))
    return sentences[:3]


def _rewrite_three_sentences(pipeline: RAGPipeline, answer: str) -> str:
    prompt = _REWRITE_PROMPT.format(answer=answer)
    try:
        rewritten = pipeline.llm.generate(
            prompt=prompt,
            max_tokens=140,
            temperature=0.3,
        )
        return rewritten.strip() or answer
    except Exception:
        return answer


def _split_sentence_for_tts(sentence: str, max_words: int = 8, max_chars: int = 90) -> list[str]:
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
    fade_len = int(sr * (fade_ms / 1000.0))
    fade_len = min(fade_len, audio.size // 2)
    if fade_len <= 1:
        return audio
    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    audio = audio.astype(np.float32, copy=False)
    audio[:fade_len] *= fade_in
    audio[-fade_len:] *= fade_out
    return audio


def _prepare_audio_for_playback(audio: np.ndarray, sr: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = np.clip(audio, -1.0, 1.0)
    audio = _apply_fade(audio, sr, fade_ms=12)
    return audio


def _select_audio_player(preferred: str | None = None) -> list[str] | None:
    if preferred in {"none", "off"}:
        return None
    if preferred in {"aplay", "ffplay"}:
        path = shutil.which(preferred)
        if not path:
            return None
        if preferred == "ffplay":
            return [path, "-autoexit", "-nodisp", "-loglevel", "error"]
        return [path]
    for candidate in ("aplay", "ffplay"):
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
    answer = pipeline.ask(question)
    answer = _sanitize_answer(answer)
    answer = _rewrite_three_sentences(pipeline, answer)
    answer = _sanitize_answer(answer)

    target_count = _target_sentence_count(question)
    if target_count == 3:
        sentences = _force_three_sentences(answer)
    else:
        sentences = _split_sentences(answer)
        sentences = _postprocess_sentences(sentences, target_count)
    if not sentences:
        sentences = [answer] if answer else ["답변을 생성할 수 없습니다."]

    audio_chunks = []
    sr = None
    player_cmd = None
    if use_tts:
        config_path = Path("models/melo_yae/config.json")
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        sr = cfg["data"]["sampling_rate"]
        player_cmd = _select_audio_player(player)

    for sentence in sentences:
        sentence = _ensure_sentence(sentence.strip())
        if not sentence:
            continue
        print(sentence)

        if use_tts:
            tts_segments = _split_sentence_for_tts(sentence)
            sentence_chunks = []
            for segment in tts_segments:
                audio = infer_tts_onnx(
                    onnx_path="models/melo_yae/melo_yae.onnx",
                    bert_onnx_path="models/melo_yae/bert_kor.onnx",
                    config_path="models/melo_yae/config.json",
                    text=segment,
                    speaker_id=0,
                    language="KR",
                    device=device,
                    out_path=None,
                )
                if sr is not None:
                    audio = _prepare_audio_for_playback(audio, sr)
                sentence_chunks.append(audio)
                if sr is not None:
                    silence = np.zeros(int(sr * 0.12), dtype=np.float32)
                    sentence_chunks.append(silence)

            if sentence_chunks:
                sentence_audio = np.concatenate(sentence_chunks, axis=0)
                audio_chunks.append(sentence_audio)
                if player_cmd and sr is not None:
                    playback = _prepare_audio_for_playback(sentence_audio, sr)
                    tail = np.zeros(int(sr * 0.10), dtype=np.float32)
                    playback = np.concatenate([playback, tail], axis=0)
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        sf.write(tmp.name, playback, sr, subtype="PCM_16")
                    subprocess.run(player_cmd + [tmp.name], check=False)

    if use_tts and audio_chunks:
        out_path = Path("data/answer/answer.wav")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        audio = np.concatenate(audio_chunks, axis=0)
        audio = _prepare_audio_for_playback(audio, sr)
        sf.write(out_path, audio, sr, subtype="PCM_16")


def main() -> None:
    """
    반복 및 단일 질의 실행
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default=None)
    parser.add_argument("--tts", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--player", default="ffplay", choices=["auto", "aplay", "ffplay", "none"])
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
