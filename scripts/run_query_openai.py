"""
OpenAI LLM 버전 실행 스크립트

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

_TTS_MODEL_PATH = Path("models/melo_yae/melo_yae.onnx")
_TTS_BERT_PATH = Path("models/melo_yae/bert_kor.onnx")
_TTS_CONFIG_PATH = Path("models/melo_yae/config.json")
_TTS_OUTPUT_PATH = Path("data/answer/answer.wav")


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


def _ensure_sentence(text: str) -> str:
    if re.search(r"[.!?。！？]\s*$", text):
        return text
    return f"{text}."


def _force_three_sentences(text: str) -> list[str]:
    sentences = _split_sentences(text)
    sentences = [s for s in (s.strip() for s in sentences) if s]
    if len(sentences) < 3:
        sentences += ["모른다"] * (3 - len(sentences))
    return sentences[:3]


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


def _run_once(pipeline: OpenAIRAGPipeline, question: str, use_tts: bool, device: str, player: str) -> None:
    answer = pipeline.ask(question)
    answer = _sanitize_answer(answer)
    sentences = _force_three_sentences(answer)
    if not sentences:
        sentences = ["답변을 생성할 수 없습니다."]

    audio_chunks: list[np.ndarray] = []
    sr = None
    player_cmd = None
    if use_tts:
        with _TTS_CONFIG_PATH.open("r", encoding="utf-8") as f:
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
        _TTS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        audio = np.concatenate(audio_chunks, axis=0)
        audio = _prepare_audio_for_playback(audio, sr)
        sf.write(_TTS_OUTPUT_PATH, audio, sr, subtype="PCM_16")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default=None)
    parser.add_argument("--tts", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--player", default="ffplay", choices=["auto", "ffplay", "none"])
    args = parser.parse_args()

    pipeline = OpenAIRAGPipeline()

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
