import os
import queue
import threading
import uuid
from pathlib import Path
from typing import Callable, Optional
import subprocess

from tts_runtime.infer_onnx import infer_tts_onnx


class TTSWorker:
    """
    문장 단위 TTS 합성/재생을 큐로 처리하는 워커.
    """

    def __init__(
        self,
        model_path: Path,
        bert_path: Path,
        config_path: Path,
        out_dir: str,
        device: str,
        player_cmd: Optional[list[str]],
        sanitize_fn: Callable[[str], str],
        split_fn: Callable[[str], list[str]],
    ) -> None:
        self._model_path = model_path
        self._bert_path = bert_path
        self._config_path = config_path
        self._out_dir = out_dir
        self._device = device
        self._player_cmd = player_cmd
        self._sanitize = sanitize_fn
        self._split = split_fn
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._lock = threading.Lock()
        self._current_proc: Optional[subprocess.Popen] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        os.makedirs(self._out_dir, exist_ok=True)
        self._thread.start()

    def enqueue(self, sentence: str) -> None:
        self._queue.put(sentence)

    def close(self) -> None:
        self._queue.put(None)
        self._queue.join()
        self._thread.join(timeout=2.0)

    def cancel(self) -> None:
        # 현재 재생 중인 프로세스를 중지하고 큐를 비운다.
        self._stop_event.set()
        with self._lock:
            if self._current_proc and self._current_proc.poll() is None:
                self._current_proc.terminate()
                try:
                    self._current_proc.wait(timeout=1.0)
                except Exception:
                    self._current_proc.kill()
            self._current_proc = None
        while True:
            try:
                item = self._queue.get_nowait()
                self._queue.task_done()
                if item is None:
                    continue
            except queue.Empty:
                break
        self._stop_event.clear()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                if self._stop_event.is_set():
                    continue
                clean_sent = self._sanitize(item)
                if not clean_sent:
                    continue
                segments = self._split(clean_sent)
                for segment in segments:
                    if self._stop_event.is_set():
                        break
                    out_path = os.path.join(self._out_dir, f"tts_{uuid.uuid4().hex}.wav")
                    audio = infer_tts_onnx(
                        onnx_path=str(self._model_path),
                        bert_onnx_path=str(self._bert_path),
                        config_path=str(self._config_path),
                        text=segment,
                        speaker_id=0,
                        language="KR",
                        device=self._device,
                        out_path=out_path,
                        log=False,
                    )
                    if self._player_cmd:
                        with self._lock:
                            self._current_proc = subprocess.Popen(self._player_cmd + [out_path])
                        self._current_proc.wait()
            finally:
                self._queue.task_done()
