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

    def start(self) -> None:
        os.makedirs(self._out_dir, exist_ok=True)
        self._thread.start()

    def enqueue(self, sentence: str) -> None:
        self._queue.put(sentence)

    def close(self) -> None:
        self._queue.put(None)
        self._queue.join()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                clean_sent = self._sanitize(item)
                if not clean_sent:
                    continue
                segments = self._split(clean_sent)
                for segment in segments:
                    out_path = os.path.join(self._out_dir, f"tts_{uuid.uuid4().hex}.wav")
                    infer_tts_onnx(
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
                        subprocess.run(self._player_cmd + [out_path], check=False)
            finally:
                self._queue.task_done()
