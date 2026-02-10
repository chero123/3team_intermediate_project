import json
import os
import queue
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Callable, Optional
import subprocess

import soundfile as sf

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
        # 합성/재생은 백그라운드 스레드에서 직렬 처리한다.
        self._thread = threading.Thread(target=self._run, daemon=True)
        # 재생 프로세스 접근 동기화용 (중단/교체 시 충돌 방지)
        self._lock = threading.Lock()
        # 현재 재생 중인 외부 플레이어 프로세스 핸들
        self._current_proc: Optional[subprocess.Popen] = None
        # cancel 시 즉시 중단 플래그
        self._stop_event = threading.Event()
        self._last_path: Optional[str] = None
        # 세션 단위 누적 출력 파일 (문장별 합성 결과를 한 파일로 저장)
        self._session_out_path = os.path.join(self._out_dir, f"tts_{uuid.uuid4().hex}.wav")
        # soundfile 핸들 (열려있는 동안 계속 write)
        self._sf: Optional[sf.SoundFile] = None
        self._sample_rate = self._load_sample_rate()

    def _load_sample_rate(self) -> int:
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            return int(cfg["data"]["sampling_rate"])
        except Exception:
            return 22050

    def start(self) -> None:
        os.makedirs(self._out_dir, exist_ok=True)
        # 문장별 합성을 하나의 파일로 누적 저장한다.
        self._sf = sf.SoundFile(
            self._session_out_path,
            mode="w",
            samplerate=self._sample_rate,
            channels=1,
            subtype="PCM_16",
        )
        # 백그라운드 스레드 시작
        self._thread.start()

    def enqueue(self, sentence: str) -> None:
        # 합성할 문장을 큐에 넣는다. None은 종료 신호로 사용됨.
        self._queue.put(sentence)

    def close(self) -> None:
        # 종료 신호를 넣고 큐 처리 완료를 기다린다.
        self._queue.put(None)
        self._queue.join()
        self._thread.join(timeout=2.0)
        if self._sf is not None:
            try:
                self._sf.close()
            except Exception:
                pass
            self._sf = None
        # 마지막 세션 파일 경로를 유지
        self._last_path = self._session_out_path

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
        self._last_path = None
        if self._sf is not None:
            try:
                self._sf.close()
            except Exception:
                pass
            self._sf = None
        # 큐에 남아있는 문장을 모두 버린다.
        while True:
            try:
                item = self._queue.get_nowait()
                self._queue.task_done()
                if item is None:
                    continue
            except queue.Empty:
                break
        self._stop_event.clear()

    def last_path(self) -> Optional[str]:
        """
        마지막으로 생성된 TTS 파일 경로를 반환한다.
        """
        return self._last_path

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                if self._stop_event.is_set():
                    continue
                # 전처리(정리/정규화) 후 짧은 조각으로 분리한다.
                clean_sent = self._sanitize(item)
                if not clean_sent:
                    continue
                segments = self._split(clean_sent)
                for segment in segments:
                    if self._stop_event.is_set():
                        break
                    out_path = os.path.join(self._out_dir, f"tts_{uuid.uuid4().hex}.wav")
                    self._last_path = out_path
                    # ONNX 모델로 단일 세그먼트 합성
                    audio = infer_tts_onnx(
                        onnx_path=str(self._model_path),
                        bert_onnx_path=str(self._bert_path),
                        config_path=str(self._config_path),
                        text=segment,
                        speaker_id=0,
                        language="KR",
                        device=self._device,
                        out_path=None,
                        log=False,
                    )
                    # 세션 누적 파일에 기록 (한 파일로 이어 붙임)
                    if self._sf is not None:
                        try:
                            self._sf.write(audio)
                        except Exception:
                            pass
                    # 재생은 문장 단위 임시 파일로 수행 (플레이어 제어 때문)
                    if self._player_cmd:
                        try:
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                                sf.write(tmp.name, audio, self._sample_rate, subtype="PCM_16")
                                temp_path = tmp.name
                            with self._lock:
                                self._current_proc = subprocess.Popen(self._player_cmd + [temp_path])
                            self._current_proc.wait()
                        finally:
                            if "temp_path" in locals() and temp_path:
                                try:
                                    os.unlink(temp_path)
                                except Exception:
                                    pass
            finally:
                self._queue.task_done()
