#------------------- doc --------------------

# # 🇺🇸 'a' => american english, 🇬🇧 'b' => british english
# # 🇪🇸 'e' => spanish es
# # 🇫🇷 'f' => french fr-fr
# # 🇮🇳 'h' => hindi hi
# # 🇮🇹 'i' => italian it
# # 🇯🇵 'j' => japanese: pip install misaki[ja]
# # 🇧🇷 'p' => brazilian portuguese pt-br
# # 🇨🇳 'z' => mandarin chinese: pip install misaki[zh]

#---------------------------------------------

import queue
import threading
import numpy as np
import sounddevice as sd
from collections import deque
import os
import orjson
from typing import Any, Optional

# try to import onnxruntime, fallback to torch if not available or requested
try:
    import onnxruntime as ort
except ImportError:
    ort = None

import torch
from kokoro import KPipeline, KModel
from huggingface_hub import hf_hub_download

class KokoroStreamer:
    """
    streams audio from kokoro tts model.
    """
    def __init__(
        self,
        interrupted: threading.Event,
        lang_code: str = 'a',
        voice: str = 'af_bella',
        speed: float = 1.15,
        chunk_size: int = 10,
        aec_ref: Optional[Any] = None, # AECReferenceBuffer | None
        use_onnx: bool = True          # default to onnx for speed
    ) -> None:  
        """
        initialize the kokoro streamer

        args:
            interrupted (threading.Event): event to signal interruption
            lang_code (str): language code for tts
            voice (str): voice name to use
            speed (float): playback speed
            chunk_size (int): number of words to buffer before synthesis
            aec_ref (Optional[Any]): optional aec reference buffer
            use_onnx (bool): whether to use onnx for inference

        returns:
            None
        """
        self.use_onnx = use_onnx and (ort is not None)
        self.repo_id = 'hexgrad/Kokoro-82M'
        self.lang_code = lang_code
        self.voice = voice
        self.speed = speed
        self.sample_rate = 24000
        self.chunk_size = chunk_size
        self.sentence_endings = frozenset('.!?,;:')
        self.interrupted = interrupted
        self._aec_ref = aec_ref

        if self.use_onnx:
            model_path = os.path.join(os.path.dirname(__file__), 'kokoro_model', 'model_fp16.onnx')
            if not os.path.exists(model_path):
                print(f"Warning: ONNX model not found at {model_path}. Falling back to PyTorch.")
                self.use_onnx = False
            else:
                try:
                    self.ort_session = ort.InferenceSession(model_path)
                    # load vocab for onnx
                    config_path = hf_hub_download(repo_id=self.repo_id, filename='config.json')
                    with open(config_path, 'rb') as f:
                        config = orjson.loads(f.read())
                    self.vocab = config['vocab']
                    # initialize pipeline without the heavy pytorch model
                    self.pipeline = KPipeline(lang_code=lang_code, repo_id=self.repo_id, model=False, trf=True)
                except Exception as e:
                    print(f"Error initializing ONNX: {e}. Falling back to PyTorch.")
                    self.use_onnx = False

        if not self.use_onnx:
            self.pipeline = KPipeline(lang_code=lang_code, repo_id=self.repo_id, trf=True)

        self._audio: deque[np.ndarray] = deque()
        self._playback_pos = 0
        self._synth_done = threading.Event()
        self._audio_lock = threading.Lock()

        try:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self._audio_callback,
                dtype='float32',
            )
            self.stream.start()
        except Exception as e:
            print(f"Warning: Could not open audio output stream: {e}")
            self.stream = None


    def _audio_callback(self, outdata: np.ndarray, frames: int, time: Any, status: Any) -> None:
        """
        callback for the audio output stream

        args:
            outdata (np.ndarray): buffer to fill with audio data
            frames (int): number of frames to provide
            time (Any): time information
            status (Any): status flags

        returns:
            None
        """
        if self.interrupted and self.interrupted.is_set():
            with self._audio_lock:
                self._audio.clear()
                self._playback_pos = 0
            outdata[:] = np.zeros((frames, 1), dtype='float32')
            if self._aec_ref is not None:
                self._aec_ref.write(np.zeros(frames, dtype='float32'))
            return

        out = np.zeros(frames, dtype='float32')
        offset = 0
        with self._audio_lock:
            while offset < frames and self._audio:
                chunk = self._audio[0]
                available = len(chunk) - self._playback_pos
                needed = frames - offset

                if available <= needed:
                    out[offset : offset + available] = chunk[self._playback_pos :]
                    offset += available
                    self._audio.popleft()
                    self._playback_pos = 0
                else:
                    out[offset:] = chunk[self._playback_pos : self._playback_pos + needed]
                    self._playback_pos += needed
                    offset = frames

        outdata[:] = out.reshape(-1, 1)
        if self._aec_ref is not None:
            self._aec_ref.write(out.copy())

    def _synthesize_text(self, text: str) -> None:
        """
        synthesize text into audio using the kokoro model

        args:
            text (str): text to synthesize

        returns:
            None
        """
        text = text.strip()
        if not text:
            return
        
        if self.use_onnx:
            # use onnx inference
            for _, ps, _ in self.pipeline(text, voice=self.voice, speed=self.speed):
                if not ps:
                    continue
                
                # convert phonemes to input_ids
                input_ids = [0] + [self.vocab[p] for p in ps if p in self.vocab] + [0]
                if len(input_ids) > 512:
                    input_ids = input_ids[:511] + [0]
                
                # load voice pack (style embedding)
                pack = self.pipeline.load_voice(self.voice)
                # style index matches len(ps) - 1 in kokoro
                style_idx = min(len(ps) - 1, pack.shape[0] - 1)
                style = pack[style_idx].numpy().astype(np.float32)
                
                # prepare onnx inputs
                onnx_inputs = {
                    'input_ids': np.array([input_ids], dtype=np.int64),
                    'style': style,
                    'speed': np.array([self.speed], dtype=np.float32)
                }
                
                # run onnx inference
                audio = self.ort_session.run(['waveform'], onnx_inputs)[0]
                audio = audio.flatten() # ensure 1d
                
                with self._audio_lock:
                    self._audio.append(audio.astype('float32'))
        else:
            # original pytorch inference
            for _, _, audio in self.pipeline(text, voice=self.voice, speed=self.speed):
                if audio is not None:
                    audio_np = audio.numpy() if hasattr(audio, 'numpy') else np.array(audio)
                    audio_np = audio_np.flatten() # ensure 1d
                    with self._audio_lock:
                        self._audio.append(audio_np.astype('float32'))


    def feed_queue(self, word_queue: queue.Queue, sentinel: Any = None) -> None:
        """
        feed words from a queue into the synthesis engine

        args:
            word_queue (queue.Queue): queue containing words to synthesize
            sentinel (Any): optional value to signal the end of the queue

        returns:
            None
        """
        buffer = []
        self._synth_done.clear()

        def synth_thread():
            while True:
                word = word_queue.get()
                if word is sentinel:
                    break
                if self.interrupted and self.interrupted.is_set():
                    while True:
                        try:
                            word_queue.get_nowait()
                        except queue.Empty:
                            break
                    self._synth_done.set()
                    return

                buffer.append(word)
                if (
                    any(word.endswith(p) for p in self.sentence_endings)
                    or len(buffer) >= self.chunk_size
                ):
                    self._synthesize_text(''.join(buffer))
                    buffer.clear()

            if not (self.interrupted and self.interrupted.is_set()):
                self._synthesize_text(''.join(buffer))
            self._synth_done.set()

        threading.Thread(target=synth_thread, daemon=True).start()

    def wait_until_done(self) -> None:
        """
        wait until all audio has been synthesized and played
        """
        self._synth_done.wait()
        while self._audio:
            sd.sleep(50)
        sd.sleep(200)

    def stop(self) -> None:
        """
        stop the audio stream
        """
        if self.stream:
            self.stream.stop()
            self.stream.close()
