"""
echo cancellation buffer for speaker and mic sync
    
prevents mic from hearing tts output to avoid self-transcription
    
uses a lock-free ring buffer for low-latency audio thread access

"""
import collections
import threading
import numpy as np

class AECReferenceBuffer:
    """
    thread-safe audio ring buffer
    
    stores speaker output for mic echo cancellation
    
    syncs kokoro and mic threads while dropping stale samples
    """
    MAX_CHUNKS = 400  
    MAX_SAMPLES = 48000 # 2 seconds at 24kHz

    def __init__(self, sample_rate: int = 24000, delay_ms: int = 150):
        self._buf: collections.deque[np.ndarray] = collections.deque(maxlen=self.MAX_CHUNKS)
        self._lock = threading.Lock()
        self._total_samples = 0
        self._sample_rate = sample_rate
        self._delay_ms = delay_ms
        self.reset()

    def reset(self) -> None:
        """Clear the buffer and prime it with silence to match the expected latency."""
        with self._lock:
            self._buf.clear()
            self._total_samples = 0
            # Prime with silence to account for loopback latency
            delay_samples = int(self._sample_rate * self._delay_ms / 1000)
            if delay_samples > 0:
                silence = np.zeros(delay_samples, dtype=np.float32)
                self._buf.append(silence)
                self._total_samples = delay_samples

    def set_delay(self, delay_ms: int) -> None:
        """Update the loopback delay and reset the buffer."""
        self._delay_ms = delay_ms
        self.reset()

    def write(self, samples: np.ndarray) -> None:
        """push speaker output samples into the buffer (float32, mono).
        """
        chunk = np.asarray(samples, dtype=np.float32).flatten()

        if chunk.size == 0:
            return

        with self._lock:
            self._buf.append(chunk)
            self._total_samples += len(chunk)

            # limit total buffer size to prevent memory growth and excessive latency
            while self._total_samples > self.MAX_SAMPLES and self._buf:
                removed = self._buf.popleft()
                self._total_samples -= len(removed)

    def read(self, n_frames: int) -> np.ndarray:
        """read exactly n_frames of reference audio.
        returns zeros if not enough data (e.g. during silence).
        """
        out = np.zeros(n_frames, dtype=np.float32)
        offset = 0

        with self._lock:
            # If we don't have enough samples, we return silence for the missing part
            # but we still want to keep the "delay" primed.
            # In a real-time stream, we should always have enough samples if primed.
            
            while offset < n_frames and self._buf:
                chunk = self._buf[0]
                need = n_frames - offset

                if len(chunk) <= need:
                    out[offset:offset + len(chunk)] = chunk
                    offset += len(chunk)
                    self._total_samples -= len(chunk)
                    self._buf.popleft()
                else:
                    out[offset:] = chunk[:need]
                    self._buf[0] = chunk[need:]
                    self._total_samples -= need
                    offset = n_frames

        return out
    

        