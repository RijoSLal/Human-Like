from moonshine_voice.transcriber import (
    Transcriber,
    TranscriptEvent,
    ModelArch,
)
import numpy as np
import sounddevice as sd
from typing import Callable, Optional

# ------------------------------- AEC engine --------------------------------------

try:
    from aec_audio_processing import AudioProcessor as _WebRTC_AP
    _AEC_AVAILABLE = True
except ImportError:
    _AEC_AVAILABLE = False

class _AECProcessor:
  
    """
    aec wrapper to handle mic and tts sample rate mismatch
    resamples speaker audio to match mic rate to prevent feedback
    """

    MIC_SR = 16_000
    REF_SR = 24_000

    def __init__(self, frame_size: int = 160):
        """
        samples per frame at 16khz mic rate
        must match sounddevice callback size
        """
        if not _AEC_AVAILABLE:
            raise RuntimeError(
                "aec-audio-processing is not installed. Run:  pip install aec-audio-processing"
            )

        self.frame_size = frame_size
        # WebRTC APM: AEC, NS, AGC
        self._aec = _WebRTC_AP(enable_aec=True, enable_ns=True, enable_agc=True)
        self._aec.set_stream_format(self.MIC_SR, 1)
        self._aec.set_reverse_stream_format(self.MIC_SR, 1)
        
        # resample ratio: how many 24 kHz samples equal one 16 kHz frame
        self._ref_frame_size = int(frame_size * self.REF_SR / self.MIC_SR)  # 240
        # rolling remainder for mic audio that doesn't fill a full frame
        self._mic_remainder = np.empty(0, dtype=np.float32)


    def process(
        self,
        mic_audio: np.ndarray,         # float32, 16 kHz
        ref_buffer,                    # AECReferenceBuffer  (24 kHz)
    ) -> np.ndarray:
        
        audio = np.concatenate([self._mic_remainder, mic_audio])
        output_chunks = []

        n_frames = len(audio) // self.frame_size
        
        for i in range(n_frames):
            mic_frame = audio[i * self.frame_size : (i + 1) * self.frame_size]

            # read exactly what was playing when this mic audio was captured
            ref_24k = ref_buffer.read(self._ref_frame_size)
            # resample far-end for AEC
            ref_16k = self._resample(ref_24k, self._ref_frame_size, self.frame_size)

            # WebRTC expects 16-bit PCM bytes. Use clipping to avoid overflow and handle potential NaN/inf.
            mic_clipped = np.nan_to_num(mic_frame, nan=0.0, posinf=1.0, neginf=-1.0).clip(-1, 1)
            ref_clipped = np.nan_to_num(ref_16k, nan=0.0, posinf=1.0, neginf=-1.0).clip(-1, 1)

            mic_int16 = (mic_clipped * 32767).astype(np.int16).tobytes()
            ref_int16 = (ref_clipped * 32767).astype(np.int16).tobytes()


            # AEC needs far-end first
            self._aec.process_reverse_stream(ref_int16)
            # then process near-end to get clean audio
            cleaned_bytes = self._aec.process_stream(mic_int16)
            
            cleaned_int16 = np.frombuffer(cleaned_bytes, dtype=np.int16)
            cleaned_f32 = cleaned_int16.astype(np.float32) / 32768.0
            output_chunks.append(cleaned_f32)

        processed = n_frames * self.frame_size
        self._mic_remainder = audio[processed:].copy()

        if output_chunks:
            return np.concatenate(output_chunks)
        return np.empty(0, dtype=np.float32)

    @staticmethod
    def _resample(data: np.ndarray, src_len: int, dst_len: int) -> np.ndarray:
        """nearest-neighbour resample -> fast and good enough for AEC reference."""
        if src_len == dst_len:
            return data
        indices = np.round(
            np.linspace(0, src_len - 1, dst_len)
        ).astype(np.int32)
        return data[indices]


# -------------------------------- MicTranscriber -----------------------------------------

class MicTranscriber:
    """
    transcribes mic audio and cancels speaker echo
    
    uses aec buffer to prevent the mic from picking up tts output
    """

    def __init__(
        self,
        model_path: str,
        model_arch: ModelArch = ModelArch.TINY,
        update_interval: float = 0.5,
        device: int = None,
        samplerate: int = 16000,
        channels: int = 1,
        blocksize: int = 1024,
        options: dict = None,
        aec_ref=None,                  # AECReferenceBuffer | None
        aec_frame_size: int = 160,     # 10 ms @ 16 kHz
    ):
        self.transcriber = Transcriber(model_path, model_arch, options=options)
        self.mic_stream = self.transcriber.create_stream(update_interval)
        self._should_listen = False
        self._sd_stream = None
        self._device = device
        self._samplerate = samplerate
        self._channels = channels
        self._blocksize = blocksize

        # -------------- AEC setup ------------------
        self._aec_ref = aec_ref
        self._aec_proc: Optional[_AECProcessor] = None
        if aec_ref is not None:
            if not _AEC_AVAILABLE:
                import warnings
                warnings.warn(
                    "aec-audio-processing is not installed – AEC disabled. "
                    "Install with:  pip install aec-audio-processing",
                    RuntimeWarning,
                )
            else:
                self._aec_proc = _AECProcessor(frame_size=aec_frame_size)

    # ------------------- Moonshine Internal -----------------------

    def _start_listening(self):
        def audio_callback(in_data, frames, time, status):
            if not self._should_listen:
                return
            if status:
                print(f"MicTranscriber: {status}")
            if in_data is None:
                return

            audio_data = in_data.astype(np.float32).flatten()

            if self._aec_proc is not None and self._aec_ref is not None:
                audio_data = self._aec_proc.process(audio_data, self._aec_ref)

            self.mic_stream.add_audio(audio_data, self._samplerate)

        self._sd_stream = sd.InputStream(
            samplerate=self._samplerate,
            blocksize=self._blocksize,
            device=self._device,
            channels=self._channels,
            dtype="float32",
            callback=audio_callback,
        )
        self._sd_stream.start()

    def start(self):
        self.mic_stream.start()
        if self._sd_stream is None:
            self._start_listening()
        self._should_listen = True

    def stop(self):
        self._should_listen = False
        self.mic_stream.stop()

    def close(self):
        self.mic_stream.close()
        self.transcriber.close()

    def add_listener(self, listener: Callable[[TranscriptEvent], None]) -> None:
        self.mic_stream.add_listener(listener)

    def remove_listener(self, listener: Callable[[TranscriptEvent], None]) -> None:
        self.mic_stream.remove_listener(listener)

    def remove_all_listeners(self):
        self.mic_stream.remove_all_listeners()