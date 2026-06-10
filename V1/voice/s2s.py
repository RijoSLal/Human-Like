# TODO 
# model local loading

import queue
import time
import threading
import sys
import warnings

# Suppress the pynvml deprecation warning from torch/cuda
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")

from moonshine_voice.transcriber import TranscriptEventListener, TranscriptLine
from moonshine_voice import get_model_for_language
from openai import OpenAI

from aec_bridge import AECReferenceBuffer
from stt.stt import MicTranscriber
from tts.tts import KokoroStreamer

# ------------------ model selection ------------------------

model_path, model_arch = get_model_for_language(
    wanted_language="en", wanted_model_arch=None
)


# ----------------------- LLM streaming client ---------------------------------

# --------------------------- !!! Temporary !!! -----------------------------

class StreamingChatClient:
    def __init__(self, base_url, api_key, model, interrupted):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Always reply in short sentences and do not use emojis."
                ),
            }
        ]
        self.llm_queue: queue.Queue = queue.Queue()
        self.interrupted = interrupted
        self.count = 0

    def chat(self, user_input: str):
        # clear the queue of any stale data from previous turns
        while not self.llm_queue.empty():
            try:
                self.llm_queue.get_nowait()
            except:
                break
        
        if self.count > 5:
           self.messages.pop(0)
           self.count = 0
        self.messages.append({"role": "user", "content": user_input})

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            stream=True,
            response_format={"type": "text"},
        )
        full_content = ""
        for chunk in stream:
            if self.interrupted.is_set():
                stream.close()
                break
            content = chunk.choices[0].delta.content
            if content:
                if not full_content:
                    sys.stdout.write("\nAssistant: ")
                sys.stdout.write(content)
                sys.stdout.flush()
                self.llm_queue.put(content)
                full_content+=f"{content}" 
        self.llm_queue.put(None)  
        self.messages.append({"role": "assistant", "content": full_content})
        self.count+=1

# --------------------------- !!! Temporary !!! -----------------------------

# ----------------------- STT listener ---------------------------

class TerminalListener(TranscriptEventListener):
    def __init__(self, interrupted):
        self.last_line_text_length = 0
        self.stt_queue: queue.Queue = queue.Queue()
        self.interrupted = interrupted

    def update_last_terminal_line(self, line: TranscriptLine):
        return line.text

    def on_line_started(self, event):          # user started speaking -> interrupt
        self.interrupted.set()
        self.last_line_text_length = 0

    def on_line_text_changed(self, event):
        self.update_last_terminal_line(event.line)

    def on_line_completed(self, event):
        complete = self.update_last_terminal_line(event.line)
        self.stt_queue.put(complete)
        print(f"\n[Agent: Voice] User Input: {complete}")


# ----------------------- Conversation cycle --------------------------------

class Conversation:
    def __init__(self):
        self.interrupted = threading.Event()

        # shared AEC reference buffer (TTS writes, STT reads)
        self.aec_ref = AECReferenceBuffer()

        self.auditor = TerminalListener(self.interrupted)

        # MicTranscriber receives aec_ref -> will cancel speaker echo
        self.mic_transcriber = MicTranscriber(
            model_path=model_path,
            model_arch=model_arch,
            aec_ref=self.aec_ref,       # AEC wired in
        )

        self.chat_client = StreamingChatClient(
            base_url = "http://localhost:11434/v1/",
            api_key = "ollama",
            model = "qwen3:1.7b",
            interrupted=self.interrupted,
        )

        # KokoroStreamer receives aec_ref -> writes reference audio
        self.tts = KokoroStreamer(
            interrupted=self.interrupted,
            aec_ref=self.aec_ref,       # AEC wired in
        )

    def start(self):
        self.mic_transcriber.add_listener(self.auditor)
        self.mic_transcriber.start()
        print("Listening to the microphone, press Ctrl+C to stop...", file=sys.stderr)
        try:
            while True:
                speech = self.auditor.stt_queue.get()
                self.interrupted.clear()
                # start LLM in background so TTS can start speaking immediately
                threading.Thread(target=self.chat_client.chat, args=(speech,), daemon=True).start()
                
                self.tts.feed_queue(self.chat_client.llm_queue)
                self.tts.wait_until_done()
                time.sleep(0.1)
        finally:
            self.mic_transcriber.stop()
            self.mic_transcriber.close()
            self.tts.stop()


if __name__ == "__main__":
    conversation = Conversation()
    conversation.start()
