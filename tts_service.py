from kokoro import KPipeline
import sounddevice as sd
import numpy as np
import traceback
import logging
import torch
import time
import os
# hard-disable cuDNN for Kokoro
torch.backends.cudnn.enabled = False  # NEW: timestamps for stream events

# NEW: WS broadcaster for UI
try:
    from tts_ws import WS
except Exception:
    WS = None
    logging.warning("tts_ws.WS not available; UI streaming disabled.")

class TextToSpeechService:
    """Drop-in: keeps sd+WS behavior; adds GPU→CPU fallback + env toggles."""
    def __init__(self):
        logging.info("Initializing TextToSpeechService...")
        logging.getLogger('numba').setLevel(logging.WARNING)
        logging.getLogger('torch').setLevel(logging.WARNING)

        # Tunables
        self.voice = os.getenv("ELYSIA_TTS_VOICE", "af_heart")
        self.sample_rate = int(os.getenv("ELYSIA_TTS_SR", "24000"))
        prefer_gpu = os.getenv("ELYSIA_TTS_GPU", "0") == "1"
        if os.getenv("ELYSIA_TTS_DISABLE_CUDNN", "0") == "1":
            torch.backends.cudnn.enabled = False

        # Init engine
        self.engine = None
        if prefer_gpu and torch.cuda.is_available():
            try:
                self.engine = KPipeline(device="cuda", lang_code="a")
                logging.info("Kokoro TTS initialized on CUDA.")
            except Exception as e:
                logging.warning(f"CUDA TTS init failed: {e}")

        if self.engine is None:
            try:
                self.engine = KPipeline(device="cpu", lang_code="a")
                logging.info("Kokoro TTS initialized on CPU.")
            except Exception as e:
                logging.error(f"Failed to initialize Kokoro TTS on CPU: {e}")
                logging.error(traceback.format_exc())
                raise

    def speak(self, text: str):
        if not text:
            return
        try:
            logging.info(f"TTS generating audio for: {text!r}")
            msg_id = f"msg_{int(time.time()*1000)}"
            if WS: WS.tts_begin(self.sample_rate, msg_id)

            chunks = []
            for result in self.engine(text=text, voice=self.voice):
                if result.audio is None:
                    continue
                audio = result.audio.detach().cpu().numpy().astype(np.float32)
                chunks.append(audio)
                if WS:
                    WS.tts_chunk(msg_id, time.time(), audio.tobytes())

            if not chunks:
                logging.warning("TTS generated no audio chunks.")
                if WS: WS.tts_end(msg_id)
                return

            audio = np.concatenate(chunks, axis=0)
            sd.play(audio, self.sample_rate)
            sd.wait()
            if WS: WS.tts_end(msg_id)
        except Exception:
            logging.error("Error in TTS service:\n" + traceback.format_exc())
            if WS:
                WS.state("error")
if __name__ == '__main__':
    # Example usage
    tts = TextToSpeechService()

    # Example with emphasis control
    expressive_text = "I [really](+1) think this is a [fantastic](+2) idea. Let's do it."
    tts.speak(expressive_text)

    # Example with pronunciation control
    pronunciation_text = "The model is called [Kokoro](/kˈOkəɹO/)."
    tts.speak(pronunciation_text)
