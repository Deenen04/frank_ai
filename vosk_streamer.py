import asyncio
import json
import logging
import os
import time
import audioop
import urllib.request
import zipfile
from typing import Optional, List, Dict, Any

import numpy as np

try:
    import vosk
except ImportError:  # graceful fallback so the whole project can still import even if dep missing
    vosk = None  # type: ignore

__all__ = ["VoskStreamer"]

log = logging.getLogger("vosk_streamer")

# Model URLs for German, English, and French
# Using telephony/narrowband models that natively support 8kHz audio
VOSK_MODELS = {
    "en": {
        # Indian English model is specifically designed for telecom and broadcast
        "small": "https://alphacephei.com/vosk/models/vosk-model-small-en-in-0.4.zip",
        "large": "https://alphacephei.com/vosk/models/vosk-model-en-in-0.5.zip"
    },
    "de": {
        # German telephony model - specifically designed for telephony and server
        "small": "https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip", 
        "large": "https://alphacephei.com/vosk/models/vosk-model-de-0.21.zip"  # This is the telephony model
    },
    "fr": {
        # LINTO French model has good performance for various audio conditions
        "small": "https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip",
        "large": "https://alphacephei.com/vosk/models/vosk-model-fr-0.6-linto-2.2.0.zip"  # LINTO model
    }
}


class VoskStreamer:  # pylint: disable=too-many-instance-attributes
    """A drop-in replacement for *WhisperStreamer* that performs *local* real-time
    transcription using *Vosk*.

    Vosk natively supports 8 kHz audio and was trained on it, making it ideal for
    telephony applications. The public callback signatures are identical to the original
    Whisper implementation so existing application code does **not** have to change.
    """

    # ---------------------------------------------------------------------
    # Construction / configuration
    # ---------------------------------------------------------------------

    def __init__(
        self,
        api_key: str | None = None,  # ignored, kept for signature compatibility
        *,
        encoding: str = "mulaw",
        sample_rate: int = 8000,
        interim_results: bool = False,
        vad_events: bool = True,
        punctuate: bool = True,  # vosk already outputs punctuation
        model: str = "small",  # vosk model size to load ("small", "large")
        language: str = "en",  # "en", "de", "fr" or "multi" for auto-detection
        allowed_languages: Optional[List[str]] = None,
        use_amplitude_vad: bool = True,
        amplitude_threshold_db: float = -5.0,
        silence_timeout: float = 0.7,
        partial_interval: float = 0.6,  # seconds between partial transcripts
        models_dir: str = "./vosk_models"  # directory to store downloaded models
    ):
        # --- public config ------------------------------------------------
        self.encoding = encoding.lower()
        self.sample_rate_in = sample_rate
        self._use_amp_vad = use_amplitude_vad
        self._silence_timeout = silence_timeout
        self._punctuate = punctuate
        self._interim_results = interim_results
        self._model_size = model
        self._models_dir = models_dir
        
        # Language configuration
        self.language = language.lower() if language else "en"
        if self.language not in ["en", "de", "fr", "multi"]:
            log.warning("[Vosk] Unknown language '%s'. Falling back to 'en'.", self.language)
            self.language = "en"
            
        self._allowed_languages = None if allowed_languages is None else [l.lower() for l in allowed_languages]
        if self._allowed_languages:
            # Filter allowed languages to only supported ones
            supported = {"en", "de", "fr"}
            self._allowed_languages = [l for l in self._allowed_languages if l in supported]
            if not self._allowed_languages:
                log.warning("[Vosk] No supported languages in allowed_languages. Using English.")
                self._allowed_languages = ["en"]

        # Vosk models and recognizers
        self._models: Dict[str, Any] = {}  # language -> vosk.Model
        self._recognizers: Dict[str, Any] = {}  # language -> vosk.KaldiRecognizer
        self._current_language: str = self.language if self.language != "multi" else "en"

        # --- amplitude VAD ------------------------------------------------
        if self.encoding in {"mulaw", "ulaw"}:
            # Fixed minimum RMS for μ-law (same as whisper_streamer)
            self._amp_threshold_linear = 2000.0
        else:
            # Linear PCM: convert dB value to linear scale
            self._amp_threshold_linear = 32768 * (10 ** (amplitude_threshold_db / 20.0))

        self._last_voice_ts: Optional[float] = None
        self._speaking = False
        self._speech_end_detected = False

        # Buffer that stores 8 kHz PCM bytes belonging to the *current* speech segment
        # Note: Vosk works natively with 8kHz, no upsampling needed!
        self._speech_buffer: bytearray = bytearray()

        # Timestamp of the last partial transcription
        self._last_partial_ts: float | None = None
        self._partial_interval = max(0.3, partial_interval)

        # Async helpers ----------------------------------------------------
        self._loop = asyncio.get_event_loop()
        self._close_evt = asyncio.Event()
        self._speech_end_timer_task: Optional[asyncio.Task] = None

        # callbacks: identical keys to the Whisper/Deepgram version
        self._callbacks = {
            "on_speech_start": None,
            "on_transcript": None,
            "on_speech_end": None,
            "on_utterance": None,
        }

        # Language detection state
        self._language_locked = False
        self._language_detection_buffer = bytearray()
        self._language_detection_threshold = 16000 * 2  # 2 seconds of 8kHz 16-bit audio

        log.info(
            "[Vosk-INIT] Using model size '%s', language '%s', amp-threshold %.1f dB (→ %.0f), silence %.1fs, allowed=%s",
            self._model_size,
            self.language,
            amplitude_threshold_db,
            self._amp_threshold_linear,
            silence_timeout,
            self._allowed_languages,
        )

    # ------------------------------------------------------------------
    # Public helpers (unchanged API)
    # ------------------------------------------------------------------

    def on(self, event: str, callback):
        if event not in self._callbacks:
            raise ValueError(f"Unknown event '{event}'")
        self._callbacks[event] = callback

    def set_vad_threshold(self, amplitude_threshold_db: float):
        self._amp_threshold_linear = 32768 * (10 ** (amplitude_threshold_db / 20.0))
        log.info("[VAD-ADJUST] New threshold: %.1f dB → %.0f", amplitude_threshold_db, self._amp_threshold_linear)

    def set_rms_threshold(self, rms_threshold: float):
        self._amp_threshold_linear = rms_threshold
        log.info("[VAD-ADJUST] New RMS threshold: %.0f", self._amp_threshold_linear)

    # ------------------------------------------------------------------
    # Life-cycle helpers
    # ------------------------------------------------------------------

    async def connect(self):
        """Download and load Vosk models for the configured languages."""
        if vosk is None:
            raise RuntimeError(
                "vosk not installed. Add `vosk` to requirements.txt and pip-install it."
            )
        
        # Create models directory
        os.makedirs(self._models_dir, exist_ok=True)
        
        # Determine which languages to load models for
        languages_to_load = []
        if self.language == "multi":
            # Load models for all allowed languages, or default set
            languages_to_load = self._allowed_languages or ["en", "de", "fr"]
        else:
            languages_to_load = [self.language]
        
        # Download and load models
        for lang in languages_to_load:
            if lang in VOSK_MODELS:
                await self._load_model_for_language(lang)
            else:
                log.warning("[Vosk] No model available for language '%s'", lang)
        
        if not self._models:
            raise RuntimeError("No Vosk models could be loaded")
        
        # Set current language to first available model
        self._current_language = list(self._models.keys())[0]
        log.info("[Vosk] Models loaded for languages: %s. Current: %s", 
                list(self._models.keys()), self._current_language)

    async def _load_model_for_language(self, language: str):
        """Download and load a Vosk model for the specified language."""
        model_info = VOSK_MODELS[language][self._model_size]
        model_filename = os.path.basename(model_info).replace('.zip', '')
        model_path = os.path.join(self._models_dir, model_filename)
        
        # Download model if it doesn't exist
        if not os.path.exists(model_path):
            log.info("[Vosk] Downloading %s model for %s...", self._model_size, language)
            await self._download_and_extract_model(model_info, model_path)
        
        # Load the model
        try:
            # Set log level to reduce Vosk verbosity
            vosk.SetLogLevel(-1)
            
            model = vosk.Model(model_path)
            recognizer = vosk.KaldiRecognizer(model, self.sample_rate_in)
            
            # Enable partial results if requested
            if self._interim_results:
                recognizer.SetPartialWords(True)
            
            self._models[language] = model
            self._recognizers[language] = recognizer
            log.info("[Vosk] Loaded %s model for %s from %s", self._model_size, language, model_path)
            
        except Exception as e:
            log.error("[Vosk] Failed to load model for %s: %s", language, e)
            # Clean up potentially corrupted model directory
            if os.path.exists(model_path):
                import shutil
                shutil.rmtree(model_path, ignore_errors=True)

    async def _download_and_extract_model(self, model_url: str, extract_path: str):
        """Download and extract a Vosk model."""
        zip_path = extract_path + ".zip"
        
        try:
            # Download with progress (simplified)
            log.info("[Vosk] Downloading from %s", model_url)
            urllib.request.urlretrieve(model_url, zip_path)
            
            # Extract
            log.info("[Vosk] Extracting to %s", extract_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract to temporary directory first
                temp_extract = extract_path + "_temp"
                zip_ref.extractall(temp_extract)
                
                # Find the actual model directory (usually has a specific name)
                extracted_items = os.listdir(temp_extract)
                if len(extracted_items) == 1 and os.path.isdir(os.path.join(temp_extract, extracted_items[0])):
                    # Move the inner directory to the final location
                    import shutil
                    shutil.move(os.path.join(temp_extract, extracted_items[0]), extract_path)
                    shutil.rmtree(temp_extract)
                else:
                    # Direct extraction worked
                    os.rename(temp_extract, extract_path)
            
            # Clean up zip file
            os.remove(zip_path)
            log.info("[Vosk] Model extraction completed")
            
        except Exception as e:
            log.error("[Vosk] Failed to download/extract model: %s", e)
            # Clean up
            for path in [zip_path, extract_path, extract_path + "_temp"]:
                if os.path.exists(path):
                    if os.path.isfile(path):
                        os.remove(path)
                    else:
                        import shutil
                        shutil.rmtree(path, ignore_errors=True)
            raise

    def is_open(self) -> bool:
        return not self._close_evt.is_set()

    async def finish(self):
        if not self._close_evt.is_set():
            self._close_evt.set()
        # Clean up recognizers
        self._recognizers.clear()
        self._models.clear()

    # ------------------------------------------------------------------
    # Audio ingestion ---------------------------------------------------
    # ------------------------------------------------------------------

    async def send(self, audio_chunk: bytes):
        """Receive μ-law data from Twilio and feed it into the VAD / ASR stack."""

        # 1) μ-law → 16-bit PCM -----------------------------------------
        try:
            pcm16 = audioop.ulaw2lin(audio_chunk, 2)
        except audioop.error as e:  # pragma: no cover
            log.debug("audioop.ulaw2lin failed: %s", e)
            return

        # 2) Vosk works natively with 8kHz, no upsampling needed!
        pcm16_8k = pcm16

        # 3) Amplitude VAD ---------------------------------------------
        if self._use_amp_vad:
            rms = audioop.rms(pcm16, 2)
            now_ts = time.time()

            # RMS debugging each 2 s
            if not hasattr(self, "_last_rms_log_ts") or now_ts - self._last_rms_log_ts > 2.0:
                log.debug("[VAD] rms %.0f (thr %.0f) speaking=%s", rms, self._amp_threshold_linear, self._speaking)
                self._last_rms_log_ts = now_ts

            if rms > self._amp_threshold_linear:
                # Voice *present*
                self._last_voice_ts = now_ts
                if not self._speaking:
                    self._trigger_speech_start()
                self._speech_buffer.extend(pcm16_8k)

                # Language detection for multi-language mode
                if self.language == "multi" and not self._language_locked:
                    self._language_detection_buffer.extend(pcm16_8k)
                    if len(self._language_detection_buffer) >= self._language_detection_threshold:
                        await self._detect_language()

                # Emit *partial* transcription every self._partial_interval s
                if (
                    self._interim_results
                    and (self._last_partial_ts is None or now_ts - self._last_partial_ts >= self._partial_interval)
                ):
                    # Use recent audio for partial transcription (last 2 seconds)
                    max_samples = int(2.0 * self.sample_rate_in) * 2  # 2 seconds of 8kHz 16-bit
                    pcm_tail = bytes(self._speech_buffer[-max_samples:]) if len(self._speech_buffer) > max_samples else bytes(self._speech_buffer)
                    self._last_partial_ts = now_ts
                    # Fire off background partial transcription
                    self._loop.create_task(self._transcribe_and_callback(pcm_tail, is_final=False))
            else:
                # Below threshold
                if self._speaking and self._last_voice_ts and (now_ts - self._last_voice_ts) >= self._silence_timeout:
                    # Speech ended
                    self._trigger_speech_end()
                elif self._speaking:
                    # Still speaking — append audio so current segment contains trailing silence
                    self._speech_buffer.extend(pcm16_8k)
        else:
            # VAD disabled → treat everything as speech
            self._speech_buffer.extend(pcm16_8k)

    async def _detect_language(self):
        """Detect language from accumulated audio buffer."""
        if not self._language_detection_buffer:
            return
        
        # Simple language detection: try each model and see which gives best confidence
        best_lang = self._current_language
        best_confidence = 0.0
        
        detection_audio = bytes(self._language_detection_buffer)
        
        for lang, recognizer in self._recognizers.items():
            if self._allowed_languages and lang not in self._allowed_languages:
                continue
                
            try:
                # Reset recognizer and try transcription
                recognizer.Reset()
                if recognizer.AcceptWaveform(detection_audio):
                    result = json.loads(recognizer.Result())
                    confidence = result.get('confidence', 0.0)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_lang = lang
            except Exception as e:
                log.debug("[Vosk] Language detection failed for %s: %s", lang, e)
        
        if best_lang != self._current_language:
            log.info("[Vosk] Language detected: %s (confidence: %.2f)", best_lang, best_confidence)
            self._current_language = best_lang
        
        self._language_locked = True
        self._language_detection_buffer.clear()

    # ------------------------------------------------------------------
    # Internal helpers --------------------------------------------------
    # ------------------------------------------------------------------

    def _trigger_speech_start(self):
        self._speaking = True
        self._speech_end_detected = False
        self._speech_buffer = bytearray()  # reset buffer for new segment
        self._language_locked = False  # allow language detection for new speech
        cb = self._callbacks.get("on_speech_start")
        if cb:
            try:
                cb()
            except Exception as e:  # pragma: no cover
                log.error("on_speech_start callback error: %s", e)

    def _trigger_speech_end(self):
        self._speaking = False
        self._speech_end_detected = True
        cb_end = self._callbacks.get("on_speech_end")
        if cb_end:
            try:
                cb_end()
            except Exception as e:  # pragma: no cover
                log.error("on_speech_end callback error: %s", e)

        # Fire asynchronous transcription task
        if self._speech_buffer:
            # Copy bytes so modifications to buffer don't affect task
            segment = bytes(self._speech_buffer)
            self._speech_buffer = bytearray()  # reset for next segment
            self._loop.create_task(self._transcribe_and_callback(segment))

    async def _transcribe_and_callback(self, pcm8k_bytes: bytes, *, is_final: bool = True):
        """Run Vosk on *pcm8k_bytes* and invoke transcript callbacks.

        If *is_final* is False, the result is treated as an *interim* (partial)
        transcript. Callbacks receive this via **on_transcript** so that the
        main application can interrupt TTS early when confidence ≥ threshold.
        """
        if not self._recognizers:
            # Models not loaded yet
            await self.connect()

        if self._current_language not in self._recognizers:
            log.warning("[Vosk] No recognizer for current language %s", self._current_language)
            return

        recognizer = self._recognizers[self._current_language]
        
        try:
            # Reset recognizer for new audio segment
            if is_final:
                recognizer.Reset()
            
            # Process audio
            if recognizer.AcceptWaveform(pcm8k_bytes):
                # Final result
                result_json = recognizer.Result()
            else:
                # Partial result
                result_json = recognizer.PartialResult()
            
            result = json.loads(result_json)
            text = result.get('text', '').strip()
            confidence = result.get('confidence', 0.8)  # Vosk doesn't always provide confidence
            
            if not text:
                return  # No transcription
            
            log.debug("[Vosk] Transcribed (%s, final=%s, conf=%.2f): '%s'", 
                     self._current_language, is_final, confidence, text)

            # Invoke transcript callback
            cb_transcript = self._callbacks.get("on_transcript")
            if cb_transcript:
                try:
                    # Call with 3 arguments: transcript, is_final, confidence
                    # This matches the enhanced callback signature from whisper_streamer
                    cb_transcript(text, is_final, confidence)
                except TypeError:
                    # Fallback to 2-argument signature for backward compatibility
                    try:
                        cb_transcript(text, is_final)
                    except Exception as e:
                        log.error("on_transcript callback error: %s", e)
                except Exception as e:
                    log.error("on_transcript callback error: %s", e)

            # Invoke utterance callback for final results
            if is_final:
                cb_utterance = self._callbacks.get("on_utterance")
                if cb_utterance:
                    try:
                        utterance_data = {
                            'text': text,
                            'confidence': confidence,
                            'language': self._current_language
                        }
                        cb_utterance(utterance_data)
                    except Exception as e:
                        log.error("on_utterance callback error: %s", e)

        except Exception as e:
            log.error("[Vosk] Transcription failed: %s", e)

    # ------------------------------------------------------------------
    # Compatibility methods
    # ------------------------------------------------------------------

    def get_vad_stats(self) -> dict:
        """Return VAD statistics for debugging."""
        return {
            "speaking": self._speaking,
            "last_voice_ts": self._last_voice_ts,
            "speech_buffer_size": len(self._speech_buffer),
            "current_language": self._current_language,
            "available_languages": list(self._models.keys()),
            "language_locked": self._language_locked
        }


def gpu_available() -> bool:
    """Check if GPU acceleration is available for Vosk."""
    # Vosk with GPU support requires specific builds
    # For now, return False as most Vosk installations are CPU-only
    # This could be enhanced to check for CUDA-enabled Vosk builds
    return False 