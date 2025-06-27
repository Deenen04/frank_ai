import asyncio
import logging
import time
import audioop
from typing import Optional, List

import numpy as np

try:
    from faster_whisper import WhisperModel
except ImportError:  # graceful fallback so the whole project can still import even if dep missing
    WhisperModel = None  # type: ignore

__all__ = ["WhisperStreamer"]

log = logging.getLogger("deepgram_streamer")  # keep original logger name for compatibility


class WhisperStreamer:  # pylint: disable=too-many-instance-attributes
    """A drop-in replacement for *DeepgramStreamer* that performs *local* real-time
    transcription using *Faster-Whisper*.

    Only μ-law @ 8 kHz audio is accepted (as emitted by Twilio).  The audio is
    converted to 16-bit PCM and up-sampled to 16 kHz before being fed into the
    Whisper model.  The public callback signatures are identical to the original
    Deepgram implementation so existing application code does **not** have to
    change.
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
        interim_results: bool = False,  # not implemented – kept for compat
        vad_events: bool = True,
        punctuate: bool = True,  # whisper already outputs punctuation
        model: str = "small",  # whisper model size to load ("tiny", "base", "small", "medium", "large")
        language: str = "multi",
        allowed_languages: Optional[List[str]] = None,
        use_amplitude_vad: bool = True,
        amplitude_threshold_db: float = -5.0,
        silence_timeout: float = 0.7,
        partial_interval: float = 0  # seconds between partial transcripts
    ):
        # --- public config ------------------------------------------------
        self.encoding = encoding.lower()
        self.sample_rate_in = sample_rate
        self._use_amp_vad = use_amplitude_vad
        self._silence_timeout = silence_timeout
        self._punctuate = punctuate
        self._interim_results = interim_results
        self._model_name = model

        # ------------------------------------------------------------------
        # Alias & validation: allow popular community model repo names
        # ------------------------------------------------------------------
        alias_map = {
            # Groq alias → CT2 optimized repo for Faster-Whisper
            "groq-whisper-large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
            # Alternate community aliases
            "whisper-large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
        }

        if self._model_name in alias_map:
            log.info("[Whisper] Resolving alias '%s' → '%s'", self._model_name, alias_map[self._model_name])
            self._model_name = alias_map[self._model_name]

        # Allow *any* HuggingFace repo path or built-in size alias.
        # Only fallback to 'small' if we detect a *size* alias that is unknown.
        size_aliases = {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "large-v3-turbo"}

        if "/" not in self._model_name and self._model_name not in size_aliases:
            log.warning("[Whisper] Unknown size alias '%s'. Falling back to 'small'.", self._model_name)
            self._model_name = "small"

        self._model: Optional[WhisperModel] = None

        # --- amplitude VAD ------------------------------------------------
        # --------------------------------------------------------------
        # Amplitude threshold (linear RMS value)
        # --------------------------------------------------------------
        # Previous versions used a *fixed* RMS of ≈2000 for μ-law which
        # often proved too insensitive – barge-in was recognised only
        # once the full utterance had already been spoken.  We now
        # honour the *amplitude_threshold_db* parameter **regardless of
        # encoding** so callers can tune the VAD sensitivity (e.g.
        # ‑30 dB for very sensitive, ‑10 dB for robust).

        # Convert dBFS → linear RMS referenced to 32768 (16-bit max).
        self._amp_threshold_linear = 32768 * (10 ** (amplitude_threshold_db / 20.0))

        # Guard against pathological values
        self._amp_threshold_linear = max(100.0, self._amp_threshold_linear)

        self._last_voice_ts: Optional[float] = None
        self._speaking = False
        self._speech_end_detected = False

        # Buffer that stores 16 kHz PCM bytes belonging to the *current* speech
        # segment.
        self._speech_buffer: bytearray = bytearray()

        # Timestamp of the last partial transcription – to throttle expensive
        # Whisper inference calls.
        self._last_partial_ts: float | None = None

        # Minimum time between partial transcriptions (seconds)
        self._partial_interval = max(0.3, partial_interval)

        # State for audioop.ratecv (for seamless resampling)
        self._ratecv_state = None

        # Async helpers ----------------------------------------------------
        self._loop = asyncio.get_event_loop()
        self._close_evt = asyncio.Event()
        self._speech_end_timer_task: Optional[asyncio.Task] = None

        # callbacks: identical keys to the Deepgram version
        self._callbacks = {
            "on_speech_start": None,
            "on_transcript": None,
            "on_speech_end": None,
            "on_utterance": None,
        }

        self._language_locked = False  # once model detects, we lock for this call
        self.language = language.lower() if language else "multi"
        # If *allowed_languages* is None, allow *all* languages.
        self._allowed_languages = None if allowed_languages is None else [l.lower() for l in allowed_languages]

        log.info(
            "[Whisper-INIT] Using faster-whisper model '%s', amp-thr %.1f dB → %.0f, silence %.1fs, allowed=%s",
            self._model_name,
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
    # Life-cycle helpers — kept to preserve external call sites
    # ------------------------------------------------------------------

    async def connect(self):
        """Instantiate Whisper on the first call. No network connections."""
        if WhisperModel is None:
            raise RuntimeError(
                "faster-whisper not installed. Add `faster-whisper` to requirements.txt and pip-install it."
            )
        if self._model is None:
            device = "cuda" if torch_available_and_gpu() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            self._model = WhisperModel(self._model_name, device=device, compute_type=compute_type)
            log.info("[Whisper] Model loaded on %s (compute=%s)", device, compute_type)

    def is_open(self) -> bool:
        return not self._close_evt.is_set()

    async def finish(self):
        if not self._close_evt.is_set():
            self._close_evt.set()
        # Nothing to close for local inference.

    # ------------------------------------------------------------------
    # Audio ingestion ---------------------------------------------------
    # ------------------------------------------------------------------

    async def send(self, audio_chunk: bytes):
        """Receive μ-law data from Twilio and feed it into the VAD / ASR stack."""

        # 1) μ-law → 16-bit PCM -----------------------------------------
        try:
            pcm16 = audioop.ulaw2lin(audio_chunk, 2)
        except audioop.error as e:  # pragma: no cover — rare encode errors
            log.debug("audioop.ulaw2lin failed: %s", e)
            return

        # 2) Up-sample 8 kHz → 16 kHz so Whisper is happy
        try:
            pcm16_16k, self._ratecv_state = audioop.ratecv(
                pcm16,
                2,  # width in bytes
                1,  # mono
                8000,
                16000,
                self._ratecv_state,
            )
        except audioop.error as e:
            log.debug("audioop.ratecv failed: %s", e)
            return

        # 3) Amplitude VAD ---------------------------------------------
        if self._use_amp_vad:
            rms = audioop.rms(pcm16, 2)
            now_ts = time.time()

            # rms debugging each 2 s
            if not hasattr(self, "_last_rms_log_ts") or now_ts - self._last_rms_log_ts > 2.0:
                log.debug("[VAD] rms %.0f (thr %.0f) speaking=%s", rms, self._amp_threshold_linear, self._speaking)
                self._last_rms_log_ts = now_ts

            if rms > self._amp_threshold_linear:
                # voice *present*
                self._last_voice_ts = now_ts
                if not self._speaking:
                    self._trigger_speech_start()
                self._speech_buffer.extend(pcm16_16k)

                # ----------------------------------------------------------
                # Emit *partial* transcription every self._partial_interval s
                # ----------------------------------------------------------
                if (
                    self._interim_results  # user requested interim results
                    and (self._last_partial_ts is None or now_ts - self._last_partial_ts >= self._partial_interval)
                ):
                    # Copy only the **last** 2.5 seconds of audio to keep latency
                    # low and inference fast (2.5s * 16000 * 2 ≈ 80kB)
                    max_samples = int(2.5 * 16000) * 2  # bytes
                    pcm_tail = bytes(self._speech_buffer[-max_samples:]) if len(self._speech_buffer) > max_samples else bytes(self._speech_buffer)
                    self._last_partial_ts = now_ts
                    # Fire off background partial transcription
                    self._loop.create_task(self._transcribe_and_callback(pcm_tail, is_final=False))
            else:
                # below threshold
                if self._speaking and self._last_voice_ts and (now_ts - self._last_voice_ts) >= self._silence_timeout:
                    # speech ended
                    self._trigger_speech_end()
                elif self._speaking:
                    # still speaking — append audio so current segment contains little trailing silence
                    self._speech_buffer.extend(pcm16_16k)
        else:
            # VAD disabled → treat everything as speech
            self._speech_buffer.extend(pcm16_16k)

    # ------------------------------------------------------------------
    # Internal helpers --------------------------------------------------
    # ------------------------------------------------------------------

    def _trigger_speech_start(self):
        self._speaking = True
        self._speech_end_detected = False
        self._speech_buffer = bytearray()  # reset buffer for new segment
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

        # fire asynchronous transcription task
        if self._speech_buffer:
            # copy bytes so modifications to buffer don't affect task
            segment = bytes(self._speech_buffer)
            self._speech_buffer = bytearray()  # reset for next segment
            self._loop.create_task(self._transcribe_and_callback(segment))

    async def _transcribe_and_callback(self, pcm16k_bytes: bytes, *, is_final: bool = True):
        """Run Whisper on *pcm16k_bytes* and invoke transcript callbacks.

        If *is_final* is False, the result is treated as an *interim* (partial)
        transcript.  Callbacks receive this via **on_transcript** so that the
        main application can interrupt TTS early when confidence ≥ threshold.
        """
        if self._model is None:
            # model was not loaded (connect() not awaited?) — best effort load now
            await self.connect()

        # convert bytes → numpy float32  (Whisper expects float32 in  range)
        audio_int16 = np.frombuffer(pcm16k_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype("float32") / 32768.0

        # ------------------------------------------------------------------
        # Language selection / restriction
        # ------------------------------------------------------------------
        whisper_language: Optional[str]
        if self.language == "multi":
            # Use model's language detection on the segment.
            try:
                dl_ret = self._model.detect_language(audio_float32)
                # faster-whisper may return either (lang, probs) or just probs
                if isinstance(dl_ret, tuple):
                    # (detected_lang, prob) OR (prob_dict, something)
                    if isinstance(dl_ret[0], str):
                        whisper_language = dl_ret[0]
                    elif isinstance(dl_ret[0], dict):
                        whisper_language = max(dl_ret[0], key=dl_ret[0].get)
                    else:
                        whisper_language = None
                elif isinstance(dl_ret, dict):
                    whisper_language = max(dl_ret, key=dl_ret.get)
                else:
                    whisper_language = None
            except Exception as e:  # pragma: no cover
                log.error("Language detection failed: %s", e)
                whisper_language = None

            if self._allowed_languages and (whisper_language is None or whisper_language not in self._allowed_languages):
                log.debug("[Whisper] Skipping segment – detected lang '%s' not in %s", whisper_language, self._allowed_languages)
                return  # ignore this segment entirely
        else:
            whisper_language = self.language

        try:
            segments_iter, _info = self._model.transcribe(
                audio_float32,
                language=whisper_language,
                beam_size=5,
                word_timestamps=False,
                temperature=0.0,  # reduce randomness for higher precision
                condition_on_previous_text=False,  # avoid bias from prior segments
            )
            segments = list(segments_iter)  # materialise for repeated use
            text_parts: List[str] = [seg.text.strip() for seg in segments]
            transcript = " ".join(text_parts).strip()
        except Exception as e:  # pragma: no cover
            log.error("Whisper transcription failed: %s", e, exc_info=True)
            transcript = ""

        if transcript:
            # --------------------------------------------------------------
            # Compute a simple confidence metric for the utterance.
            # --------------------------------------------------------------
            try:
                if segments:
                    probs = [np.exp(s.avg_logprob) for s in segments if s.avg_logprob is not None]
                    confidence = float(sum(probs) / len(probs)) * 100.0 if probs else 0.0
                else:
                    confidence = 0.0
            except Exception as _:
                confidence = 0.0

            log.debug("[Whisper] Utterance confidence: %.1f%% — '%s'", confidence, transcript)

            # --------------------------------------------------------------
            # on_transcript  (interim / final)  — forwards confidence when available
            # --------------------------------------------------------------
            cb_tr = self._callbacks.get("on_transcript")
            if cb_tr:
                try:
                    # Pass *is_final* flag as second positional arg for b/c
                    # with DeepgramStreamer signature.  We additionally append
                    # *confidence* as third arg so advanced handlers can use it.
                    cb_tr(transcript, is_final, confidence)
                except Exception as e:
                    log.error("on_transcript callback error: %s", e)

            # on_utterance --------------------------------------------------------------
            cb_utt = self._callbacks.get("on_utterance")
            if cb_utt and is_final:
                try:
                    # Forward confidence as second positional argument and detected language as third
                    cb_utt(transcript, confidence, whisper_language)
                except Exception as e:
                    log.error("on_utterance callback error: %s", e)

    # ------------------------------------------------------------------
    # Misc helpers ------------------------------------------------------
    # ------------------------------------------------------------------

    def get_vad_stats(self) -> dict:
        return {
            "is_speaking": self._speaking,
            "threshold_linear": self._amp_threshold_linear,
            "last_voice_timestamp": self._last_voice_ts,
            "silence_timeout": self._silence_timeout,
            "use_amplitude_vad": self._use_amp_vad,
            "speech_end_detected": self._speech_end_detected,
        }


# -----------------------------------------------------------------------
# small utility ----------------------------------------------------------
# -----------------------------------------------------------------------

def torch_available_and_gpu() -> bool:
    """Return *True* if *torch* is importable **and** a CUDA device is available."""
    try:
        import torch  # pylint: disable=import-error

        return torch.cuda.is_available()
    except Exception:  # pragma: no cover
        return False 