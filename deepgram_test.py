import asyncio
import json
import time  # Used for amplitude-VAD timing
import websockets
import logging  # Add logging
import audioop  # For simple audio amplitude / VAD detection
from typing import Optional, List

# Configure root logger for terminal output
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("deepgram_streamer")  # Create a logger


class DeepgramStreamer:
    """
    Asynchronous Deepgram live transcription via WebSocket with amplitude-based VAD events only.

    Events:
      on_speech_start() -> called once when user begins speaking
      on_transcript(transcript: str, is_final: bool) -> interim/final transcripts
      on_speech_end() -> called once when user stops (amp-VAD)
      on_utterance(transcript: str) -> when Deepgram signals speech_final (complete utterance)

    Usage:
      streamer = DeepgramStreamer(api_key, encoding, sample_rate)
      streamer.on('on_speech_start', callback)
      streamer.on('on_transcript', callback)
      streamer.on('on_speech_end', callback)
      streamer.on('on_utterance', callback)
      await streamer.connect()
      await streamer.send(audio_bytes)
      await streamer.finish()
    """
    def __init__(
        self,
        api_key: str,
        encoding: str = 'mulaw',
        sample_rate: int = 8000,
        interim_results: bool = True,
        vad_events: bool = True,     # retained for compatibility, unused
        punctuate: bool = True,
        profanity_filter: bool = True,
        model: str = 'nova-3',  # Default to nova-3 as requested
        language: str = 'multi',  # 'multi' enables all languages
        # --- Local amplitude-based VAD parameters ---
        use_amplitude_vad: bool = True,
        amplitude_threshold_db: float = -10.0,  # Much more sensitive threshold for speech detection
        silence_timeout: float = 2,  # Increased timeout for better speech end detection
    ):
        self.api_key = api_key
        self.encoding = encoding
        self.sample_rate = sample_rate
        self.interim_results = interim_results
        self.punctuate = punctuate
        self.profanity_filter = profanity_filter
        self.model = model
        self.language = language

        # Amplitude-VAD settings
        self._use_amp_vad = use_amplitude_vad
        # Based on real testing, human speech RMS is typically above 1000-5000
        # Setting threshold to 500 to catch quiet speech but avoid background noise
        if encoding.lower() in ['mulaw', 'ulaw']:
            # For μ-law, use a direct RMS threshold based on observed speech levels
            self._amp_threshold_linear = 500.0  # Realistic threshold for human speech
        else:
            # For linear PCM, convert dB to linear (keeping old calculation as fallback)
            self._amp_threshold_linear = 32768 * (10 ** (amplitude_threshold_db / 20.0))
        
        self._silence_timeout = silence_timeout
        self._last_voice_ts: Optional[float] = None

        # Transcript accumulation for handling partial results
        self._current_transcript_parts: List[str] = []
        self._last_transcript_time: Optional[float] = None
        self._speech_end_detected = False
        self._speech_end_timer_task: Optional[asyncio.Task] = None
        self._deepgram_response_timeout = 0.5  # Wait 3 seconds after speech end for Deepgram response

        log.info(f"[VAD-INIT] Amplitude threshold: {amplitude_threshold_db} dB -> linear: {self._amp_threshold_linear:.2f}")
        log.info(f"[VAD-INIT] Silence timeout: {silence_timeout}s, Encoding: {encoding}")
        log.info(f"[VAD-INIT] Deepgram response timeout: {self._deepgram_response_timeout}s")

        # Build Deepgram WebSocket URL following requested parameter set.
        self.url = (
            "wss://api.deepgram.com/v1/listen?"
            f"model={self.model}"
            f"&language={self.language}"
            f"&encoding={self.encoding}"
            f"&sample_rate={self.sample_rate}"
            "&smart_format=true"
            f"&punctuate={str(self.punctuate).lower()}"
            f"&profanity_filter={str(self.profanity_filter).lower()}"
            f"&interim_results={str(self.interim_results).lower()}"
            "&utterances=true"
            # Note: Deepgram endpointing is left on but local VAD controls start/end
            "&endpointing=1000"
        )
        self._ws = None
        self._close_evt = asyncio.Event()
        self._speaking = False
        self._callbacks = {
            'on_speech_start': None,
            'on_transcript': None,
            'on_speech_end': None,
            'on_utterance': None,
        }

    def on(self, event: str, callback):
        if event not in self._callbacks:
            raise ValueError(f"Unknown event '{event}'")
        self._callbacks[event] = callback

    def set_vad_threshold(self, amplitude_threshold_db: float):
        """Adjust the VAD threshold at runtime for debugging purposes."""
        if self.encoding.lower() in ['mulaw', 'ulaw']:
            # For μ-law, treat the dB value as a direct RMS threshold multiplier
            # E.g., -10 dB -> 1000 RMS, -5 dB -> 500 RMS, 0 dB -> 100 RMS
            if amplitude_threshold_db <= -10:
                self._amp_threshold_linear = 1000.0
            elif amplitude_threshold_db <= -5:
                self._amp_threshold_linear = 500.0
            elif amplitude_threshold_db <= 0:
                self._amp_threshold_linear = 200.0
            else:
                self._amp_threshold_linear = 100.0
        else:
            self._amp_threshold_linear = 32768 * (10 ** (amplitude_threshold_db / 20.0))
        
        log.info(f"[VAD-ADJUST] New threshold: {amplitude_threshold_db} dB -> linear: {self._amp_threshold_linear:.2f}")

    def set_rms_threshold(self, rms_threshold: float):
        """Directly set the RMS threshold for more intuitive control."""
        self._amp_threshold_linear = rms_threshold
        log.info(f"[VAD-ADJUST] New RMS threshold: {self._amp_threshold_linear:.2f}")

    def get_vad_stats(self) -> dict:
        """Get current VAD statistics for debugging."""
        return {
            "is_speaking": self._speaking,
            "threshold_linear": self._amp_threshold_linear,
            "last_voice_timestamp": self._last_voice_ts,
            "silence_timeout": self._silence_timeout,
            "use_amplitude_vad": self._use_amp_vad,
            "speech_end_detected": self._speech_end_detected,
            "current_transcript_parts": len(self._current_transcript_parts),
            "accumulated_transcript": " ".join(self._current_transcript_parts).strip(),
            "deepgram_response_timeout": self._deepgram_response_timeout
        }

    def is_open(self) -> bool:
        if not self._ws:
            return False
        closed_attr = getattr(self._ws, "closed", None)
        try:
            if closed_attr is None:
                close_code = getattr(self._ws, "close_code", None)
                return close_code is None
            if isinstance(closed_attr, asyncio.Future):
                return not closed_attr.done()
            return not bool(closed_attr)
        except Exception:
            return False

    async def connect(self):
        headers = {'Authorization': f'Token {self.api_key}'}
        try:
            self._ws = await websockets.connect(self.url, additional_headers=headers)
            asyncio.create_task(self._receiver())
            log.info("[DG] WebSocket connection established.")
        except Exception as e:
            log.error(f"[DG] Failed to connect to WebSocket: {e}")
            self._ws = None
            self._close_evt.set()
            raise

    async def send(self, audio_chunk: bytes):
        # --- Local amplitude-based VAD ---
        if self._use_amp_vad:
            try:
                linear_pcm = audioop.ulaw2lin(audio_chunk, 2)
                rms = audioop.rms(linear_pcm, 2)
                now_ts = time.time()

                # Log RMS values periodically for debugging
                if hasattr(self, '_last_rms_log_ts'):
                    if now_ts - self._last_rms_log_ts > 2.0:  # Log every 2 seconds
                        log.debug(f"[VAD-DEBUG] RMS: {rms:.2f}, Threshold: {self._amp_threshold_linear:.2f}, Speaking: {self._speaking}")
                        self._last_rms_log_ts = now_ts
                else:
                    self._last_rms_log_ts = now_ts
                    log.debug(f"[VAD-DEBUG] RMS: {rms:.2f}, Threshold: {self._amp_threshold_linear:.2f}, Speaking: {self._speaking}")

                if rms > self._amp_threshold_linear:
                    self._last_voice_ts = now_ts
                    if not self._speaking:
                        self._speaking = True
                        log.info(f"[VAD] User started speaking. RMS: {rms:.2f} > Threshold: {self._amp_threshold_linear:.2f}")
                        
                        # Reset transcript accumulation for new speech event
                        self._reset_transcript_accumulation()
                        
                        cb = self._callbacks.get('on_speech_start')
                        if cb:
                            try:
                                cb()
                            except Exception as e:
                                log.error(f"Error in on_speech_start callback (amp-VAD): {e}")
                else:
                    if self._speaking and self._last_voice_ts and (now_ts - self._last_voice_ts) >= self._silence_timeout:
                        self._speaking = False
                        self._speech_end_detected = True
                        log.info(f"[VAD] User stopped speaking. Silence duration: {now_ts - self._last_voice_ts:.2f}s")
                        
                        # Check if we already have transcript - if yes, send immediately
                        accumulated = " ".join(self._current_transcript_parts).strip()
                        if not accumulated and getattr(self, "_last_interim_text", ""):
                            accumulated = self._last_interim_text         # promote interim
                        if accumulated:
                            log.info(f"[VAD-IMMEDIATE] Sending transcript immediately: '{accumulated}'")
                            # Trigger the utterance callback immediately
                            cb = self._callbacks.get('on_utterance')
                            if cb:
                                try:
                                    cb(accumulated)
                                except Exception as e:
                                    log.error(f"Error in on_utterance callback: {e}")
                            # Reset for next speech event
                            self._reset_transcript_accumulation()
                        else:
                            # No transcript yet, wait for Deepgram
                            log.info(f"[VAD] No transcript yet. Waiting {self._deepgram_response_timeout}s for Deepgram...")
                            self._speech_end_timer_task = asyncio.create_task(self._delayed_speech_end_handler())
                        
                        # Still trigger immediate speech end callback for interruption purposes
                        cb = self._callbacks.get('on_speech_end')
                        if cb:
                            try:
                                cb()
                            except Exception as e:
                                log.error(f"Error in on_speech_end callback (amp-VAD): {e}")
            except Exception as e:
                log.debug(f"[DG] Amplitude VAD error: {e}")

        if self.is_open():
            try:
                await self._ws.send(audio_chunk)
            except websockets.exceptions.ConnectionClosed as e:
                log.warning(f"[DG] Tried to send on a closed/closing connection: {e}")
                await self.finish()
            except Exception as e:
                log.error(f"[DG] Error sending audio chunk: {e}")
                await self.finish()

    async def finish(self):
        if self.is_open():
            log.info("[DG] Sending CloseStream message.")
            try:
                await self._ws.send(json.dumps({'type': 'CloseStream'}))
            except websockets.exceptions.ConnectionClosed:
                log.warning("[DG] Connection already closed when trying to send CloseStream.")
            except Exception as e:
                log.error(f"[DG] Error sending CloseStream: {e}")
        
        # Cancel any pending speech end timer
        if self._speech_end_timer_task and not self._speech_end_timer_task.done():
            log.debug("[DG] Cancelling pending speech end timer")
            self._speech_end_timer_task.cancel()
        
        if self._ws:
            log.info("[DG] Closing WebSocket connection.")
            try:
                await self._ws.close()
            except websockets.exceptions.ConnectionClosed:
                log.info("[DG] Connection was already closed when trying to close.")
            except Exception as e:
                log.error(f"[DG] Error closing Deepgram websocket: {e}")
            self._ws = None
        if not self._close_evt.is_set():
            self._close_evt.set()

    async def _receiver(self):
        if not self._ws:
            log.warning("[DG_RECEIVER] WebSocket not initialized, receiver cannot start.")
            if not self._close_evt.is_set():
                self._close_evt.set()
            return

        try:
            async for raw in self._ws:
                data = json.loads(raw)
                if data.get('type') == 'Results':
                    channel = data.get('channel', {})
                    alternatives = channel.get('alternatives', [])
                    if not alternatives:
                        continue
                    alt = alternatives[0]
                    transcript = alt.get('transcript', '').strip()

                    if transcript:
                        # Add transcript to accumulation instead of immediate callback
                        self._add_transcript_part(transcript, bool(data.get('is_final')))
                        
                        # Still trigger the transcript callback for live display
                        cb = self._callbacks.get('on_transcript')
                        if cb:
                            try:
                                cb(transcript, bool(data.get('is_final')))
                            except Exception as e:
                                log.error(f"Error in on_transcript callback: {e}")

        except websockets.exceptions.ConnectionClosed as e:
            log.warning(f"[DG_RECEIVER] Connection closed: {e.code} {e.reason}")
        except Exception as e:
            log.error(f"[DG_RECEIVER] Unexpected error: {e}", exc_info=True)
        finally:
            log.info("[DG_RECEIVER] Receiver loop finished.")
            if self._speaking:
                self._speaking = False
                log.info("[VAD] User stopped speaking.")
                cb = self._callbacks.get('on_speech_end')
                if cb:
                    try:
                        cb()
                    except Exception as e:
                        log.error(f"Error in on_speech_end callback (receiver finally): {e}")
            
            # Cancel any pending speech end timer
            if self._speech_end_timer_task and not self._speech_end_timer_task.done():
                log.debug("[DG_RECEIVER] Cancelling pending speech end timer")
                self._speech_end_timer_task.cancel()
            
            if not self._close_evt.is_set():
                self._close_evt.set()

    def _reset_transcript_accumulation(self):
        """Reset transcript accumulation for a new speech event."""
        self._current_transcript_parts = []
        self._last_transcript_time = None
        self._speech_end_detected = False
        if self._speech_end_timer_task and not self._speech_end_timer_task.done():
            self._speech_end_timer_task.cancel()
        self._speech_end_timer_task = None

    def _add_transcript_part(self, transcript: str, is_final: bool = False):
        if not transcript.strip():
            return
        self._last_transcript_time = time.time()

        # Append unique transcript segments so we don't lose the beginning
        if not self._current_transcript_parts or transcript not in self._current_transcript_parts[-1]:
            self._current_transcript_parts.append(transcript)
        else:
            # Overwrite the last part if Deepgram sent a revised interim/final version
            self._current_transcript_parts[-1] = transcript
        if is_final:
            log.debug(f"[TRANSCRIPT-FINAL] '{transcript}'")
        else:
            self._last_interim_text = transcript      # NEW

    async def _delayed_speech_end_handler(self):
        """Wait for Deepgram response after VAD detects speech end."""
        await asyncio.sleep(self._deepgram_response_timeout)
        
        # Check if we received any meaningful transcript during the wait
        accumulated = " ".join(self._current_transcript_parts).strip()
        if not accumulated and getattr(self, "_last_interim_text", ""):
            accumulated = self._last_interim_text         # promote interim
        
        if accumulated:
            log.info(f"[VAD-CONFIRMED] Speech end confirmed with transcript: '{accumulated}'")
            # Trigger the utterance callback with accumulated transcript
            cb = self._callbacks.get('on_utterance')
            if cb:
                try:
                    cb(accumulated)
                except Exception as e:
                    log.error(f"Error in on_utterance callback: {e}")
        else:
            log.info("[VAD-DISCARDED] Speech end detected but no transcript received - likely background noise")
        
        # Reset for next speech event
        self._reset_transcript_accumulation()
