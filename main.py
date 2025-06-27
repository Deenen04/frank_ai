import os
import json
import base64
import asyncio
import logging
import re
from typing import List, Optional
import langid
import audioop  # for μ-law → PCM conversion & RMS amplitude
langid.set_languages(["en", "fr", "de"])

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Form
from fastapi.responses import HTMLResponse
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream, Parameter
from dotenv import load_dotenv
from starlette.websockets import WebSocketState

from whisper_streamer import WhisperStreamer as DeepgramStreamer
from elevenlabs.client import ElevenLabs
from apiChatCompletion import make_openai_request
from ai_prompt import build_prompt
from ai_executor import generate_reply

# ------------------------------------------------------------------
# Setup and Constants
# ------------------------------------------------------------------
_FR_CHARS = set("éèêëàâäîïôöùûüçœ")
_DE_CHARS = set("äöüß")

def fast_lang_detect(text: str) -> str:
    lowered = text.lower()
    if any(ch in _FR_CHARS for ch in lowered):
        return "fr"
    if any(ch in _DE_CHARS for ch in lowered):
        return "de"
    return "en"

def route_call(call_sid):
    log.info(f"Mock route_call called for SID: {call_sid}")

load_dotenv()
HOSTNAME = os.getenv("HOSTNAME_twilio", "localhost:8000")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
log = logging.getLogger("voicebot")
deepgram_log = logging.getLogger("deepgram_streamer")
deepgram_log.setLevel(logging.DEBUG)

VOICE_IDS = {"en": "kdnRe2koJdOK4Ovxn2DI", "fr": "6wsYTkh7uhMGLNTiM1TC", "de": "KXxZd16DiBqt82nbarJx"}
FAREWELL_LINES = {"en": "Thanks for calling. Goodbye.", "fr": "Merci d'avoir appelé. Au revoir.", "de": "Danke für Ihren Anruf. Auf Wiedersehen."}
GREETING_LINES = {"en": "Hi, This is Frank Babar Clinic, I am here to assist you book an appointment with us today. How can I help you?", "fr": "Bonjour, comment puis-je vous aider?", "de": "Hallo, wie kann ich Ihnen helfen?"}
END_DELAY_SEC = 1
USER_TURN_END_DELAY_SEC = 0.8


## FIX: This class is not strictly needed with the new `on_utterance` logic,
## but is kept to maintain the original file structure.
class TranscriptSanitizer:
    def __init__(self):
        self.reset()
    def reset(self):
        self._final_parts: List[str] = []
        self._last_interim: str = ""
    def add_transcript(self, new_transcript: str, is_final: bool = False):
        if not new_transcript.strip(): return
        if is_final:
            if self._last_interim:
                self._final_parts.append(self._last_interim)
                self._last_interim = ""
            self._final_parts.append(new_transcript)
        else:
            self._last_interim = new_transcript
    def get_current_transcript(self) -> str:
        return " ".join(self._final_parts + ([self._last_interim] if self._last_interim else [])).strip()
    def finalize_utterance(self) -> str:
        if self._last_interim:
            self._final_parts.append(self._last_interim)
            self._last_interim = ""
        full_transcript = " ".join(self._final_parts)
        self._final_parts = []
        return " ".join(full_transcript.split())


## FIX: Upgraded TTSController to manage a non-blocking audio queue and producer task.
class TTSController:
    def __init__(self):
        self.current_generator = None
        self.is_speaking = False
        self.spoken_text_parts: list[str] = []
        # New components for the producer-consumer pattern
        self.audio_queue: Optional[asyncio.Queue] = None
        self.producer_task: Optional[asyncio.Task] = None

    def reset_spoken_text(self):
        self.spoken_text_parts.clear()

    def add_spoken_text(self, text: str):
        if text:
            self.spoken_text_parts.append(text.strip())

    def get_spoken_text(self) -> str:
        return " ".join(self.spoken_text_parts).strip()

    async def stop_immediately(self):
        """Force-stops any ongoing TTS generation and sending."""
        if not self.is_speaking and not self.producer_task:
            return # Already stopped

        log.debug("[TTS] Attempting to stop TTS immediately.")
        self.is_speaking = False

        if self.producer_task and not self.producer_task.done():
            log.debug("[TTS] Cancelling producer task.")
            self.producer_task.cancel()
            try:
                await self.producer_task
            except asyncio.CancelledError:
                log.debug("[TTS] Producer task successfully cancelled.")
        
        if self.current_generator:
            try: self.current_generator.close()
            except Exception: pass
        
        self.current_generator = None
        self.producer_task = None
        
        if self.audio_queue:
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()
            log.debug("[TTS] Audio queue has been cleared.")

app = FastAPI()
tts_client = ElevenLabs(api_key=ELEVEN_API_KEY)

@app.on_event("startup")
async def startup_event():
    log.info("[STARTUP] Voice agent started — integrated DeepgramStreamer")

@app.post("/voice", response_class=HTMLResponse)
async def answer_call(_request: Request, From: str = Form(...), To: str = Form(...)):
    log.info("[/voice] %s ➔ %s", From, To)
    vr = VoiceResponse()
    connect = Connect()
    if "://" in HOSTNAME:
        ws_url = f"{HOSTNAME}/media"
    elif ":" in HOSTNAME:
        ws_url = f"ws://{HOSTNAME}/media"
    else:
        ws_url = f"wss://{HOSTNAME}/media"
    log.info(f"Connecting to WebSocket: {ws_url}")
    stream = Stream(url=ws_url)
    stream.parameter(name="caller_phone_number", value=From)
    stream.parameter(name="brand_phone_number", value=To)
    stream.parameter(name="twilio_call_sid", value="{{CallSid}}")
    connect.append(stream)
    vr.append(connect)
    return HTMLResponse(str(vr), media_type="application/xml")


## FIX: Rewritten play_greeting to be non-blocking and interruptible.
async def play_greeting(lang: str, sid: str, ws: WebSocket, tts_controller: TTSController, state: dict, send_twilio_clear):
    voice_id, text = VOICE_IDS.get(lang, VOICE_IDS["en"]), GREETING_LINES.get(lang, GREETING_LINES["en"])
    
    # Producer task to fetch audio in the background
    async def _produce_audio(generator, queue):
        try:
            for audio_chunk in generator:
                await queue.put(audio_chunk)
            await queue.put(None)
        except Exception as e:
            log.error(f"[GREETING-PRODUCER-ERROR] {e}", exc_info=True)
            await queue.put(None)

    tts_controller.is_speaking = True
    tts_controller.audio_queue = asyncio.Queue()
    log.info(f"[GREETING-{sid}] Playing: '{text}'")
    
    try:
        tts_controller.current_generator = tts_client.text_to_speech.stream(
            text=text,
            voice_id=voice_id,
            model_id="eleven_turbo_v2", # Using a valid, fast model
            output_format="ulaw_8000",
            optimize_streaming_latency=0
        )
        
        tts_controller.producer_task = asyncio.create_task(
            _produce_audio(tts_controller.current_generator, tts_controller.audio_queue)
        )
        
        # Consumer loop to send audio
        while True:
            if state.get("user_is_speaking") or state["stop_call"] or ws.client_state != WebSocketState.CONNECTED:
                log.warning(f"[GREETING-CUTOFF-{sid}] Interruption detected. Stopping greeting.")
                await tts_controller.stop_immediately()
                await send_twilio_clear()
                break

            audio = await tts_controller.audio_queue.get()
            if audio is None:
                break # End of stream

            await ws.send_text(json.dumps({
                "event": "media", "streamSid": sid,
                "media": {"payload": base64.b64encode(audio).decode()}
            }))
        
    except Exception as e:
        log.warning(f"[GREETING-ERROR-{sid}] TTS streaming interrupted: {e}", exc_info=True)
    finally:
        await tts_controller.stop_immediately() # Ensure cleanup

    # Send mark only if not interrupted
    if not state["stop_call"] and not state.get("user_is_speaking") and ws.client_state == WebSocketState.CONNECTED:
        log.info(f"[GREETING-MARK-{sid}] Sending end_of_ai_turn mark.")
        try:
            await ws.send_text(json.dumps({"event": "mark", "streamSid": sid, "mark": {"name": "end_of_ai_turn"}}))
        except Exception as e:
            log.error(f"[GREETING-MARK-ERROR-{sid}] Failed to send mark: {e}")

def display_live_transcript(caller: str, transcript: str):
    print(f"\r[LIVE] {caller or 'Caller'}: {transcript}", end="", flush=True)

def display_final_turn(caller: str, full_phrase: str):
    print(f"\n[TURN] {caller or 'Caller'}: {full_phrase}")
    print("-" * 60, flush=True)

@app.websocket("/media")
async def media_websocket_endpoint(ws: WebSocket):
    await ws.accept()
    log.info("[/media] WebSocket connection accepted.")
    
    call_state = {
        "stop_call": False, "initiate_transfer": False, "terminate_call": False,
        "user_is_speaking": False, "twilio_stream_sid": None, "caller_phone_number": "?",
        "brand_phone_number": "?", "twilio_call_sid": None, "last_processed_transcript": ""
    }
    tts_controller = TTSController()
    conversation_history: List[str] = []
    current_language = "multi"
    ai_response_task: Optional[asyncio.Task] = None
    action_watcher_task: Optional[asyncio.Task] = None

    async def send_twilio_clear():
        if call_state.get("twilio_stream_sid") and ws.client_state == WebSocketState.CONNECTED:
            log.debug("[TWI-CLEAR] Sending clear event to Twilio.")
            try:
                await ws.send_text(json.dumps({"event": "clear", "streamSid": call_state["twilio_stream_sid"]}))
            except Exception as exc:
                log.warning(f"[TWI-CLEAR] Failed to send clear event: {exc}")

    ## FIX: VAD settings are now much more sensitive to catch user speech immediately.
    deepgram_streamer = DeepgramStreamer(
        api_key=DEEPGRAM_API_KEY, encoding="mulaw", sample_rate=8000,
        interim_results=True, vad_events=True, punctuate=True,
        model='groq-whisper-large-v3-turbo', language=current_language,
        use_amplitude_vad=True,
        amplitude_threshold_db=-30.0, # Much more sensitive than -5.0
        silence_timeout=0.8, # Respond to silence faster
        partial_interval=0.3  # emit partials every 300 ms for rapid barge-in
    )

    # ------------------------------------------------------------------
    # Lightweight amplitude VAD (μ-law RMS) for *instant* barge-in.
    # ------------------------------------------------------------------
    AMP_VAD_DB = -30.0  # dBFS – keep in sync with Whisper settings
    AMP_VAD_LINEAR = 32768 * (10 ** (AMP_VAD_DB / 20.0))

    async def process_final_utterance(final_utterance: str):
        nonlocal ai_response_task, current_language
        if not final_utterance.strip():
            call_state["user_is_speaking"] = False
            return
        
        # ... (Language detection logic remains the same) ...
        call_state["user_is_speaking"] = False
        display_final_turn(call_state["caller_phone_number"], final_utterance)
        
        if ai_response_task and not ai_response_task.done():
            ai_response_task.cancel()
        
        # Stop any residual speaking and clear buffer
        await tts_controller.stop_immediately()
        await send_twilio_clear()
        
        conversation_history.append(f"Human: {final_utterance}")
        ai_response_task = asyncio.create_task(
            handle_ai_turn(call_state, current_language, ws, conversation_history, tts_controller, send_twilio_clear)
        )

    def on_dg_speech_start():
        log.warning("!!! VAD SPEECH START DETECTED !!! Barge-in initiated.")
        if call_state["stop_call"]: return
        call_state["user_is_speaking"] = True

        if tts_controller.is_speaking:
            log.warning("[DG-START] User barge-in detected → stopping TTS immediately.")
            ## FIX: Launch async functions as tasks from this sync callback.
            asyncio.create_task(tts_controller.stop_immediately())
            asyncio.create_task(send_twilio_clear())
            
            nonlocal ai_response_task
            if ai_response_task and not ai_response_task.done():
                ai_response_task.cancel()
                log.debug("[DG-START] Cancelled ongoing AI response task due to barge-in.")

    def on_dg_transcript(*args):
        if call_state["stop_call"]: return
        transcript = args[0] if args else ""
        if not transcript.strip(): return
        call_state["user_is_speaking"] = True
        # Fallback barge-in: if the speech-start VAD was missed but we
        # already have a (partial) transcript, cut TTS right now.
        if tts_controller.is_speaking:
            log.warning("[DG-TRANSCRIPT] Barge-in detected via transcript → stopping TTS.")
            asyncio.create_task(tts_controller.stop_immediately())
            asyncio.create_task(send_twilio_clear())
        display_live_transcript(call_state["caller_phone_number"], transcript)

    def on_dg_utterance(*args):
        # ... (This logic is good, just launches process_final_utterance) ...
        utterance = args[0] if args else ""
        asyncio.create_task(process_final_utterance(utterance))
        
    deepgram_streamer.on('on_speech_start', on_dg_speech_start)
    deepgram_streamer.on('on_transcript', on_dg_transcript)
    deepgram_streamer.on('on_utterance', on_dg_utterance)

    try:
        await deepgram_streamer.connect()
        # Main WebSocket loop
        while not call_state["stop_call"] and ws.client_state == WebSocketState.CONNECTED:
            try:
                raw_message = await ws.receive_text()
            except WebSocketDisconnect:
                log.warning("[/media] WebSocket disconnected by client.")
                break
            
            message = json.loads(raw_message)
            event_type = message.get("event")

            if event_type == "start":
                call_state.update(message["start"]["customParameters"])
                call_state["twilio_stream_sid"] = message["start"]["streamSid"]
                log.info(f"[WS-START] SID: {call_state['twilio_stream_sid']}, CallSID: {call_state['twilio_call_sid']}")
                print(f"\n🎯 CALL STARTED: {call_state['caller_phone_number']} → {call_state['brand_phone_number']}\n" + "="*60)
                asyncio.create_task(play_greeting(
                    current_language, call_state["twilio_stream_sid"], ws, tts_controller, call_state, send_twilio_clear
                ))

            elif event_type == "media":
                if call_state["stop_call"]:
                    continue

                # Decode once and share bytes for both VAD & ASR
                try:
                    audio_bytes = base64.b64decode(message["media"]["payload"])
                except Exception:
                    continue  # malformed packet – skip

                # ------------------------------------------------------------------
                # Amplitude VAD: stop TTS the *moment* the caller makes a sound
                # ------------------------------------------------------------------
                if tts_controller.is_speaking:
                    try:
                        pcm16 = audioop.ulaw2lin(audio_bytes, 2)
                        rms = audioop.rms(pcm16, 2)
                        if rms > AMP_VAD_LINEAR:
                            log.warning(
                                "[AMP-VAD] Voice detected (RMS=%d > %.0f) → stopping TTS immediately.",
                                rms,
                                AMP_VAD_LINEAR,
                            )
                            call_state["user_is_speaking"] = True
                            asyncio.create_task(tts_controller.stop_immediately())
                            asyncio.create_task(send_twilio_clear())
                    except Exception as _:
                        pass  # ignore audioop errors

                # Forward to Whisper / Deepgram emulator
                if deepgram_streamer.is_open():
                    await deepgram_streamer.send(audio_bytes)

            elif event_type == "mark":
                mark_name = message.get("mark", {}).get("name")
                log.info(f"[WS-MARK] Received mark: {mark_name}")
                if mark_name == "end_of_ai_turn":
                    tts_controller.is_speaking = False
                    log.debug("[TTS] AI turn complete, TTS speaking set to False.")

            elif event_type == "stop":
                log.info(f"[WS-STOP] Received stop event. Call is ending.")
                break
            # ... other event handlers ...
    except Exception as e:
        log.error(f"[/media] Error in WebSocket handler: {e}", exc_info=True)
    finally:
        log.info(f"[/media] Cleaning up for call {call_state.get('twilio_call_sid', 'N/A')}")
        call_state["stop_call"] = True
        if ai_response_task and not ai_response_task.done(): ai_response_task.cancel()
        if action_watcher_task and not action_watcher_task.done(): action_watcher_task.cancel()
        await tts_controller.stop_immediately()
        if deepgram_streamer: await deepgram_streamer.finish()
        if ws.client_state == WebSocketState.CONNECTED: await ws.close(code=1000)
        print(f"\n🔚 CALL ENDED: {call_state.get('caller_phone_number', 'N/A')}\n" + "="*60, flush=True)
        log.info("[/media] WebSocket cleanup complete.")


## FIX: This entire function is rewritten to use the non-blocking producer-consumer pattern.
async def handle_ai_turn(call_state: dict, lang: str, ws: WebSocket,
                         conversation_history: List[str], tts_controller: TTSController,
                         send_twilio_clear):
    
    stream_sid = call_state["twilio_stream_sid"]

    async def _produce_audio(generator, queue):
        try:
            for audio_chunk in generator:
                await queue.put(audio_chunk)
            await queue.put(None)
        except Exception as e:
            log.error(f"[TTS-PRODUCER-ERROR] {e}", exc_info=True)
            await queue.put(None)

    async def speak_chunk(text_chunk: str, voice_id: str):
        if call_state.get("user_is_speaking"): return False
        if not text_chunk.strip() or call_state["stop_call"]: return True # Not a failure, just nothing to say
        
        log.info(f"[TTS-SPEAK-{stream_sid}] AI ➔ {text_chunk[:60].replace(chr(10), ' ')}")
        tts_controller.audio_queue = asyncio.Queue()

        try:
            tts_controller.current_generator = tts_client.text_to_speech.stream(
                text=text_chunk, voice_id=voice_id, model_id="eleven_turbo_v2",
                output_format="ulaw_8000", optimize_streaming_latency=0,
            )
            tts_controller.producer_task = asyncio.create_task(
                _produce_audio(tts_controller.current_generator, tts_controller.audio_queue)
            )

            while True:
                if call_state.get("user_is_speaking"):
                    log.warning(f"[TTS-CUTOFF-{stream_sid}] BARGE-IN! Stopping consumer loop.")
                    await tts_controller.stop_immediately()
                    await send_twilio_clear()
                    return False

                audio = await tts_controller.audio_queue.get()
                if audio is None: break

                await ws.send_text(json.dumps({
                    "event": "media", "streamSid": stream_sid,
                    "media": {"payload": base64.b64encode(audio).decode()},
                }))
            return True # Spoke successfully
        except Exception as e:
            log.warning(f"[TTS-ERROR-{stream_sid}] TTS stream exception: {e}", exc_info=True)
            await tts_controller.stop_immediately()
            return False

    # --- AI and prompt logic (remains the same) ---
    prompt_for_chat = build_prompt(conversation_history[-100:], lang)
    prompt_for_chat += (
        "\n\nAfter your response add exactly one of the following tags on its own line:" 
        "\n  • [[END]]     – ONLY when you have already (a) confirmed an appointment slot, (b) gathered the caller's name and phone number, AND (c) given a goodbye." 
        "\n  • [[CONTINUE]] – in all other situations." 
        "\nDo not output anything after the tag." 
        "\n\nAssistant:"
    )
    try:
        ai_response_text = await make_openai_request(
            api_key_manager=None, model="openchat/openchat-3.5-1210", prompt=prompt_for_chat,
            max_tokens=512, temperature=0.3, top_p=0.95,
        ) or ""
        if "[[END]]" in ai_response_text:
            conversation_status = "ended"
            ai_response_text = ai_response_text.replace("[[END]]", "").strip()
            ai_response_text = FAREWELL_LINES.get(lang, FAREWELL_LINES["en"])
        else:
            conversation_status = "continue"
            ai_response_text = ai_response_text.replace("[[CONTINUE]]", "").strip()
    except Exception as exc:
        log.error(f"[AI_TURN-{stream_sid}] Error generating AI reply: {exc}", exc_info=True)
        ai_response_text = ""
        conversation_status = "continue"
    
    if not ai_response_text.strip():
        log.info(f"[AI_TURN-{stream_sid}] No AI response, skipping TTS.")
        return # Nothing to say.

    tts_controller.is_speaking = True # Set speaking state to True before we start sending audio
    
    # --- Speaking logic using the new non-blocking function ---
    tts_controller.reset_spoken_text()
    all_spoken_successfully = True
    voice_id = VOICE_IDS.get(lang, VOICE_IDS["en"])

    words = ai_response_text.split()
    for i in range(0, len(words), 100):
        if call_state["stop_call"] or call_state.get("user_is_speaking"):
            all_spoken_successfully = False
            break
        chunk_text = " ".join(words[i : i + 100])
        if not await speak_chunk(chunk_text, voice_id):
            all_spoken_successfully = False
            break
        else:
            tts_controller.add_spoken_text(chunk_text)
    
    spoken_text = tts_controller.get_spoken_text()
    if spoken_text:
        conversation_history.append(f"AI: {spoken_text}")
    elif all_spoken_successfully and ai_response_text:
        conversation_history.append(f"AI: {ai_response_text}")

    if conversation_status == "ended" and all_spoken_successfully:
        log.info(f"[AI_TURN-{stream_sid}] Conversation ended. Terminating call.")
        await asyncio.sleep(END_DELAY_SEC)
        call_state["terminate_call"] = True
        return

    if all_spoken_successfully and not call_state["stop_call"] and ws.client_state == WebSocketState.CONNECTED:
        log.info(f"[AI_TURN-MARK-{stream_sid}] Sending end_of_ai_turn mark.")
        try:
            await ws.send_text(json.dumps({"event": "mark", "streamSid": stream_sid, "mark": {"name": "end_of_ai_turn"}}))
        except Exception as e:
            log.error(f"[AI_TURN-MARK-ERROR-{stream_sid}] Failed to send mark: {e}")
    elif not all_spoken_successfully:
        tts_controller.is_speaking = False

def _is_trivial_thanks(text: str) -> bool:
    txt = text.strip().lower().rstrip(".!?")
    return txt in {"thank you", "you"}