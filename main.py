import os
import json
import base64
# import time # Not explicitly used for time.sleep, asyncio.sleep is used
import asyncio
import logging
import re
from typing import List, Optional
import langid  # language identification library
langid.set_languages(["en", "fr", "de"])

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Form
from fastapi.responses import HTMLResponse
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream, Parameter # Parameter not used in current code
from dotenv import load_dotenv
from starlette.websockets import WebSocketState # Import for checking WebSocket state


from whisper_streamer import WhisperStreamer as DeepgramStreamer
from elevenlabs.client import ElevenLabs
# Streaming generation helper
from apiChatCompletion import make_openai_request
# Decision prompt template is imported below; we build it with simple replace.
# Import language‐specific prompt templates
from ai_prompt import (
    UNIFIED_CONVERSATION_AGENT,
    UNIFIED_CONVERSATION_AGENT_FR,
    UNIFIED_CONVERSATION_AGENT_DE,
    DECISION_PROMPT,
    build_messages,
)
# (generate_reply is kept for non-streaming fallbacks if needed)
from ai_executor import generate_reply  # Import the real AI function (fallback)
# from route_call import route_call # Assuming this is defined elsewhere

# ------------------------------------------------------------------
# Ultra-fast heuristic language detector (English / French / German)
# ------------------------------------------------------------------
_FR_CHARS = set("éèêëàâäîïôöùûüçœ")
_DE_CHARS = set("äöüß")


def fast_lang_detect(text: str) -> str:
    """Return "fr", "de" or "en" using a cheap character-based heuristic.

    The heuristic looks for any accented characters that are unique to
    French or German. If none are found we default to English.
    This runs in O(len(text)) time and on short strings typically takes
    <0.01 ms – well below the 0.01 s requirement.
    """
    lowered = text.lower()
    if any(ch in _FR_CHARS for ch in lowered):
        return "fr"
    if any(ch in _DE_CHARS for ch in lowered):
        return "de"
    return "en"

def route_call(call_sid):
    log.info(f"Mock route_call called for SID: {call_sid}")


load_dotenv()
HOSTNAME = os.getenv("HOSTNAME_twilio", "localhost:8000") # Ensure port if uvicorn runs on non-80
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")  # optional now – kept for backwards compat
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")

# Local Whisper no longer needs the Deepgram key – do not error if missing.

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
log = logging.getLogger("voicebot")

# Enable DEBUG logging for Deepgram VAD debugging
deepgram_log = logging.getLogger("deepgram_streamer")
deepgram_log.setLevel(logging.DEBUG)

VOICE_IDS = {"en": "21m00Tcm4TlvDq8ikWAM", "fr": "ZQFCSsF1tIcjtMZJ6VCA", "de": "v3V1d2rk6528UrLKRuy8"}
FAREWELL_LINES = {"en": "Thanks for calling. Goodbye.", "fr": "Merci d'avoir appelé. Au revoir.", "de": "Danke für Ihren Anruf. Auf Wiedersehen."}
GREETING_LINES = {"en": "Hi, This is Frank Babar Clinic, I am here to assist you book an appointment with us today. How can I help you?", "fr": "Bonjour, comment puis-je vous aider?", "de": "Hallo, wie kann ich Ihnen helfen?"}
END_DELAY_SEC = 1 # Reduced for faster testing
USER_TURN_END_DELAY_SEC = 0.8 # Reduced for faster response

class TranscriptSanitizer:
    def __init__(self):
        self.reset()

    def reset(self):
        self._final_parts: List[str] = []
        self._last_interim: str = ""

    def add_transcript(self, new_transcript: str, is_final: bool = False):
        """Add a new transcript part, properly handling interim vs final results."""
        if not new_transcript.strip():
            return
        
        # Handle based on whether this is a final or interim result
        if is_final:
            # This is a final result from Deepgram ASR
            # Add any existing interim to final parts first
            if self._last_interim:
                self._final_parts.append(self._last_interim)
                self._last_interim = ""
            # Add the final transcript
            self._final_parts.append(new_transcript)
        else:
            # This is an interim result - replace the last interim
            # Deepgram interim results usually replace previous interims for the same utterance
            self._last_interim = new_transcript

    def get_current_transcript(self) -> str:
        """Get the current complete transcript including final parts and current interim."""
        return " ".join(self._final_parts + ([self._last_interim] if self._last_interim else [])).strip()

    def finalize_utterance(self) -> str:
        """Finalize the current utterance and reset for next one."""
        if self._last_interim:
            self._final_parts.append(self._last_interim)
            self._last_interim = ""
        full_transcript = " ".join(self._final_parts)
        self._final_parts = []  # Reset for next full utterance
        return " ".join(full_transcript.split())  # Normalize spaces


class TTSController:
    def __init__(self): self.current_generator = None; self.is_speaking = False
    def stop_immediately(self):
        if self.current_generator:
            log.debug("[TTS] Attempting to stop TTS generator.")
            try: self.current_generator.close() # For generator-based streams
            except GeneratorExit:
                log.debug("[TTS] GeneratorExit caught.")
            except Exception as e:
                log.warning(f"[TTS] Exception closing generator: {e}")
            self.current_generator = None
        self.is_speaking = False

app = FastAPI()
tts_client = ElevenLabs(api_key=ELEVEN_API_KEY)

@app.on_event("startup")
async def startup_event(): # Renamed to avoid conflict with `startup` variable if any
    log.info("[STARTUP] Voice agent started — integrated DeepgramStreamer")

@app.post("/voice", response_class=HTMLResponse)
async def answer_call(_request: Request, From: str = Form(...), To: str = Form(...)): # Renamed `answer`
    log.info("[/voice] %s ➔ %s", From, To)
    vr = VoiceResponse()
    connect = Connect() # Corrected variable name
    # Ensure HOSTNAME includes scheme for wss if it's not just the host
    ws_url = f"wss://{HOSTNAME}/media" if not HOSTNAME.startswith("localhost") else f"ws://{HOSTNAME}/media"
    
    # Check if HOSTNAME from .env already has ws:// or wss://
    if "://" in HOSTNAME:
        ws_url = f"{HOSTNAME}/media" # Assume full URL like wss://myhost.com or ws://localhost:8000
    elif ":" in HOSTNAME: # e.g. localhost:8000
        ws_url = f"ws://{HOSTNAME}/media"
    else: # e.g. mydomain.com
        ws_url = f"wss://{HOSTNAME}/media"


    log.info(f"Connecting to WebSocket: {ws_url}")
    stream = Stream(url=ws_url)
    # Twilio sends these as custom parameters in the 'start' message
    stream.parameter(name="caller_phone_number", value=From) # Parameter class has lowercase 'p'
    stream.parameter(name="brand_phone_number", value=To)
    # CallSid is available as a template variable in TwiML
    stream.parameter(name="twilio_call_sid", value="{{CallSid}}")
    connect.append(stream)
    vr.append(connect)
    return HTMLResponse(str(vr), media_type="application/xml")

async def play_greeting(lang: str, sid: str, ws: WebSocket, tts_controller: TTSController, state: dict):
    voice_id, text = VOICE_IDS.get(lang, VOICE_IDS["en"]), GREETING_LINES.get(lang, GREETING_LINES["en"])
    
    # state["user_is_speaking"] = False # This should be controlled by Deepgram events
    tts_controller.is_speaking = True
    log.info(f"[GREETING-{sid}] Playing: '{text}'")
    try:
        tts_controller.current_generator = tts_client.text_to_speech.stream(
            text=text, voice_id=voice_id, model_id="eleven_multilingual_v2", # eleven_flash_v2_5 is not a public model name. Use eleven_multilingual_v2 or eleven_turbo_v2
            output_format="ulaw_8000", optimize_streaming_latency=0
        )
        for audio_chunk_count, audio in enumerate(tts_controller.current_generator):
            if state.get("user_is_speaking"):
                log.warning(f"[GREETING-CUTOFF-{sid}] User started speaking. Stopping greeting.")
                tts_controller.stop_immediately() # This will break out of the loop
                break
            if state["stop_call"]: # Renamed for clarity
                log.info(f"[GREETING-STOP-{sid}] Call stop requested. Stopping greeting.")
                tts_controller.stop_immediately()
                break
            if ws.client_state != WebSocketState.CONNECTED:
                log.warning(f"[GREETING-DISCONNECTED-{sid}] WebSocket no longer connected. Stopping TTS.")
                tts_controller.stop_immediately()
                break
            
            # log.debug(f"[GREETING-{sid}] Sending audio chunk {audio_chunk_count}")
            await ws.send_text(json.dumps({
                "event": "media", "streamSid": sid,
                "media": {"payload": base64.b64encode(audio).decode()}
            }))
            await asyncio.sleep(0.01) # Small sleep to allow other tasks to run, prevent hogging

    except Exception as e:
        log.warning(f"[GREETING-ERROR-{sid}] TTS streaming interrupted: {e}", exc_info=True)
    finally:
        log.debug(f"[GREETING-FINALLY-{sid}] Cleaning up TTS.")
        tts_controller.is_speaking = False # Ensure this is reset
        # current_generator is set to None in stop_immediately

    if not state["stop_call"] and ws.client_state == WebSocketState.CONNECTED:
        log.info(f"[GREETING-MARK-{sid}] Sending end_of_ai_turn mark.")
        try:
            await ws.send_text(json.dumps({"event": "mark", "streamSid": sid,
                                           "mark": {"name": "end_of_ai_turn"}}))
        except RuntimeError as e: # Catch specific error for sending on closed socket
            if "WebSocket is not connected" in str(e) or "after sending 'websocket.close'" in str(e):
                log.warning(f"[GREETING-MARK-ERROR-{sid}] WebSocket already closed when trying to send mark: {e}")
            else:
                log.error(f"[GREETING-MARK-ERROR-{sid}] Unexpected RuntimeError: {e}", exc_info=True)
            # raise # Optionally re-raise if it's truly unexpected
        except Exception as e:
            log.error(f"[GREETING-MARK-ERROR-{sid}] Failed to send mark: {e}", exc_info=True)
    else:
        log.info(f"[GREETING-MARK-SKIP-{sid}] Skipping end_of_ai_turn mark (stop_call={state['stop_call']}, ws_state={ws.client_state}).")


def display_live_transcript(caller: str, transcript: str):
    print(f"\r[LIVE] {caller or 'Caller'}: {transcript}", end="", flush=True)

def display_final_turn(caller: str, full_phrase: str):
    print(f"\n[TURN] {caller or 'Caller'}: {full_phrase}")
    print("-" * 60, flush=True)

@app.websocket("/media")
async def media_websocket_endpoint(ws: WebSocket): # Renamed `media`
    await ws.accept()
    log.info("[/media] WebSocket connection accepted.")
    
    # State and helpers
    # Use more descriptive state keys
    call_state = {
        "stop_call": False, 
        "initiate_transfer": False, # Was "route_call"
        "terminate_call": False,    # Was "end_call"
        "user_is_speaking": False,
        "twilio_stream_sid": None,
        "caller_phone_number": "?",
        "brand_phone_number": "?",
        "twilio_call_sid": None, # From Twilio custom parameters
        "last_processed_transcript": "",  # Track what transcript we've already processed
        "pending_transcript_parts": [],   # Store partial transcripts for aggregation
    }
    tts_controller = TTSController()
    # DeepgramStreamer now emits complete utterances, so we no longer need TranscriptSanitizer.
    current_language = "multi" # Default language
    conversation_history: List[str] = []
    
    # Task management
    ai_response_task: Optional[asyncio.Task] = None
    # user_turn_end_timer: Optional[asyncio.Task] = None # No longer needed with on_utterance callback
    action_watcher_task: Optional[asyncio.Task] = None


    deepgram_streamer = DeepgramStreamer( # Renamed from streamer
        api_key=DEEPGRAM_API_KEY,
        encoding="mulaw", # Twilio default is mulaw
        sample_rate=8000, # Twilio default is 8000Hz
        interim_results=False,
        vad_events=True,  # Keep VAD for fallback even though endpointing is used
        punctuate=True,
        model='nova-3',  # Use nova-3 as requested
        language=current_language, # Start with default, can be changed
        # Enhanced amplitude-based VAD parameters
        use_amplitude_vad=True,
        amplitude_threshold_db=-5.0,  # This translates to 700 RMS - more reliable speech detection
        silence_timeout=2.0  # Longer timeout for natural speech patterns
    )

    async def process_final_utterance(final_utterance: str):
        """Handle a complete user utterance coming from DeepgramStreamer."""
        nonlocal ai_response_task, current_language
        
        # --------------------------------------------------------------
        # Detect language on the very first non-empty utterance and lock it
        # --------------------------------------------------------------
        if final_utterance and current_language == "multi":
            lang_code, lang_conf = langid.classify(final_utterance)
            if lang_code in ("fr", "de", "en"):
                current_language = lang_code
                log.info(f"[LANG-DETECT] Detected {lang_code.upper()} (conf={lang_conf:.2f}) – locking conversation language.")
            else:
                current_language = "en"  # default fallback
                log.info(f"[LANG-DETECT] Unknown language ({lang_code}), defaulting to EN.")
        
        # Handle empty transcripts (initial VAD detection)
        if not final_utterance:
            log.info("[PROCESS_UTTERANCE] Received empty utterance - VAD detected speech end, waiting for transcript...")
            # Don't process empty utterances, but mark user as not speaking
            call_state["user_is_speaking"] = False
            # Cancel any ongoing AI if user was detected speaking
            if tts_controller.is_speaking:
                log.info("[PROCESS_UTTERANCE] Stopping AI audio due to speech detection.")
                tts_controller.stop_immediately()
            return

        # Check if this is an aggregated transcript (contains previous parts)
        if call_state["last_processed_transcript"] and call_state["last_processed_transcript"] in final_utterance:
            log.info(f"[PROCESS_UTTERANCE] Received aggregated transcript: '{final_utterance}' (previous: '{call_state['last_processed_transcript']}')")
            # This is an aggregated transcript, interrupt any current AI and process the complete version
            if tts_controller.is_speaking:
                log.info("[PROCESS_UTTERANCE] Interrupting AI for aggregated transcript.")
                tts_controller.stop_immediately()
            if ai_response_task and not ai_response_task.done():
                log.warning("[PROCESS_UTTERANCE] Cancelling previous AI task for aggregated transcript.")
                ai_response_task.cancel()
            
            # Remove the old entry from conversation history if it exists
            if call_state["last_processed_transcript"]:
                for i in range(len(conversation_history) - 1, -1, -1):
                    if conversation_history[i].startswith("Human: ") and call_state["last_processed_transcript"] in conversation_history[i]:
                        log.info(f"[PROCESS_UTTERANCE] Removing old conversation entry: {conversation_history[i]}")
                        conversation_history.pop(i)
                        break
        else:
            log.info(f"[PROCESS_UTTERANCE] Processing new transcript: '{final_utterance}'")

        # Mark that user finished speaking
        call_state["user_is_speaking"] = False
        call_state["last_processed_transcript"] = final_utterance

        display_final_turn(call_state["caller_phone_number"], final_utterance)

        # Cancel any ongoing AI generation / TTS (in case not already done above)
        if ai_response_task and not ai_response_task.done():
            log.warning("[PROCESS_UTTERANCE] Cancelling previous AI task – user spoke over AI.")
            ai_response_task.cancel()
        tts_controller.stop_immediately()

        # Add to conversation history
        conversation_history.append(f"Human: {final_utterance}")

        # Launch AI response
        ai_response_task = asyncio.create_task(
            handle_ai_turn(call_state, current_language, ws, conversation_history, tts_controller)
        )

        # In process_final_utterance, after stripping final_utterance (we removed earlier), add duplicate check
        if final_utterance.lower() == call_state.get("last_processed_transcript", "").lower():
            log.info("[PROCESS_UTTERANCE] Duplicate utterance detected — ignoring.")
            return

    def on_dg_speech_start():
        # Pure VAD signal – we now wait for actual transcript before interrupting TTS.
        # Leaving this as a debug notification only.
        if call_state["stop_call"]:
            return
        log.debug("[DG] Speech start detected (waiting for transcript before taking action).")

    def on_dg_transcript(transcript: str, is_final: bool):
        """Hard stop TTS when any transcript is received from Deepgram."""
        if call_state["stop_call"]:
            return
            
        # Immediately stop TTS when we receive ANY transcript from Deepgram
        # This ensures we don't talk over the user
        if transcript.strip():
            # Mark that the caller is indeed speaking (confirmed by ASR text)
            call_state["user_is_speaking"] = True

        if tts_controller.is_speaking and transcript.strip():
            log.info(f"[DG-TRANSCRIPT] Hard stopping TTS due to transcript: '{transcript}' (final: {is_final})")
            tts_controller.stop_immediately()
            
        # Display live transcript for debugging
        if transcript.strip():
            display_live_transcript(call_state["caller_phone_number"], transcript)

    def on_dg_speech_end():
        # Optional: this is already covered by on_utterance, but good for state tracking
        if call_state["stop_call"]:
            return
        call_state["user_is_speaking"] = False
        log.info("[DG] Speech end detected.")

    def on_dg_utterance(utterance: str):
        if call_state["stop_call"]:
            return
        # This is triggered by the enhanced VAD system:
        # 1. VAD detects speech end and sends transcript immediately (even if empty)
        # 2. If more transcript parts arrive later, they are aggregated and sent again
        # 3. The main processing handles interruption of current AI responses
        asyncio.create_task(process_final_utterance(utterance))

    # Register callbacks with the streamer
    deepgram_streamer.on('on_transcript', on_dg_transcript)  # Add transcript callback for hard stop
    deepgram_streamer.on('on_speech_start', on_dg_speech_start)
    deepgram_streamer.on('on_speech_end', on_dg_speech_end)
    deepgram_streamer.on('on_utterance', on_dg_utterance)

    try:
        log.info("[/media] Connecting to Deepgram...")
        await deepgram_streamer.connect()
        log.info("[/media] Deepgram connection successful.")
    except Exception as e:
        log.error(f"[/media] Failed to connect to Deepgram: {e}", exc_info=True)
        # Send error response to client and close
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.close(code=1011, reason="Failed to connect to speech service")
        return

    try:
        # Watch for routing/end signals - this can be simplified or integrated
        async def call_action_watcher():
            log.debug("[ACTION_WATCHER] Started.")
            while not call_state["stop_call"]: # Loop as long as call is active
                if call_state["initiate_transfer"] and call_state["twilio_call_sid"] and '{{' not in call_state["twilio_call_sid"]:
                    log.info(f"[/media] Routing call {call_state['twilio_call_sid']}")
                    # await asyncio.get_event_loop().run_in_executor(None, route_call, call_state["twilio_call_sid"])
                    route_call(call_state["twilio_call_sid"]) # Assuming route_call is not IO-bound blocking
                    call_state["stop_call"] = True # Mark to stop processing
                    break 
                if call_state["terminate_call"]:
                    log.info(f"[/media] Terminating call {call_state['twilio_call_sid']}")
                    call_state["stop_call"] = True # Mark to stop processing
                    break
                await asyncio.sleep(0.1)
            log.debug("[ACTION_WATCHER] Exited.")
            # Ensure WebSocket is closed if watcher decides to stop the call
            if ws.client_state == WebSocketState.CONNECTED:
                log.info("[ACTION_WATCHER] Closing WebSocket due to call action.")
                await ws.close(code=1000)


        action_watcher_task = asyncio.create_task(call_action_watcher())

        # Main receive loop for messages from Twilio
        while not call_state["stop_call"] and ws.client_state == WebSocketState.CONNECTED:
            try:
                raw_message = await ws.receive_text()
            except WebSocketDisconnect:
                log.warning("[/media] WebSocket disconnected by client (Twilio).")
                call_state["stop_call"] = True
                break # Exit main loop

            message = json.loads(raw_message)
            event_type = message.get("event")

            if event_type == "start":
                call_state["twilio_stream_sid"] = message["start"]["streamSid"]
                custom_params = message["start"].get("customParameters", {})
                call_state["caller_phone_number"] = custom_params.get("caller_phone_number", "?")
                call_state["brand_phone_number"] = custom_params.get("brand_phone_number", "?")
                call_state["twilio_call_sid"] = custom_params.get("twilio_call_sid") # Store this
                
                log.info(f"[WS-START] SID: {call_state['twilio_stream_sid']}, "
                         f"From: {call_state['caller_phone_number']}, To: {call_state['brand_phone_number']}, "
                         f"CallSID: {call_state['twilio_call_sid']}")
                print(f"\n🎯 CALL STARTED: {call_state['caller_phone_number']} → {call_state['brand_phone_number']}")
                print("="*60)
                
                # Start greeting
                asyncio.create_task(play_greeting(
                    current_language, call_state["twilio_stream_sid"], ws, tts_controller, call_state
                ))

            elif event_type == "media":
                if call_state["stop_call"] or not deepgram_streamer.is_open():  # DG connection check
                    continue 
                
                payload = base64.b64decode(message["media"]["payload"])
                try:
                    await deepgram_streamer.send(payload)
                except Exception as e:
                    log.warning(f"[/media] Error sending audio to Deepgram: {e}")
                    # Continue processing, don't break the loop for individual send errors

            elif event_type == "mark":
                mark_name = message.get("mark", {}).get("name")
                sid_logged = message.get("streamSid") or call_state.get("twilio_stream_sid", "?")
                log.info(f"[WS-MARK] Received mark: {mark_name} for SID {sid_logged}")
                if mark_name == "end_of_ai_turn":
                    # This confirms AI has finished speaking its part.
                    # Useful if you need to take action after AI speaks.
                    # Currently, user can interrupt, so this might not always be the true end of AI's turn.
                    pass


            elif event_type == "stop":
                stop_obj = message.get("stop", {}) if isinstance(message, dict) else {}
                sid_logged = stop_obj.get("streamSid") or message.get("streamSid") or call_state.get("twilio_stream_sid", "?")
                log.info(f"[WS-STOP] Received stop event from Twilio for SID {sid_logged}. Call is ending.")
                call_state["stop_call"] = True
                break # Exit main loop
            
            elif event_type == "dtmf":
                log.info(f"[WS-DTMF] Received DTMF digit: {message['dtmf']['digit']} for SID {message['streamSid']}")
                # Handle DTMF if needed, e.g., to interrupt AI or trigger actions

            else:
                log.warning(f"[/media] Received unknown event type: {event_type}")
                log.debug(f"Unknown message: {message}")

    except WebSocketDisconnect:
        log.warning("[/media] WebSocket disconnected unexpectedly during processing.")
    except Exception as e:
        log.error(f"[/media] Error in WebSocket handler: {e}", exc_info=True)
    finally:
        log.info(f"[/media] Cleaning up for call {call_state.get('twilio_call_sid', 'N/A')}")
        call_state["stop_call"] = True # Ensure flag is set for all cleanup tasks

        # Cancel all pending tasks
        if ai_response_task and not ai_response_task.done():
            log.debug("[/media] Cancelling AI response task.")
            ai_response_task.cancel()
        if action_watcher_task and not action_watcher_task.done():
            log.debug("[/media] Cancelling action watcher task.")
            action_watcher_task.cancel()
        
        # Stop any active TTS
        tts_controller.stop_immediately()

        # Clean up Deepgram connection
        if deepgram_streamer:
            log.info("[/media] Finishing Deepgram streamer.")
            try:
                await deepgram_streamer.finish()
            except Exception as e:
                log.error(f"[/media] Error finishing Deepgram streamer: {e}", exc_info=True)

        # Ensure WebSocket is closed if not already
        if ws.client_state == WebSocketState.CONNECTED:
            log.info("[/media] Closing WebSocket connection in finally block.")
            try:
                await ws.close(code=1000)
            except Exception as e:
                log.error(f"[/media] Error closing WebSocket in finally: {e}")
        
        print(f"\n🔚 CALL ENDED: {call_state.get('caller_phone_number', 'N/A')}")
        print("="*60, flush=True)
        log.info("[/media] WebSocket cleanup complete.")


async def handle_ai_turn(call_state: dict, lang: str, ws: WebSocket,
                         conversation_history: List[str], tts_controller: TTSController):
    
    lang_selected = lang  # mutable copy – may switch after first AI words
    voice_id = VOICE_IDS.get(lang_selected, VOICE_IDS["en"])
    stream_sid = call_state["twilio_stream_sid"]  # cached for quick access/logging
    lang_locked = False   # becomes True after first successful detection
    if lang_selected in ("en", "fr", "de"):
        lang_locked = True  # user language already known from earlier detection

    # Replace previous speak_chunk definition with an updated one that
    # does a one-time language detection.
    async def speak_chunk(text_chunk: str):
        nonlocal voice_id, lang_locked, lang_selected
        # --------------------------------------------------------------
        # One-time language detection based on the first chunk
        # --------------------------------------------------------------
        if not lang_locked and text_chunk.strip():
            detected = fast_lang_detect(text_chunk)
            log.info(f"[LANG] Detected {detected.upper()} from first AI chunk → switching voice.")
            lang_selected = detected
            voice_id = VOICE_IDS.get(lang_selected, VOICE_IDS["en"])
            lang_locked = True
        # --------------------------------------------------------------
        # Existing early-exit checks
        # --------------------------------------------------------------
        if call_state["stop_call"] or not text_chunk.strip() or call_state.get("user_is_speaking"):
            log.info(f"[TTS-SPEAK] Skipping speak: stop_call={call_state['stop_call']}, "
                     f"empty_chunk={not text_chunk.strip()}, user_speaking={call_state.get('user_is_speaking')}")
            if call_state.get("user_is_speaking"):
                tts_controller.stop_immediately()
            return False
        # --------------------------------------------------------------
        # Streaming TTS with the (possibly updated) voice_id
        # --------------------------------------------------------------
        log.info(f"[TTS-SPEAK-{stream_sid}] AI ➔ {text_chunk[:60].replace(chr(10), ' ')}")
        tts_controller.is_speaking = True
        try:
            tts_controller.current_generator = tts_client.text_to_speech.stream(
                text=text_chunk,
                voice_id=voice_id,
                model_id="eleven_multilingual_v2",
                output_format="ulaw_8000",
                optimize_streaming_latency=0,
            )
            for audio_chunk_count, audio in enumerate(tts_controller.current_generator):
                if call_state.get("user_is_speaking"):
                    log.warning(f"[TTS-CUTOFF-{stream_sid}] User started speaking. Stopping AI TTS.")
                    tts_controller.stop_immediately()
                    return False
                if call_state["stop_call"] or ws.client_state != WebSocketState.CONNECTED:
                    log.warning(f"[TTS-STOP-{stream_sid}] Call stop or WS disconnected. Stopping AI TTS.")
                    tts_controller.stop_immediately()
                    return False
                await ws.send_text(json.dumps({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": base64.b64encode(audio).decode()},
                }))
                await asyncio.sleep(0.01)
            return True
        except asyncio.CancelledError:
            log.warning(f"[TTS-CANCELLED-{stream_sid}] TTS task was cancelled.")
            tts_controller.stop_immediately()
            return False
        except Exception as e:
            log.warning(f"[TTS-ERROR-{stream_sid}] TTS stream exception: {e}", exc_info=True)
            tts_controller.stop_immediately()
            return False
        finally:
            pass

    # ------------------------------------------------------------------
    # Build the prompt in the same way as ai_executor.generate_reply but
    # we will *stream* the answer instead of waiting for the entire text.
    # ------------------------------------------------------------------

    MAX_HISTORY_LINES = 20  # Keep in sync with ai_executor
    history_for_prompt = conversation_history[-MAX_HISTORY_LINES:]
    full_conversation_text = "\n".join(history_for_prompt)

    # Build chat messages for new backend
    messages_for_chat = build_messages(history_for_prompt, lang_selected)

    # --------------------------------------------------------------
    # Fetch the full assistant reply in one request (no streaming)
    # --------------------------------------------------------------
    try:
        ai_response_text = await make_openai_request(
            api_key_manager=None,
            model="vanilj/smaug-llama-3-70b-instruct:Q2_K",
            messages=messages_for_chat,
            max_tokens=512,
            temperature=0.3,
            top_p=0.95,
        ) or ""
    except Exception as exc:
        log.error(f"[AI_TURN-{stream_sid}] Error while generating AI reply: {exc}", exc_info=True)
        ai_response_text = ""

    all_spoken_successfully = True

    # --------------------------------------------------------------
    # Speak the reply in CHUNK_WORD_COUNT-word chunks
    # --------------------------------------------------------------
    CHUNK_WORD_COUNT = 15
    words = ai_response_text.split()
    for i in range(0, len(words), CHUNK_WORD_COUNT):
        if call_state["stop_call"]:
            break
        chunk_text = " ".join(words[i : i + CHUNK_WORD_COUNT])
        if not await speak_chunk(chunk_text):
            all_spoken_successfully = False

    # Append reply to history (AFTER we know it)
    if ai_response_text:
        conversation_history.append(f"AI: {ai_response_text}")
        print(
            f"[{call_state['caller_phone_number']}] HISTORY (now {len(conversation_history)} lines):\n"
            f"{json.dumps(conversation_history, indent=2, ensure_ascii=False)}\n",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Decide ROUTE / END / CONTINUE using the full reply
    # ------------------------------------------------------------------
    conversation_status = "continue"
    try:
        decision_prompt = DECISION_PROMPT.replace("{ai_reply}", ai_response_text)
        decision_raw = await make_openai_request(
            api_key_manager=None,
            model="vanilj/smaug-llama-3-70b-instruct:Q2_K",
            messages=[{"role": "user", "content": decision_prompt}],
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
        ) or ""
        dec = decision_raw.strip().upper()
        if dec == "ROUTE":
            conversation_status = "routed"
        elif dec == "END":
            conversation_status = "ended"
    except Exception as e:
        log.error(f"[AI_TURN-{stream_sid}] Failed to classify conversation status: {e}")

    if call_state["twilio_call_sid"] and conversation_status == "routed":
        call_state["initiate_transfer"] = True

    tts_controller.is_speaking = False  # ensure flag reset when turn done / interrupted

    # --------------------------------------------------------------
    # Handle END behaviour (farewell etc.)
    # --------------------------------------------------------------
    if conversation_status == "ended" and all_spoken_successfully:
        log.info(f"[AI_TURN-{stream_sid}] Conversation marked as ended. Terminating after delay…")
        await asyncio.sleep(END_DELAY_SEC)
        call_state["terminate_call"] = True
        return

    # --------------------------------------------------------------
    # Send Twilio mark if we completed speaking successfully
    # --------------------------------------------------------------
    if all_spoken_successfully and not call_state["stop_call"] and not call_state.get("user_is_speaking") and ws.client_state == WebSocketState.CONNECTED:
        log.info(f"[AI_TURN-MARK-{stream_sid}] Sending end_of_ai_turn mark.")
        try:
            await ws.send_text(json.dumps({"event": "mark", "streamSid": stream_sid,
                                           "mark": {"name": "end_of_ai_turn"}}))
        except RuntimeError as e:
            if "WebSocket is not connected" in str(e) or "after sending 'websocket.close'" in str(e):
                log.warning(f"[AI_TURN-MARK-ERROR-{stream_sid}] WebSocket already closed when trying to send mark: {e}")
            else:
                log.error(f"[AI_TURN-MARK-ERROR-{stream_sid}] Unexpected RuntimeError: {e}", exc_info=True)
        except Exception as e:
            log.error(f"[AI_TURN-MARK-ERROR-{stream_sid}] Failed to send mark: {e}", exc_info=True)
    else:
        log.info(f"[AI_TURN-MARK-SKIP-{stream_sid}] Skipping end_of_ai_turn mark (spoken_successfully={all_spoken_successfully}, stop_call={call_state['stop_call']}, user_speaking={call_state.get('user_is_speaking')}).")
