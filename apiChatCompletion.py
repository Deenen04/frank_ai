import logging
import httpx
import asyncio
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
import json
import os

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ----------------------------------------------------------------------
# Endpoint & timeout settings
# ----------------------------------------------------------------------

LLM_API_BASE = os.getenv("LLM_API_BASE", "http://127.0.0.1:8000")

# For *chat-style* requests (preferred) we target the OpenAI-compatible endpoint
#     POST <base>/v1/chat/completions
# For *raw-prompt* requests (legacy) we keep the old "/completion" path so
# existing callers that rely on a single prompt string continue to work.

CHAT_COMPLETION_ENDPOINT = f"{LLM_API_BASE.rstrip('/')}/v1/chat/completions"
LEGACY_COMPLETION_ENDPOINT = f"{LLM_API_BASE.rstrip('/')}/completion"

CHAT_TIMEOUT = 60  # seconds

# ----------------------------------------------------------------------
# Helper: Convert chat-style messages to single prompt string expected by backend
# ----------------------------------------------------------------------

def _messages_to_prompt(messages: List[Dict[str, Any]]) -> str:
    """Convert a list of {role, content} dicts to a single prompt string.

    The backend expects one string beginning with ``<|begin_of_text|>`` and then
    repeating ``<|role|>\n<content>`` for each message. Roles are mapped as:
        * system     → <|system|>
        * user       → <|user|>
        * assistant  → <|assistant|>
    Unknown roles default to ``<|user|>``.
    """
    role_map = {
        "system": "<|system|>",
        "user": "<|user|>",
        "assistant": "<|assistant|>",
    }
    parts: List[str] = ["<|begin_of_text|>"]
    for msg in messages:
        role_tag = role_map.get(msg.get("role", "user"), "<|user|>")
        parts.append(f"{role_tag}\n{msg.get('content', '')}")
    # The backend will continue generation from the last assistant tag
    # if the final message is from the user. Ensure we end with the tag.
    if messages and messages[-1].get("role") != "assistant":
        parts.append("<|assistant|>")
    return "\n".join(parts)

# ----------------------------------------------------------------------
# make_openai_request – now points to your hosted LLM
# ----------------------------------------------------------------------
async def make_openai_request(
    *,
    api_key_manager: Optional[object],  # not used anymore
    model: str,
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    response_format: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Send prompt to hosted model instead of OpenAI.

    The hosted LLM expects a *single* prompt string, not the Chat Completions
    format.  We therefore flatten the ``messages`` list into a prompt and send
    the JSON body like this::

        {
            "prompt": "<|begin_of_text|><|system|>…",
            "n_predict": 200,
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9,
            "stream": false
        }
    """
    try:
        # ------------------------------------------------------------------
        # Decide which endpoint & payload format to use
        # ------------------------------------------------------------------

        if messages is not None:
            # Preferred *chat completions* style.
            url = CHAT_COMPLETION_ENDPOINT
            payload: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "stream": False,
            }
        else:
            # Legacy single-prompt style (fallback / internal use)
            if prompt is None:
                raise ValueError("make_openai_request: Either 'prompt' or 'messages' must be provided.")

            # Some backends still expect a flat prompt – keep compatibility.
            prompt = _messages_to_prompt(messages) if prompt is None else prompt

            payload = {
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": temperature,
                "top_k": 40,
                "top_p": top_p,
                "stream": False,
            }
            url = LEGACY_COMPLETION_ENDPOINT

        async with httpx.AsyncClient(timeout=CHAT_TIMEOUT) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            # Compatible parsing – try common response shapes
            if isinstance(data, dict):
                # Many backends return {"content": "…"}
                if data.get("content"):
                    return str(data["content"]).strip()
                # GPT4All style maybe {"response": "…"}
                if data.get("response"):
                    return str(data["response"]).strip()
                # OpenAI compatibility shim (some proxies still use it)
                if "choices" in data and data["choices"]:
                    choice0 = data["choices"][0]
                    if isinstance(choice0, dict):
                        for key in ("content", "text"):
                            if key in choice0 and choice0[key]:
                                return str(choice0[key]).strip()
                        if choice0.get("message") and isinstance(choice0["message"], dict):
                            m_content = choice0["message"].get("content")
                            if m_content:
                                return str(m_content).strip()
                # Fallback top-level "text"
                if data.get("text"):
                    return str(data["text"]).strip()
            logger.warning("make_openai_request: Could not find assistant content in response JSON: %s", data)
            return ""

    except Exception as e:
        # Helpful debug information on failure
        if isinstance(e, httpx.HTTPStatusError):
            logger.error(
                "Hosted LLM request failed: HTTP %s – %s", e.response.status_code, e.response.text
            )
        else:
            logger.error("Hosted LLM request failed: %s", e)
        return None


# ----------------------------------------------------------------------
# Dummy APIKeyManager to preserve compatibility with ai_executor
# ----------------------------------------------------------------------
class APIKeyManager:
    """Stub class for compatibility; not used in hosted model."""
    @classmethod
    def from_env(cls, env_var: str):
        return cls()

    async def get_next_api_key(self) -> str:
        return ""  # not used

# ----------------------------------------------------------------------
# Streaming variant – yields the reply incrementally (word-level) so that
# the caller can start doing TTS much earlier.
# ----------------------------------------------------------------------
async def make_openai_request_stream(
    *,
    model: str,
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.1,
    top_p: float = 0.1,
    chunk_size_words: int = 1,
) -> AsyncGenerator[str, None]:
    """Same semantics as *make_openai_request* but returns an **async generator**
    that yields the assistant reply in *chunk_size_words* word chunks.

    It tries to use server-side streaming when available (payload["stream"] = True).
    If the upstream endpoint does **not** support streaming it falls back to the
    non-streaming request and simply chunks the full response.
    """

    if prompt is None:
        if messages is None:
            raise ValueError("make_openai_request_stream: Either 'prompt' or 'messages' must be provided.")
        prompt = _messages_to_prompt(messages)

    payload = {
        "prompt": prompt,
        "stream": True,
        "temperature": temperature,
        "top_k": 40,
        "top_p": top_p,
        "n_predict": 2048,
    }

    url_stream = CHAT_ENDPOINT

    buffer: List[str] = []  # buffer for chunks we will yield
    cumulative_text: str = ""  # track text already emitted to avoid duplicates when endpoint sends cumulative payloads
    last_word: Optional[str] = None  # track last emitted word to avoid duplicates

    def _yield_buffer(force: bool = False):
        """Helper to yield *chunk_size_words* words at a time."""
        nonlocal buffer
        while len(buffer) >= chunk_size_words:
            to_emit = buffer[:chunk_size_words]
            buffer = buffer[chunk_size_words:]
            yield " ".join(to_emit)
        if force and buffer:
            to_emit = buffer
            buffer = []
            yield " ".join(to_emit)

    try:
        # Attempt streaming request first.
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url_stream, json=payload) as response:
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    # Some servers use SSE style "data: <json>" lines – strip prefix.
                    if line.startswith("data:"):
                        line = line[len("data:"):].strip()
                    # Many implementations send "[DONE]" when finished.
                    if line.strip() in {"[DONE]", "DONE"}:
                        break

                    # Extract the text content from whatever JSON schema we get.
                    try:
                        data = json.loads(line)
                        # Possible keys
                        content_part = (
                            data.get("content")
                            or data.get("response")
                            or data.get("text")
                            or ""
                        )
                    except Exception:
                        # If line isn't JSON, treat as plain text token.
                        content_part = line

                    # If the endpoint sends cumulative text, compute only the *new tail*
                    if content_part.startswith(cumulative_text):
                        new_part = content_part[len(cumulative_text):]
                    else:
                        new_part = content_part

                    cumulative_text = content_part

                    # Split into words and add to buffer
                    for w in new_part.split():
                        if w and w != last_word:  # ignore empty strings and duplicates
                            buffer.append(w)
                            last_word = w
                    # Yield any complete chunks available
                    for chunk in _yield_buffer():
                        yield chunk

        # Finished streaming – flush any remainder
        for chunk in _yield_buffer(force=True):
            yield chunk
        return

    except Exception as e:
        # Streaming failed – fall back to non-streaming request.
        logger.warning("Streaming request failed (falling back to full request): %s", e)

    # Fallback path – reuse normal helper
    full_text = await make_openai_request(
        api_key_manager=None,
        model=model,
        prompt=prompt,
        max_tokens=2048,
        temperature=temperature,
        top_p=top_p,
    ) or ""

    # Yield in chunks of *chunk_size_words*
    for word in full_text.split():
        if word == last_word:
            continue
        buffer.append(word)
        last_word = word
        if len(buffer) >= chunk_size_words:
            yield " ".join(buffer[:chunk_size_words])
            buffer = buffer[chunk_size_words:]

    if buffer:
        yield " ".join(buffer)
