import logging
import httpx
import asyncio
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
import json

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Timeout settings
CHAT_TIMEOUT = 60  # seconds

# ----------------------------------------------------------------------
# make_openai_request – now points to your hosted LLM
# ----------------------------------------------------------------------
async def make_openai_request(
    *,
    api_key_manager: Optional[object],  # not used anymore
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    response_format: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Send prompt to hosted model instead of OpenAI."""
    try:
        # ------------------------------------------------------------------
        # New backend expects OpenAI-style chat payload (see user spec)
        # ------------------------------------------------------------------
        payload = {
            "model": model,
            "stream": False,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }
        url = "http://localhost:21434/api/chat"

        async with httpx.AsyncClient(timeout=CHAT_TIMEOUT) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            # Try standard OpenAI format first
            if isinstance(data, dict):
                if "choices" in data and data["choices"]:
                    try:
                        return data["choices"][0]["message"]["content"].strip()
                    except Exception:
                        pass
                # Fallbacks used by other servers
                if "response" in data:
                    return str(data["response"]).strip()
                if "text" in data:
                    return str(data["text"]).strip()
            return ""

    except Exception as e:
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
    messages: List[Dict[str, Any]],
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

    payload = {
        "model": model,
        "stream": True,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }

    url_stream = "http://localhost:21434/api/chat"

    buffer: List[str] = []  # buffer for chunks we will yield
    cumulative_text: str = ""  # track text already emitted to avoid duplicates when endpoint sends cumulative payloads
    last_word: Optional[str] = None  # track last emitted word to avoid duplicates

    async def _yield_buffer(force: bool = False):
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
                        # Common possibilities
                        content_part = (
                            # OpenAI streaming delta format
                            data.get("choices", [{}])[0].get("delta", {}).get("content")
                            or data.get("choices", [{}])[0].get("message", {}).get("content")
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
                        # Fallback – treat whole content as new (some endpoints send only delta)
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
        messages=messages,
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
