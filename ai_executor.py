#!/usr/bin/env python3
# ============================  ai_executor.py  =============================
"""Core orchestration: stateless helper that generates a reply and a decision
based on a capped conversation history."""
from __future__ import annotations

import logging
import os
from string import Formatter
from typing import Any, Dict, List

from apiChatCompletion import APIKeyManager, make_openai_request
from ai_prompt import DECISION_PROMPT, build_messages
from dotenv import load_dotenv

load_dotenv()

# ───────────────────────────────────────── Config & Logging ─────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

DEBUG_PROMPTS = True
API_MODEL     = "llama3-70b-instruct"
OPENAI_KEYS   = APIKeyManager.from_env("API_KEYS_THINKING_VERSION2")

# ✅ NEW: Add a constant to control how many turns of history we send.
# This prevents the context from growing too large and causing model amnesia.
MAX_HISTORY_LINES = 100 # (10 turns of Human/AI)

# ───────────────────────────────────────────────── Prompt-safety helper (Unchanged) ───────────────────
def safe_format(template: str, **kwargs) -> str:
    placeholders = {fname for fname, *_ in Formatter().parse(template) if fname}
    missing = placeholders - kwargs.keys()
    if missing:
        logger.error("Missing placeholders in template: %s", ", ".join(sorted(missing)))
        for m in missing: kwargs[m] = ""
    return template.format(**kwargs)

# ───────────────────────────────────────────── Public entry point ───────────────────────────
async def generate_reply(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stateless helper.  `payload` **must** contain:
        • conversation_history: List[str]  (['Human: …', 'AI: …', …])
    Returns {ai_message, decision} where decision in {'continue','routed','ended'}
    """
    history: List[str] = payload["conversation_history"]

    # ✅ FIX: Take only the last N lines of history to avoid exceeding the model's context limit.
    if len(history) > MAX_HISTORY_LINES:
        logger.warning(f"History has {len(history)} lines, trimming to last {MAX_HISTORY_LINES}.")
        history = history[-MAX_HISTORY_LINES:]

    # Build chat-style messages for the hosted backend
    messages = build_messages(history, lang="en")

    # ✅ DEBUG: Log messages being sent to the AI.
    if DEBUG_PROMPTS:
        logger.info("--- Messages Sent to AI ---\n%s\n-------------------------", messages)

    # Call hosted LLM using the new chat-completions format
    raw_text = await make_openai_request(
        api_key_manager=None,
        model=API_MODEL,
        messages=messages,
        max_tokens=100,
        temperature=0.3,
        top_p=0.95,
    )
    raw_text = raw_text or ""

    # ------------------------------------------------------------------
    # Secondary call: classify END / CONTINUE based on the *user's* last
    # message (not the assistant reply).
    # ------------------------------------------------------------------
    # Find last user message in history
    last_user_msg = ""
    for line in reversed(history):
        if line.startswith("Human:") or line.startswith("User:"):
            last_user_msg = line.split(":", 1)[1].strip()
            break

    class_prompt = DECISION_PROMPT.replace("{user_reply}", last_user_msg)

    decision_raw = await make_openai_request(
        api_key_manager=None,
        model=API_MODEL,
        prompt=class_prompt,
        max_tokens=10,
        temperature=0.0,
        top_p=1.0,
    ) or ""

    dec = decision_raw.strip().upper()
    status = "continue"
    if dec == "END":
        status = "ended"

    return { "ai_message": raw_text, "conversation_status": status }