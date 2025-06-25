"""Centralised configuration module.

Loads environment variables from a local ``.env`` file (if present) and
exposes **two** interfaces so existing code keeps working:

1. **Module‑level constants** (e.g. ``REDIS_URL``) – backwards‑compatible
   with the snippet you posted.
2. A **Settings** object – imported via ``from config import settings`` –
   that bundles everything as attributes (what the new *main.py* expects).
"""
from __future__ import annotations

import os
import logging
from types import SimpleNamespace
from typing import List, Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
#                               .env support
# ---------------------------------------------------------------------------

load_dotenv()  # noqa – quietly ignore missing file

# ---------------------------------------------------------------------------
#                         Raw environment variables
# ---------------------------------------------------------------------------

REDIS_URL: str | None = os.getenv("REDIS")
if not REDIS_URL:
    raise ValueError("REDIS environment variable not set.")

LOGGING_LEVEL: str = os.getenv("LOGGING_LEVEL", "INFO").upper()
ALLOWED_ORIGINS: List[str] = os.getenv("ALLOWED_ORIGINS", "*").split(",")
CONCURRENT_TASKS_LIMIT: int = int(os.getenv("CONCURRENT_TASKS_LIMIT", 100))
INACTIVITY_THRESHOLD: int = int(os.getenv("INACTIVITY_THRESHOLD", 600))

HOSTNAME: str | None = os.getenv("HOSTNAME")  # e.g. api.example.com
if not HOSTNAME:
    raise ValueError("HOSTNAME environment variable not set.")

DEEPGRAM_API_KEY: str | None = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    raise ValueError("DEEPGRAM_API_KEY environment variable not set.")

ELEVEN_API_KEY: str | None = os.getenv("ELEVEN_API_KEY")
if not ELEVEN_API_KEY:
    raise ValueError("ELEVEN_API_KEY environment variable not set.")

# OpenAI / embeddings --------------------------------------------------------

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_EMBEDDING_URL = "https://api.openai.com/v1/embeddings"
MAX_RETRIES = 2
RETRY_BACKOFF_FACTOR = 1.2
DEFAULT_TIMEOUT = 30
EMBEDDING_MODEL = "text-embedding-ada-002"

# Neo4j ---------------------------------------------------------------------

NEO4J_URI: Optional[str] = os.getenv("NEO4J_URI")
NEO4J_USER: Optional[str] = os.getenv("NEO4J_USER")
NEO4J_PASSWORD: Optional[str] = os.getenv("NEO4J_PASSWORD")
NEO4J_URI_BOLT: Optional[str] = os.getenv("NEO4J_URI_BOLT")

# ---------------------------------------------------------------------------
#                            Pydantic‑style object
# ---------------------------------------------------------------------------

class _Settings(SimpleNamespace):
    """Bundle all values into an attribute‑style object."""

    def __init__(self):  # noqa: D401 (simple docstring OK)
        super().__init__(
            # Core credentials
            HOSTNAME=HOSTNAME,
            DEEPGRAM_API_KEY=DEEPGRAM_API_KEY,
            ELEVEN_API_KEY=ELEVEN_API_KEY,

            # Redis + runtime limits
            REDIS_URL=REDIS_URL,
            CONCURRENT_TASKS_LIMIT=CONCURRENT_TASKS_LIMIT,
            INACTIVITY_THRESHOLD=INACTIVITY_THRESHOLD,

            # Logging / CORS
            LOGGING_LEVEL=LOGGING_LEVEL,
            ALLOWED_ORIGINS=ALLOWED_ORIGINS,

            # OpenAI
            OPENAI_API_URL=OPENAI_API_URL,
            OPENAI_EMBEDDING_URL=OPENAI_EMBEDDING_URL,
            MAX_RETRIES=MAX_RETRIES,
            RETRY_BACKOFF_FACTOR=RETRY_BACKOFF_FACTOR,
            DEFAULT_TIMEOUT=DEFAULT_TIMEOUT,
            EMBEDDING_MODEL=EMBEDDING_MODEL,

            # Neo4j
            NEO4J_URI=NEO4J_URI,
            NEO4J_USER=NEO4J_USER,
            NEO4J_PASSWORD=NEO4J_PASSWORD,
            NEO4J_URI_BOLT=NEO4J_URI_BOLT,
        )


settings = _Settings()

# ---------------------------------------------------------------------------
#                          Helper: configure logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=LOGGING_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
