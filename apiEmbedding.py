"""
apiEmbedding.py – replacement version that uses a local SentenceTransformer
model and keeps the original public interface expected by vector.py.

Changes:
* Eliminates the global variable `model` that was being shadowed.
* Stores the SentenceTransformer instance in a module‑level constant `_ST_MODEL`.
* Returns embeddings as `np.float32` arrays the way Neo4j expects.
* Keeps `APIKeyManager`, `APIClient`, and `make_openai_embedding_request` so
  no other file needs to change.
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer



from utils import configure_logging
from config import EMBEDDING_MODEL  # value unused but preserved for signature

# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
configure_logging()
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Load the SentenceTransformer once at import time (thread‑safe)
# --------------------------------------------------------------------------- #
logger.info("Loading SentenceTransformer model 'intfloat/e5-base-v2' …")
_ST_MODEL = SentenceTransformer("intfloat/e5-base-v2")
logger.info("Model loaded – 768‑D embeddings ready")

# --------------------------------------------------------------------------- #
# Compatibility: APIKeyManager is kept even though we do not need real keys
# --------------------------------------------------------------------------- #
class APIKeyManager:
    """Dummy key‑manager so existing code keeps working unchanged."""

    def __init__(self, api_keys: List[str]):
        self.api_keys = deque(api_keys)

    @classmethod
    def from_env(cls, env_var: str) -> "APIKeyManager":
        # Always returns a stub instance; env var is ignored
        return cls(["local-model"])

    async def get_next_api_key(self) -> str:  # noqa: D401
        return "local-model"


# --------------------------------------------------------------------------- #
# APIClient – singleton wrapper around the local model
# --------------------------------------------------------------------------- #
class APIClient:
    _instance: Optional["APIClient"] = None

    def __new__(cls, api_key_manager: APIKeyManager, timeout: int = 30):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, api_key_manager: APIKeyManager, timeout: int = 30):
        if getattr(self, "_initialised", False):
            return  # already set up
        self.api_key_manager = api_key_manager
        self._initialised = True
        logger.info("APIClient initialised – using local SentenceTransformer")

    async def make_openai_embedding_request(
        self,
        input_text: str,
        model: str = EMBEDDING_MODEL,
    ) -> Optional[np.ndarray]:
        """Return a float32 numpy vector for *input_text* or *None* on error."""
        try:
            # E5 models expect the prefix "query: " for query embeddings
            formatted = f"query: {input_text}"

            loop = asyncio.get_running_loop()
            vector: np.ndarray = await loop.run_in_executor(
                None,
                lambda txt=formatted: _ST_MODEL.encode(
                    txt,
                    convert_to_numpy=True,
                    normalize_embeddings=False,  # match OpenAI default
                ),
            )
            return vector.astype(np.float32, copy=False)
        except Exception:
            logger.exception("Error generating local embedding")
            return None

    async def close(self) -> None:  # noqa: D401
        """Nothing to clean up for a local model, but kept for interface parity."""
        return None


# --------------------------------------------------------------------------- #
# Module‑level helper exposed for vector.py
# --------------------------------------------------------------------------- #
_global_client: Optional[APIClient] = None
_global_api_key_manager: Optional[APIKeyManager] = None


async def make_openai_embedding_request(
    api_key_manager: APIKeyManager,
    input_text: str,
    model: str = EMBEDDING_MODEL,
) -> Optional[np.ndarray]:
    """Vector.py calls this – signature must remain intact."""
    global _global_client, _global_api_key_manager

    if _global_client is None or _global_api_key_manager is not api_key_manager:
        _global_api_key_manager = api_key_manager
        _global_client = APIClient(api_key_manager)

    return await _global_client.make_openai_embedding_request(
        input_text=input_text,
        model=model,
    )


# --------------------------------------------------------------------------- #
# Quick self‑test (run `python apiEmbedding.py` to verify installation)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    async def _demo():
        mgr = APIKeyManager.from_env("OPENAI_API_KEYS")
        vec = await make_openai_embedding_request(
            api_key_manager=mgr,
            input_text="This is a test sentence for local embedding generation.",
        )
        if vec is not None:
            print("✓ Embedding generated – shape:", vec.shape, "dtype:", vec.dtype)
        else:
            print("✗ Failed to generate embedding")

    asyncio.run(_demo())
