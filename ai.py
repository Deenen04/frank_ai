# =============================  ai_service.py  ==============================
"""
Light-weight wrapper around ai_executor.* that used to be served as HTTP
end-points.  You can now call `handle_ai_call()` and `cancel_last_call()`
directly from your own code, a task queue, a CLI, a scheduler, etc.

Example
-------
>>> from ai_service import handle_ai_call
>>> await handle_ai_call({
...     "brand_phone_number": "+1-800-PHONX",
...     "user_phone_number": "+49-151-123456",
...     "call_id": "abc-123",
...     "question": "Where is my order?",
... })
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Union

from pydantic import BaseModel, Field, ValidationError

from ai_executor import process_call, process_cancel  # your orchestration logic

__all__ = ["CallRequest", "CancelRequest", "handle_ai_call", "cancel_last_call"]

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# --------------------------------------------------------------------------- #
# Request models (kept for validation + autocomplete)
# --------------------------------------------------------------------------- #


class CallRequest(BaseModel):
    brand_phone_number: str = Field(..., alias="brand_phone_number")
    user_phone_number: str = Field(..., alias="user_phone_number")
    call_id: str
    question: str


class CancelRequest(BaseModel):
    """Payload to cancel the last AI call for a user."""

    user_phone_number: str = Field(..., alias="user_phone_number")


# --------------------------------------------------------------------------- #
# Public API – plain async functions
# --------------------------------------------------------------------------- #


async def handle_ai_call(
    payload: Union[CallRequest, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Run a full AI call.

    Parameters
    ----------
    payload : CallRequest | dict
        Same keys you sent to the old HTTP endpoint.  Can be a pydantic model
        or a raw dict – it will be validated either way.

    Returns
    -------
    dict
        Whatever `ai_executor.process_call` returns, plus a `"processing_time"`
        key (seconds, float).
    """
    # --- validation --------------------------------------------------------
    if isinstance(payload, dict):
        try:
            payload = CallRequest.model_validate(payload)
        except ValidationError as exc:
            logger.error("Validation failed for CallRequest – %s", exc)
            raise
    # ----------------------------------------------------------------------

    start = time.perf_counter()
    try:
        result: Dict[str, Any] = await process_call(payload.model_dump(by_alias=True))
    except Exception as exc:  # bubble up – caller decides what to do
        logger.exception("Unhandled error for call_id=%s", payload.call_id)
        raise
    result["processing_time"] = time.perf_counter() - start
    return result


async def cancel_last_call(
    payload: Union[CancelRequest, Dict[str, Any]]
) -> None:
    """
    Cancel (delete) the last AI call for a user.

    Parameters
    ----------
    payload : CancelRequest | dict
        Must contain ``user_phone_number``.
    """
    if isinstance(payload, dict):
        try:
            payload = CancelRequest.model_validate(payload)
        except ValidationError as exc:
            logger.error("Validation failed for CancelRequest – %s", exc)
            raise

    try:
        await process_cancel(payload.user_phone_number)
    except Exception as exc:
        logger.exception("Failed to cancel for user=%s", payload.user_phone_number)
        raise
