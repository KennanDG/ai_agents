import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def transfer_to_human(reason: str) -> Dict[str, Any]:
    """Simulate transferring the call to a human agent.

    In a production system, this would integrate with a call-center
    or queuing platform (Twilio, Amazon Connect, etc.).
    """
    if not reason or not reason.strip():
        logger.warning("Transfer requested without a reason")
        return {"error": "A transfer reason is required."}

    # Simulate a successful transfer (always succeeds in this stub).
    logger.info("Transferring call to human agent. Reason: %s", reason)
    return {
        "status": "transferred",
        "reason": reason.strip()[:200],
        "estimated_wait_seconds": 30,
    }
