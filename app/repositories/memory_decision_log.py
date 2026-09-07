import json
from datetime import datetime, timezone
from pathlib import Path

from app.core.settings import MEMORY_DECISION_LOG_PATH
from app.core.logging import logger
from app.models.memory import MEMORY_SCHEMA


def append_memory_decision_log(
    user_id: str,
    result: dict,
    *,
    source: str = "structured_memory_storage",
    timestamp: str | None = None,
    log_path: str | None = None,
) -> dict:
    """Best-effort audit metadata, never a prerequisite for memory-write success."""
    target_path = Path(log_path or MEMORY_DECISION_LOG_PATH)
    category, field = result.get("category"), result.get("field")

    entry = {
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "category": category if isinstance(category, str) and category in MEMORY_SCHEMA else None,
        "field": field if isinstance(field, str) and any(field in keys for keys in MEMORY_SCHEMA.values()) else None,
        "has_proposed_value": result.get("proposed_value") is not None,
        "has_existing_value": result.get("existing_value") is not None,
        "decision": result.get("decision"),
        "reason": result.get("reason"),
        "source": source,
    }

    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except (OSError, TypeError, ValueError) as exc:
        logger.error(
            "[MEMORY AUDIT]: Decision log append failed (%s); memory decision remains %s",
            type(exc).__name__, entry["decision"],
        )
        return {**entry, "audit_error": type(exc).__name__}

    return entry
