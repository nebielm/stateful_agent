import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from app.core.logging import logger
from app.core.identity import require_user_id
from app.repositories.memory_decision_log import append_memory_decision_log
from app.core.settings import USER_INFO_PATH
from app.models.memory import IMMUTABLE_KEYS, MEMORY_SCHEMA, structured_value_error


class MemoryStoreError(RuntimeError):
    """The memory file could not be read or safely replaced."""


def load_user_data(file_path: str | None = None) -> Dict[str, Any]:
    path = Path(file_path if file_path is not None else USER_INFO_PATH)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise MemoryStoreError(f"Cannot read memory file {path}; original file preserved.") from exc
    if not isinstance(data, dict) or any(
        not isinstance(profile, dict)
        or any(category in profile and not isinstance(profile[category], dict) for category in MEMORY_SCHEMA)
        for profile in data.values()
    ):
        raise MemoryStoreError(f"Invalid memory structure in {path}; original file preserved.")
    return data


def _write_user_data(data: Dict[str, Any]) -> None:
    """Replace only a complete file; this is not a multi-writer transaction."""
    path = Path(USER_INFO_PATH)
    temporary_path = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(data, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except (OSError, ValueError, TypeError) as exc:
        raise MemoryStoreError(f"Cannot replace memory file {path}; original file preserved.") from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                logger.error("[MEMORY STORAGE]: Could not remove temporary memory file")


def lookup_user_value(user_data: Dict[str, Any], key: str):
    if key in user_data:
        return user_data[key]

    for category_values in user_data.values():
        if isinstance(category_values, dict) and key in category_values:
            return category_values[key]

    return None


def is_valid_key(key, category):
    return isinstance(key, str) and isinstance(category, str) and key in MEMORY_SCHEMA.get(category, [])


def build_structured_storage_result(
    decision: str,
    field: str,
    category: str,
    existing_value=None,
    proposed_value=None,
    reason: str | None = None,
):
    return {
        "decision": decision,
        "field": field,
        "category": category,
        "existing_value": existing_value,
        "proposed_value": proposed_value,
        "reason": reason,
    }


def controlled_structured_data_storage(user_id: str, key: str, value: str, category: str):
    require_user_id(user_id)
    logger.info("[MEMORY STORAGE]: Started structured user data storage")

    if not is_valid_key(key, category):
        logger.info("[MEMORY STORAGE]: IGNORED: Invalid schema key/category")
        result = build_structured_storage_result(
            decision="ignored",
            field=key,
            category=category,
            proposed_value=value,
            reason="invalid schema key",
        )
        append_memory_decision_log(user_id=user_id, result=result)
        return result

    if reason := structured_value_error(key, value):
        result = build_structured_storage_result(
            "ignored", key, category, proposed_value=value, reason=reason,
        )
        append_memory_decision_log(user_id=user_id, result=result)
        return result

    try:
        data = load_user_data(USER_INFO_PATH)
        existing_user_data = data.get(user_id, {})
        existing_value = lookup_user_value(existing_user_data, key)

        if key in IMMUTABLE_KEYS and existing_value is not None:
            if str(existing_value) == str(value):
                logger.info(f"[MEMORY STORAGE]: No change for {key}")
                result = build_structured_storage_result(
                    decision="no_change",
                    field=key,
                    category=category,
                    existing_value=existing_value,
                    proposed_value=value,
                    reason="same immutable value",
                )
                append_memory_decision_log(user_id=user_id, result=result)
                return result
            logger.info(
                f"[MEMORY STORAGE]: IGNORED: Key: {key} is write-once immutable and cannot be overwritten."
            )
            result = build_structured_storage_result(
                decision="needs_confirmation",
                field=key,
                category=category,
                existing_value=existing_value,
                proposed_value=value,
                reason="immutable field conflict",
            )
            append_memory_decision_log(user_id=user_id, result=result)
            return result

        if user_id not in data:
            data[user_id] = {}
        if category not in data[user_id]:
            data[user_id][category] = {}

        user_data = data[user_id]
        if key not in user_data[category] or user_data[category][key] != value:
            user_data[category][key] = value
            _write_user_data(data)
            logger.info("[MEMORY STORAGE]: Structured user data stored: %s", key)
            result = build_structured_storage_result(
                decision="stored",
                field=key,
                category=category,
                existing_value=existing_value,
                proposed_value=value,
                reason="value stored",
            )
            append_memory_decision_log(user_id=user_id, result=result)
            return result
        logger.info(f"[MEMORY STORAGE]: No change for {key}")
        result = build_structured_storage_result(
            decision="no_change",
            field=key,
            category=category,
            existing_value=user_data[category][key],
            proposed_value=value,
            reason="same value",
        )
        append_memory_decision_log(user_id=user_id, result=result)
        return result
    except MemoryStoreError as exc:
        logger.error("[MEMORY STORAGE]: %s", exc)
        result = build_structured_storage_result(
            "failed", key, category, proposed_value=value, reason="memory persistence failed",
        )
        append_memory_decision_log(user_id, result)
        return result


def apply_confirmed_structured_correction(
    user_id: str,
    key: str,
    value: str,
    category: str,
    *,
    expected_existing_value=None,
):
    require_user_id(user_id)
    logger.info("[MEMORY STORAGE]: Applying confirmed structured correction")

    if not is_valid_key(key, category):
        return build_structured_storage_result(
            decision="ignored",
            field=key,
            category=category,
            proposed_value=value,
            reason="invalid schema key",
        )

    if key not in IMMUTABLE_KEYS:
        return build_structured_storage_result(
            "ignored", key, category, proposed_value=value,
            reason="field is not part of immutable confirmation workflow",
        )
    if reason := structured_value_error(key, value):
        return build_structured_storage_result(
            "ignored", key, category, proposed_value=value, reason=reason,
        )

    data = load_user_data(USER_INFO_PATH)
    existing_user_data = data.get(user_id, {})
    existing_value = existing_user_data.get(category, {}).get(key)

    if expected_existing_value is None or existing_value is None or existing_value != expected_existing_value:
        return build_structured_storage_result(
            decision="ignored",
            field=key,
            category=category,
            existing_value=existing_value,
            proposed_value=value,
            reason="pending confirmation mismatch",
        )

    if user_id not in data:
        data[user_id] = {}
    if category not in data[user_id]:
        data[user_id][category] = {}

    data[user_id][category][key] = value
    _write_user_data(data)

    return build_structured_storage_result(
        decision="confirmed_update_applied",
        field=key,
        category=category,
        existing_value=existing_value,
        proposed_value=value,
        reason="user confirmed immutable update",
    )


def retrieve_structured_memory(
    user_id: str,
    relevant_categories: List[str] = None,
    relevant_keys: List[str] = None,
    k: int = 5,
) -> List[Dict[str, Any]]:
    require_user_id(user_id)
    logger.info("[RETRIEVAL SYSTEM]: Retrieving structured user data from user DB")
    data = load_user_data(USER_INFO_PATH)
    if not data:
        logger.info("[RETRIEVAL SYSTEM]: No structured memory stored yet")
        return []
    try:
        user_data = data.get(user_id, {})
        results = []

        for category, items in user_data.items():
            if relevant_categories and category not in relevant_categories:
                continue

            for key, value in items.items():
                if relevant_keys and key not in relevant_keys:
                    continue
                if not is_valid_key(key, category) or structured_value_error(key, value):
                    logger.warning("[RETRIEVAL SYSTEM]: Skipped invalid stored structured value")
                    continue

                score = 1.0 if relevant_categories and category in relevant_categories else 0.7

                results.append(
                    {
                        "key": key,
                        "value": value,
                        "category": category,
                        "score": score,
                    }
                )
        logger.info("[RETRIEVAL SYSTEM]: ✅ Successfully retrieved structured user data.")
        return results[:k]
    except Exception as e:
        logger.error(f"[RETRIEVAL SYSTEM]: ❌ Error while retrieving structured user data: {str(e)}")
        return []
