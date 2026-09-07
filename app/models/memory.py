import math
import re

from app.utils.dates import parse_birthdate


# Session-only extraction keys. Persistent goals use dynamic.current_goal below.
ALLOWED_KEYS = {
    "weight",
    "target_weight",
    "location",
    "goal",
    "balance",
    "calories",
    "height",
    "age",
}

IMMUTABLE_KEYS = ["name", "birthdate", "country of origin", "skin type"]

MEMORY_SCHEMA = {
    "profile": ["birthdate", "city", "job", "education"],
    "preferences": ["favorite_food", "hobbies", "diet"],
    "health": ["allergies"],
    "household": ["household_size"],
    "kitchen": ["appliances"],
    "dynamic": ["current_goal", "mood", "weight"],
}

ALLOWED_TYPES = ["habit", "preference", "diet", "dislike", "behavior", "context"]


def structured_value_error(key: str, value) -> str | None:
    """Check JSON values before they become authoritative structured facts."""
    if key == "birthdate":
        if not isinstance(value, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
            return "birthdate must use YYYY-MM-DD"
        try:
            parse_birthdate(value)
        except ValueError:
            return "invalid or future birthdate"
        return None
    if key in {"weight", "household_size"}:
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            return "expected a positive number"
        try:
            number = float(value)
        except (ValueError, OverflowError):
            return "expected a positive number"
        if not math.isfinite(number) or number <= 0:
            return "expected a positive finite number"
        if key == "household_size" and not number.is_integer():
            return "household_size must be a whole number"
        return None
    if key in {"hobbies", "allergies", "appliances"} and isinstance(value, list):
        if not 1 <= len(value) <= 50:
            return "expected between 1 and 50 text values"
        if all(isinstance(item, str) and item.strip() and len(item) <= 2000 for item in value):
            return None
        return "expected non-empty text values"
    if not isinstance(value, str) or not value.strip() or len(value) > 2000:
        return "expected non-empty text up to 2000 characters"
    return None


def memory_item_error(item, kind: str) -> str | None:
    if not isinstance(item, dict):
        return "memory item must be an object"
    required = {"key", "category", "value"} if kind == "structured" else {"text", "type"}
    if set(item) - required:
        return "unexpected memory fields; model-supplied ownership is not allowed"
    if not required.issubset(item):
        return "missing required memory fields"
    if kind == "structured":
        if not isinstance(item["key"], str) or not isinstance(item["category"], str):
            return "memory key and category must be text"
    else:
        if not isinstance(item["text"], str) or not 5 <= len(item["text"].strip()) <= 2000:
            return "memory text must contain 5 to 2000 characters"
        if not isinstance(item["type"], str) or not item["type"].strip():
            return "memory type must be non-empty text"
    return None


def user_stated_birthdate(text: str, value) -> bool:
    """Conservatively accept an unquoted, first-person ISO birthdate statement."""
    match = re.fullmatch(
        r"\s*(?:actually[,\s]+)?(?:i was born on|my (?:birthdate|date of birth) is)"
        r"\s+(?:actually\s+)?([0-9]{4}-[0-9]{2}-[0-9]{2})[.!]?\s*",
        text,
        flags=re.IGNORECASE,
    )
    return match is not None and match.group(1) == value
