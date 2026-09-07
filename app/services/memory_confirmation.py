import re

from langchain_core.messages import AIMessage

from app.core.identity import require_user_id
from app.core.logging import logger
from app.models.memory import IMMUTABLE_KEYS, structured_value_error
from app.repositories.memory_decision_log import append_memory_decision_log
from app.repositories.user_memory import MemoryStoreError, apply_confirmed_structured_correction, build_structured_storage_result, is_valid_key


def build_pending_memory_confirmation(result: dict, *, user_id: str):
    require_user_id(user_id)
    if not isinstance(result, dict) or result.get("decision") != "needs_confirmation":
        return None

    return {
        "user_id": user_id,
        "field": result.get("field"),
        "category": result.get("category"),
        "existing_value": result.get("existing_value"),
        "proposed_value": result.get("proposed_value"),
        "reason": result.get("reason"),
    }


def build_confirmation_message(pending_confirmation: dict) -> str:
    return (
        f"I currently have {pending_confirmation['existing_value']} saved as your "
        f"{pending_confirmation['field']}. Do you want me to replace it with "
        f"{pending_confirmation['proposed_value']}?"
    )


def apply_confirmation_prompt_to_state(state: dict, pending_confirmation: dict):
    confirmation_message = build_confirmation_message(pending_confirmation)

    if state.get("messages") and isinstance(state["messages"][-1], AIMessage):
        last_ai_message = state["messages"][-1]
        if isinstance(last_ai_message.content, str) and last_ai_message.content.strip():
            state["messages"][-1] = AIMessage(
                content=f"{last_ai_message.content}\n\n{confirmation_message}",
                id=last_ai_message.id,
            )
        else:
            state["messages"][-1] = AIMessage(content=confirmation_message, id=last_ai_message.id)
    else:
        state.setdefault("messages", []).append(AIMessage(content=confirmation_message))

    return state


def classify_confirmation_reply(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    normalized = normalized.strip(".,!?;:'\"")

    confirm_replies = {
        "yes",
        "y",
        "correct",
        "exactly",
        "stimmt",
        "ja",
    }
    reject_replies = {
        "no",
        "n",
        "nope",
        "keep old",
        "nein",
    }

    if normalized in confirm_replies:
        return "confirm"
    if normalized in reject_replies:
        return "reject"
    return "unclear"


def resolve_pending_confirmation(user_id: str, reply_text: str, pending_confirmation: dict):
    require_user_id(user_id)
    pending = pending_confirmation if isinstance(pending_confirmation, dict) else {}
    field, category = pending.get("field"), pending.get("category")
    if (
        pending.get("user_id") != user_id
        or not is_valid_key(field, category)
        or field not in IMMUTABLE_KEYS
        or pending.get("existing_value") is None
        or structured_value_error(field, pending.get("proposed_value"))
    ):
        result = build_structured_storage_result(
            "ignored", field if isinstance(field, str) else None,
            category if isinstance(category, str) else None,
            reason="invalid pending confirmation or owner mismatch",
        )
        append_memory_decision_log(user_id, result, source="confirmation_resolver")
        return {"status": "invalid", "message": "That pending correction is invalid. No memory was changed.", "result": result}

    classification = classify_confirmation_reply(reply_text)

    if classification == "confirm":
        try:
            result = apply_confirmed_structured_correction(
                user_id=user_id,
                key=pending_confirmation["field"],
                value=pending_confirmation["proposed_value"],
                category=pending_confirmation["category"],
                expected_existing_value=pending_confirmation["existing_value"],
            )
        except MemoryStoreError as exc:
            logger.error("[CONFIRMATION]: %s", exc)
            result = build_structured_storage_result(
                "failed", field, category, reason="memory persistence failed",
            )
            append_memory_decision_log(user_id, result, source="confirmation_resolver")
            return {
                "status": "failed",
                "message": "I couldn't save that correction. The memory file is unchanged; check the storage error before trying again.",
                "result": result,
            }
        append_memory_decision_log(
            user_id=user_id,
            result=result,
            source="confirmation_resolver",
        )
        if result.get("decision") != "confirmed_update_applied":
            return {
                "status": "failed",
                "message": (
                    "I couldn't apply that update safely. The stored value may have changed; "
                    "please state the correction again."
                ),
                "result": result,
            }
        return {
            "status": "confirmed",
            "message": (
                f"Got it - I updated your {pending_confirmation['field']} "
                f"to {pending_confirmation['proposed_value']}."
            ),
            "result": result,
        }

    if classification == "reject":
        result = build_structured_storage_result(
            decision="confirmation_rejected",
            field=pending_confirmation["field"],
            category=pending_confirmation["category"],
            existing_value=pending_confirmation["existing_value"],
            proposed_value=pending_confirmation["proposed_value"],
            reason="user rejected immutable update",
        )
        append_memory_decision_log(
            user_id=user_id,
            result=result,
            source="confirmation_resolver",
        )
        return {
            "status": "rejected",
            "message": (
                f"Okay, I kept your {pending_confirmation['field']} "
                f"as {pending_confirmation['existing_value']}."
            ),
            "result": result,
        }

    return {
        "status": "unclear",
        "message": (
            f"Please answer yes or no so I know whether to update your "
            f"{pending_confirmation['field']}."
        ),
        "result": None,
    }
