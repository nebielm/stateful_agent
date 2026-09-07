import os


def require_user_id(user_id: str) -> str:
    """Validate an owner supplied by the application, never by a model."""
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("A non-empty application user_id is required")
    return user_id


def get_user_id() -> str:
    """Use one stable local identity; configuration can select an existing owner."""
    return require_user_id(os.getenv("STATEFUL_AGENT_USER_ID", "local-user").strip())
