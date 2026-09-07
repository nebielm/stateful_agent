import json

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openrouter import ChatOpenRouter

from app.core.logging import logger
from app.core import settings


_UNINITIALIZED = object()
_llm = _UNINITIALIZED
_embeddings = _UNINITIALIZED


def get_llm():
    global _llm

    if _llm is _UNINITIALIZED:
        if not settings.OPENROUTER_API_KEY:
            raise ValueError("Missing OPENROUTER_API_KEY. Set it in the environment or project .env; only OpenRouter is supported.")

        _llm = ChatOpenRouter(
            model=settings.CHAT_MODEL_NAME,
            api_key=settings.OPENROUTER_API_KEY,
            temperature=0,
        )

    return _llm


def get_embeddings():
    global _embeddings

    if _embeddings is _UNINITIALIZED:
        _embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)

    return _embeddings


def call_llm_json(prompt: str, default):
    response = get_llm().invoke(prompt)
    content = response.content
    if not isinstance(content, str):
        content = str(content)

    try:
        cleaned_content = content.strip()

        if "```" in cleaned_content:
            cleaned_content = cleaned_content.replace("```json", "").replace("```", "").strip()

        return json.loads(cleaned_content)

    except Exception:
        logger.warning("Invalid JSON, retrying once...")
        retry_prompt = f"Fix this JSON:\n{content}"
        retry = get_llm().invoke(retry_prompt)

        try:
            retry_content = retry.content
            if not isinstance(retry_content, str):
                retry_content = str(retry_content)
            retry_content = retry_content.strip()
            if "```" in retry_content:
                retry_content = retry_content.replace("```json", "").replace("```", "").strip()
            return json.loads(retry_content)
        except Exception:
            logger.error("Failed twice → returning default")
            return default
