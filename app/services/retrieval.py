from datetime import datetime
from typing import Any, Dict, List

from app.core.logging import logger
from app.core.identity import require_user_id
from app.llm.client import call_llm_json
from app.models.memory import ALLOWED_TYPES


def build_context_fallback(all_context: dict, k: int = 5):
    fallback = {}
    for category in ("structured", "unstructured", "knowledge"):
        items = all_context.get(category, [])
        fallback[category] = items[:k] if isinstance(items, list) else []
    return fallback


def normalize_ranked_context(output, all_context: dict, k: int = 5):
    candidates = build_context_fallback(all_context, max(0, k))
    fallback = {category: [] for category in candidates}
    if not isinstance(output, dict):
        return fallback

    normalized = {}
    for category in ("structured", "unstructured", "knowledge"):
        source_items = candidates[category]

        if category not in output:
            normalized[category] = []
            continue

        ranked_items = output.get(category)
        if not isinstance(ranked_items, list):
            normalized[category] = []
            continue

        filtered = []
        for item in ranked_items:
            if item in source_items and item not in filtered:
                filtered.append(item)
            if len(filtered) >= k:
                break

        normalized[category] = filtered

    return normalized


def retrieve_unstructured_memory(
    user_vectorstore,
    message: str,
    user_id: str,
    relevant_types: List[str],
    k: int = 5,
) -> List[Dict]:
    """
    Retrieve top-k relevant unstructured memories for a user, combining the user message
    with LLM-identified types, enforced as metadata constraints.
    """
    logger.info("[RETRIEVAL SYSTEM]: Start retrieving unstructured user data.")
    require_user_id(user_id)
    relevant_types = list(dict.fromkeys(
        item for item in relevant_types if isinstance(item, str) and item in ALLOWED_TYPES
    ))
    if not relevant_types or k <= 0:
        return []

    search_query = " ".join([message] + relevant_types)
    try:
        results = user_vectorstore.similarity_search_with_score(
            query=search_query,
            k=k * 2,
            filter={"$and": [{"user_id": user_id}, {"type": {"$in": relevant_types}}]},
        )
        scored = []

        for doc, distance in results:
            if doc.metadata.get("user_id") != user_id or doc.metadata.get("type") not in relevant_types:
                continue

            timestamp = doc.metadata.get("timestamp")

            if timestamp:
                dt = datetime.fromisoformat(timestamp)
                age_days = (datetime.now() - dt).days
            else:
                age_days = 999

            recency_score = min(1, max(0, 1 - age_days / 30))
            # Lower is better. This preserves the distance/recency ordering,
            # but is a heuristic cost, not a calibrated similarity/probability.
            ranking_cost = distance * 0.7 + (1 - recency_score) * 0.3

            scored.append(
                {
                    "text": doc.page_content,
                    "type": doc.metadata.get("type"),
                    "distance": distance,
                    "ranking_cost": ranking_cost,
                }
            )

        scored = sorted(scored, key=lambda x: x["ranking_cost"])

        logger.info("[RETRIEVAL SYSTEM]: ✅ Successfully retrieved unstructured user data.")
        return scored[:k]

    except Exception as e:
        logger.error(f"[RETRIEVAL SYSTEM]: ❌ Error while retrieving unstructured user data: {str(e)}")
        return []


def retrieve_knowledge_docs(knowledge_vectorstore, message: str, k: int = 5) -> List[Dict]:
    """
    Retrieve top-k knowledge chunks relevant to the user's message.
    """
    logger.info("[RETRIEVAL SYSTEM]: Start retrieving knowledge base data.")

    try:
        results = knowledge_vectorstore.similarity_search_with_score(query=message, k=k)

        scored = []
        for doc, distance in results:
            scored.append(
                {
                    "text": doc.page_content,
                    "category": doc.metadata.get("category"),
                    "tags": doc.metadata.get("tags"),
                    "distance": distance,
                }
            )

        scored = sorted(scored, key=lambda x: x["distance"])
        logger.info("[RETRIEVAL SYSTEM]: ✅ Successfully retrieved knowledge base data.")
        return scored[:k]

    except Exception as e:
        logger.error(f"[RETRIEVAL SYSTEM]: ❌ Error while retrieving knowledge base data: {str(e)}")
        return []


def retrieve_relevant_context_for_user(all_context: dict, message: str, k: int = 5):
    """
    Takes combined structured, unstructured, and knowledge contexts and ranks the top-k
    relevant entries for the given user message using an LLM.
    """
    logger.info("[RETRIEVAL SYSTEM]: Start retrieving and ranking relevant context data.")

    ranking_prompt = f"""
    You are an AI assistant that ranks memory and knowledge context for another AI assistant.

    User message:
    "{message}"

    Your job:
    Select the MOST relevant items from the provided context.

    RULES:
    - Do NOT invent new data
    - ONLY return items from the lists
    - Keep original structure
    - Be strict and selective

    Return TOP {k} per category.

    OUTPUT (JSON ONLY):
    {{
      "structured": [...],
      "unstructured": [...],
      "knowledge": [...]
    }}

    Structured:
    {all_context.get("structured", [])}

    Unstructured:
    {all_context.get("unstructured", [])}

    Knowledge:
    {all_context.get("knowledge", [])}
    """
    fallback = {category: [] for category in ("structured", "unstructured", "knowledge")}
    try:
        output = call_llm_json(
            ranking_prompt,
            default=fallback,
        )

        normalized_output = normalize_ranked_context(output, all_context, k)
        logger.info("[RETRIEVAL SYSTEM]: ✅ Successfully retrieved relevant context data.")
        return normalized_output
    except Exception as e:
        logger.error(f"[RETRIEVAL SYSTEM]: ❌ Error while retrieving relevant context data: {str(e)}")
        return fallback
