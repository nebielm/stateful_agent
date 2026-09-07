import json
from typing import Any, Dict

from app.core.logging import logger
from app.llm.client import call_llm_json, get_llm
from app.llm.prompts.memory import DATA_SELECTION_PROMPT
from app.llm.prompts.retrieval import DATA_RELEVANCE_PROMPT
from app.models.memory import ALLOWED_KEYS, ALLOWED_TYPES, IMMUTABLE_KEYS, MEMORY_SCHEMA
from app.utils.formatting import format_prompt


def extract_ephemeral_updates(user_text: str, agent_text: str) -> Dict[str, Any]:
    """
    Extracts CURRENT state updates from user + agent messages.

    Returns:
        dict of updated state values (safe, filtered, validated)
    """
    combined_text = f"""
    USER MESSAGE:
    {user_text}

    AGENT RESPONSE:
    {agent_text}
    """

    prompt = f"""
    You are a state extraction engine.

    Extract ONLY the CURRENT user state from the conversation below.

    RULES:
    - Only extract FINAL / CURRENT values (not past values)
    - Do NOT infer or guess
    - Do NOT include historical values
    - Keep keys short and generic (e.g. weight, location, balance)
    - goal and target_weight describe session-only state; persisted goals use current_goal
    - Output MUST be valid JSON (no text, no explanation)

    Example:
    Input:
    "User: I lost 10kg. Assistant: Your new weight is 120kg"
    Output:
    {{"weight": 120}}

    Conversation:
    {combined_text}
    """

    try:
        response = get_llm().invoke(prompt)
        raw_output = response.content.strip()
    except Exception:
        return {}

    try:
        parsed = json.loads(raw_output)
    except Exception:
        return {}

    if not isinstance(parsed, dict):
        return {}

    filtered = {}
    for key, value in parsed.items():
        if key in ALLOWED_KEYS:
            filtered[key] = value

    return filtered


def extract_memory_updates(text: str) -> dict:
    logger.info("[MEMORY EXTRACTION]: Start extracting user memory updates")
    try:
        prompt = format_prompt(
            DATA_SELECTION_PROMPT,
            text=text,
            memory_schema=MEMORY_SCHEMA,
            immutable_keys=IMMUTABLE_KEYS,
            allowed_types=ALLOWED_TYPES,
        )

        output = call_llm_json(prompt, {"structured": [], "unstructured": []})

        if not isinstance(output, dict):
            logger.warning("[MEMORY EXTRACTION]: Rejected non-object extraction result")
            return {"structured": [], "unstructured": []}

        structured = output.get("structured", [])
        unstructured = output.get("unstructured", [])

        if not isinstance(structured, list):
            logger.warning("[MEMORY EXTRACTION]: Rejected non-list structured batch")
            structured = []

        if not isinstance(unstructured, list):
            logger.warning("[MEMORY EXTRACTION]: Rejected non-list unstructured batch")
            unstructured = []
        logger.info(
            "[MEMORY EXTRACTION]: Extracted update counts: structured=%s unstructured=%s",
            len(structured), len(unstructured),
        )
        return {
            "structured": structured,
            "unstructured": unstructured,
        }

    except Exception as e:
        logger.error(f"[MEMORY EXTRACTION]: ❌ Error while extracting updates on user data: {str(e)}")
        return {"structured": [], "unstructured": []}


def extract_knowledge(raw_text: str) -> dict:
    """
    Extract structured knowledge for storage.
    Returns: summary, category, tags
    """
    logger.info("[MEMORY EXTRACTION]: Start extracting knowledge updates")

    if not raw_text or len(raw_text) < 20:
        logger.info("[MEMORY EXTRACTION]: IGNORED: Text must be at least 20 characters long.")
        return {
            "summary": raw_text,
            "category": "general",
            "tags": [],
        }

    prompt = f"""
    Extract structured knowledge from the text.

    Return a JSON object with the following fields:
    - summary: a concise factual summary of the text, between 100 and 250 words, capturing all important points.
    - category: one word representing the main topic (e.g., tech, health, nutrition, fitness, general).
    - tags: a list of 3-6 keywords representing the key concepts or entities.

    Rules:
    - Be consistent in style and formatting.
    - Do NOT include explanations, commentary, or extra text.
    - Output ONLY valid JSON.
    - Focus on factual and verifiable information.
    - If the text is very short (<100 words), summarize it fully but keep it concise.

    Text:
    {raw_text}
    """
    try:
        data = call_llm_json(
            prompt,
            default={
                "summary": raw_text[:200],
                "category": "general",
                "tags": [],
            },
        )

        if not isinstance(data, dict):
            logger.error("[MEMORY EXTRACTION]: Invalid JSON format")
            return {
                "summary": raw_text[:200],
                "category": "general",
                "tags": [],
            }
        result = {
            "summary": data.get("summary", raw_text[:200]),
            "category": data.get("category", "general"),
            "tags": data.get("tags", []),
        }
        logger.info("[MEMORY EXTRACTION]:✅Extracted updates on knowledge base successfully.")
        return result
    except Exception as e:
        logger.error(f"[MEMORY EXTRACTION]: ❌ Error while extracting updates on user data: {str(e)}")
        return {
            "summary": raw_text[:200],
            "category": "general",
            "tags": [],
        }


def extract_retrieval_plan(text: str) -> dict:
    logger.info(
        "[MEMORY EXTRACTION]: Start extracting retrieval plan"
    )
    try:
        prompt = format_prompt(
            DATA_RELEVANCE_PROMPT,
            user_input=text,
            memory_schema=MEMORY_SCHEMA,
            allowed_types=ALLOWED_TYPES,
        )

        output = call_llm_json(
            prompt,
            {
                "structured_to_retrieve": [],
                "unstructured_to_retrieve": [],
            },
        )

        if not isinstance(output, dict):
            return {
                "structured_to_retrieve": [],
                "unstructured_to_retrieve": [],
            }
        logger.info(
            "[MEMORY EXTRACTION]: ✅ Extracted retrieving plan for user data and knowledge base data."
        )
        return {
            "structured_to_retrieve": output.get("structured_to_retrieve", []),
            "unstructured_to_retrieve": output.get("unstructured_to_retrieve", []),
        }
    except Exception as e:
        logger.error(
            f"[MEMORY EXTRACTION]: ❌ Error while extracting retrieving plan for user data and knowledge base data:{str(e)}"
        )
        return {
            "structured_to_retrieve": [],
            "unstructured_to_retrieve": [],
        }
