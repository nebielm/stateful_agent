import json
import time
from datetime import datetime
from typing import List

import requests
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from app.core.identity import require_user_id
from app.core.logging import logger
from app.db.vectorstores import get_core_vectorstore
from app.llm.client import call_llm_json, get_llm
from app.llm.prompts.tools import TOOL_SELECTION_PROMPT
from app.models.memory import structured_value_error
from app.repositories.user_memory import load_user_data, lookup_user_value
from app.services.memory import enrich_knowledge
from app.utils.dates import calculate_age_from_birthdate
from app.utils.formatting import format_prompt
from app.utils.math_tools import safe_eval


@tool
def calculator(expression: str) -> float:
    """Evaluate a mathematical expression like '40 + 12 * 2' safely."""
    logger.info("[TOOL]: Calculator: Evaluating expression")
    if len(expression) > 50:
        logger.info("[TOOL]: Calculator: IGNORED: Expression must be at most 50 characters.")
        return "Expression too long"
    try:
        result = safe_eval(expression)
        logger.info("[TOOL]: Calculator: Successfully evaluated expression")
        return result
    except Exception as e:
        logger.error(f"[TOOL]: ❌ Calculator: Failed to evaluate expression: {str(e)}")
        return "Invalid mathematical expression"


@tool
def get_current_time() -> str:
    """Returns the current date and time"""
    logger.info("[TOOL]: CURRENT TIME: Getting current time")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[TOOL]: ✅ CURRENT TIME: {str(current_time)}")
    return current_time


@tool
def get_current_age(runtime: ToolRuntime[dict]) -> int | None:
    """Calculate the current user's age from their stored birthdate."""
    user_id = require_user_id(runtime.state.get("user_id"))
    logger.info("[TOOL] CURRENT USER AGE: Getting current age.")
    data = load_user_data()
    if not data:
        logger.info("[TOOL] CURRENT USER AGE: No file data/user_info.json found.")
        return
    try:
        user_data = data.get(user_id, {})
        birthdate_value = lookup_user_value(user_data, "birthdate") or lookup_user_value(user_data, "birthday")
        if birthdate_value is None:
            logger.info("[TOOL] CURRENT USER AGE: No birthdate data found.")
            return

        if structured_value_error("birthdate", birthdate_value):
            logger.warning("[TOOL] CURRENT USER AGE: Invalid stored birthdate")
            return

        age = calculate_age_from_birthdate(birthdate_value)
        logger.info("[TOOL] CURRENT USER AGE: Calculated age")
        return age
    except Exception as e:
        logger.info(f"[TOOL] ❌ CURRENT USER AGE: Error while getting user info {e}.")
        return


@tool
def get_user_info(key: str, runtime: ToolRuntime[dict]) -> str:
    """Retrieve a stored fact for the current user."""
    user_id = require_user_id(runtime.state.get("user_id"))
    logger.info("[TOOL] USER INFO: Getting user info")
    data = load_user_data()
    if not data:
        logger.info("[TOOL] USER INFO: No file data/user_info.json found.")
        return ""
    try:
        user_data = data.get(user_id, {})
        value = lookup_user_value(user_data, key)
        if key in {"birthdate", "birthday"} and structured_value_error("birthdate", value):
            return f"{key}: No valid data found"
        logger.info("[TOOL] USER INFO: Retrieved value")
        if value is None:
            return f"{key}: No data found"
        return f"{key}: {value}"
    except Exception as e:
        logger.info(f"[TOOL] ❌ USER INFO: Error while getting user info {e}.")
        return ""


@tool
def semantic_scholar_search(query: str, store: bool = False) -> List[str]:
    """
    Search Semantic Scholar for scientific papers.

    Use this for:
    - health, nutrition, medical, or scientific questions

    The query should be:
    - short and keyword-based (not a full sentence)
    - include key concepts (e.g., "vitamin D immunity effects dosage")

    Avoid:
    - conversational language
    - unnecessary words
    """
    logger.info("[TOOL] SEMANTIC SCHOLAR: Search requested")
    url = "https://api.semanticscholar.org/graph/v1/paper/search"

    params = {
        "query": query,
        "limit": 5,
        "fields": "title,abstract,year",
        "minCitationCount": 5,
    }
    for attempt in range(2):
        logger.info(f"[TOOL] SEMANTIC SCHOLAR: Request Attempt: {str(attempt)}")
        try:
            response = requests.get(url, params=params, timeout=10)

            if response.status_code != 200:
                logger.info(f"[TOOL] ❌ SEMANTIC SCHOLAR: Bad status: {str(response.status_code)}")
                raise Exception(f"Bad status: {str(response.status_code)}")

            data = response.json()

            paper_texts = []
            for paper in data.get("data", []):
                abstract = paper.get("abstract", "")
                if not abstract:
                    continue
                text = (
                    f"Year of publication:{paper.get('year', '')} "
                    f"Title:{paper.get('title', '')}\n{paper.get('abstract', '')}"
                )
                paper_texts.append(text)

            if store:
                logger.info("[TOOL] SEMANTIC SCHOLAR: Saving findings in knowledge db")
                knowledge_vectorstore = get_core_vectorstore()
                for text in paper_texts:
                    enrich_knowledge(
                        knowledge_vectorstore,
                        source="SemanticScholar",
                        raw_text=text,
                        chunk_size=300,
                        chunk_overlap=0,
                    )
            logger.info(f"[TOOL] ✅ SEMANTIC SCHOLAR: Findings: {len(paper_texts)}")

            return paper_texts

        except Exception as e:
            if attempt == 1:
                logger.error(f"[TOOL] ❌ SEMANTIC SCHOLAR: Error while request: {str(e)}")
                return []
            time.sleep(1)


tools = [calculator, get_current_time, get_current_age, get_user_info, semantic_scholar_search]

tool_metadata = [
    {
        "name": current_tool.name,
        "description": current_tool.description,
    }
    for current_tool in tools
]

bound_models_cache = {}


def select_tools_via_llm(query: str):
    logger.info("[TOOL SELECTOR]: Selecting tools")
    try:
        prompt = format_prompt(
            TOOL_SELECTION_PROMPT,
            query=query,
            tool_metadata=json.dumps(tool_metadata, indent=2),
        )

        raw_output = call_llm_json(prompt, default=[])

        if not isinstance(raw_output, list):
            logger.info("[TOOL SELECTOR]: Output not a list; returning empty list")
            return []

        valid_names = {tool_info["name"] for tool_info in tool_metadata}

        cleaned = list({name for name in raw_output if name in valid_names})
        if not cleaned:
            logger.info("[TOOL SELECTOR]: No valid tool names; returning empty list")
            return []

        tool_map = {current_tool.name: current_tool for current_tool in tools}
        tools_list = [tool_map[name] for name in cleaned]
        logger.info(f"[TOOL SELECTOR]: ✅ Tools selected successfully: returning {str(tools_list)} tools")
        return tools_list

    except Exception as e:
        logger.error(f"[TOOL SELECTOR]: Error while selecting tool: {str(e)}")
        return []


def get_bound_model(selected_tools):
    """Return an LLM bound to the given list of tools, using cache if available."""
    logger.info("[TOOL SELECTOR]: Selecting Bound model.")
    if not selected_tools:
        return get_llm()

    tool_key = tuple(sorted([current_tool.name for current_tool in selected_tools]))

    if tool_key in bound_models_cache:
        return bound_models_cache[tool_key]

    model = get_llm().bind_tools(selected_tools)
    bound_models_cache[tool_key] = model
    logger.info("[TOOL SELECTOR]: Tool Bound LLM model selected.")
    return model
