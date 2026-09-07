import uuid

from langchain_core.messages import AIMessage, HumanMessage

from app.core.logging import logger
from app.core.identity import get_user_id
from app.db.vectorstores import runtime_context
from app.llm.client import get_llm
from app.repositories.user_memory import load_user_data
from app.services.graph import app


def chat():
    user_id = get_user_id()
    load_user_data()
    get_llm()
    runtime_context()

    state = {
        "request_id": str(uuid.uuid4()),
        "user_id": user_id,
        "messages": [],
        "memory_updates": {
            "structured": [],
            "unstructured": [],
        },
        "context": {},
    }

    while True:
        request_id = str(uuid.uuid4())
        state["request_id"] = request_id
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        state["messages"].append(HumanMessage(content=user_input))

        result = app.invoke(input=state, context=runtime_context())

        state = result

        human_msg = next(
            (message for message in reversed(state["messages"]) if isinstance(message, HumanMessage)),
            None,
        )
        ai_msg = next(
            (message for message in reversed(state["messages"]) if isinstance(message, AIMessage)),
            None,
        )
        if human_msg is None or ai_msg is None:
            logger.error(f"[{str(state['request_id'])}] Missing chat messages after graph execution.")
            continue

        logger.info(f"[{str(state['request_id'])}] Chat turn completed")
        print(f"\nAI: {ai_msg.content}")
