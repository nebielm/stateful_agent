import uuid
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logging import logger
from app.core.identity import require_user_id
from app.llm.extractors import extract_knowledge
from app.models.memory import ALLOWED_TYPES


def normalize_type(t: str) -> str:
    t = t.lower().strip()
    mapping = {
        "likes": "preference",
        "like": "preference",
        "dislikes": "dislike",
        "hate": "dislike",
        "routine": "habit",
    }
    if t in mapping:
        return mapping[t]
    if t not in ALLOWED_TYPES:
        return "context"
    return t


def similar_memory_exists(
    vectorstore,
    text: str,
    max_distance: float = 0.15,
    k: int = 3,
    metadata_filter: dict | None = None,
):
    """
    Check raw distance against a heuristic duplicate cutoff for our L2 stores.
    This is not a calibrated similarity threshold.
    """
    results = vectorstore.similarity_search_with_score(
        query=text,
        k=k,
        filter=metadata_filter if metadata_filter else None,
    )

    for doc, distance in results:
        if distance <= max_distance:
            return {
                "exists": True,
                "distance": distance,
                "doc": doc.page_content,
            }

    return {"exists": False}


def controlled_unstructured_data_storage(user_vectorstore, text: str, type: str, user_id: str):
    """Stores unstructured memory into Chroma vector DB"""
    require_user_id(user_id)
    logger.info("[MEMORY STORAGE]: Started unstructured user data storage")

    if not text or len(text) < 5:
        logger.info("[MEMORY STORAGE]: IGNORED: Text must be at least 5 characters long.")
        return "ignored: too short"

    similarity = similar_memory_exists(user_vectorstore, text, metadata_filter={"user_id": user_id})
    if similarity.get("exists"):
        logger.info("[MEMORY STORAGE]: IGNORED: Similar memory already exists.")
        return "ignored: similar memory already exists"

    try:
        clean_type = normalize_type(type)
        metadata = {
            "user_id": user_id,
            "type": clean_type,
            "timestamp": datetime.now().isoformat(),
        }

        user_vectorstore.add_texts(
            texts=[text],
            metadatas=[metadata],
            ids=[str(uuid.uuid4())],
        )
        logger.info("[MEMORY STORAGE]: Stored unstructured memory successfully")
        return f"stored {text}, (type: {clean_type}) "

    except Exception as e:
        logger.error(f"[MEMORY STORAGE]: ❌ Error while storing unstructured memory: {str(e)}")


def enrich_knowledge(vectorstore, source: str, raw_text: str, chunk_size: int, chunk_overlap: int):
    """
    Store important new knowledge for future use.
    Only store concise, useful, and reusable information.
    """
    logger.info(f"[MEMORY STORAGE]: Started enriching knowledge based on source: {source}")
    if not raw_text or len(raw_text) < 5:
        logger.info("[MEMORY STORAGE]: IGNORED: Text must be at least 5 characters long.")
        return "ignored: too short"

    try:
        data = extract_knowledge(raw_text)

        summary_text = data.get("summary", raw_text)
        category = data.get("category", "general").lower()
        tags = [t.strip() for t in data.get("tags", [])] if data.get("tags") else []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_text(summary_text)

        stored_count = 0
        for chunk in chunks:
            similarity = similar_memory_exists(vectorstore=vectorstore, text=chunk)
            if similarity.get("exists"):
                continue

            metadata = {
                "category": category,
                "timestamp": datetime.now().isoformat(),
                "source": source,
            }
            if tags:
                metadata["tags"] = tags

            vectorstore.add_texts(
                texts=[chunk],
                metadatas=[metadata],
                ids=[str(uuid.uuid4())],
            )
            stored_count += 1
        logger.info(
            f"[MEMORY STORAGE]: ✅ Enriching knowledge base successfully: Stored {str(stored_count)} new chunk(s) from source '{source}'"
        )
        return f"Stored {stored_count} new chunk(s) from source '{source}'."

    except Exception as e:
        logger.error(f"[MEMORY STORAGE]: ❌ Error while Enriching knowledge base: {str(e)}")
