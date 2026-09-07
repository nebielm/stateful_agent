import os
import shutil
from datetime import datetime

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logging import logger
from app.core import settings
from app.llm.client import get_embeddings


_UNINITIALIZED = object()
_core_vectorstore = _UNINITIALIZED
_user_memory_vectorstore = _UNINITIALIZED


def ingest_knowledge(
    persist_dir: str,
    collection_name: str,
    doc_folder: str | None = None,
    force_rebuild: bool | None = False,
):
    """Open a local store; optional .txt ingestion also requires unstructured."""
    logger.info(f"[System]: Start {collection_name} Knowledge base creation")
    try:
        if force_rebuild and os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)

        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=get_embeddings(),
            persist_directory=persist_dir,
        )

        if doc_folder and os.path.exists(doc_folder) and collection_name == settings.CORE_KNOWLEDGE_COLLECTION:
            loader = DirectoryLoader(doc_folder, glob="**/*.txt")
            documents = loader.load()

            if documents:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50,
                )
                chunks = text_splitter.split_documents(documents)

                for chunk in chunks:
                    source = chunk.metadata.get("source", "").lower()
                    if "recipes" in source:
                        category = "recipe"
                    elif "nutrition" in source:
                        category = "nutrition"
                    elif "safety" in source:
                        category = "safety"
                    else:
                        category = "general"
                    chunk.metadata = {
                        "category": category,
                        "timestamp": datetime.now().isoformat(),
                        "source": "core",
                    }

                vectorstore.add_documents(chunks)

        logger.info(f"[System]:✅ {collection_name} Knowledge base created successfully")
        return vectorstore
    except Exception as e:
        logger.error(f"[System]: ❌ Error while creating {collection_name} Knowledge base:{str(e)}")


def get_core_vectorstore():
    global _core_vectorstore

    if _core_vectorstore is _UNINITIALIZED:
        _core_vectorstore = ingest_knowledge(
            persist_dir=settings.CORE_KNOWLEDGE_DIR,
            collection_name=settings.CORE_KNOWLEDGE_COLLECTION,
        )

    return _core_vectorstore


def get_user_memory_vectorstore():
    global _user_memory_vectorstore

    if _user_memory_vectorstore is _UNINITIALIZED:
        _user_memory_vectorstore = ingest_knowledge(
            persist_dir=settings.USER_MEMORY_DIR,
            collection_name=settings.USER_MEMORY_COLLECTION,
        )

    return _user_memory_vectorstore


def runtime_context():
    return {
        "knowledge_vectorstore": get_core_vectorstore(),
        "user_vectorstore": get_user_memory_vectorstore(),
    }
