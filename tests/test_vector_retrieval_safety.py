from datetime import datetime

import pytest
from chromadb.api.types import validate_metadata, validate_where
from langchain_core.documents import Document

from app.db import vectorstores
from app.services import memory, retrieval


class FakeStore:
    def __init__(self, results):
        self.results, self.calls = results, []

    def similarity_search_with_score(self, **kwargs):
        self.calls.append(kwargs)
        return self.results


def doc(text, kind="preference", owner="owner", **metadata):
    return Document(page_content=text, metadata={"type": kind, "user_id": owner, **metadata})


def test_unstructured_retrieval_enforces_owner_and_requested_types():
    store = FakeStore([
        (doc("likes pasta"), 0.1), (doc("avoids pork", "dislike"), 0.2),
        (doc("irrelevant habit", "habit"), 0.0), (doc("foreign preference", owner="other"), 0.0),
        (Document(page_content="legacy missing metadata"), 0.0),
    ])
    results = retrieval.retrieve_unstructured_memory(store, "dinner?", "owner", ["preference", "dislike", "preference", "invented"])
    where = store.calls[0]["filter"]
    assert where == {"$and": [{"user_id": "owner"}, {"type": {"$in": ["preference", "dislike"]}}]}
    validate_where(where)
    assert [item["text"] for item in results] == ["likes pasta", "avoids pork"]


@pytest.mark.parametrize("types", [[], ["invented"], [None, {}]])
def test_no_valid_types_do_not_query_all_memories(types):
    store = FakeStore([])
    assert retrieval.retrieve_unstructured_memory(store, "hello", "owner", types) == []
    assert store.calls == []


def test_raw_distances_and_recency_cost_preserve_ranking(monkeypatch):
    class FixedDateTime(datetime):
        @classmethod
        def now(cls):
            return cls(2026, 5, 4, 12)

    monkeypatch.setattr(retrieval, "datetime", FixedDateTime)
    store = FakeStore([
        (doc("old preference", timestamp="2026-04-01T12:00:00"), 0.2),
        (doc("recent preference", timestamp="2026-05-04T12:00:00"), 0.3),
        (doc("distant preference"), 8.0),
    ])
    result = retrieval.retrieve_unstructured_memory(store, "food", "owner", ["preference"])
    assert [item["text"] for item in result] == ["recent preference", "old preference", "distant preference"]
    assert result[0]["distance"] == 0.3
    assert result[0]["ranking_cost"] == pytest.approx(0.21)
    assert result[-1]["ranking_cost"] > 1
    assert all("score" not in item and "similarity" not in item for item in result)


def test_knowledge_results_are_ordered_by_raw_distance():
    store = FakeStore([(doc("far"), 9.0), (doc("near"), 0.5)])
    results = retrieval.retrieve_knowledge_docs(store, "question", k=1)
    assert [item["text"] for item in results] == ["near"]
    assert results[0]["distance"] == 0.5
    assert "score" not in results[0]


@pytest.mark.parametrize("distance,exists", [(0.0, True), (0.15, True), (0.151, False), (5.0, False)])
def test_duplicate_cutoff_uses_raw_distance(distance, exists):
    store = FakeStore([(doc("candidate"), distance)])
    result = memory.similar_memory_exists(store, "candidate", metadata_filter={"user_id": "owner"})
    assert result["exists"] is exists
    assert store.calls[0]["filter"] == {"user_id": "owner"}
    if exists:
        assert result["distance"] == distance
        assert "similarity" not in result


def test_optional_ingestion_uses_valid_chroma_metadata(tmp_path, monkeypatch):
    captured = []

    class Store:
        def add_documents(self, documents):
            for document in documents:
                validate_metadata(document.metadata)
            captured.extend(documents)

    class Loader:
        def __init__(self, *args, **kwargs):
            pass

        def load(self):
            return [Document(page_content="A simple recipe", metadata={"source": "recipes.txt"})]

    monkeypatch.setattr(vectorstores, "get_embeddings", lambda: object())
    monkeypatch.setattr(vectorstores, "Chroma", lambda **kwargs: Store())
    monkeypatch.setattr(vectorstores, "DirectoryLoader", Loader)
    assert vectorstores.ingest_knowledge(str(tmp_path / "store"), "core_knowledge", str(tmp_path)) is not None
    assert captured[0].metadata["category"] == "recipe"
    assert "tags" not in captured[0].metadata


def test_enrichment_omits_empty_tags(monkeypatch):
    captured = []

    class Store(FakeStore):
        def add_texts(self, **kwargs):
            for metadata in kwargs["metadatas"]:
                validate_metadata(metadata)
                captured.append(metadata)

    monkeypatch.setattr(memory, "extract_knowledge", lambda text: {"summary": text, "category": "general", "tags": []})
    result = memory.enrich_knowledge(Store([]), "test", "Some useful knowledge.", 500, 50)
    assert result.startswith("Stored 1")
    assert "tags" not in captured[0]
