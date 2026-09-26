import pytest

from query_expansion.agents import postprocess
from query_expansion.data_loader import Document, Query
from query_expansion.evaluation import evaluate, retrieval_metrics
from query_expansion.retriever import BM25


def test_full_corpus_does_not_inject_positives():
    docs = [Document(f"a{i:03d}", "en", "noise") for i in range(110)]
    docs.append(Document("z-positive", "en", "noise"))
    report = evaluate(
        [Query("q", "query", "en", split="test", qrels={"z-positive": 1})],
        BM25(docs),
        lambda q, lang: (postprocess(q, "", False), 3),
        split="test",
    )
    row = report["queries"][0]
    assert row["baseline"]["recall@100"] == 0
    assert row["expanded"]["ndcg@10"] == 0
    assert row["fallback"]
    assert report["scope"] == "full_corpus"


def test_graded_metrics_and_duplicate_document_ids():
    metrics = retrieval_metrics(["a", "a", "b"], {"a": 2, "b": 1})
    assert metrics == {"ndcg@10": 1, "mrr@10": 1, "recall@100": 1}
    with pytest.raises(ValueError, match="positive"):
        retrieval_metrics([], {})
