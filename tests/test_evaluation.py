from query_expansion.contracts import Completion, Document, Record
from query_expansion.evaluate import evaluate
from query_expansion.retriever import BM25


def test_full_corpus_evaluation_does_not_inject_positive(make_trainer, monkeypatch):
    bundle, _ = make_trainer()
    # All scores tie; the positive sorts beyond top 100. Candidate injection would hide this.
    docs = [Document(f"a{i:03d}", "en", "unrelated") for i in range(110)]
    docs.append(Document("z-positive", "en", "unrelated"))
    index = BM25(docs)
    monkeypatch.setattr(
        "query_expansion.evaluate.generate", lambda *args, **kwargs: Completion([2], "", False)
    )
    result = evaluate(
        bundle.model,
        bundle.tokenizer,
        [Record("eval", "query", "en", positive_doc_ids=("z-positive",))],
        index,
    )
    row = result["queries"][0]
    assert row["baseline"]["recall@100"] == 0
    assert row["expanded"]["ndcg@10"] == 0
    assert row["fallback"]
