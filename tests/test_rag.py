"""Unit tests for the similarity threshold in rag.search. OpenAI and Supabase are faked."""
from types import SimpleNamespace

import rag

ROWS = [
    {"content": "Domicilios", "similarity": 0.42},
    {"content": "Medios de pago", "similarity": 0.31},
    {"content": "Carimañolas", "similarity": 0.12},
]


def fake_backend(monkeypatch, rows):
    """Replace embed() and the Supabase RPC so search() runs offline."""
    monkeypatch.setattr(rag, "embed", lambda texts: [[0.0] * 1536])
    rpc = lambda name, params: SimpleNamespace(execute=lambda: SimpleNamespace(data=rows))
    monkeypatch.setattr(rag, "db", lambda: SimpleNamespace(rpc=rpc))


def test_default_keeps_all_results(monkeypatch):
    fake_backend(monkeypatch, ROWS)
    assert len(rag.search("Do you deliver to Riomar?")) == 3


def test_drops_results_below_threshold(monkeypatch):
    fake_backend(monkeypatch, ROWS)
    results = rag.search("Do you deliver to Riomar?", min_similarity=0.20)
    assert [r["content"] for r in results] == ["Domicilios", "Medios de pago"]


def test_returns_empty_when_nothing_passes(monkeypatch):
    fake_backend(monkeypatch, ROWS)
    assert rag.search("What's the weather like?", min_similarity=0.50) == []
