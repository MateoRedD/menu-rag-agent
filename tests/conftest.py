"""Shared fixtures."""
from types import SimpleNamespace

import pytest

import tools


@pytest.fixture
def inserted(monkeypatch):
    """Fake the Supabase writes in tools.py: records every insert as (table, row) and returns id 1.
    Reads (rag.search) are not affected, so integration tests still retrieve from the real menu."""
    rows = []

    def table(name):
        def insert(row):
            rows.append((name, row))
            return SimpleNamespace(execute=lambda: SimpleNamespace(data=[{"id": 1, **row}]))
        return SimpleNamespace(insert=insert)

    monkeypatch.setattr(tools, "db", lambda: SimpleNamespace(table=table))
    return rows
