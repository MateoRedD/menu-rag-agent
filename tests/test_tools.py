"""Unit tests for the agent tools. OpenAI and Supabase are faked, so nothing is written to the database."""
from types import SimpleNamespace

import pytest

import tools


@pytest.fixture
def inserted(monkeypatch):
    """Fake Supabase client: records every insert as (table, row) and returns id 1."""
    rows = []

    def table(name):
        def insert(row):
            rows.append((name, row))
            return SimpleNamespace(execute=lambda: SimpleNamespace(data=[{"id": 1, **row}]))
        return SimpleNamespace(insert=insert)

    monkeypatch.setattr(tools, "db", lambda: SimpleNamespace(table=table))
    return rows


def arepas(quantity=2):
    return [{"name": "arepa de huevo", "quantity": quantity}]


# ---------- search_menu ----------

def test_search_menu_joins_chunks(monkeypatch):
    monkeypatch.setattr(tools, "search", lambda *a, **kw: [{"content": "Domicilios..."}, {"content": "Medios de pago..."}])
    assert tools.search_menu.invoke({"query": "domicilios"}) == "Domicilios...\n\n---\n\nMedios de pago..."


def test_search_menu_says_when_nothing_matches(monkeypatch):
    monkeypatch.setattr(tools, "search", lambda *a, **kw: [])
    assert tools.search_menu.invoke({"query": "clima de mañana"}) == tools.NO_RESULTS


def test_search_menu_uses_item_strategy_and_threshold(monkeypatch):
    calls = []
    monkeypatch.setattr(tools, "search", lambda *a, **kw: calls.append(kw) or [])
    tools.search_menu.invoke({"query": "sushi"})
    assert calls == [{"strategy": "item", "k": 4, "min_similarity": tools.MIN_SIMILARITY}]


# ---------- create_order ----------

def test_quote_is_not_saved(inserted):
    result = tools.create_order.invoke(
        {"customer_name": "Ana", "items": arepas(), "order_type": "dine_in", "confirmed": False})
    assert result["status"] == "quote_not_saved"
    assert result["total"] == 14000
    assert inserted == []


def test_confirmed_order_is_saved_with_total_from_code(inserted):
    result = tools.create_order.invoke({
        "customer_name": "Ana", "items": arepas(), "order_type": "delivery",
        "confirmed": True, "address": "Calle 80 # 52-10, Riomar"})
    assert result["status"] == "saved" and result["order_id"] == 1
    table, row = inserted[0]
    assert table == "orders"
    assert (row["total"], row["delivery_fee"], row["order_type"]) == (19000, 5000, "delivery")
    assert row["items"][0] == {"name": "Arepa de huevo", "quantity": 2, "unit_price": 7000, "notes": None}


def test_invalid_order_returns_error_and_saves_nothing(inserted):
    result = tools.create_order.invoke(
        {"customer_name": "Ana", "items": [{"name": "sushi", "quantity": 1}], "order_type": "dine_in", "confirmed": True})
    assert "not on the menu" in result["error"]
    assert inserted == []


# ---------- handoff_to_human ----------

def test_handoff_is_saved(inserted):
    message = tools.handoff_to_human.invoke({"reason": "allergy", "summary": "Severe peanut allergy, asks about traces."})
    assert inserted == [("handoffs", {"reason": "allergy", "summary": "Severe peanut allergy, asks about traces."})]
    assert "Handoff #1" in message


def test_handoff_rejects_unknown_reason(inserted):
    with pytest.raises(ValueError, match="literal_error"):  # pydantic's ValidationError subclasses ValueError
        tools.handoff_to_human.invoke({"reason": "angry", "summary": "..."})
    assert inserted == []
