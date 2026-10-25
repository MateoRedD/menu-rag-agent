"""Unit tests for price parsing and order totals. No API calls."""
from datetime import date

import pytest

from menu import DELIVERY_FEE, DELIVERY_ZONES, FREE_DELIVERY_OVER, MENU_PATH, PRICES, WEEKEND_ONLY, price_order

TUESDAY, SATURDAY = date(2026, 10, 20), date(2026, 10, 24)


def order(*items):
    return [{"name": name, "quantity": qty} for name, qty in items]


# ---------- Parsing ----------

def test_parses_every_dish_and_skips_policies():
    assert len(PRICES) == 16  # 15 dishes, juice split into two variants
    assert PRICES["Arepa de huevo"] == 7000
    assert "Horario de atención" not in PRICES


def test_dish_with_two_prices_becomes_two_entries():
    assert PRICES["Jugo natural en agua"] == 6000
    assert PRICES["Jugo natural en leche"] == 7500
    assert "Jugo natural" not in PRICES


def test_delivery_constants_still_match_the_menu():
    menu = MENU_PATH.read_text(encoding="utf-8")
    assert f"{DELIVERY_FEE:,}".replace(",", ".") + " COP" in menu
    assert f"mayores a {FREE_DELIVERY_OVER:,}".replace(",", ".") in menu
    assert all(zone in menu for zone in DELIVERY_ZONES)
    for dish in WEEKEND_ONLY:
        assert "Solo sábados y domingos" in menu.split(f"## {dish}")[1].split("##")[0]


# ---------- Totals ----------

def test_dine_in_total_has_no_delivery_fee():
    result = price_order(order(("Arepa de huevo", 2), ("Limonada de coco", 1)), "dine_in")
    assert result["delivery_fee"] == 0
    assert result["total"] == 2 * 7000 + 9000


@pytest.mark.parametrize("items, expected_fee", [
    (order(("Arepa de huevo", 1)), DELIVERY_FEE),                                                  # 7.000
    (order(("Mojarra frita", 1), ("Posta negra cartagenera", 1), ("Jugo natural en agua", 1)), DELIVERY_FEE),  # exactly 80.000
    (order(("Mojarra frita", 3)), 0),                                                              # 114.000
])
def test_delivery_fee_is_free_only_above_80000(items, expected_fee):
    result = price_order(items, "delivery", address="Calle 80 # 52-10, Riomar")
    assert result["delivery_fee"] == expected_fee


def test_names_match_ignoring_case_and_accents():
    result = price_order(order(("CARIMANOLAS", 1), ("ceviche de camaron", 1)), "dine_in")
    assert [line["name"] for line in result["items"]] == ["Carimañolas", "Ceviche de camarón"]


def test_sancocho_only_on_weekends():
    with pytest.raises(ValueError, match="Saturdays and Sundays"):
        price_order(order(("Sancocho de gallina", 1)), "dine_in", today=TUESDAY)
    assert price_order(order(("Sancocho de gallina", 1)), "dine_in", today=SATURDAY)["total"] == 30000


def test_notes_are_kept():
    result = price_order([{"name": "Jugo natural en leche", "quantity": 1, "notes": "lulo"}], "dine_in")
    assert result["items"][0]["notes"] == "lulo"


# ---------- Validation ----------

@pytest.mark.parametrize("items, order_type, address, error", [
    (order(("Sushi", 1)), "dine_in", None, "not on the menu"),
    (order(("Jugo natural", 1)), "dine_in", None, "not on the menu"),  # must say agua or leche
    (order(("Arepa de huevo", 0)), "dine_in", None, "between 1 and"),
    (order(("Arepa de huevo", 21)), "dine_in", None, "between 1 and"),
    ([], "dine_in", None, "no items"),
    (order(("Arepa de huevo", 1)), "takeaway", None, "order_type"),
    (order(("Arepa de huevo", 1)), "delivery", None, "zones"),
    (order(("Arepa de huevo", 1)), "delivery", "Calle 30, Soledad", "zones"),
])
def test_invalid_orders_are_rejected(items, order_type, address, error):
    with pytest.raises(ValueError, match=error):
        price_order(items, order_type, address)
