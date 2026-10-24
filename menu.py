"""Prices and order totals, computed in code (never by the LLM).

Prices are parsed from data/menu.md, so the menu stays the single source of truth.
"""
import re
import unicodedata
from pathlib import Path

from rag import chunk_by_item

MENU_PATH = Path(__file__).parent / "data" / "menu.md"

# Delivery policy from the "Domicilios" section of menu.md (tests check they still match).
DELIVERY_FEE = 5000
FREE_DELIVERY_OVER = 80000  # "gratis en pedidos mayores a 80.000 COP" -> strictly greater
DELIVERY_ZONES = ["Riomar", "Alto Prado", "Villa Country", "El Golf"]
MAX_QUANTITY = 20  # larger orders are unusual enough to go to a human


def normalize(text: str) -> str:
    """Lowercase, strip accents and extra spaces: 'Carimañolas ' -> 'carimanolas'."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return " ".join(text.lower().split())


def parse_prices(text: str) -> dict[str, int]:
    """Map each dish to its price in COP. A dish with variants gets one entry per variant:
    'Precio: 6.000 COP en agua, 7.500 COP en leche' -> 'Jugo natural en agua', 'Jugo natural en leche'.
    """
    prices = {}
    for chunk in chunk_by_item(text):
        header, *lines = chunk.splitlines()
        name = header.split(" > ", 1)[1]
        price_line = next((line for line in lines if line.startswith("Precio:")), None)
        if price_line is None:  # policies (hours, delivery...) have no price
            continue
        for amount, variant in re.findall(r"([\d.]+) COP(?: en (\w+))?", price_line):
            key = f"{name} en {variant}" if variant else name
            prices[key] = int(amount.replace(".", ""))
    return prices


PRICES = parse_prices(MENU_PATH.read_text(encoding="utf-8"))
_BY_NORMALIZED_NAME = {normalize(name): name for name in PRICES}


def price_order(items: list[dict], order_type: str, address: str | None = None) -> dict:
    """Validate an order and compute its total.

    items: [{"name": "Arepa de huevo", "quantity": 2, "notes": "optional"}]
    Raises ValueError with a message the agent can show or act on.
    """
    if not items:
        raise ValueError("The order has no items.")
    if order_type not in ("dine_in", "delivery"):
        raise ValueError("order_type must be 'dine_in' or 'delivery'.")

    lines = []
    for item in items:
        name = _BY_NORMALIZED_NAME.get(normalize(item["name"]))
        if name is None:
            raise ValueError(f"'{item['name']}' is not on the menu. Valid items: {', '.join(PRICES)}.")
        quantity = item["quantity"]
        if not 1 <= quantity <= MAX_QUANTITY:
            raise ValueError(f"Quantity for '{name}' must be between 1 and {MAX_QUANTITY}.")
        lines.append({"name": name, "quantity": quantity, "unit_price": PRICES[name], "notes": item.get("notes")})

    subtotal = sum(line["unit_price"] * line["quantity"] for line in lines)

    delivery_fee = 0
    if order_type == "delivery":
        if not address or not any(normalize(zone) in normalize(address) for zone in DELIVERY_ZONES):
            raise ValueError(f"Delivery address must include one of our zones: {', '.join(DELIVERY_ZONES)}.")
        delivery_fee = 0 if subtotal > FREE_DELIVERY_OVER else DELIVERY_FEE

    return {"items": lines, "delivery_fee": delivery_fee, "total": subtotal + delivery_fee}
