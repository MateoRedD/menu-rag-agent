"""The three tools the agent can call. Docstrings are what the LLM reads to decide when and how to use them."""
from typing import Literal, NotRequired, TypedDict

from langchain.tools import tool

from menu import price_order
from rag import MIN_SIMILARITY, db, search

NO_RESULTS = "NO_RESULTS: nothing in the approved menu or policies matches this query."


class OrderItem(TypedDict):
    name: str
    quantity: int
    notes: NotRequired[str]


@tool(parse_docstring=True)
def search_menu(query: str) -> str:
    """Search the restaurant's approved menu and policies (dishes, prices, hours, delivery, payments,
    reservations, allergens, location). Use it before answering any question about the restaurant.

    Args:
        query: What to look up, written in Spanish because the menu is in Spanish
            (e.g. "domicilios a Riomar", "postres sin gluten").
    """
    # k=4: list questions ("what fish dishes do you have?") need more than one chunk.
    results = search(query, strategy="item", k=4, min_similarity=MIN_SIMILARITY)
    if not results:
        return NO_RESULTS
    return "\n\n---\n\n".join(r["content"] for r in results)


@tool(parse_docstring=True)
def create_order(
    customer_name: str,
    items: list[OrderItem],
    order_type: Literal["dine_in", "delivery"],
    confirmed: bool,
    address: str | None = None,
) -> dict:
    """Price an order and, once the customer has confirmed it, save it. The total is always computed
    here: never calculate prices yourself. First call with confirmed=false to get the total and show it
    to the customer; call again with confirmed=true only after they explicitly accept it.

    Args:
        customer_name: Name of the customer.
        items: Dishes with their exact menu name and quantity. Use notes for flavors or changes
            (e.g. {"name": "Jugo natural en leche", "quantity": 1, "notes": "lulo"}).
        order_type: "dine_in" to eat at the restaurant or "delivery" for home delivery.
        confirmed: false to get a quote only, true to save the confirmed order.
        address: Required for delivery. Must include the neighborhood.
    """
    try:
        order = price_order(items, order_type, address)
    except ValueError as e:
        return {"error": str(e)}  # returned, not raised, so the LLM can read it and fix the order

    if not confirmed:
        return {"status": "quote_not_saved", **order}

    row = db().table("orders").insert({
        "customer_name": customer_name,
        "items": order["items"],
        "delivery_fee": order["delivery_fee"],
        "total": order["total"],
        "order_type": order_type,
        "address": address,
    }).execute().data[0]
    return {"status": "saved", "order_id": row["id"], **order}


@tool(parse_docstring=True)
def handoff_to_human(
    reason: Literal["complaint", "allergy", "group_booking", "unknown_answer", "other"],
    summary: str,
) -> str:
    """Hand the conversation off to a human. Use it for complaints, severe allergies, group bookings
    (6 or more people), when the customer asks for a person, or when search_menu does not answer the question.

    Args:
        reason: Why a human is needed.
        summary: Short summary of the conversation so the human does not have to ask again.
    """
    row = db().table("handoffs").insert({"reason": reason, "summary": summary}).execute().data[0]
    return f"Handoff #{row['id']} registered. Tell the customer a person from the team will follow up."


TOOLS = [search_menu, create_order, handoff_to_human]
