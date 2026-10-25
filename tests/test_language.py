"""Unit tests for the reply-language detection. No API calls."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent import detect_language, reply_language


@pytest.mark.parametrize("text, expected", [
    ("Do you deliver to Riomar? How much is it?", "English"),
    ("Is there parking?", "English"),
    ("I want 2 arepas de huevo please", "English"),  # dish names in Spanish don't flip it
    ("¿Tienen sushi?", "Spanish"),
    ("Quiero dos arepas para domicilio", "Spanish"),
    ("hola", "Spanish"),
    ("ok", None),
    ("2", None),
])
def test_detect_language(text, expected):
    assert detect_language(text) == expected


def test_short_reply_keeps_the_previous_language():
    messages = [HumanMessage("Do you deliver to Riomar?"), AIMessage("Yes, we do. Confirm?"), HumanMessage("ok")]
    assert reply_language(messages) == "English"


def test_latest_message_wins_when_customer_switches():
    messages = [HumanMessage("Do you deliver to Riomar?"), AIMessage("Yes."), HumanMessage("¿Y cuánto cuesta?")]
    assert reply_language(messages) == "Spanish"


def test_defaults_to_spanish():
    assert reply_language([HumanMessage("ok")]) == "Spanish"
