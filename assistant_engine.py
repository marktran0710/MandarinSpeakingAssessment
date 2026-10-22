"""Small, privacy-friendly conversation coach for guided speaking practice.

The local coach is deterministic and works without an API key.  A future provider
can replace this module without changing the browser contract.
"""
from __future__ import annotations

import re


MENUS = {
    "menu_a": {
        "name": "Menu A",
        "items": {
            "burger": "$6 (no onions)",
            "sandwich": "$5 (no cheese)",
            "salad": "$4 (dressing on the side)",
            "orange juice": "$2 (large +$1)",
            "water": "$1",
        },
    },
    "menu_b": {
        "name": "Menu B",
        "items": {
            "chicken wrap": "$6 (sold out)",
            "vegetable sandwich": "$5",
            "apple juice": "$2",
            "water": "$1",
        },
    },
}

HELP_WORDS = re.compile(r"\b(idk|don't know|do not know|help|stuck|不知道|不會|不会|幫我|帮我)\b", re.I)
FOOD_WORDS = {"burger", "sandwich", "salad", "juice", "water", "wrap", "hamburger"}


def _level(history: list[dict], requested: int | None) -> int:
    if requested is not None and int(requested) > 0:
        return max(0, min(4, int(requested)))
    user_messages = [str(item.get("content", "")) for item in history if item.get("role") == "user"]
    if not user_messages:
        return 0
    if any(HELP_WORDS.search(message) for message in user_messages):
        return min(4, 1 + sum(bool(HELP_WORDS.search(message)) for message in user_messages))
    return 0


def _menu_text(menu: dict) -> str:
    return ", ".join(f"{name} {price}" for name, price in menu["items"].items())


def _support(level: int, topic: str) -> dict:
    if level == 0:
        return {"level": 0, "label": "Independent", "prompt": "What would you like to order?"}
    if level == 1:
        return {"level": 1, "label": "Question", "prompt": "Which food or drink would you like?"}
    if level == 2:
        return {"level": 2, "label": "Sentence starter", "prompt": "Try starting with: I’d like…"}
    if level == 3:
        return {"level": 3, "label": "Fill in", "prompt": "I’d like a ___, please. (burger / sandwich)"}
    return {"level": 4, "label": "Example", "prompt": "Example: I’d like a burger, please. Now change it to a sandwich."}


def respond(message: str, history: list[dict] | None = None, topic: str = "restaurant", menu_id: str = "menu_a",
            requested_level: int | None = None, current_sample: str = "", weakest_tone: int | None = None,
            language: str = "en") -> dict:
    history = history or []
    menu = MENUS.get(menu_id, MENUS["menu_a"])
    level = _level(history, requested_level)
    text = str(message or "").strip()
    lowered = text.casefold()
    support = _support(level, topic)
    sold_out = menu_id == "menu_b" and ("chicken wrap" in lowered or "wrap" in lowered)
    asked_price = "price" in lowered or "how much" in lowered or "cost" in lowered
    selected = next((item for item in menu["items"] if item in lowered), None)

    if topic == "free":
        reply = "Tell me one thing you like. You can start with: I like…"
        next_level = max(0, level - 1)
    elif sold_out:
        reply = "Sorry, the chicken wrap is sold out. Would you like a vegetable sandwich or apple juice?"
        next_level = max(level, 1)
    elif asked_price and selected:
        price = menu["items"][selected].split(" ")[0]
        reply = f"The {selected} is {price}. Would you like to order it?"
        next_level = max(0, level - 1)
    elif selected:
        reply = f"Good choice. How many {selected} would you like?"
        next_level = max(0, level - 1)
    elif not text:
        reply = support["prompt"]
        next_level = level
    else:
        reply = "Thanks. Please choose an item from the menu and tell me what you would like."
        next_level = level

    if level > 0 and not selected and not asked_price:
        reply = f"{reply} {support['prompt']}"

    return {
        "reply": reply,
        "support": support,
        "next_support_level": next_level,
        "topic": topic,
        "menu": {"id": menu_id, "name": menu["name"], "items": menu["items"]},
        "coach_note": (
            f"Keep the target tone in mind: T{weakest_tone}."
            if weakest_tone in (1, 2, 3, 4, 5) else ""
        ),
        "language": language,
    }
