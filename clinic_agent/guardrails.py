"""Safety rules that live in code, so they hold even when the model misbehaves."""

from __future__ import annotations

import re

CONFIRM_MARKER = "Is that correct?"

_EMERGENCY = re.compile(
    r"chest pain|can'?t breathe|cannot breathe|trouble breathing|difficulty breathing|not breathing"
    r"|unconscious|heart attack|\bstroke\b|severe bleeding|bleeding (?:a lot|heavily)|overdos|suicid|kill myself"
    r"|medical emergency|ambulance|(?<!not )(?<!n't )\ban emergency",
    re.I,
)
_AFFIRMATIVE_WORDS = {
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "correct", "confirm", "confirmed",
    "absolutely", "definitely", "right",
}
_AFFIRMATIVE_PHRASES = ("that's right", "that is correct", "go ahead", "sounds good", "do it", "that works")
_NEGATIVE_WORDS = {"no", "nope", "nah", "stop", "wait", "don't", "dont"}
_NEGATIVE_PHRASES = ("forget it", "never mind", "nevermind", "cancel that", "not right", "wrong")

EMERGENCY_REPLY = (
    "This sounds like an emergency. Please hang up and call your local emergency number right now. "
    "I am alerting our staff as well."
)


def is_emergency(text: str) -> bool:
    return bool(_EMERGENCY.search(text))


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def is_negative(text: str) -> bool:
    words = _words(text)
    low = text.lower()
    return bool(words) and (words[0] in _NEGATIVE_WORDS or any(p in low for p in _NEGATIVE_PHRASES))


def is_affirmative(text: str) -> bool:
    """True when the caller clearly agrees. Used as a hard gate before any booking change."""
    if is_negative(text):
        return False
    words = _words(text)
    low = text.lower().strip()
    return bool(words) and (words[0] in _AFFIRMATIVE_WORDS or any(low.startswith(p) for p in _AFFIRMATIVE_PHRASES))
