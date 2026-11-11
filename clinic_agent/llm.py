"""LLM interface with two implementations: OpenAI function calling and an offline test double."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Protocol

from clinic_agent.guardrails import CONFIRM_MARKER, is_affirmative, is_negative


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON string, exactly as an LLM API returns it


@dataclass
class LLMReply:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLM(Protocol):
    name: str

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMReply: ...


# --------------------------------------------------------------------------- OpenAI
class OpenAILLM:
    """Chat Completions with function calling. ``client`` is injectable for tests."""

    def __init__(self, model: str | None = None, client: Any = None) -> None:
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.name = f"openai:{self.model}"
        if client is None:
            from openai import OpenAI  # imported lazily so offline use needs no package or key

            client = OpenAI()
        self.client = client

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMReply:
        wire = [{k: v for k, v in m.items() if not (m["role"] == "tool" and k == "name")} for m in messages]
        resp = self.client.chat.completions.create(model=self.model, messages=wire, tools=tools, temperature=0)
        msg = resp.choices[0].message
        calls = [ToolCall(c.id, c.function.name, c.function.arguments) for c in (msg.tool_calls or [])]
        return LLMReply(content=msg.content, tool_calls=calls)


def get_llm(today: date | None = None) -> LLM:
    kind = os.environ.get("LLM", "mock").lower()
    if kind == "openai":
        return OpenAILLM()
    if kind == "mock":
        return MockLLM(today or date.today())
    raise ValueError(f"LLM must be 'mock' or 'openai', got {kind!r}")


# --------------------------------------------------------------------------- Mock
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
_MEDICAL_ADVICE = re.compile(
    r"should i take|which (?:medication|medicine|drug)|what (?:medication|medicine)|dosage|\bdose\b"
    r"|side effects?|diagnos|what(?:'s| is) wrong|is it safe|can i take",
    re.I,
)
_REASONS = ("check-up", "checkup", "follow-up", "vaccination", "flu shot", "consultation", "blood test")


def detect_intent(text: str) -> str | None:
    t = text.lower()
    if re.search(r"\b(reschedul\w*|move|postpone|push)\b", t):
        return "reschedule"
    if re.search(r"\b(cancel\w*|call off)\b", t):
        return "cancel"
    if re.search(r"\b(book\w*|schedul\w*|appointment|slot|see (?:the|a) doctor|come in)\b", t):
        return "book"
    return None


def parse_date(text: str, today: date) -> date | None:
    t = text.lower()
    if m := re.search(r"\b(\d{4}-\d{2}-\d{2})\b", t):
        try:
            return date.fromisoformat(m.group(1))
        except ValueError:
            return None
    if "day after tomorrow" in t:
        return today + timedelta(days=2)
    if "tomorrow" in t:
        return today + timedelta(days=1)
    if re.search(r"\btoday\b", t):
        return today
    if m := re.search(r"\b(" + "|".join(_WEEKDAYS) + r")\b", t):
        delta = (_WEEKDAYS.index(m.group(1)) - today.weekday()) % 7
        return today + timedelta(days=delta or 7)
    month = "(" + "|".join(_MONTHS) + r")[a-z]*\.?"
    m = re.search(rf"\b{month}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", t) or None
    day_first = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{month}\b", t)
    if m or day_first:
        mon, day = (m.group(1), m.group(2)) if m else (day_first.group(2), day_first.group(1))
        try:
            d = date(today.year, _MONTHS.index(mon) + 1, int(day))
        except ValueError:
            return None
        return d if d > today else d.replace(year=today.year + 1)
    return None


def parse_time(text: str) -> str | None:
    t = text.lower()
    if m := re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)", t):
        h, mins, mer = int(m.group(1)), int(m.group(2) or 0), m.group(3)[0]
        if not 1 <= h <= 12:
            return None
        h = (h % 12) + (12 if mer == "p" else 0)
        return f"{h:02d}:{mins:02d}"
    if m := re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", t):
        return f"{int(m.group(1)):02d}:{m.group(2)}"
    if "noon" in t:
        return "12:00"
    return None


_NAME_CUE = re.compile(
    r"(?i:my name is|this is|i am|i'm|name's|name is)\s+([A-Z][a-z'\-]+(?:\s+[A-Z][a-z'\-]+)+)"
)
_BARE_NAME = re.compile(r"^[A-Za-z][A-Za-z'\-]+(?: [A-Za-z][A-Za-z'\-]+){1,2}\.?$")


def extract_name(text: str, prev_assistant: str) -> str | None:
    if m := _NAME_CUE.search(text):
        return m.group(1).title()
    if "full name" in prev_assistant.lower() and _BARE_NAME.match(text.strip()):
        return text.strip().rstrip(".").title()
    return None


@dataclass
class _State:
    intent: str | None = None
    name: str | None = None
    day: date | None = None
    time: str | None = None
    reason: str = "general visit"
    last_user: str = ""
    last_assistant: str = ""
    results: dict[tuple[str, str], dict] = field(default_factory=dict)


class MockLLM:
    """Deterministic, rule-based stand-in for an LLM.

    It re-derives the conversation state from the full message history on every call (like a real
    stateless chat API) and emits the same message shapes, including tool calls. It exists so that
    the agent loop, guardrails, tools, voice layer and simulator can be tested offline. It is a test
    double, not a model: it understands a narrow set of phrasings and its simulator scores say nothing
    about how well a real LLM would do.
    """

    name = "mock"

    def __init__(self, today: date) -> None:
        self.today = today
        self._ids = 0

    # ---- public --------------------------------------------------------
    def chat(self, messages: list[dict], tools: list[dict]) -> LLMReply:
        s = self._parse(messages)
        last = messages[-1]
        if last["role"] == "tool":
            if last["name"] == "transfer_to_staff":
                return self._say("I am passing you to a member of our staff now. Thank you for calling.")
            if last["name"] in {"book_appointment", "cancel_appointment", "reschedule_appointment"}:
                return self._say(self._write_result(json.loads(last["content"])))
        return self._decide(s)

    # ---- helpers -------------------------------------------------------
    def _say(self, text: str) -> LLMReply:
        return LLMReply(content=text)

    def _call(self, name: str, **args: Any) -> LLMReply:
        self._ids += 1
        return LLMReply(tool_calls=[ToolCall(f"call_{self._ids}", name, json.dumps(args))])

    def _parse(self, messages: list[dict]) -> _State:
        s, pending, prev_assistant = _State(), {}, ""
        for m in messages:
            if m["role"] == "user":
                text = m["content"]
                s.intent = s.intent or detect_intent(text)
                s.name = extract_name(text, prev_assistant) or s.name
                s.day = parse_date(text, self.today) or s.day
                s.time = parse_time(text) or s.time
                for r in _REASONS:
                    if r in text.lower():
                        s.reason = r
                s.last_user = text
            elif m["role"] == "assistant":
                for tc in m.get("tool_calls") or []:
                    pending[tc["id"]] = (tc["function"]["name"], json.loads(tc["function"]["arguments"]))
                if m.get("content"):
                    prev_assistant = s.last_assistant = m["content"]
            elif m["role"] == "tool":
                name, args = pending.get(m["tool_call_id"], (m.get("name", ""), {}))
                key = args.get("date") or args.get("patient_name", "").lower()
                s.results[(name, key)] = json.loads(m["content"])
        return s

    def _decide(self, s: _State) -> LLMReply:
        confirming = CONFIRM_MARKER in s.last_assistant
        if confirming and is_negative(s.last_user):
            return self._say("No problem. I have not changed anything. Is there anything else I can help with?")
        if _MEDICAL_ADVICE.search(s.last_user):
            return self._call("transfer_to_staff", reason="caller asked for medical advice")
        if s.intent is None:
            if re.search(r"\b(hi|hello|hey)\b", s.last_user.lower()):
                return self._say("Hello, I can book, move or cancel appointments. What would you like to do?")
            return self._say(
                "I can only help with booking, moving or cancelling appointments. "
                "Is there an appointment I can help you with?"
            )
        if not s.name:
            return self._say("Happy to help. May I have your full name?")

        if s.intent == "book":
            plan = self._plan_book(s)
        else:
            plan = self._plan_existing(s)
        if isinstance(plan, LLMReply):
            return plan
        tool, args, summary = plan
        if confirming and is_affirmative(s.last_user):
            return self._call(tool, **args)
        return self._say(f"{summary} {CONFIRM_MARKER}")

    def _availability(self, s: _State) -> LLMReply | list[str]:
        """Return a reply that still needs to be sent, or the list of free slots for the chosen day."""
        if s.day is None:
            return self._say("What day would you like?")
        res = s.results.get(("check_availability", s.day.isoformat()))
        if res is None:
            return self._call("check_availability", date=s.day.isoformat())
        if not res["ok"]:
            s.day = None
            return self._say(f"I cannot book that day: {res['message']}. What other day works for you?")
        return res["available"]

    def _time_choice(self, s: _State, free: list[str]) -> LLMReply | None:
        if s.time is None:
            shown = ", ".join(free[:6]) or "none"
            return self._say(f"On {s.day:%A %Y-%m-%d} we have openings at {shown}. What time suits you?")
        if s.time not in free:
            h, m = s.time.split(":")
            target = int(h) * 60 + int(m)
            near = sorted(free, key=lambda x: abs(int(x[:2]) * 60 + int(x[3:]) - target))[:3]
            return self._say(
                f"Sorry, {s.time} is not available on {s.day:%Y-%m-%d}. "
                f"The closest free times are {', '.join(sorted(near))}. Which would you like?"
            )
        return None

    def _plan_book(self, s: _State):
        free = self._availability(s)
        if isinstance(free, LLMReply):
            return free
        if reply := self._time_choice(s, free):
            return reply
        args = {"patient_name": s.name, "date": s.day.isoformat(), "time": s.time, "reason": s.reason}
        return "book_appointment", args, f"To confirm: a {s.reason} for {s.name} on {s.day:%A %Y-%m-%d} at {s.time}."

    def _plan_existing(self, s: _State):
        res = s.results.get(("find_appointment", s.name.lower()))
        if res is None:
            return self._call("find_appointment", patient_name=s.name)
        if not res["appointments"]:
            return self._say(f"I could not find an upcoming appointment for {s.name}. Could you check the spelling?")
        appt = res["appointments"][0]
        current = f"{appt['appointment_id']} on {appt['day']} at {appt['slot']}"
        if s.intent == "cancel":
            return "cancel_appointment", {"appointment_id": appt["appointment_id"]}, f"I will cancel appointment {current}."
        if s.day is None:
            return self._say(f"I found appointment {current}. What new day would you like?")
        free = self._availability(s)
        if isinstance(free, LLMReply):
            return free
        if reply := self._time_choice(s, free):
            return reply
        args = {"appointment_id": appt["appointment_id"], "date": s.day.isoformat(), "time": s.time}
        return "reschedule_appointment", args, f"I will move appointment {appt['appointment_id']} to {s.day:%A %Y-%m-%d} at {s.time}."

    @staticmethod
    def _write_result(res: dict) -> str:
        if not res["ok"]:
            alt = res.get("alternatives") or []
            extra = f" The closest free times are {', '.join(alt)}." if alt else ""
            if res["error"] == "confirmation_required":
                return f"I need your confirmation first. {CONFIRM_MARKER}"
            return f"I could not complete that: {res['message']}.{extra} Would you like to try something else?"
        if "cancelled" in res:
            a = res["cancelled"]
            return f"Done. Appointment {a['appointment_id']} on {a['day']} at {a['slot']} is cancelled. Anything else?"
        a = res["appointment"]
        return f"All set. Appointment {a['appointment_id']} for {a['patient_name']} is on {a['day']} at {a['slot']}. Anything else?"
