"""Tool definitions (OpenAI function-calling schema) and a validating executor."""

from __future__ import annotations

from typing import Any

from clinic_agent.scheduling import Calendar, SchedulingError, parse_iso_date

WRITE_TOOLS = frozenset({"book_appointment", "cancel_appointment", "reschedule_appointment"})

TOOL_SPECS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "List the free appointment times on a date.",
            "parameters": {
                "type": "object",
                "properties": {"date": {"type": "string", "description": "YYYY-MM-DD"}},
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_appointment",
            "description": "Find a patient's upcoming appointments by full name.",
            "parameters": {
                "type": "object",
                "properties": {"patient_name": {"type": "string"}},
                "required": ["patient_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Book an appointment. Only call after the patient confirmed the exact details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {"type": "string"},
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                    "time": {"type": "string", "description": "HH:MM, 24-hour"},
                    "reason": {"type": "string"},
                },
                "required": ["patient_name", "date", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_appointment",
            "description": "Cancel an appointment by id. Only call after the patient confirmed.",
            "parameters": {
                "type": "object",
                "properties": {"appointment_id": {"type": "string"}},
                "required": ["appointment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reschedule_appointment",
            "description": "Move an appointment to a new date and time. Only call after the patient confirmed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "appointment_id": {"type": "string"},
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                    "time": {"type": "string", "description": "HH:MM, 24-hour"},
                },
                "required": ["appointment_id", "date", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_to_staff",
            "description": "Hand the call to a human. Use for medical questions or anything outside scheduling.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
        },
    },
]


class ToolExecutor:
    """Runs a tool call against the calendar and always returns a JSON-serialisable dict."""

    def __init__(self, calendar: Calendar) -> None:
        self.calendar = calendar
        self.handoff_reason: str | None = None

    def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        handler = getattr(self, f"_{name}", None)
        if handler is None:
            return {"ok": False, "error": "unknown_tool", "message": f"no tool named {name!r}"}
        try:
            return handler(**args)
        except SchedulingError as exc:
            return {"ok": False, "error": exc.code, "message": exc.message, "alternatives": exc.alternatives}
        except TypeError as exc:  # missing or unexpected arguments from the model
            return {"ok": False, "error": "bad_arguments", "message": str(exc)}

    def _check_availability(self, date: str) -> dict:
        day = parse_iso_date(date)
        return {"ok": True, "date": day.isoformat(), "available": self.calendar.available(day)}

    def _find_appointment(self, patient_name: str) -> dict:
        return {"ok": True, "appointments": [a.to_dict() for a in self.calendar.find(patient_name)]}

    def _book_appointment(self, patient_name: str, date: str, time: str, reason: str = "general visit") -> dict:
        appt = self.calendar.book(patient_name, parse_iso_date(date), time, reason)
        return {"ok": True, "appointment": appt.to_dict()}

    def _cancel_appointment(self, appointment_id: str) -> dict:
        return {"ok": True, "cancelled": self.calendar.cancel(appointment_id).to_dict()}

    def _reschedule_appointment(self, appointment_id: str, date: str, time: str) -> dict:
        appt = self.calendar.reschedule(appointment_id, parse_iso_date(date), time)
        return {"ok": True, "appointment": appt.to_dict()}

    def _transfer_to_staff(self, reason: str) -> dict:
        self.handoff_reason = reason
        return {"ok": True, "handoff": True}
