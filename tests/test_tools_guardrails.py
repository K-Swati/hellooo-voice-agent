import pytest

from clinic_agent.guardrails import is_affirmative, is_emergency, is_negative
from clinic_agent.tools import ToolExecutor


def test_executor_success_and_structured_errors(calendar):
    ex = ToolExecutor(calendar)
    ok = ex.execute("book_appointment", {"patient_name": "Ann Lee", "date": "2026-11-17", "time": "10:00"})
    assert ok["ok"] and ok["appointment"]["appointment_id"] == "A-0001"
    dup = ex.execute("book_appointment", {"patient_name": "Bo Ray", "date": "2026-11-17", "time": "10:00"})
    assert dup["error"] == "slot_unavailable" and dup["alternatives"]
    assert ex.execute("check_availability", {"date": "17/11/2026"})["error"] == "invalid_date"
    assert ex.execute("check_availability", {})["error"] == "bad_arguments"
    assert ex.execute("check_availability", {"date": "2026-11-17", "extra": 1})["error"] == "bad_arguments"
    assert ex.execute("drop_table", {})["error"] == "unknown_tool"


def test_transfer_sets_handoff(calendar):
    ex = ToolExecutor(calendar)
    ex.execute("transfer_to_staff", {"reason": "billing"})
    assert ex.handoff_reason == "billing"


@pytest.mark.parametrize("text", ["yes", "Yes, please", "yeah that works", "ok", "Correct.", "go ahead", "sounds good"])
def test_affirmative(text):
    assert is_affirmative(text)


@pytest.mark.parametrize("text", ["no", "No, forget it", "wait", "please cancel my appointment", "hmm", ""])
def test_not_affirmative(text):
    assert not is_affirmative(text)


def test_negative():
    assert is_negative("nope") and is_negative("never mind") and not is_negative("yes")


@pytest.mark.parametrize("text", ["I have chest pain", "he is not breathing", "this is a medical emergency", "call an ambulance"])
def test_emergency_detected(text):
    assert is_emergency(text)


@pytest.mark.parametrize("text", ["this is not an emergency, I just need a checkup", "book me for tuesday", "it isn't an emergency"])
def test_emergency_not_triggered(text):
    assert not is_emergency(text)
