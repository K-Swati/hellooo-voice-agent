from datetime import date

import pytest

from clinic_agent.scheduling import SchedulingError

TUE, WED, SAT = date(2026, 11, 17), date(2026, 11, 18), date(2026, 11, 21)


def code(fn, *a):
    with pytest.raises(SchedulingError) as e:
        fn(*a)
    return e.value


def test_book_and_list(calendar):
    a = calendar.book("maria lopez", TUE, "9:00")
    assert (a.patient_name, a.slot, a.appointment_id) == ("Maria Lopez", "09:00", "A-0001")
    assert "09:00" not in calendar.available(TUE)


def test_double_booking_rejected_with_alternatives(calendar):
    calendar.book("A One", TUE, "10:00")
    err = code(calendar.book, "B Two", TUE, "10:00")
    assert err.code == "slot_unavailable" and "10:00" not in err.alternatives and err.alternatives


@pytest.mark.parametrize(
    "day,slot,expected",
    [
        (date(2026, 11, 15), "10:00", "past_date"),
        (date(2026, 11, 16), "10:00", "past_date"),  # same day is closed to new bookings
        (SAT, "10:00", "clinic_closed"),
        (TUE, "08:00", "outside_hours"),
        (TUE, "17:00", "outside_hours"),
        (TUE, "10:15", "outside_hours"),
        (TUE, "noon", "invalid_time"),
        (date(2027, 6, 1), "10:00", "too_far_ahead"),
    ],
)
def test_rules(calendar, day, slot, expected):
    assert code(calendar.book, "Ann Lee", day, slot).code == expected


def test_last_slot_ends_at_closing(calendar):
    assert calendar.all_slots(TUE)[-1] == "16:30"
    assert calendar.book("Ann Lee", TUE, "16:30").slot == "16:30"


def test_cancel_and_reschedule(calendar):
    a = calendar.book("Ann Lee", TUE, "10:00")
    calendar.reschedule(a.appointment_id, WED, "11:00")
    assert (a.day, a.slot) == (WED, "11:00") and calendar.available(TUE)[0] == "09:00"
    assert "10:00" in calendar.available(TUE)
    calendar.cancel(a.appointment_id)
    assert calendar.appointments() == []
    assert code(calendar.cancel, a.appointment_id).code == "not_found"


def test_reschedule_to_own_slot_is_allowed(calendar):
    a = calendar.book("Ann Lee", TUE, "10:00")
    assert calendar.reschedule(a.appointment_id, TUE, "10:00").slot == "10:00"


def test_reschedule_into_taken_slot_keeps_original(calendar):
    calendar.book("Bob Ray", WED, "10:00")
    a = calendar.book("Ann Lee", TUE, "10:00")
    assert code(calendar.reschedule, a.appointment_id, WED, "10:00").code == "slot_unavailable"
    assert (a.day, a.slot) == (TUE, "10:00")


def test_find_is_case_insensitive_and_future_only(calendar):
    calendar.book("Ann Lee", TUE, "10:00")
    assert len(calendar.find("  ANN   lee ")) == 1
    assert calendar.find("Nobody Here") == []


def test_name_required(calendar):
    assert code(calendar.book, " ", TUE, "10:00").code == "invalid_name"
