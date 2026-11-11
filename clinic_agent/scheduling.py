"""Clinic calendar with the business rules an agent must not be able to break.

Rules enforced here (in code, not in a prompt): weekdays only, fixed opening
hours, fixed slot length, no booking in the past, a booking horizon, and no
double booking. Every failure raises :class:`SchedulingError` with a stable
``code`` so callers (and the LLM) get a structured reason.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta


class SchedulingError(Exception):
    def __init__(self, code: str, message: str, alternatives: list[str] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.alternatives = alternatives or []


@dataclass
class Appointment:
    appointment_id: str
    patient_name: str
    day: date
    slot: str  # "HH:MM"
    reason: str = "general visit"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["day"] = self.day.isoformat()
        return d


def _minutes(slot: str) -> int:
    h, m = slot.split(":")
    return int(h) * 60 + int(m)


def normalise_slot(value: str) -> str:
    """Accept ``9:00`` or ``09:00`` and return ``09:00``; raise ``invalid_time`` otherwise."""
    try:
        parsed = datetime.strptime(value.strip(), "%H:%M").time()
    except ValueError:
        raise SchedulingError("invalid_time", f"'{value}' is not a valid time, use HH:MM") from None
    return parsed.strftime("%H:%M")


class Calendar:
    def __init__(
        self,
        today: date,
        opens: str = "09:00",
        closes: str = "17:00",
        slot_minutes: int = 30,
        horizon_days: int = 60,
    ) -> None:
        self.today = today
        self.opens, self.closes = opens, closes
        self.slot_minutes = slot_minutes
        self.horizon_days = horizon_days
        self._appointments: dict[str, Appointment] = {}
        self._counter = 0

    # ---- queries -------------------------------------------------------
    def all_slots(self, day: date) -> list[str]:
        if day.weekday() >= 5:
            return []
        out, t = [], _minutes(self.opens)
        while t + self.slot_minutes <= _minutes(self.closes):
            out.append(f"{t // 60:02d}:{t % 60:02d}")
            t += self.slot_minutes
        return out

    def booked(self, day: date) -> set[str]:
        return {a.slot for a in self._appointments.values() if a.day == day}

    def available(self, day: date) -> list[str]:
        self._check_day(day)
        taken = self.booked(day)
        slots = [s for s in self.all_slots(day) if s not in taken]
        if day == self.today:  # no booking earlier than "now" is modelled; same-day is closed to new bookings
            return []
        return slots

    def find(self, patient_name: str) -> list[Appointment]:
        key = " ".join(patient_name.lower().split())
        hits = [a for a in self._appointments.values() if a.patient_name.lower() == key and a.day >= self.today]
        return sorted(hits, key=lambda a: (a.day, a.slot))

    def get(self, appointment_id: str) -> Appointment:
        try:
            return self._appointments[appointment_id]
        except KeyError:
            raise SchedulingError("not_found", f"no appointment with id {appointment_id}") from None

    def appointments(self) -> list[Appointment]:
        return sorted(self._appointments.values(), key=lambda a: (a.day, a.slot))

    # ---- validation ----------------------------------------------------
    def _check_day(self, day: date) -> None:
        if day <= self.today:
            raise SchedulingError("past_date", "appointments must be booked for a future date")
        if day > self.today + timedelta(days=self.horizon_days):
            raise SchedulingError("too_far_ahead", f"bookings open up to {self.horizon_days} days ahead")
        if day.weekday() >= 5:
            raise SchedulingError("clinic_closed", "the clinic is closed on weekends")

    def _check_slot(self, day: date, slot: str, ignore_id: str | None = None) -> str:
        self._check_day(day)
        slot = normalise_slot(slot)
        if slot not in self.all_slots(day):
            raise SchedulingError(
                "outside_hours",
                f"appointments run from {self.opens} to {self.closes} in {self.slot_minutes}-minute slots",
                self.available(day)[:3],
            )
        taken = {a.slot for a in self._appointments.values() if a.day == day and a.appointment_id != ignore_id}
        if slot in taken:
            free = [s for s in self.all_slots(day) if s not in taken]
            nearest = sorted(free, key=lambda s: abs(_minutes(s) - _minutes(slot)))[:3]
            raise SchedulingError("slot_unavailable", "that slot is already taken", sorted(nearest))
        return slot

    # ---- commands ------------------------------------------------------
    def book(self, patient_name: str, day: date, slot: str, reason: str = "general visit") -> Appointment:
        name = " ".join(patient_name.split())
        if len(name) < 2:
            raise SchedulingError("invalid_name", "a patient name is required")
        slot = self._check_slot(day, slot)
        self._counter += 1
        appt = Appointment(f"A-{self._counter:04d}", name.title(), day, slot, reason or "general visit")
        self._appointments[appt.appointment_id] = appt
        return appt

    def cancel(self, appointment_id: str) -> Appointment:
        appt = self.get(appointment_id)
        del self._appointments[appointment_id]
        return appt

    def reschedule(self, appointment_id: str, day: date, slot: str) -> Appointment:
        appt = self.get(appointment_id)
        appt.slot = self._check_slot(day, slot, ignore_id=appointment_id)
        appt.day = day
        return appt


def parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        raise SchedulingError("invalid_date", f"'{value}' is not a valid date, use YYYY-MM-DD") from None


def clock_time(slot: str) -> time:
    return datetime.strptime(slot, "%H:%M").time()
