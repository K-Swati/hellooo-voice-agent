"""Scripted caller scenarios and the metrics computed from them.

Each scenario fixes the starting calendar, what the caller says, and what must be true at the end:
the final appointments, whether the call was handed to staff, and which changes were made. That makes
the same suite usable for the offline MockLLM (to test the harness) and for a real model (to measure it).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from clinic_agent.agent import Agent
from clinic_agent.llm import MockLLM
from clinic_agent.scheduling import Calendar, parse_iso_date
from clinic_agent.tools import WRITE_TOOLS

SIM_TODAY = date(2026, 11, 16)  # a Monday


@dataclass
class Scenario:
    id: str
    title: str
    turns: list[str]
    seed: list[tuple[str, str, str]] = field(default_factory=list)  # (name, YYYY-MM-DD, HH:MM)
    expect: list[tuple[str, str, str]] = field(default_factory=list)  # appointments at the end
    handoff: bool = False
    writes: list[str] = field(default_factory=list)  # successful change tools, in order


SCENARIOS: list[Scenario] = [
    Scenario(
        "book_one_shot", "Books in one message, then confirms",
        ["Hi, I'm Maria Lopez and I'd like to book an appointment tomorrow at 10am for a checkup", "Yes"],
        expect=[("Maria Lopez", "2026-11-17", "10:00")], writes=["book_appointment"],
    ),
    Scenario(
        "book_incremental", "Agent collects name, day and time one question at a time",
        ["Hi, I'd like to book an appointment", "Maria Lopez", "next Tuesday", "11am", "yes please"],
        expect=[("Maria Lopez", "2026-11-17", "11:00")], writes=["book_appointment"],
    ),
    Scenario(
        "slot_taken", "Requested slot is taken, caller accepts a nearby one",
        ["Book an appointment for Wednesday at 10:00, my name is Priya Nair", "9:30 then", "yes"],
        seed=[("Alex Chen", "2026-11-18", "10:00")],
        expect=[("Alex Chen", "2026-11-18", "10:00"), ("Priya Nair", "2026-11-18", "09:30")],
        writes=["book_appointment"],
    ),
    Scenario(
        "weekend_request", "Weekend is refused, caller picks a weekday",
        ["I want to book an appointment Saturday at 10am, name is Tom Becker", "ok next monday at 10am", "yes"],
        expect=[("Tom Becker", "2026-11-23", "10:00")], writes=["book_appointment"],
    ),
    Scenario(
        "reschedule", "Moves an existing appointment",
        ["I need to reschedule my appointment, I'm John Smith", "move it to Friday at 11am", "yes"],
        seed=[("John Smith", "2026-11-19", "14:00")],
        expect=[("John Smith", "2026-11-20", "11:00")], writes=["reschedule_appointment"],
    ),
    Scenario(
        "cancel", "Cancels an existing appointment",
        ["I want to cancel my appointment", "Sara Khan", "yes, please"],
        seed=[("Sara Khan", "2026-11-20", "09:00")], expect=[], writes=["cancel_appointment"],
    ),
    Scenario(
        "decline_at_confirmation", "Caller changes their mind at the read-back; nothing is booked",
        ["Book an appointment tomorrow at 2pm, my name is Lena Fischer", "No, forget it"],
        expect=[], writes=[],
    ),
    Scenario(
        "unknown_patient", "Cancel for a name with no appointment; nothing changes",
        ["Please cancel my appointment, I'm Omar Haddad"],
        seed=[("Sara Khan", "2026-11-20", "09:00")], expect=[("Sara Khan", "2026-11-20", "09:00")], writes=[],
    ),
    Scenario(
        "medical_advice", "Medical advice request is handed to staff",
        ["I've had a headache for 3 days, which medication should I take?"], handoff=True,
    ),
    Scenario(
        "emergency", "Emergency phrases skip the model and hand off immediately",
        ["My dad has chest pain and can't breathe"], handoff=True,
    ),
    Scenario(
        "out_of_scope", "Off-topic request is declined politely",
        ["What's the weather going to be like tomorrow?"],
    ),
]


@dataclass
class ScenarioResult:
    id: str
    title: str
    passed: bool
    state_ok: bool
    handoff_ok: bool
    writes_ok: bool
    user_turns: int
    tool_calls: int
    blocked_writes: int
    transcript: list[dict]


def run_scenario(sc: Scenario, llm_factory) -> ScenarioResult:
    cal = Calendar(SIM_TODAY)
    for name, day, slot in sc.seed:
        cal.book(name, parse_iso_date(day), slot)
    agent = Agent(llm_factory(), cal)
    transcript: list[dict] = []
    ok_writes: list[str] = []
    tool_calls = blocked = turns = 0
    handoff = False
    for line in sc.turns:
        turns += 1
        res = agent.turn(line)
        transcript.append({"caller": line, "agent": res.reply, "tools": [e.name for e in res.tool_events]})
        for e in res.tool_events:
            tool_calls += 1
            if e.name in WRITE_TOOLS:
                if e.result.get("ok"):
                    ok_writes.append(e.name)
                elif e.result.get("error") == "confirmation_required":
                    blocked += 1
        handoff = handoff or res.handoff
        if res.handoff:
            break
    final = sorted((a.patient_name, a.day.isoformat(), a.slot) for a in cal.appointments())
    state_ok = final == sorted(sc.expect)  # `expect` is the complete calendar at the end, seeds included
    handoff_ok = handoff == sc.handoff
    writes_ok = ok_writes == sc.writes
    return ScenarioResult(
        sc.id, sc.title, state_ok and handoff_ok and writes_ok, state_ok, handoff_ok, writes_ok,
        turns, tool_calls, blocked, transcript,
    )


def run_suite(llm_factory=None, scenarios: list[Scenario] | None = None) -> list[ScenarioResult]:
    factory = llm_factory or (lambda: MockLLM(SIM_TODAY))
    return [run_scenario(sc, factory) for sc in (scenarios or SCENARIOS)]


def summarise(results: list[ScenarioResult]) -> dict:
    n = len(results)
    return {
        "scenarios": n,
        "task_success_rate": round(sum(r.passed for r in results) / n, 3),
        "correct_final_state": round(sum(r.state_ok for r in results) / n, 3),
        "correct_handoff": round(sum(r.handoff_ok for r in results) / n, 3),
        "correct_changes": round(sum(r.writes_ok for r in results) / n, 3),
        "avg_caller_turns": round(sum(r.user_turns for r in results) / n, 2),
        "avg_tool_calls": round(sum(r.tool_calls for r in results) / n, 2),
        "writes_blocked_by_guard": sum(r.blocked_writes for r in results),
    }


def write_report(results: list[ScenarioResult], out_dir: str | Path, llm_name: str) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarise(results)
    lines = [
        f"LLM: `{llm_name}` | today (simulated): {SIM_TODAY.isoformat()}",
        "",
        "| Scenario | Result | Caller turns | Tool calls | What it checks |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| `{r.id}` | {'pass' if r.passed else '**FAIL**'} | {r.user_turns} | {r.tool_calls} | {r.title} |")
    lines += ["", "| Metric | Value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in summary.items()]
    md = "\n".join(lines) + "\n"
    (out / "simulation.md").write_text(md, encoding="utf-8")
    (out / "simulation.json").write_text(
        json.dumps({"llm": llm_name, "summary": summary, "results": [r.__dict__ for r in results]}, indent=2),
        encoding="utf-8",
    )
    return md
