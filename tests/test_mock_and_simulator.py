from datetime import date

import pytest

from clinic_agent.llm import parse_date, parse_time
from clinic_agent.simulator import SCENARIOS, run_suite, summarise, write_report

TODAY = date(2026, 11, 16)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("tomorrow", date(2026, 11, 17)),
        ("day after tomorrow", date(2026, 11, 18)),
        ("on Friday", date(2026, 11, 20)),
        ("next monday", date(2026, 11, 23)),
        ("2026-12-01", date(2026, 12, 1)),
        ("november 25th", date(2026, 11, 25)),
        ("3 dec", date(2026, 12, 3)),
        ("january 5", date(2027, 1, 5)),
        ("whenever", None),
    ],
)
def test_parse_date(text, expected):
    assert parse_date(text, TODAY) == expected


@pytest.mark.parametrize(
    "text,expected",
    [("10am", "10:00"), ("2:30 pm", "14:30"), ("12 am", "00:00"), ("12pm", "12:00"), ("14:00", "14:00"),
     ("noon", "12:00"), ("at 10", None), ("13pm", None)],
)
def test_parse_time(text, expected):
    assert parse_time(text) == expected


@pytest.mark.parametrize("sc", SCENARIOS, ids=lambda s: s.id)
def test_every_scenario_passes_with_mock(sc):
    (res,) = run_suite(scenarios=[sc])
    assert res.passed, res.transcript


def test_summary_and_report(tmp_path):
    results = run_suite()
    s = summarise(results)
    assert s["scenarios"] == len(SCENARIOS) and s["task_success_rate"] == 1.0
    md = write_report(results, tmp_path, "mock")
    assert "book_one_shot" in md and (tmp_path / "simulation.json").exists()


def test_simulator_detects_failure():
    """The suite must be able to fail: an LLM that never books should score below 1."""
    from clinic_agent.llm import LLMReply

    class Lazy:
        name = "lazy"

        def chat(self, messages, tools):
            return LLMReply(content="Please call back later.")

    s = summarise(run_suite(lambda: Lazy()))
    assert s["task_success_rate"] < 0.5
