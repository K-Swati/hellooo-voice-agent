from datetime import date

import pytest

from clinic_agent.agent import Agent
from clinic_agent.llm import MockLLM
from clinic_agent.scheduling import Calendar

TODAY = date(2026, 11, 16)  # Monday


@pytest.fixture
def calendar():
    return Calendar(TODAY)


@pytest.fixture
def agent(calendar):
    return Agent(MockLLM(TODAY), calendar)
