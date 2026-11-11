"""Command line: ``python -m clinic_agent chat|simulate|serve``."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date

from clinic_agent.agent import new_agent
from clinic_agent.llm import get_llm
from clinic_agent.simulator import SIM_TODAY, run_suite, summarise, write_report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="clinic_agent", description="Voice appointment-scheduling agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("chat", help="talk to the agent in the terminal (type 'quit' to stop)")
    sim = sub.add_parser("simulate", help="run the scripted caller scenarios")
    sim.add_argument("--reports", default="reports")
    sv = sub.add_parser("serve", help="run the HTTP API")
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    if args.cmd == "chat":
        agent = new_agent(get_llm(date.today()))
        print(f"({agent.llm.name}) Riverside Clinic. How can I help?")
        while not agent.ended:
            try:
                line = input("you> ")
            except EOFError:
                break
            if line.strip().lower() in {"quit", "exit"}:
                break
            print("agent>", agent.turn(line).reply)
        return 0

    if args.cmd == "simulate":
        results = run_suite(lambda: get_llm(SIM_TODAY))
        print(write_report(results, args.reports, os.environ.get("LLM", "mock")))
        return 0 if summarise(results)["task_success_rate"] == 1.0 else 1

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("clinic_agent.api:app", host=args.host, port=args.port)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
