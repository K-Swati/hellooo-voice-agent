"""FastAPI: text chat and voice endpoints, one in-memory calendar and agent per session."""

from __future__ import annotations

import base64
import os
import uuid
from datetime import date

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from clinic_agent.agent import Agent
from clinic_agent.llm import LLM, get_llm
from clinic_agent.scheduling import Calendar
from clinic_agent.voice import Synthesizer, Transcriber, VoiceSession, get_speech

MAX_SESSIONS = 200
MAX_AUDIO_BYTES = 5 * 1024 * 1024


class ChatRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


class ChatResponse(BaseModel):
    reply: str
    tools: list[str]
    handoff: bool


def create_app(llm_factory=None, stt: Transcriber | None = None, tts: Synthesizer | None = None) -> FastAPI:
    app = FastAPI(title="Voice appointment agent", version="1.0.0")
    sessions: dict[str, Agent] = {}
    default_stt, default_tts = get_speech()
    stt, tts = stt or default_stt, tts or default_tts

    def make_llm() -> LLM:
        return llm_factory() if llm_factory else get_llm(date.today())

    def session(session_id: str) -> Agent:
        if session_id not in sessions:
            raise HTTPException(404, "unknown session")
        return sessions[session_id]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "sessions": len(sessions), "llm": os.environ.get("LLM", "mock")}

    @app.post("/sessions", status_code=201)
    def create_session() -> dict:
        if len(sessions) >= MAX_SESSIONS:
            raise HTTPException(503, "too many active sessions")
        sid = uuid.uuid4().hex
        sessions[sid] = Agent(make_llm(), Calendar(date.today()))
        return {"session_id": sid}

    @app.post("/sessions/{session_id}/chat", response_model=ChatResponse)
    def chat(session_id: str, req: ChatRequest) -> ChatResponse:
        res = session(session_id).turn(req.text)
        return ChatResponse(reply=res.reply, tools=[e.name for e in res.tool_events], handoff=res.handoff)

    @app.post("/sessions/{session_id}/voice")
    async def voice(session_id: str, request: Request) -> dict:
        agent = session(session_id)
        audio = await request.body()
        if not audio:
            raise HTTPException(400, "send audio bytes in the request body")
        if len(audio) > MAX_AUDIO_BYTES:
            raise HTTPException(413, "audio too large")
        turn = VoiceSession(agent, stt, tts).handle(audio)
        return {
            "transcript": turn.transcript,
            "reply": turn.reply_text,
            "spoken_text": turn.spoken_text,
            "audio_base64": base64.b64encode(turn.audio).decode(),
            "handoff": turn.handoff,
        }

    @app.get("/sessions/{session_id}/appointments")
    def appointments(session_id: str) -> dict:
        return {"appointments": [a.to_dict() for a in session(session_id).calendar.appointments()]}

    return app


app = create_app()
