import base64
from types import SimpleNamespace

from fastapi.testclient import TestClient

from clinic_agent.agent import Agent
from clinic_agent.api import create_app
from clinic_agent.llm import MockLLM, OpenAILLM
from clinic_agent.voice import (
    OpenAISynthesizer,
    OpenAITranscriber,
    TextSynthesizer,
    TextTranscriber,
    VoiceSession,
    to_speech_text,
)


def test_to_speech_text():
    out = to_speech_text("Appointment A-0001 is on 2026-11-17 at 14:30 and at 09:00. **Thanks**")
    assert out == "Appointment A 0 0 0 1 is on Tuesday, November 17 at 2:30 PM and at 9 AM. Thanks"
    assert to_speech_text("2026-13-45") == "2026-13-45"


def test_voice_session_round_trip(agent):
    vs = VoiceSession(agent, TextTranscriber(), TextSynthesizer())
    turn = vs.handle(b"Hi, I'm Maria Lopez and I'd like to book an appointment tomorrow at 10am")
    assert turn.transcript.startswith("Hi, I'm Maria") and "Is that correct?" in turn.reply_text
    assert "Tuesday, November 17" in turn.spoken_text and turn.audio == turn.spoken_text.encode()
    assert vs.handle(b"yes").reply_text.startswith("All set")
    assert agent.calendar.appointments()[0].patient_name == "Maria Lopez"


def test_silence_gets_reprompt_without_calling_llm(agent):
    turn = VoiceSession(agent, TextTranscriber(), TextSynthesizer()).handle(b"   ")
    assert "did not catch" in turn.reply_text and len(agent.messages) == 1


def test_openai_speech_wrappers_use_client():
    seen = {}
    client = SimpleNamespace(
        audio=SimpleNamespace(
            transcriptions=SimpleNamespace(create=lambda **kw: seen.update(stt=kw) or SimpleNamespace(text="hello")),
            speech=SimpleNamespace(create=lambda **kw: seen.update(tts=kw) or SimpleNamespace(content=b"mp3")),
        )
    )
    assert OpenAITranscriber(client=client).transcribe(b"wav") == "hello"
    assert OpenAISynthesizer(client=client).synthesize("hi") == b"mp3"
    assert seen["stt"]["model"] == "whisper-1" and seen["tts"]["input"] == "hi"


def test_openai_llm_parses_tool_calls_and_strips_tool_name():
    sent = {}

    def create(**kw):
        sent.update(kw)
        call = SimpleNamespace(id="c1", function=SimpleNamespace(name="check_availability", arguments='{"date":"2026-11-17"}'))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[call]))])

    llm = OpenAILLM(model="m", client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    reply = llm.chat(
        [{"role": "user", "content": "hi"}, {"role": "tool", "tool_call_id": "x", "name": "t", "content": "{}"}], []
    )
    assert reply.tool_calls[0].name == "check_availability" and reply.content is None
    assert "name" not in sent["messages"][1] and sent["temperature"] == 0


def test_http_flow(tmp_path):
    client = TestClient(create_app(llm_factory=lambda: MockLLM(__import__("datetime").date.today())))
    assert client.get("/health").json()["status"] == "ok"
    sid = client.post("/sessions").json()["session_id"]
    r = client.post(f"/sessions/{sid}/chat", json={"text": "What's the weather like?"}).json()
    assert "appointments" in r["reply"] and not r["handoff"]
    r = client.post(f"/sessions/{sid}/chat", json={"text": "I have chest pain"}).json()
    assert r["handoff"]
    assert client.post("/sessions/nope/chat", json={"text": "hi"}).status_code == 404
    assert client.post(f"/sessions/{sid}/chat", json={"text": ""}).status_code == 422
    assert client.get(f"/sessions/{sid}/appointments").json() == {"appointments": []}


def test_http_voice():
    client = TestClient(create_app(llm_factory=lambda: MockLLM(__import__("datetime").date.today())))
    sid = client.post("/sessions").json()["session_id"]
    r = client.post(f"/sessions/{sid}/voice", content=b"hello there").json()
    assert r["transcript"] == "hello there" and base64.b64decode(r["audio_base64"]).decode() == r["spoken_text"]
    assert client.post(f"/sessions/{sid}/voice", content=b"").status_code == 400
    assert client.post(f"/sessions/{sid}/voice", content=b"x" * (5 * 1024 * 1024 + 1)).status_code == 413


def test_agent_class_importable():
    assert Agent
