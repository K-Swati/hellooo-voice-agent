"""Speech in and out. Offline mode treats 'audio' as UTF-8 text so everything is testable without a model."""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol

from clinic_agent.agent import Agent


class Transcriber(Protocol):
    def transcribe(self, audio: bytes) -> str: ...


class Synthesizer(Protocol):
    def synthesize(self, text: str) -> bytes: ...


class TextTranscriber:
    """Offline stand-in: the 'audio' is already UTF-8 text."""

    def transcribe(self, audio: bytes) -> str:
        return audio.decode("utf-8", errors="replace")


class TextSynthesizer:
    def synthesize(self, text: str) -> bytes:
        return text.encode("utf-8")


class OpenAITranscriber:
    def __init__(self, model: str = "whisper-1", client: Any = None) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client, self.model = client, model

    def transcribe(self, audio: bytes) -> str:
        buf = io.BytesIO(audio)
        buf.name = "caller.wav"
        return self.client.audio.transcriptions.create(model=self.model, file=buf).text


class OpenAISynthesizer:
    def __init__(self, model: str = "tts-1", voice: str = "alloy", client: Any = None) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client, self.model, self.voice = client, model, voice

    def synthesize(self, text: str) -> bytes:
        resp = self.client.audio.speech.create(model=self.model, voice=self.voice, input=text, response_format="mp3")
        return resp.content


ELEVENLABS_URL = "https://api.elevenlabs.io/v1"


class ElevenLabsSynthesizer:
    """Text to speech through the ElevenLabs REST API. ``client`` (httpx.Client) is injectable for tests."""

    def __init__(self, api_key: str | None = None, voice_id: str | None = None, model: str = "eleven_turbo_v2_5",
                 client: Any = None) -> None:
        import httpx

        self.api_key = api_key or os.environ["ELEVENLABS_API_KEY"]
        self.voice_id = voice_id or os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        self.model = model
        self.client = client or httpx.Client(timeout=30)

    def synthesize(self, text: str) -> bytes:
        resp = self.client.post(
            f"{ELEVENLABS_URL}/text-to-speech/{self.voice_id}",
            headers={"xi-api-key": self.api_key},
            json={"text": text, "model_id": self.model},
        )
        resp.raise_for_status()
        return resp.content


class ElevenLabsTranscriber:
    """Speech to text with ElevenLabs Scribe."""

    def __init__(self, api_key: str | None = None, model: str = "scribe_v1", client: Any = None) -> None:
        import httpx

        self.api_key = api_key or os.environ["ELEVENLABS_API_KEY"]
        self.model = model
        self.client = client or httpx.Client(timeout=60)

    def transcribe(self, audio: bytes) -> str:
        resp = self.client.post(
            f"{ELEVENLABS_URL}/speech-to-text",
            headers={"xi-api-key": self.api_key},
            data={"model_id": self.model},
            files={"file": ("caller.wav", audio)},
        )
        resp.raise_for_status()
        return resp.json()["text"]


def get_speech() -> tuple[Transcriber, Synthesizer]:
    kind = os.environ.get("SPEECH", "text").lower()
    if kind == "openai":
        return OpenAITranscriber(), OpenAISynthesizer()
    if kind == "elevenlabs":
        return ElevenLabsTranscriber(), ElevenLabsSynthesizer()
    if kind == "text":
        return TextTranscriber(), TextSynthesizer()
    raise ValueError(f"SPEECH must be 'text', 'openai' or 'elevenlabs', got {kind!r}")


# ---- making text sound right when read aloud ---------------------------------
def _spoken_time(m: re.Match) -> str:
    h, mins = int(m.group(1)), int(m.group(2))
    suffix = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12} {suffix}" if mins == 0 else f"{h12}:{mins:02d} {suffix}"


def _spoken_date(m: re.Match) -> str:
    try:
        d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return m.group(0)
    return f"{d:%A}, {d:%B} {d.day}"


def _spoken_id(m: re.Match) -> str:
    return "A " + " ".join(m.group(1))


def to_speech_text(text: str) -> str:
    """Turn ``2026-11-17`` into "Tuesday, November 17", ``14:30`` into "2:30 PM", ``A-0001`` into "A 0 0 0 1"."""
    text = re.sub(r"[*#_`]", "", text)
    text = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", _spoken_date, text)
    text = re.sub(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", _spoken_time, text)
    text = re.sub(r"\bA-(\d{4})\b", _spoken_id, text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class VoiceTurn:
    transcript: str
    reply_text: str
    spoken_text: str
    audio: bytes
    handoff: bool


class VoiceSession:
    """One phone call: audio in -> transcript -> agent -> speakable text -> audio out."""

    def __init__(self, agent: Agent, stt: Transcriber, tts: Synthesizer) -> None:
        self.agent, self.stt, self.tts = agent, stt, tts

    def handle(self, audio: bytes) -> VoiceTurn:
        transcript = self.stt.transcribe(audio).strip()
        if not transcript:
            reply = "Sorry, I did not catch that. Could you say it again?"
            spoken = reply
            return VoiceTurn("", reply, spoken, self.tts.synthesize(spoken), False)
        result = self.agent.turn(transcript)
        spoken = to_speech_text(result.reply)
        return VoiceTurn(transcript, result.reply, spoken, self.tts.synthesize(spoken), result.handoff)
