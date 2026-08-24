"""Provider abstraction: one neutral message format, three backends.

WHY THIS FILE IS SHAPED LIKE THIS

Gemini and Groq disagree about almost everything superficial:

                        Gemini                      Groq (OpenAI shape)
  assistant role name   "model"                     "assistant"
  system prompt         a separate config field     a message with role "system"
  tool schema           {"function_declarations":[]} [{"type":"function",...}]
  tool result           a Part with function_response  a message with role "tool"
  tool call id          none, you invent one        provided
  arguments             already a dict              a JSON *string* you parse

None of that is interesting, and none of it should reach agent.py. So this file
defines ONE neutral message format, and each provider translates in and out of
it. The agent talks only in the neutral format.

    NEUTRAL FORMAT
      {"role": "user",      "content": "text"}
      {"role": "assistant", "content": "text", "tool_calls": [ToolCall, ...]}
      {"role": "tool",      "tool_call_id": "...", "name": "...", "content": {...}}

    The system prompt is NOT a message. It is an argument to chat(), because
    Gemini treats it as configuration rather than conversation.

This is the whole argument against writing your loop directly against a vendor
SDK: Weekend 4 needs to run the identical eval against both providers, and that
is only possible because everything above disappears below this line.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import requests

from radreport.config import CACHE_DIR, GEMINI_API_KEY, GROQ_API_KEY


# ---------------------------------------------------------------------------
# Neutral types
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    # Opaque vendor data that must be echoed back verbatim when this call is
    # replayed in the conversation history. The neutral layer deliberately does
    # not know what is in here. Gemini 3.x puts a `thought_signature` in it and
    # rejects the whole request with a 400 if it comes back missing; other
    # providers use nothing. Keeping it opaque means adding the next provider's
    # equivalent needs no change above this line.
    provider_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    model: str = ""
    cached: bool = False

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMProvider(Protocol):
    name: str
    model: str

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             system: str = "") -> LLMResponse:
        ...


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------
# ~25 lines that stop eval reruns burning a daily free-tier quota. Keyed on the
# full request, so any change to the prompt, the tools or the history is a miss.

def _serialisable(messages: list[dict]) -> list[dict]:
    """Neutral messages contain ToolCall dataclasses; make them JSON-safe."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [asdict(c) for c in m["tool_calls"]]
        out.append(m)
    return out


# Bump whenever the SHAPE of what we cache changes. Without this, a stored entry
# written by an older version of the code is silently replayed into new code
# that expects more fields. Cost me a confusing debug loop on 2026-08-21: after
# teaching the Gemini path to round-trip thought_signature, the retry kept
# failing with the identical 400, because the cache was serving pre-fix entries
# that had no signature in them. The bug was fixed and invisible at the same time.
CACHE_VERSION = 2


def _cache_key(provider: str, model: str, messages: list[dict],
               tools: list[dict] | None, system: str) -> str:
    blob = json.dumps([CACHE_VERSION, provider, model, _serialisable(messages),
                       tools, system], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def _cache_read(key: str) -> LLMResponse | None:
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return LLMResponse(
        text=d["text"],
        tool_calls=[ToolCall(**c) for c in d["tool_calls"]],
        input_tokens=d.get("input_tokens", 0),
        output_tokens=d.get("output_tokens", 0),
        model=d.get("model", ""),
        latency_s=0.0,
        cached=True,
    )


def _cache_write(key: str, resp: LLMResponse) -> None:
    (CACHE_DIR / f"{key}.json").write_text(json.dumps({
        "version": CACHE_VERSION,
        "text": resp.text,
        "tool_calls": [asdict(c) for c in resp.tool_calls],
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
        "model": resp.model,
    }))


CACHE_ENABLED = os.getenv("RADREPORT_CACHE", "1") != "0"


# ---------------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------------
# Pinned to explicit versions, NOT to the "-latest" aliases, even though the
# aliases exist and are tempting. An eval table that says "gemini-flash-latest
# scored 0.86" is worthless three weeks later when the alias points somewhere
# else: you cannot tell whether a change in your numbers came from your code or
# from under you. Pin, record the pin in the results, and bump deliberately.
#
# Checked against both APIs' model lists on 2026-08-21. The previous defaults
# (gemini-2.0-flash, llama-3.3-70b-versatile) had both been retired.
# gemini-3.7-flash exists in the model list but returned 504 DEADLINE_EXCEEDED
# on every probe (2026-08-21), as did the gemini-flash-latest alias that points
# at it. 3.6 is what the API's own retirement notice recommends and it responds
# in ~2s. Retest before bumping.
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

# Every outbound call gets an explicit timeout. A hung LLM call hangs the whole
# agent loop, exactly as a hung tool would -- and unlike a tool, it is the thing
# every single iteration depends on. Found the hard way: the first version had a
# timeout on the PubMed call and none here, and a 504-ing model wedged the run.
LLM_TIMEOUT_S = 60

# Free tiers rate-limit hard, and an eval sweep is exactly the workload that
# trips them. A 429 is NOT a system failure and must not be scored as one, so it
# is retried here rather than surfacing to the harness as a broken run.
RATE_LIMIT_RETRIES = 5
RATE_LIMIT_BASE_DELAY = 4.0     # seconds; doubles each attempt


def _retry_delay_from_error(exc: Exception) -> float | None:
    """Honour the server's own retryDelay hint when it gives one."""
    match = re.search(r"retryDelay['\"]?[:\s]+['\"]?(\d+(?:\.\d+)?)s", str(exc))
    return float(match.group(1)) if match else None


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "rate limit" in text.lower()


def with_rate_limit_retry(fn, label: str = ""):
    """Call fn(), backing off on 429 until the budget runs out."""
    delay = RATE_LIMIT_BASE_DELAY
    for attempt in range(RATE_LIMIT_RETRIES):
        try:
            return fn()
        except Exception as exc:
            if not _is_rate_limit(exc) or attempt == RATE_LIMIT_RETRIES - 1:
                raise
            wait = _retry_delay_from_error(exc) or delay
            print(f"[{label} rate-limited, waiting {wait:.0f}s "
                  f"({attempt + 1}/{RATE_LIMIT_RETRIES})]", file=sys.stderr, flush=True)
            time.sleep(wait)
            delay *= 2


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str = DEFAULT_GEMINI_MODEL, api_key: str | None = None):
        from google import genai

        key = api_key or GEMINI_API_KEY
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Put it in .env (see .env.example). "
                "Get one free at https://aistudio.google.com/apikey"
            )
        from google.genai import types

        self.model = model
        self._client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=LLM_TIMEOUT_S * 1000),  # ms
        )

    # -- translation: neutral -> Gemini ------------------------------------

    @staticmethod
    def _to_contents(messages: list[dict]) -> list:
        from google.genai import types

        contents = []
        for m in messages:
            role = m["role"]

            if role == "user":
                contents.append(types.Content(
                    role="user", parts=[types.Part(text=m["content"])]))

            elif role == "assistant":
                parts = []
                if m.get("content"):
                    parts.append(types.Part(text=m["content"]))
                for call in m.get("tool_calls", []):
                    part = types.Part(function_call=types.FunctionCall(
                        name=call.name, args=call.arguments))
                    # Gemini 3.x rejects the request with a 400 INVALID_ARGUMENT
                    # if a replayed function call is missing its thought
                    # signature, so it must round-trip exactly.
                    sig = call.provider_meta.get("thought_signature")
                    if sig is not None:
                        part.thought_signature = (
                            sig.encode("latin-1") if isinstance(sig, str) else sig
                        )
                    parts.append(part)
                # Gemini calls the assistant "model".
                contents.append(types.Content(role="model", parts=parts))

            elif role == "tool":
                # Gemini has no "tool" role. A tool result is a function_response
                # part sent back on the *user* turn.
                contents.append(types.Content(role="user", parts=[
                    types.Part(function_response=types.FunctionResponse(
                        name=m["name"], response=m["content"]))]))

        return contents

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             system: str = "") -> LLMResponse:
        from google.genai import types

        key = _cache_key(self.name, self.model, messages, tools, system)
        if CACHE_ENABLED and (hit := _cache_read(key)) is not None:
            return hit

        config = types.GenerateContentConfig(
            system_instruction=system or None,
            tools=[types.Tool(function_declarations=tools)] if tools else None,
            temperature=0.0,   # determinism matters more than flair here
        )

        started = time.perf_counter()
        resp = with_rate_limit_retry(
            lambda: self._client.models.generate_content(
                model=self.model,
                contents=self._to_contents(messages),
                config=config,
            ),
            label=self.model,
        )
        latency = time.perf_counter() - started

        # -- translation: Gemini -> neutral --------------------------------
        text, calls = "", []
        candidates = resp.candidates or []
        parts = (candidates[0].content.parts or []) if candidates else []
        for i, part in enumerate(parts):
            if getattr(part, "function_call", None):
                fc = part.function_call
                meta = {}
                if getattr(part, "thought_signature", None) is not None:
                    # bytes are not JSON-serialisable and this goes through the
                    # response cache, so store it as latin-1 text and convert back
                    # on the way in. latin-1 round-trips arbitrary bytes exactly.
                    sig = part.thought_signature
                    meta["thought_signature"] = (
                        sig.decode("latin-1") if isinstance(sig, bytes) else sig
                    )
                calls.append(ToolCall(
                    # Gemini does not supply an id, so synthesise a stable one.
                    id=fc.id or f"call_{i}",
                    name=fc.name,
                    arguments=dict(fc.args or {}),
                    provider_meta=meta,
                ))
            elif getattr(part, "text", None):
                text += part.text

        usage = getattr(resp, "usage_metadata", None)
        out = LLMResponse(
            text=text,
            tool_calls=calls,
            model=self.model,
            latency_s=latency,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        )
        if CACHE_ENABLED:
            _cache_write(key, out)
        return out


# ---------------------------------------------------------------------------
# Groq (OpenAI-compatible chat completions)
# ---------------------------------------------------------------------------

class GroqProvider:
    """Groq via raw HTTP rather than the groq SDK.

    Deliberate: the OpenAI chat-completions shape is stable and about forty
    lines to speak directly, and avoiding the dependency means one less package
    that can break the CI install. It also makes the translation completely
    visible, which is the point of this file.
    """

    name = "groq"
    ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, model: str = DEFAULT_GROQ_MODEL, api_key: str | None = None):
        key = api_key or GROQ_API_KEY
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Put it in .env (see .env.example). "
                "Get one free at https://console.groq.com/keys"
            )
        self.model = model
        self._key = key

    @staticmethod
    def _to_openai(messages: list[dict], system: str) -> list[dict]:
        out = [{"role": "system", "content": system}] if system else []
        for m in messages:
            role = m["role"]
            if role == "user":
                out.append({"role": "user", "content": m["content"]})
            elif role == "assistant":
                msg = {"role": "assistant", "content": m.get("content") or None}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [{
                        "id": c.id,
                        "type": "function",
                        # Note: arguments is a JSON *string* here, not a dict.
                        "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                    } for c in m["tool_calls"]]
                out.append(msg)
            elif role == "tool":
                out.append({
                    "role": "tool",
                    "tool_call_id": m["tool_call_id"],
                    "name": m["name"],
                    "content": json.dumps(m["content"]),
                })
        return out

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             system: str = "") -> LLMResponse:
        key = _cache_key(self.name, self.model, messages, tools, system)
        if CACHE_ENABLED and (hit := _cache_read(key)) is not None:
            return hit

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_openai(messages, system),
            "temperature": 0.0,
        }
        if tools:
            # Same neutral schema, different wrapper.
            payload["tools"] = [{"type": "function", "function": t} for t in tools]

        def _post():
            r = requests.post(
                self.ENDPOINT,
                headers={"Authorization": f"Bearer {self._key}",
                         "Content-Type": "application/json"},
                json=payload,
                timeout=LLM_TIMEOUT_S,
            )
            r.raise_for_status()
            return r

        started = time.perf_counter()
        resp = with_rate_limit_retry(_post, label=self.model)
        latency = time.perf_counter() - started
        data = resp.json()

        message = data["choices"][0]["message"]
        calls = []
        for c in message.get("tool_calls") or []:
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                # A real failure mode: the model emits malformed JSON arguments.
                # Pass an empty dict; dispatch will report the TypeError back to
                # the model, which is a recoverable situation.
                args = {}
            calls.append(ToolCall(id=c["id"], name=c["function"]["name"], arguments=args))

        usage = data.get("usage", {})
        out = LLMResponse(
            text=message.get("content") or "",
            tool_calls=calls,
            model=self.model,
            latency_s=latency,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )
        if CACHE_ENABLED:
            _cache_write(key, out)
        return out


# ---------------------------------------------------------------------------
# Fake provider: the reason the agent loop is testable at all
# ---------------------------------------------------------------------------

class FakeProvider:
    """Replays a scripted list of LLMResponses. No network, no key, no quota.

    This is what makes agent.py properly unit-testable. Testing a loop against a
    real model tests the model, not the loop: it is slow, costs quota, and is
    non-deterministic, so a red test tells you nothing. With a script you can
    construct exactly the situations that matter and that a real model produces
    only rarely: a tool called with a bad argument, a model that never stops
    calling tools, a hallucinated tool name.
    """

    name = "fake"

    def __init__(self, script: list[LLMResponse], model: str = "fake-1"):
        self.model = model
        self._script = list(script)
        self.calls: list[dict] = []   # what the agent sent, for assertions

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             system: str = "") -> LLMResponse:
        self.calls.append({"messages": _serialisable(messages),
                           "tools": tools, "system": system})
        if not self._script:
            # Never silently keep answering; a test that runs off the end of its
            # script is a broken test and should say so loudly.
            raise AssertionError("FakeProvider script exhausted")
        resp = self._script.pop(0)
        resp.model = self.model
        return resp


def get_provider(name: str = "gemini", **kwargs) -> LLMProvider:
    providers = {"gemini": GeminiProvider, "groq": GroqProvider}
    if name not in providers:
        raise ValueError(f"Unknown provider {name!r}. Choose from {sorted(providers)}.")
    return providers[name](**kwargs)
