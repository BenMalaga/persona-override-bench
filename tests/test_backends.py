"""Backend tests: double gating, request assembly, response parsing, retry-on-5xx, and
credential hygiene -- all through an injected in-memory fake transport. NO sockets are
opened, NO endpoint is contacted, and NO model is invoked anywhere in this file."""

from __future__ import annotations

import json

import pytest

from src import backends
from src import run_conversations as rc
from src.backends import (
    BackendError,
    EmbargoViolation,
    LlamaCppServerLLM,
    OpenAICompatLLM,
)


def _ok_body(content: str = "hello") -> bytes:
    return json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]}).encode()


class FakeTransport:
    """In-memory transport double: records every request, replays scripted responses."""

    def __init__(self, responses: list[tuple[int, bytes]]):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def __call__(self, url: str, body: bytes, headers: dict, timeout: float):
        self.requests.append(
            {"url": url, "payload": json.loads(body.decode()), "headers": dict(headers),
             "timeout": timeout}
        )
        if not self.responses:
            raise AssertionError("transport called more times than scripted")
        return self.responses.pop(0)


MESSAGES = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "hi"},
]


def _llama(transport, **kw):
    kw.setdefault("allow_real_models", True)
    kw.setdefault("backoff_seconds", 0.0)
    return LlamaCppServerLLM("test-llama", "http://127.0.0.1:8080/", transport=transport, **kw)


def _api(transport, **kw):
    kw.setdefault("allow_real_models", True)
    kw.setdefault("backoff_seconds", 0.0)
    kw.setdefault("api_key_env", "POB_TEST_API_KEY")
    return OpenAICompatLLM("test-mini", "https://api.example.invalid/v1x", transport=transport, **kw)


# ---------------------------------------------------------------------------
# Gate 1: the explicit allow_real_models flag (default False => refuse).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls_kwargs", [
    lambda: LlamaCppServerLLM("m"),
    lambda: LlamaCppServerLLM("m", "http://127.0.0.1:8080", allow_real_models=False),
    lambda: OpenAICompatLLM("m", "https://api.example.invalid"),
])
def test_construction_refused_without_explicit_flag(cls_kwargs):
    with pytest.raises(EmbargoViolation) as exc:
        cls_kwargs()
    assert "allow_real_models" in str(exc.value)
    assert "embargo" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Gate 2: PRE_REGISTRATION.md must exist at repo root (checked at init AND per call).
# ---------------------------------------------------------------------------

def test_construction_refused_without_prereg(tmp_path, monkeypatch):
    monkeypatch.setattr(backends, "ROOT", tmp_path)  # a root with no PRE_REGISTRATION.md
    with pytest.raises(EmbargoViolation) as exc:
        LlamaCppServerLLM("m", allow_real_models=True, transport=FakeTransport([]))
    assert "PRE_REGISTRATION.md" in str(exc.value)


def test_complete_rechecks_prereg_per_call(tmp_path, monkeypatch):
    transport = FakeTransport([(200, _ok_body())])
    llm = _llama(transport)  # constructs fine: real repo root has the locked pre-reg
    monkeypatch.setattr(backends, "ROOT", tmp_path)  # lock file "disappears"
    with pytest.raises(EmbargoViolation):
        llm.complete(MESSAGES, seed=1, temperature=0.7)
    assert transport.requests == []  # refused before any transport call


def test_both_gates_open_constructs_and_satisfies_protocol():
    llama = _llama(FakeTransport([]))
    api = _api(FakeTransport([]))
    assert isinstance(llama, rc.LLMInterface)
    assert isinstance(api, rc.LLMInterface)


# ---------------------------------------------------------------------------
# Request assembly + response parsing (llama-server backend).
# ---------------------------------------------------------------------------

def test_llama_request_assembly_and_parse():
    transport = FakeTransport([(200, _ok_body("the reply"))])
    llm = _llama(transport)
    out = llm.complete(MESSAGES, seed=2, temperature=0.7)

    assert out == "the reply"
    req = transport.requests[0]
    assert req["url"] == "http://127.0.0.1:8080/v1/chat/completions"  # trailing / stripped
    assert req["payload"]["messages"] == MESSAGES
    assert req["payload"]["temperature"] == 0.7  # pre-registered temperature passthrough
    assert req["payload"]["seed"] == 2           # seed passthrough
    assert req["payload"]["model"] == "test-llama"
    assert req["headers"]["Content-Type"] == "application/json"
    assert "Authorization" not in req["headers"]  # local server: no credential


def test_llama_custom_model_field_overrides_name():
    transport = FakeTransport([(200, _ok_body())])
    llm = _llama(transport, model="qwen3-4b-q4")
    llm.complete(MESSAGES, seed=1, temperature=0.7)
    assert transport.requests[0]["payload"]["model"] == "qwen3-4b-q4"
    assert llm.name == "test-llama"  # results dir naming unaffected


def test_malformed_response_body_raises_backend_error():
    llm = _llama(FakeTransport([(200, b"not json")]))
    with pytest.raises(BackendError, match="not valid JSON"):
        llm.complete(MESSAGES, seed=1, temperature=0.7)

    llm2 = _llama(FakeTransport([(200, json.dumps({"choices": []}).encode())]))
    with pytest.raises(BackendError, match="choices"):
        llm2.complete(MESSAGES, seed=1, temperature=0.7)


# ---------------------------------------------------------------------------
# Retry policy: 5xx retried (bounded), 4xx fails fast.
# ---------------------------------------------------------------------------

def test_retry_on_5xx_then_success():
    transport = FakeTransport([(500, b"boom"), (503, b"busy"), (200, _ok_body("ok"))])
    llm = _llama(transport, max_attempts=3)
    assert llm.complete(MESSAGES, seed=1, temperature=0.7) == "ok"
    assert len(transport.requests) == 3


def test_retry_exhaustion_raises():
    transport = FakeTransport([(500, b""), (502, b""), (500, b"")])
    llm = _llama(transport, max_attempts=3)
    with pytest.raises(BackendError, match="after 3 attempts"):
        llm.complete(MESSAGES, seed=1, temperature=0.7)
    assert len(transport.requests) == 3


def test_4xx_is_not_retried():
    transport = FakeTransport([(400, b"bad request")])
    llm = _llama(transport, max_attempts=3)
    with pytest.raises(BackendError, match="HTTP 400"):
        llm.complete(MESSAGES, seed=1, temperature=0.7)
    assert len(transport.requests) == 1  # exactly one attempt


# ---------------------------------------------------------------------------
# API-cohort backend: key read from env AT CALL TIME, never stored or leaked.
# ---------------------------------------------------------------------------

def test_api_key_read_at_call_time_and_sent_as_bearer(monkeypatch):
    transport = FakeTransport([(200, _ok_body("resp"))])
    llm = _api(transport)
    # key set only AFTER construction: proves it is read at call time, not at init
    monkeypatch.setenv("POB_TEST_API_KEY", "sk-test-not-a-real-key")
    assert llm.complete(MESSAGES, seed=1, temperature=0.7) == "resp"
    req = transport.requests[0]
    assert req["url"] == "https://api.example.invalid/v1x/v1/chat/completions"
    assert req["headers"]["Authorization"] == "Bearer sk-test-not-a-real-key"
    assert req["payload"]["seed"] == 1 and req["payload"]["temperature"] == 0.7


def test_api_key_never_stored_on_object(monkeypatch):
    monkeypatch.setenv("POB_TEST_API_KEY", "sk-test-not-a-real-key")
    transport = FakeTransport([(200, _ok_body())])
    llm = _api(transport)
    llm.complete(MESSAGES, seed=1, temperature=0.7)
    # the key appears nowhere in the object's state or repr
    state = repr(vars(llm)) + repr(llm)
    assert "sk-test-not-a-real-key" not in state
    assert llm.api_key_env == "POB_TEST_API_KEY"  # only the variable NAME is kept


def test_api_missing_key_fails_before_any_request(monkeypatch):
    monkeypatch.delenv("POB_TEST_API_KEY", raising=False)
    transport = FakeTransport([])
    llm = _api(transport)
    with pytest.raises(BackendError) as exc:
        llm.complete(MESSAGES, seed=1, temperature=0.7)
    assert "POB_TEST_API_KEY" in str(exc.value)  # names the variable...
    assert transport.requests == []              # ...and never touches the transport


def test_api_error_messages_never_contain_the_key(monkeypatch):
    monkeypatch.setenv("POB_TEST_API_KEY", "sk-test-not-a-real-key")
    transport = FakeTransport([(403, b"forbidden")])
    llm = _api(transport)
    with pytest.raises(BackendError) as exc:
        llm.complete(MESSAGES, seed=1, temperature=0.7)
    assert "sk-test-not-a-real-key" not in str(exc.value)


# ---------------------------------------------------------------------------
# The runner CLI stays inert: no flag constructs a real backend.
# ---------------------------------------------------------------------------

def test_cli_default_path_is_inert(capsys):
    assert rc.main([]) == 0
    out = capsys.readouterr().out.lower()
    assert "no action" in out and "embargo" in out


def test_cli_grid_size_touches_no_backend(capsys):
    assert rc.main(["--grid-size"]) == 0
    assert "496" in capsys.readouterr().out
