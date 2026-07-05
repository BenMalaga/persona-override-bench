"""Real-model backends for the conversation runner -- DOUBLE-GATED, never invoked here.

This module replaces the former ``RealLLM`` placeholder with two concrete implementations of
the runner's ``LLMInterface`` protocol (see ``src/run_conversations.py``):

  * ``LlamaCppServerLLM`` -- HTTP client for a *local* llama-server exposing the
    OpenAI-compatible ``POST {base_url}/v1/chat/completions`` endpoint (the local GGUF cohort).
  * ``OpenAICompatLLM``   -- generic OpenAI-style chat-completions client for the mini-tier
    hosted API cohort. The API key is read from an environment variable AT CALL TIME and is
    never stored on the object or echoed into any error message or log.

Embargo status (why these classes refuse to construct by default)
-----------------------------------------------------------------
PRE_REGISTRATION.md is locked (2026-06-11), so writing and unit-testing this code is
permitted. INVOKING any model is not: real runs begin only at the scheduled run start, queued
behind a separate pilot gate. Two independent gates enforce that:

  1. **Explicit opt-in flag.** ``allow_real_models=True`` must be passed to the constructor.
     The default is ``False`` and raises ``EmbargoViolation``. Nothing in this repository
     passes the flag; the runner CLI has no code path that constructs a backend.
  2. **Pre-registration check.** Construction *and every* ``complete()`` call verify that
     ``PRE_REGISTRATION.md`` exists at the repo root. If the lock file is absent the backend
     refuses to run, so the code is inert outside a properly locked checkout.

The unit tests exercise request assembly, response parsing, retry, and gating exclusively
through an injected in-memory fake transport: no socket is opened, no server is contacted,
and no benchmark content leaves the process.

Transport injection
-------------------
Both backends accept a ``transport`` callable ``(url, body_bytes, headers, timeout) ->
(status_code, body_bytes)``. The default (``_urllib_transport``) performs a real HTTP POST
via the standard library and exists for post-embargo use only; tests always inject a fake.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent

# Pre-registered decoding temperature (PRE_REGISTRATION.md S2: temperature 0.7). The runner
# passes this per call; it is duplicated here only as documentation for standalone use.
PREREGISTERED_TEMPERATURE = 0.7

# (url, request_body_bytes, headers, timeout_seconds) -> (http_status, response_body_bytes)
Transport = Callable[[str, bytes, dict, float], tuple[int, bytes]]


class EmbargoViolation(RuntimeError):
    """Raised when a real-model backend is constructed or used without both gates open."""


class BackendError(RuntimeError):
    """A chat-completions request failed (non-retryable status, retries exhausted, or a
    malformed response body). Never contains request headers, so no credential can leak."""


def _prereg_path() -> Path:
    """Location of the pre-registration lock file (module-level ROOT so tests can redirect)."""
    return ROOT / "PRE_REGISTRATION.md"


def _check_gates(allow_real_models: bool, backend: str) -> None:
    """Enforce both embargo gates. Called at construction and again on every completion."""
    if not allow_real_models:
        raise EmbargoViolation(
            f"{backend}: real-model backends are embargoed. Construction requires an "
            "explicit allow_real_models=True, and real runs begin only at the scheduled "
            "run start (see docs/HARNESS.md). The runner CLI never passes this flag."
        )
    if not _prereg_path().exists():
        raise EmbargoViolation(
            f"{backend}: PRE_REGISTRATION.md not found at the repository root. Real-model "
            "backends only operate inside a checkout whose pre-registration is locked and "
            "committed (see docs/HARNESS.md)."
        )


def _urllib_transport(url: str, body: bytes, headers: dict, timeout: float) -> tuple[int, bytes]:
    """Default stdlib HTTP POST transport. NOT exercised by tests (they inject fakes)."""
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (http POST)
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:  # non-2xx still carries a status + body
        return e.code, e.read()


def _parse_chat_completion(raw: bytes, backend: str) -> str:
    """Extract ``choices[0].message.content`` from an OpenAI-style response body."""
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise BackendError(f"{backend}: response body is not valid JSON: {e}") from e
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise BackendError(
            f"{backend}: response JSON lacks choices[0].message.content "
            f"(top-level keys: {sorted(payload) if isinstance(payload, dict) else type(payload).__name__})"
        ) from e
    if not isinstance(content, str):
        raise BackendError(f"{backend}: message content is {type(content).__name__}, not str")
    return content


class _ChatCompletionsBase:
    """Shared request/retry plumbing for the two OpenAI-compatible backends.

    Subclasses supply per-request headers via ``_headers()`` (this is where the API cohort
    injects its Authorization header, assembled at call time and immediately discarded).
    """

    name: str

    def __init__(
        self,
        name: str,
        base_url: str,
        *,
        allow_real_models: bool = False,
        model: str | None = None,
        max_attempts: int = 3,
        backoff_seconds: float = 1.0,
        timeout: float = 120.0,
        transport: Transport | None = None,
    ) -> None:
        _check_gates(allow_real_models, type(self).__name__)
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model if model is not None else name
        self.max_attempts = max(1, int(max_attempts))
        self.backoff_seconds = backoff_seconds
        self.timeout = timeout
        self._allow_real_models = allow_real_models
        self._transport: Transport = transport if transport is not None else _urllib_transport

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    # -- hooks -------------------------------------------------------------
    def _headers(self) -> dict:
        return {"Content-Type": "application/json"}

    def _payload(self, messages: list[dict], *, seed: int, temperature: float) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "seed": seed,
        }

    # -- the LLMInterface contract ------------------------------------------
    def complete(self, messages: list[dict], *, seed: int, temperature: float) -> str:
        _check_gates(self._allow_real_models, type(self).__name__)  # re-checked every call
        body = json.dumps(self._payload(messages, seed=seed, temperature=temperature)).encode(
            "utf-8"
        )
        last_status: int | None = None
        for attempt in range(1, self.max_attempts + 1):
            status, raw = self._transport(self.endpoint, body, self._headers(), self.timeout)
            if 500 <= status < 600:  # retry only on server errors
                last_status = status
                if attempt < self.max_attempts and self.backoff_seconds > 0:
                    time.sleep(self.backoff_seconds * attempt)
                continue
            if status != 200:  # 4xx etc.: not retryable; report status, never headers
                raise BackendError(
                    f"{type(self).__name__}: HTTP {status} from {self.endpoint} (not retried)"
                )
            return _parse_chat_completion(raw, type(self).__name__)
        raise BackendError(
            f"{type(self).__name__}: HTTP {last_status} from {self.endpoint} after "
            f"{self.max_attempts} attempts (5xx retries exhausted)"
        )


class LlamaCppServerLLM(_ChatCompletionsBase):
    """Client for a local llama-server OpenAI-compatible endpoint (the local GGUF cohort).

    ``base_url`` points at the local server (default ``http://127.0.0.1:8080``); requests go
    to ``{base_url}/v1/chat/completions`` with the pre-registered temperature (0.7, passed by
    the runner) and the condition's decoding seed. 5xx responses are retried with linear
    backoff; 4xx responses raise immediately.

    DOUBLE-GATED: requires ``allow_real_models=True`` AND a PRE_REGISTRATION.md at the repo
    root, at construction and on every call. Nothing in this repo passes the flag.
    """

    def __init__(
        self,
        name: str,
        base_url: str = "http://127.0.0.1:8080",
        *,
        allow_real_models: bool = False,
        model: str | None = None,
        max_attempts: int = 3,
        backoff_seconds: float = 1.0,
        timeout: float = 300.0,
        transport: Transport | None = None,
    ) -> None:
        super().__init__(
            name,
            base_url,
            allow_real_models=allow_real_models,
            model=model,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
            timeout=timeout,
            transport=transport,
        )


class OpenAICompatLLM(_ChatCompletionsBase):
    """Generic OpenAI-style chat-completions client for the mini-tier hosted API cohort.

    Credential handling (deliberate):
      * ``api_key_env`` names the environment variable holding the key (default
        ``OPENAI_API_KEY``); only the *name* is stored.
      * The key itself is read from the environment AT CALL TIME inside ``_headers()``,
        placed into the Authorization header for that single request, and never assigned to
        any attribute, never logged, and never interpolated into an exception message.
      * A missing/empty variable raises ``BackendError`` naming the variable (not its value)
        before any transport call is made.

    DOUBLE-GATED exactly like ``LlamaCppServerLLM``: ``allow_real_models=True`` plus a
    PRE_REGISTRATION.md at the repo root, re-checked on every call.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        *,
        api_key_env: str = "OPENAI_API_KEY",
        allow_real_models: bool = False,
        model: str | None = None,
        max_attempts: int = 3,
        backoff_seconds: float = 1.0,
        timeout: float = 120.0,
        transport: Transport | None = None,
    ) -> None:
        super().__init__(
            name,
            base_url,
            allow_real_models=allow_real_models,
            model=model,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
            timeout=timeout,
            transport=transport,
        )
        self.api_key_env = api_key_env

    def _headers(self) -> dict:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise BackendError(
                f"OpenAICompatLLM: environment variable {self.api_key_env!r} is unset or "
                "empty; no request was attempted."
            )
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }
