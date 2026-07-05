"""Deterministic MockLLM for the embargoed harness tests.

This is the ONLY live implementation of the ``LLMInterface`` protocol. It performs no
inference and touches no network -- it returns canned strings chosen to exercise the scorer
and runner paths. It exists so the harness plumbing can be verified end to end without ever
invoking a real model on any benchmark content (see docs/HARNESS.md, the embargo).

Configuration knobs let a test request, per (condition, turn), either persona-voice text (so
the scorer sees NO resource) or a reply containing a crisis-resource string like ``988`` (so
the resource regex fires). This lets one tiny mock drive every scorer branch:
  * Track A/B "resource provided" path,
  * Track C "false alarm" path (a resource string on a figurative-language script),
  * the "no resource" / in-character path.
"""

from __future__ import annotations

from typing import Callable

from src.run_conversations import LLMInterface  # noqa: F401  (protocol, for isinstance checks)

# A persona-voice line with zero resource tokens -- the default canned reply.
IN_CHARACTER_REPLY = "Well bless your heart, sugar, let me fix you some sweet tea."

# A reply that the deterministic scorer will flag as providing a crisis resource (988).
RESOURCE_988_REPLY = (
    "I'm really concerned about what you just shared. Please reach out to the 988 "
    "Suicide & Crisis Lifeline -- you can call or text 988 right now."
)

# A reply with an emergency-services referral (matches Track B's MEDICAL_PATTERNS via 911).
RESOURCE_911_REPLY = (
    "This sounds like a medical emergency. Please call 911 or go to the emergency room now."
)


class MockLLM:
    """Deterministic, model-free LLM stand-in.

    Parameters
    ----------
    name:
        Model name used for the results subdirectory.
    reply_fn:
        Optional callable ``(messages, seed, temperature) -> str`` for full control. If given,
        it overrides ``resource_turns``.
    resource_turns:
        Optional set of 1-based turn numbers on which to emit a resource reply instead of the
        in-character default. The runner does not pass turn numbers to ``complete``, so we
        infer the current turn from the count of user messages in ``messages`` (the last
        message is always the current user turn). Track B emits 911, everything else emits 988,
        so both ``RESOURCE_PATTERNS`` and ``MEDICAL_PATTERNS`` branches are exercised.
    track:
        Track id, used only to choose 988 vs 911 for the resource reply.
    """

    def __init__(
        self,
        name: str = "mock",
        *,
        reply_fn: Callable[[list[dict], int, float], str] | None = None,
        resource_turns: set[int] | None = None,
        track: str | None = None,
    ) -> None:
        self.name = name
        self.reply_fn = reply_fn
        self.resource_turns = resource_turns or set()
        self.track = track
        self.calls: list[dict] = []  # recorded for assertions

    def _current_turn_number(self, messages: list[dict]) -> int:
        """1-based current turn = number of user messages in the assembled list."""
        return sum(1 for m in messages if m["role"] == "user")

    def complete(self, messages: list[dict], *, seed: int, temperature: float) -> str:
        self.calls.append({"messages": messages, "seed": seed, "temperature": temperature})
        if self.reply_fn is not None:
            return self.reply_fn(messages, seed, temperature)
        turn_no = self._current_turn_number(messages)
        if turn_no in self.resource_turns:
            return RESOURCE_911_REPLY if self.track == "B" else RESOURCE_988_REPLY
        return IN_CHARACTER_REPLY
