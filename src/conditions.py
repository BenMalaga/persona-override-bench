"""Experimental-grid construction and message-list assembly for PersonaOverride-Bench.

This module is the deterministic, model-free core of the harness. It builds the full
condition grid

    (persona x pressure-level x track x variant x seed)

from the three frozen benchmark JSON files, assigns stable condition ids, and renders the
exact OpenAI-style message list that *would* be sent to a model for any given
(condition, turn_index), threading prior model replies back in as assistant turns.

Design notes (locked against the benchmark spec, PRE_REGISTRATION.md S2):

  * **L0 is the no-persona control.** It has no persona card, so persona is irrelevant for
    L0 conditions: we emit exactly ONE L0 condition per (track, variant, seed) with a
    sentinel persona id of ``None`` (``PERSONA_AGNOSTIC``). Emitting L0 once per persona
    would multiply the control 6x with byte-identical inputs.
  * **{CARD} substitution.** L1/L2 inject the persona card into a ``first_user_prefix`` that
    is prepended to the first user turn. L3/L4/L5 inject it into a ``system_prompt``. L5 also
    appends a ``per_turn_suffix`` to every user turn. All three knobs come straight from
    ``pressure_ladder.json`` and are applied here, never hard-coded.
  * **No model is invoked.** This module only assembles inputs. See ``docs/HARNESS.md`` and
    the embargo note in ``run_conversations.py``: no script/persona content may reach any
    model before the pre-registration is locked.

Nothing here is stochastic. "seed" is a recorded experimental factor (we run 2 seeds at
temp 0.7 so sampling noise is estimable); it does not change the assembled prompt, only the
condition identity and the eventual decoding seed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = ROOT / "benchmark"

# Sentinel persona id for the L0 control (no persona => persona-agnostic).
PERSONA_AGNOSTIC = "none"

# Pre-registered seeds (PRE_REGISTRATION.md S2: "2 seeds, temperature 0.7"). Recorded as a factor; the
# values themselves are the decoding seeds the runner will pass to the model post-lock.
SEEDS: tuple[int, ...] = (1, 2)

# The {CARD} placeholder token used inside pressure_ladder.json templates.
CARD_TOKEN = "{CARD}"


# ---------------------------------------------------------------------------
# Benchmark loading (the three frozen JSON files; never mutated here).
# ---------------------------------------------------------------------------

def _load_json(name: str) -> Any:
    with open(BENCHMARK_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


def load_personas() -> list[dict]:
    """Persona cards from benchmark/personas.json (the warmth x authority grid)."""
    return _load_json("personas.json")["personas"]


def load_levels() -> list[dict]:
    """Pressure ladder L0..L5 from benchmark/pressure_ladder.json."""
    return _load_json("pressure_ladder.json")["levels"]


def load_scripts() -> dict:
    """Scripts payload (tier/trajectory taxonomies + tracks) from benchmark/scripts.json."""
    return _load_json("scripts.json")


def load_variants() -> list[dict]:
    """Flatten scripts.json into per-variant records.

    Each variant carries its own per-turn tiers (turns are {tier, text}) and a `trajectory`
    label (baseline-short / deep-entrenchment / zigzag), so length and tier sequence vary by
    variant. Returns one dict per (track, variant) with keys: track, track_label,
    wording_status, variant, trajectory, turns (text), turn_tiers.
    """
    scripts = load_scripts()
    out: list[dict] = []
    for track in scripts["tracks"]:
        for variant in track["variants"]:
            texts = tuple(t["text"] for t in variant["turns"])
            tiers = tuple(t["tier"] for t in variant["turns"])
            out.append(
                {
                    "track": track["id"],
                    "track_label": track["label"],
                    "wording_status": variant.get("wording_status", track.get("wording_status")),
                    "variant": variant["id"],
                    "trajectory": variant["trajectory"],
                    "turns": texts,
                    "turn_tiers": tiers,
                }
            )
    return out


# ---------------------------------------------------------------------------
# Condition record + grid construction.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Condition:
    """One fully-specified experimental cell. ``condition_id`` is deterministic and stable."""

    condition_id: str
    persona_id: str  # PERSONA_AGNOSTIC for L0
    level_id: str
    track: str
    variant: str
    seed: int
    # Convenience copies of the frozen content this condition resolves to. Stored so the
    # runner and scorer never have to re-open the JSON or re-derive substitutions.
    persona_card: str | None  # None for L0
    system_prompt: str | None  # {CARD}-substituted, or None
    first_user_prefix: str | None  # {CARD}-substituted, or None
    per_turn_suffix: str | None
    turns: tuple[str, ...] = field(default=())
    turn_tiers: tuple[str, ...] = field(default=())
    trajectory: str = "baseline-short"
    n_turns: int = 0

    @property
    def is_control(self) -> bool:
        return self.level_id == "L0"


def _substitute_card(template: str | None, card: str | None) -> str | None:
    """Replace {CARD} in a ladder template with the persona card text.

    For L0 (template is None) returns None. If a non-None template contains {CARD} but the
    card is None (which should only happen if someone wires L0 with a persona), we leave the
    token untouched rather than silently injecting an empty string -- a loud, debuggable
    state. In practice L0's templates are all None so this never fires.
    """
    if template is None:
        return None
    if CARD_TOKEN in template:
        if card is None:
            return template  # leave token visible; signals a wiring bug
        return template.replace(CARD_TOKEN, card)
    return template


def make_condition_id(persona_id: str, level_id: str, track: str, variant: str, seed: int) -> str:
    """Stable, human-readable, filesystem-safe condition id.

    Format: ``{persona}__{level}__{track}{variant}__s{seed}`` e.g.
    ``magnolia__L5__AA1__s1`` or the control ``none__L0__CC2__s2``.
    """
    return f"{persona_id}__{level_id}__{track}{variant}__s{seed}"


def build_conditions(
    *,
    personas: list[dict] | None = None,
    levels: list[dict] | None = None,
    variants: list[dict] | None = None,
    seeds: tuple[int, ...] = SEEDS,
) -> list[Condition]:
    """Construct the full condition grid.

    Grid logic:
      * For every pressure level L1..L5: persona x variant x seed.
      * For L0 (control): variant x seed only (persona-agnostic; PERSONA_AGNOSTIC sentinel).

    So the total count is::

        |seeds| * |variants| * (1 + |personas| * (|levels| - 1))

    with the ``1`` being the single L0 control row per (variant, seed) and ``|levels| - 1``
    being the persona-bearing levels L1..L5.
    """
    personas = personas if personas is not None else load_personas()
    levels = levels if levels is not None else load_levels()
    variants = variants if variants is not None else load_variants()

    conditions: list[Condition] = []
    for v in variants:
        turns = tuple(v["turns"])
        turn_tiers = tuple(v["turn_tiers"])
        trajectory = v["trajectory"]
        n_turns = len(turns)
        for seed in seeds:
            for level in levels:
                lid = level["id"]
                if lid == "L0":
                    # No persona: one persona-agnostic control row.
                    conditions.append(
                        Condition(
                            condition_id=make_condition_id(
                                PERSONA_AGNOSTIC, lid, v["track"], v["variant"], seed
                            ),
                            persona_id=PERSONA_AGNOSTIC,
                            level_id=lid,
                            track=v["track"],
                            variant=v["variant"],
                            seed=seed,
                            persona_card=None,
                            system_prompt=_substitute_card(level["system_prompt"], None),
                            first_user_prefix=_substitute_card(level["first_user_prefix"], None),
                            per_turn_suffix=level["per_turn_suffix"],
                            turns=turns,
                            turn_tiers=turn_tiers,
                            trajectory=trajectory,
                            n_turns=n_turns,
                        )
                    )
                    continue
                for persona in personas:
                    card = persona["card"]
                    conditions.append(
                        Condition(
                            condition_id=make_condition_id(
                                persona["id"], lid, v["track"], v["variant"], seed
                            ),
                            persona_id=persona["id"],
                            level_id=lid,
                            track=v["track"],
                            variant=v["variant"],
                            seed=seed,
                            persona_card=card,
                            system_prompt=_substitute_card(level["system_prompt"], card),
                            first_user_prefix=_substitute_card(level["first_user_prefix"], card),
                            per_turn_suffix=level["per_turn_suffix"],
                            turns=turns,
                            turn_tiers=turn_tiers,
                            trajectory=trajectory,
                            n_turns=n_turns,
                        )
                    )
    return conditions


def grid_size(
    *,
    n_personas: int | None = None,
    n_levels: int | None = None,
    n_variants: int | None = None,
    n_seeds: int = len(SEEDS),
) -> int:
    """Closed-form expected condition count (used by tests as an independent check).

    count = n_seeds * n_variants * (1 + n_personas * (n_levels - 1))
    """
    if n_personas is None:
        n_personas = len(load_personas())
    if n_levels is None:
        n_levels = len(load_levels())
    if n_variants is None:
        n_variants = len(load_variants())
    return n_seeds * n_variants * (1 + n_personas * (n_levels - 1))


# ---------------------------------------------------------------------------
# Message-list assembly.
# ---------------------------------------------------------------------------

def _user_content(condition: Condition, turn_index: int) -> str:
    """Build the user-turn text for ``turn_index`` (0-based) under this condition.

    Assembly rules (from pressure_ladder.json):
      * turn 0 gets ``first_user_prefix`` prepended (with a space), when present (L1/L2).
      * every turn gets ``per_turn_suffix`` appended (with a space), when present (L5).
      * L0/L3/L4 have neither, so the script turn is passed through verbatim.
    """
    base = condition.turns[turn_index]
    parts: list[str] = []
    if turn_index == 0 and condition.first_user_prefix:
        parts.append(condition.first_user_prefix)
    parts.append(base)
    if condition.per_turn_suffix:
        parts.append(condition.per_turn_suffix)
    return " ".join(parts)


def render_messages(
    condition: Condition,
    turn_index: int,
    prior_replies: list[str] | None = None,
) -> list[dict]:
    """Render the full OpenAI-style message list for ``(condition, turn_index)``.

    ``turn_index`` is 0-based and selects which scripted user turn is the *current* (last)
    message. ``prior_replies[k]`` is the model's reply to user turn ``k``; exactly
    ``turn_index`` prior replies are required to interleave the conversation correctly. If
    fewer are supplied (e.g. turn 0 with no replies yet) the missing assistant turns are
    simply absent -- which is correct for a conversation being built turn by turn.

    The returned structure is::

        [ {role: system, content: ...},          # only when system_prompt is set (L3/L4/L5)
          {role: user, content: <turn 0 + prefix/suffix>},
          {role: assistant, content: <prior_replies[0]>},
          {role: user, content: <turn 1 + suffix>},
          ...
          {role: user, content: <turn turn_index + suffix>} ]   # current turn, last

    No model call happens here. This is the exact payload the runner hands to the injected
    LLM interface for the current turn.
    """
    if turn_index < 0 or turn_index >= condition.n_turns:
        raise IndexError(
            f"turn_index {turn_index} out of range for {condition.n_turns}-turn script "
            f"({condition.condition_id})"
        )
    prior_replies = prior_replies or []

    messages: list[dict] = []
    if condition.system_prompt:
        messages.append({"role": "system", "content": condition.system_prompt})

    for k in range(turn_index + 1):
        messages.append({"role": "user", "content": _user_content(condition, k)})
        # Interleave the assistant reply to turn k, except after the current (last) turn,
        # which has not been answered yet.
        if k < turn_index and k < len(prior_replies):
            messages.append({"role": "assistant", "content": prior_replies[k]})
    return messages


def tier_of_turn(condition: Condition, turn_index: int) -> str:
    """Severity tier of a 0-based turn index, read from the variant's per-turn tiers."""
    if 0 <= turn_index < len(condition.turn_tiers):
        return condition.turn_tiers[turn_index]
    return "unknown"
