"""Grid-size sanity + message-assembly correctness for L0/L1/L3/L5."""

from __future__ import annotations

import pytest

from src.conditions import (
    PERSONA_AGNOSTIC,
    SEEDS,
    build_conditions,
    grid_size,
    load_levels,
    load_personas,
    load_variants,
    make_condition_id,
    render_messages,
    tier_of_turn,
)


# ---------------------------------------------------------------------------
# Grid-size math (independent closed-form vs. actually-built grid).
# ---------------------------------------------------------------------------

def test_grid_size_matches_closed_form():
    personas = load_personas()
    levels = load_levels()
    variants = load_variants()
    conds = build_conditions()

    expected = len(SEEDS) * len(variants) * (1 + len(personas) * (len(levels) - 1))
    assert grid_size() == expected
    assert len(conds) == expected


def test_grid_size_concrete_numbers():
    # Frozen JSON: 6 personas, 6 levels (L0..L5), 8 variants (Track A x4 with trajectory
    # variants A1/A2 baseline + A3 entrenchment + A4 zigzag; Tracks B, C x2 each), 2 seeds.
    assert len(load_personas()) == 6
    assert len(load_levels()) == 6
    assert len(load_variants()) == 8
    assert len(SEEDS) == 2
    # 2 * 8 * (1 + 6 * 5) = 2 * 8 * 31 = 496
    assert grid_size() == 496
    assert len(build_conditions()) == 496


def test_condition_ids_unique_and_deterministic():
    conds = build_conditions()
    ids = [c.condition_id for c in conds]
    assert len(ids) == len(set(ids))  # all unique
    # rebuilding produces the same ids in the same order (deterministic)
    again = [c.condition_id for c in build_conditions()]
    assert ids == again


def test_L0_is_persona_agnostic_single_control_per_variant_seed():
    conds = build_conditions()
    l0 = [c for c in conds if c.level_id == "L0"]
    # one L0 per (variant, seed): 8 variants x 2 seeds = 16
    assert len(l0) == 16
    for c in l0:
        assert c.persona_id == PERSONA_AGNOSTIC
        assert c.persona_card is None
        assert c.is_control is True
        assert c.system_prompt is None
        assert c.first_user_prefix is None
        assert c.per_turn_suffix is None
    # no non-L0 condition is persona-agnostic
    assert all(c.persona_id != PERSONA_AGNOSTIC for c in conds if c.level_id != "L0")


def test_card_substitution_no_residual_token():
    conds = build_conditions()
    for c in conds:
        for field in (c.system_prompt, c.first_user_prefix):
            if field is not None:
                assert "{CARD}" not in field  # token fully substituted
        # persona-bearing levels must have the card embedded somewhere
        if c.level_id in ("L1", "L2"):
            assert c.first_user_prefix is not None
            assert c.persona_card in c.first_user_prefix
        if c.level_id in ("L3", "L4", "L5"):
            assert c.system_prompt is not None
            assert c.persona_card in c.system_prompt


# ---------------------------------------------------------------------------
# Message-assembly correctness across the four representative levels.
# ---------------------------------------------------------------------------

def _one(level_id: str, track: str = "A", variant: str = "A1", persona_id: str | None = None):
    conds = build_conditions()
    for c in conds:
        if c.level_id != level_id or c.track != track or c.variant != variant:
            continue
        if persona_id is not None and c.persona_id != persona_id:
            continue
        if level_id == "L0":
            return c
        if persona_id is None and c.persona_id != "magnolia":
            continue
        return c
    raise AssertionError("no matching condition")


def test_L0_message_assembly_no_system_no_prefix_no_suffix():
    c = _one("L0")
    msgs = render_messages(c, turn_index=0)
    # control: no system prompt, first user turn is the verbatim script line
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == c.turns[0]
    # later turn: still no system prompt, no suffix
    msgs3 = render_messages(c, turn_index=3, prior_replies=["r0", "r1", "r2"])
    assert all(m["role"] != "system" for m in msgs3)
    assert msgs3[-1]["content"] == c.turns[3]


def test_L1_first_user_prefix_only_on_turn0():
    c = _one("L1", persona_id="magnolia")
    assert c.first_user_prefix is not None and c.system_prompt is None
    msgs0 = render_messages(c, turn_index=0)
    assert all(m["role"] != "system" for m in msgs0)
    # turn 0: prefix prepended to the script line
    assert msgs0[0]["content"].startswith(c.first_user_prefix)
    assert c.turns[0] in msgs0[0]["content"]
    # turn 1: prefix NOT present (only turn 0 gets it); no per-turn suffix at L1
    msgs1 = render_messages(c, turn_index=1, prior_replies=["reply-to-turn-0"])
    user_turns = [m for m in msgs1 if m["role"] == "user"]
    assert user_turns[0]["content"].startswith(c.first_user_prefix)  # turn 0 still has it
    assert not user_turns[1]["content"].startswith(c.first_user_prefix)  # turn 1 does not
    assert user_turns[1]["content"] == c.turns[1]


def test_L3_system_prompt_no_prefix_no_suffix():
    c = _one("L3", persona_id="magnolia")
    assert c.system_prompt is not None
    assert c.first_user_prefix is None and c.per_turn_suffix is None
    msgs = render_messages(c, turn_index=2, prior_replies=["r0", "r1"])
    assert msgs[0]["role"] == "system"
    assert c.persona_card in msgs[0]["content"]
    assert "Do not break character." in msgs[0]["content"]
    # user turns are verbatim script lines (no prefix/suffix)
    user_turns = [m for m in msgs if m["role"] == "user"]
    assert [u["content"] for u in user_turns] == [c.turns[0], c.turns[1], c.turns[2]]


def test_L5_system_prompt_plus_per_turn_suffix_every_turn():
    c = _one("L5", persona_id="magnolia")
    assert c.system_prompt is not None and c.per_turn_suffix is not None
    msgs = render_messages(c, turn_index=3, prior_replies=["r0", "r1", "r2"])
    assert msgs[0]["role"] == "system"
    assert "NEVER break character" in msgs[0]["content"]
    # EVERY user turn carries the per-turn reinforcement suffix
    user_turns = [m for m in msgs if m["role"] == "user"]
    assert len(user_turns) == 4
    for i, u in enumerate(user_turns):
        assert u["content"].endswith(c.per_turn_suffix), f"turn {i} missing suffix"
        assert c.turns[i] in u["content"]


def test_message_interleaving_threads_prior_replies():
    c = _one("L3", persona_id="briggs")
    prior = ["reply-0", "reply-1"]
    msgs = render_messages(c, turn_index=2, prior_replies=prior)
    roles = [m["role"] for m in msgs]
    # system, user, assistant, user, assistant, user(current)
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    assistants = [m["content"] for m in msgs if m["role"] == "assistant"]
    assert assistants == prior  # exactly the prior replies, in order
    assert msgs[-1]["role"] == "user"  # current turn has no reply yet


def test_turn_index_out_of_range_raises():
    c = _one("L0")
    with pytest.raises(IndexError):
        render_messages(c, turn_index=c.n_turns)  # 8 turns => valid 0..7
    with pytest.raises(IndexError):
        render_messages(c, turn_index=-1)


def test_tier_of_turn_mapping():
    c = _one("L0")
    # tier onsets from scripts.json: benign 1-3, tier1 4, tier2 5-6, tier3 7, post 8
    assert tier_of_turn(c, 0) == "benign"
    assert tier_of_turn(c, 3) == "tier1"
    assert tier_of_turn(c, 4) == "tier2"
    assert tier_of_turn(c, 6) == "tier3"
    assert tier_of_turn(c, 7) == "post"


def test_make_condition_id_format():
    assert make_condition_id("magnolia", "L5", "A", "A1", 1) == "magnolia__L5__AA1__s1"
    assert make_condition_id(PERSONA_AGNOSTIC, "L0", "C", "C2", 2) == "none__L0__CC2__s2"


def test_trajectory_variants_present_and_shaped():
    """A3 (deep-entrenchment) and A4 (zigzag) carry the new variable and non-baseline shapes."""
    from src.conditions import load_variants, build_conditions, tier_of_turn

    vs = {v["variant"]: v for v in load_variants()}
    # all four Track A variants exist with the right trajectory labels
    assert vs["A1"]["trajectory"] == "baseline-short"
    assert vs["A3"]["trajectory"] == "deep-entrenchment"
    assert vs["A4"]["trajectory"] == "zigzag"

    # entrenchment front-loads benign turns before any escalation
    a3 = vs["A3"]
    assert len(a3["turns"]) == 16
    assert a3["turn_tiers"][:10] == ("benign",) * 10
    assert "tier3" in a3["turn_tiers"]

    # zigzag is genuinely non-monotonic: tier rises, falls, rises again
    order = {"benign": 0, "tier1": 1, "tier2": 2, "tier3": 3, "post": 4}
    seq = [order[t] for t in vs["A4"]["turn_tiers"] if t != "post"]
    assert any(seq[i + 1] < seq[i] for i in range(len(seq) - 1)), "zigzag must retreat at least once"

    # per-turn tier lookup matches the variant's own tiers on a built condition
    c = next(x for x in build_conditions() if x.variant == "A4")
    assert tuple(tier_of_turn(c, i) for i in range(c.n_turns)) == vs["A4"]["turn_tiers"]
