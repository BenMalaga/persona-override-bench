"""Runner tests: MockLLM end-to-end, resume-safety, config sidecar, and protocol checks.
(The real-backend gating tests live in tests/test_backends.py.)"""

from __future__ import annotations

import json

import pytest

from src import run_conversations as rc
from src.conditions import build_conditions
from tests.mock_llm import IN_CHARACTER_REPLY, MockLLM


@pytest.fixture()
def small_grid():
    """A tiny 2-condition slice of the grid so runner tests stay fast and CPU-cheap."""
    conds = build_conditions()
    # pick one L0 control and one L5 persona condition on the same script
    l0 = next(c for c in conds if c.level_id == "L0" and c.variant == "A1" and c.seed == 1)
    l5 = next(
        c
        for c in conds
        if c.level_id == "L5" and c.variant == "A1" and c.seed == 1 and c.persona_id == "magnolia"
    )
    return [l0, l5]


@pytest.fixture()
def redirect_raw(tmp_path, monkeypatch):
    """Redirect the runner's RAW_DIR into a tmp dir so tests never write into the repo."""
    monkeypatch.setattr(rc, "RAW_DIR", tmp_path / "raw")
    return tmp_path


def test_run_model_writes_jsonl_and_config(small_grid, redirect_raw):
    llm = MockLLM(name="mock-A")
    summary = rc.run_model(llm, conditions=small_grid)

    assert summary["conditions_run"] == 2
    assert summary["conditions_skipped"] == 0

    jsonl_path, config_path = rc.result_paths("mock-A")
    assert jsonl_path.exists() and config_path.exists()

    records = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    assert len(records) == 2
    for rec in records:
        assert rec["n_turns"] == 8
        assert len(rec["turns"]) == 8
        # every turn was answered by the mock's in-character default
        assert all(t["reply"] == IN_CHARACTER_REPLY for t in rec["turns"])
        # tiers are present and correctly ordered at the boundaries
        assert rec["turns"][0]["tier"] == "benign"
        assert rec["turns"][6]["tier"] == "tier3"

    cfg = json.loads(config_path.read_text())
    assert cfg["model"] == "mock-A"
    assert cfg["temperature"] == 0.7
    assert cfg["seeds"] == [1, 2]
    # benchmark content fingerprinted so a run is tied to exact frozen JSON versions
    assert set(cfg["benchmark_sha256"]) == {
        "personas.json",
        "pressure_ladder.json",
        "scripts.json",
    }
    assert "embargo" in cfg


def test_resume_safety_rerun_skips_completed(small_grid, redirect_raw):
    llm = MockLLM(name="mock-resume")
    first = rc.run_model(llm, conditions=small_grid)
    assert first["conditions_run"] == 2

    jsonl_path, _ = rc.result_paths("mock-resume")
    size_after_first = jsonl_path.stat().st_size

    # rerun: everything already done -> zero new runs, file unchanged
    llm2 = MockLLM(name="mock-resume")
    second = rc.run_model(llm2, conditions=small_grid)
    assert second["conditions_run"] == 0
    assert second["conditions_skipped"] == 2
    assert jsonl_path.stat().st_size == size_after_first
    # the mock for the rerun was never asked to complete anything
    assert llm2.calls == []


def test_resume_safety_partial_completion(small_grid, redirect_raw):
    # run only the first condition, then the full slice; only the missing one is run
    llm = MockLLM(name="mock-partial")
    rc.run_model(llm, conditions=[small_grid[0]])

    llm2 = MockLLM(name="mock-partial")
    summary = rc.run_model(llm2, conditions=small_grid)
    assert summary["conditions_run"] == 1
    assert summary["conditions_skipped"] == 1


def test_resume_tolerates_torn_last_line(small_grid, redirect_raw):
    llm = MockLLM(name="mock-torn")
    rc.run_model(llm, conditions=small_grid)
    jsonl_path, _ = rc.result_paths("mock-torn")

    # append a torn (invalid-JSON) partial line, simulating an interrupted write
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write('{"condition_id": "partial", "turns": [')  # no newline, broken JSON

    done = rc.completed_condition_ids(jsonl_path)
    # both real conditions are still recognized as done; the torn line is ignored
    assert {c.condition_id for c in small_grid} <= done


def test_mock_llm_satisfies_protocol():
    # MockLLM is a structural match for the runtime-checkable LLMInterface protocol
    assert isinstance(MockLLM(name="x"), rc.LLMInterface)
