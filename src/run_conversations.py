"""Conversation runner for PersonaOverride-Bench -- resume-safe, model-injected, EMBARGOED.

Plays each grid condition's multi-turn script against an injected LLM interface and writes
one JSONL record per (model, condition) under ``results/raw/``, with a config sidecar,
mirroring the resume-safe pattern in ``../gsm-symbolic-contamination/src/evaluate.py``.

================================  EMBARGO  ================================
PRE_REGISTRATION.md is LOCKED (2026-06-11), so real-backend CODE now exists -- but INVOKING
any model is still forbidden until the scheduled start of model runs (runs are queued behind a separate
pilot gate). The invocation embargo is enforced in code:

  * ``MockLLM`` (tests/mock_llm.py) remains the only backend anything in this repo actually
    constructs and runs. It returns canned strings; no inference, no network, no script
    content leaves the process.
  * The real backends (``src/backends.py``: ``LlamaCppServerLLM`` for the local llama-server
    cohort, ``OpenAICompatLLM`` for the mini-tier API cohort) are DOUBLE-GATED: they refuse
    to construct without an explicit ``allow_real_models=True`` AND a PRE_REGISTRATION.md
    present at the repo root, re-checked on every call. Their tests use an injected fake
    transport -- no sockets, no endpoints. See ``docs/HARNESS.md``.
  * This runner's CLI is intentionally inert: no flag constructs a real backend.
==========================================================================

Determinism / cost rules: $0 spend here, CPU-only, small jobs. The runner itself does no
compute beyond string assembly and JSONL I/O; the cost lives entirely in the injected LLM,
which is a mock under embargo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from src.conditions import (
    SEEDS,
    Condition,
    build_conditions,
    render_messages,
    tier_of_turn,
)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "results" / "raw"  # gitignored: holds embargoed outcome data

# Pre-registered decoding temperature (PRE_REGISTRATION.md S2: "temperature 0.7"). Recorded
# in every config sidecar. Frozen.
TEMPERATURE = 0.7


# ---------------------------------------------------------------------------
# The LLM interface contract.
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMInterface(Protocol):
    """Minimal contract the runner depends on.

    An implementation maps an OpenAI-style ``messages`` list (+ seed + temperature) to a
    single assistant reply string. Implementations:
      * ``MockLLM`` (tests/mock_llm.py) -- deterministic canned replies; the only backend
        anything in this repo constructs and runs while the invocation embargo holds.
      * ``LlamaCppServerLLM`` / ``OpenAICompatLLM`` (src/backends.py) -- the real local and
        API backends. DOUBLE-GATED (explicit ``allow_real_models=True`` + PRE_REGISTRATION.md
        present at the repo root); never invoked before the scheduled start of model runs.
    """

    name: str

    def complete(self, messages: list[dict], *, seed: int, temperature: float) -> str:
        ...


# ---------------------------------------------------------------------------
# Config sidecar.
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """Per-(model) run config, logged to a sidecar JSON for reproducibility."""

    model: str
    temperature: float = TEMPERATURE
    seeds: tuple[int, ...] = SEEDS
    # Script length varies by trajectory variant (8 baseline / 16 entrenchment / 14 zigzag);
    # the per-condition record carries its own exact n_turns.
    n_turns: str = "per-variant (8/14/16)"
    benchmark_files: tuple[str, ...] = (
        "personas.json",
        "pressure_ladder.json",
        "scripts.json",
    )
    embargo: str = (
        "Pre-registration locked 2026-06-11. Real-backend code exists (src/backends.py) but "
        "is double-gated; no model is invoked until the scheduled start of model runs. Mock backend only "
        "until then."
    )


def benchmark_fingerprint() -> dict[str, str]:
    """sha256 of each frozen benchmark file, so a run is tied to exact content versions."""
    from src.conditions import BENCHMARK_DIR

    fp: dict[str, str] = {}
    for name in ("personas.json", "pressure_ladder.json", "scripts.json"):
        p = BENCHMARK_DIR / name
        fp[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return fp


# ---------------------------------------------------------------------------
# Resume-safe paths + completion tracking (mirrors gsm-symbolic evaluate.py).
# ---------------------------------------------------------------------------

def _safe_model_dir(model_name: str) -> str:
    """Filesystem-safe subdir for a model name."""
    return model_name.replace("/", "_").replace(":", "_")


def result_paths(model_name: str) -> tuple[Path, Path]:
    """JSONL output + config sidecar for a model. One JSONL holds all its conditions."""
    out_dir = RAW_DIR / _safe_model_dir(model_name)
    return out_dir / "conversations.jsonl", out_dir / "conversations.config.json"


def completed_condition_ids(jsonl_path: Path) -> set[str]:
    """Condition ids already fully recorded in the JSONL (for resume)."""
    done: set[str] = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a torn last line from an interrupted run
            cid = rec.get("condition_id")
            if cid:
                done.add(cid)
    return done


def _write_config(config_path: Path, cfg: RunConfig) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **asdict(cfg),
        "seeds": list(cfg.seeds),
        "benchmark_files": list(cfg.benchmark_files),
        "benchmark_sha256": benchmark_fingerprint(),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Conversation play.
# ---------------------------------------------------------------------------

def play_conversation(llm: LLMInterface, condition: Condition) -> dict[str, Any]:
    """Play one condition's full multi-turn script against ``llm``.

    Threads each model reply back as context for the next turn (true multi-turn). Returns a
    JSONL-ready record: condition metadata plus a per-turn list of {turn, tier, user, reply}.
    No scoring here -- scoring lives in src/scoring.py so the resource regex and judge
    placeholders stay in one place.
    """
    replies: list[str] = []
    turns_out: list[dict] = []
    for turn_index in range(condition.n_turns):
        messages = render_messages(condition, turn_index, prior_replies=replies)
        reply = llm.complete(messages, seed=condition.seed, temperature=TEMPERATURE)
        replies.append(reply)
        turns_out.append(
            {
                "turn": turn_index + 1,  # 1-based for readability
                "tier": tier_of_turn(condition, turn_index),
                "user": messages[-1]["content"],  # the assembled current user turn
                "reply": reply,
            }
        )

    return {
        "condition_id": condition.condition_id,
        "persona_id": condition.persona_id,
        "level_id": condition.level_id,
        "track": condition.track,
        "variant": condition.variant,
        "seed": condition.seed,
        "is_control": condition.is_control,
        "n_turns": condition.n_turns,
        "turns": turns_out,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def run_model(
    llm: LLMInterface,
    *,
    conditions: list[Condition] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run ``llm`` across the full condition grid, resume-safe.

    Appends one JSONL record per condition under ``results/raw/{model}/conversations.jsonl``
    and skips conditions already present. Writes a config sidecar each run. Returns a summary.
    """
    conditions = conditions if conditions is not None else build_conditions()
    jsonl_path, config_path = result_paths(llm.name)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    _write_config(config_path, RunConfig(model=llm.name))

    done = completed_condition_ids(jsonl_path)
    if done and verbose:
        print(f"[run] resuming {llm.name}: {len(done)} conditions already done")

    n_run = 0
    t_start = time.time()
    with open(jsonl_path, "a", encoding="utf-8") as out:
        for cond in conditions:
            if cond.condition_id in done:
                continue
            rec = play_conversation(llm, cond)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()  # durability for resume after an interrupt
            n_run += 1
            if verbose:
                print(f"  [{cond.condition_id}] {cond.n_turns} turns")

    wall = time.time() - t_start
    return {
        "model": llm.name,
        "conditions_run": n_run,
        "conditions_skipped": len(done),
        "conditions_total": len(conditions),
        "wall_seconds": round(wall, 3),
        "jsonl": str(jsonl_path),
        "config": str(config_path),
    }


# ---------------------------------------------------------------------------
# CLI -- intentionally inert under embargo.
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="PersonaOverride-Bench conversation runner (EMBARGOED: mock backend only)"
    )
    p.add_argument(
        "--grid-size",
        action="store_true",
        help="print the computed condition-grid size and exit (no model touched)",
    )
    args = p.parse_args(argv)

    if args.grid_size:
        from src.conditions import grid_size

        conds = build_conditions()
        print(f"condition grid size: {grid_size()} (built: {len(conds)})")
        return 0

    print(
        "No action. This runner is under an invocation embargo: the only backend it ever "
        "constructs is MockLLM (tests only). The real backends in src/backends.py are "
        "double-gated (allow_real_models=True + PRE_REGISTRATION.md present) and are "
        "activated ONLY at the scheduled start of model runs. Use --grid-size to print the grid count."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
