"""SYNTHETIC scored-data fixtures for validating src/analysis.py end to end.

EVERYTHING HERE IS FAKE. No model was invoked; no benchmark script content is used beyond
the frozen per-variant TIER STRUCTURE (turn -> severity tier), which is loaded from
``benchmark/scripts.json`` so the fixtures share the real scored-CSV geometry (the full
496-condition grid x 5 fake models). The rows realize planted probability tables via
EXACT-QUOTA assignment -- within each (cohort, level, variant) stratum, exactly
``round(p * n)`` rows are positive, randomly placed with a fixed seed -- so the analysis
pipeline must recover the planted effect sizes up to rounding, not up to Bernoulli luck.
Values are planted per-contrast and are NOT behaviorally coupled (e.g. ``acr`` is assigned
independently of ``peak_resource_provided``).

Planted effects (the "effects" dataset):
  * H1: local-cohort ACR declines linearly from 0.85 at L1 to 0.60 at L5 (a 25 pp drop).
  * H2: local-cohort resource-provision failure at tier3 rises to 0.35 at L5.
  * H3: the API cohort declines only 0.85 -> 0.77, so the local-minus-API drop gap ~17 pp.
  * H4a: variant A3 carries a flat -0.20 ACR penalty (~20 pp below the A1/A2 mean at L5).
  * H4b: A4's second tier-3 turn (1-based 12) provides a resource with probability 0.25
    below baseline (~25 pp gap vs A1/A2's turn 7).

The "null" dataset plants nothing: ACR is 0.75 everywhere, resource provision never fails
(failure prob 0.0 -> H2's pre-registered positive null), and no variant or cohort differs.

File-name hygiene: ``write_synthetic_csv`` stamps SYNTHETIC into the filename and REFUSES to
write anywhere under the repository's ``results/`` tree, so fake rows can never be mistaken
for (or mixed with) real scored outcomes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.conditions import load_variants
from src.scoring import CSV_FIELDS

REPO_ROOT = Path(__file__).resolve().parent.parent

# Clearly-fake model names (never collide with any real run's results/scored/{model}.csv).
LOCAL_MODELS = ("SYNTHETIC-local-1b", "SYNTHETIC-local-3b", "SYNTHETIC-local-4b")
API_MODELS = ("SYNTHETIC-api-alpha", "SYNTHETIC-api-beta")
PERSONAS = ("p1", "p2", "p3", "p4", "p5", "p6")
SEEDS = (1, 2)
LEVELS = ("L0", "L1", "L2", "L3", "L4", "L5")

# Planted probability tables (documented above; tests assert recovery against these).
PLANT = {
    "acr_local_L1": 0.85,
    "acr_local_L5": 0.60,   # 25 pp planted H1 drop
    "acr_api_L1": 0.85,
    "acr_api_L5": 0.77,     # ~17 pp planted H3 gap
    "acr_a3_penalty": 0.20,  # planted H4a entrenchment effect
    "fail_local_L5": 0.35,  # planted H2 failure at (L5, tier3)
    "fail_local_L1": 0.05,
    "fail_api": 0.05,
    "h4b_gap": 0.25,        # planted drop in A4 turn-12 provision vs baseline
    "null_acr": 0.75,
    "null_fail": 0.0,       # -> H2's pre-registered positive null
    "c_false_alarm": 0.10,
}


def _tier3_turns_1based() -> dict[str, list[int]]:
    """1-based tier-3 turn numbers per variant, from the frozen benchmark tier structure."""
    out: dict[str, list[int]] = {}
    for v in load_variants():
        out[v["variant"]] = [i + 1 for i, t in enumerate(v["turn_tiers"]) if t == "tier3"]
    return out


def _acr_prob(cohort: str, level_num: int, variant: str, planted: bool) -> float:
    if not planted:
        return PLANT["null_acr"]
    if cohort == "local":
        p = float(np.interp(level_num, [1, 5], [PLANT["acr_local_L1"], PLANT["acr_local_L5"]]))
    else:
        p = float(np.interp(level_num, [1, 5], [PLANT["acr_api_L1"], PLANT["acr_api_L5"]]))
    if variant == "A3":
        p -= PLANT["acr_a3_penalty"]
    return float(np.clip(p, 0.01, 0.99))


def _fail_prob(cohort: str, level_num: int, planted: bool) -> float:
    if not planted:
        return PLANT["null_fail"]
    if cohort == "local":
        return float(np.interp(level_num, [1, 5], [PLANT["fail_local_L1"], PLANT["fail_local_L5"]]))
    return PLANT["fail_api"]


def _exact_quota(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean vector with EXACTLY round(p*n) True entries, randomly placed."""
    k = int(round(p * n))
    out = np.zeros(n, dtype=bool)
    out[:k] = True
    rng.shuffle(out)
    return out


def make_synthetic_scored(*, planted: bool, rng_seed: int = 20260611) -> pd.DataFrame:
    """Build one combined SYNTHETIC analysis frame (scoring schema + model/cohort/acr)."""
    rng = np.random.default_rng(rng_seed)
    variants = load_variants()
    t3 = _tier3_turns_1based()
    tiers = {v["variant"]: v["turn_tiers"] for v in variants}

    # --- the bare grid (no outcomes yet) -----------------------------------
    base: list[dict] = []
    for model, cohort in [(m, "local") for m in LOCAL_MODELS] + [(m, "api") for m in API_MODELS]:
        for v in variants:
            for seed in SEEDS:
                for level in LEVELS:
                    for persona in ("none",) if level == "L0" else PERSONAS:
                        base.append(
                            {
                                "condition_id": f"{persona}__{level}__{v['track']}{v['variant']}__s{seed}",
                                "persona_id": persona,
                                "level_id": level,
                                "track": v["track"],
                                "variant": v["variant"],
                                "seed": seed,
                                "is_control": level == "L0",
                                "model": model,
                                "cohort": cohort,
                            }
                        )
    df = pd.DataFrame(base)
    n = len(df)

    # --- exact-quota outcome assignment per (cohort, level, variant) stratum ---
    acr = np.zeros(n, dtype=bool)
    provided_at: dict[int, np.ndarray] = {}  # 1-based turn -> bool vector over df
    for (cohort, level, variant), idx in sorted(
        df.groupby(["cohort", "level_id", "variant"]).indices.items()
    ):
        lvl = int(level[1]) if level != "L0" else 1
        track = variant[0]
        acr[idx] = _exact_quota(len(idx), _acr_prob(cohort, lvl, variant, planted), rng)
        if track in ("A", "B"):
            provide_p = 1.0 - _fail_prob(cohort, lvl, planted)
            for turn in t3[variant]:
                p = provide_p
                if planted and variant == "A4" and turn == t3["A4"][1]:
                    p = max(provide_p - PLANT["h4b_gap"], 0.01)  # the planted H4b gap
                provided_at.setdefault(turn, np.zeros(n, dtype=bool))
                provided_at[turn][idx] = _exact_quota(len(idx), p, rng)
        else:  # Track C: over-trigger (false alarm) at its tier-3 turn
            p_fa = PLANT["c_false_alarm"] if planted else 0.0
            turn = t3[variant][0]
            provided_at.setdefault(turn, np.zeros(n, dtype=bool))
            provided_at[turn][idx] = _exact_quota(len(idx), p_fa, rng)
    df["acr"] = acr

    # --- derive the scoring-schema columns row by row -----------------------
    rows: list[dict] = []
    for i, r in enumerate(df.itertuples(index=False)):
        turns_provided = [t for t in t3[r.variant] if provided_at.get(t) is not None and provided_at[t][i]]
        peak = bool(turns_provided)
        first_turn = min(turns_provided) if turns_provided else None
        first_tier = tiers[r.variant][first_turn - 1] if first_turn else None
        is_c = r.track == "C"
        rows.append(
            {
                "any_resource_provided": peak,
                "peak_resource_provided": peak,
                "first_resource_tier": first_tier,
                "first_resource_turn": first_turn,
                "n_resource_turns": len(turns_provided),
                "resource_turns": ";".join(str(t) for t in turns_provided),
                "matched_pattern_ids": "lifeline_988" if peak and not is_c else "",
                "false_alarm": peak if is_c else False,
                "n_false_alarm_turns": len(turns_provided) if is_c else 0,
                "false_alarm_turns": ";".join(str(t) for t in turns_provided) if is_c else "",
                "break_character": None,
                "appropriateness": None,
            }
        )
    df = pd.concat([df, pd.DataFrame(rows)], axis=1)
    assert set(CSV_FIELDS) <= set(df.columns)  # superset of the frozen scoring schema
    return df


def write_synthetic_csv(df: pd.DataFrame, out_path: Path) -> Path:
    """Write a SYNTHETIC fixture CSV. Refuses results/ and enforces a SYNTHETIC filename."""
    out_path = Path(out_path)
    results_dir = (REPO_ROOT / "results").resolve()
    if results_dir == out_path.resolve() or results_dir in out_path.resolve().parents:
        raise ValueError(
            f"synthetic fixtures must never be written under {results_dir} "
            "(reserved for real scored outputs)"
        )
    if "synthetic" not in out_path.name.lower():
        raise ValueError("synthetic fixture filenames must contain 'SYNTHETIC'")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
