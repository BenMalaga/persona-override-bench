"""End-to-end validation of src/analysis.py on SYNTHETIC scored data only.

Two fixtures from tests/synthetic_scored.py (both CSV-round-tripped through pandas, exactly
as real scored data would be loaded):
  * "effects": planted 25 pp L1->L5 ACR drop (H1), 35% L5 failure (H2), ~17 pp cohort gap
    (H3), 20 pp A3 entrenchment penalty (H4a), 25 pp A4 turn-12 gap (H4b).
  * "null": nothing planted -- flat ACR, zero resource failure.

The pipeline must recover every planted effect and report the null as a null. No model is
invoked anywhere; all rows are random draws from planted probability tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import analysis
from src.scoring import CSV_FIELDS
from tests.synthetic_scored import make_synthetic_scored, write_synthetic_csv

N_BOOT = 300  # plenty for stable percentile CIs at test scale; keeps the suite CPU-light


@pytest.fixture(scope="module")
def planted_df(tmp_path_factory) -> pd.DataFrame:
    tmp = tmp_path_factory.mktemp("synthetic")
    path = write_synthetic_csv(
        make_synthetic_scored(planted=True), tmp / "SYNTHETIC_scored_planted_effects.csv"
    )
    return pd.read_csv(path)  # round-trip: analysis sees CSV dtypes, as with real data


@pytest.fixture(scope="module")
def null_df(tmp_path_factory) -> pd.DataFrame:
    tmp = tmp_path_factory.mktemp("synthetic")
    path = write_synthetic_csv(
        make_synthetic_scored(planted=False), tmp / "SYNTHETIC_scored_null.csv"
    )
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Fixture hygiene.
# ---------------------------------------------------------------------------

def test_fixture_schema_is_superset_of_scoring_schema(planted_df):
    assert set(CSV_FIELDS) <= set(planted_df.columns)
    assert {"model", "cohort", "acr"} <= set(planted_df.columns)
    # full grid geometry: (3 local + 2 api models) x 496 conditions
    assert len(planted_df) == 5 * 496
    assert planted_df["model"].str.contains("SYNTHETIC").all()  # unmistakably fake


def test_synthetic_writer_refuses_results_dir(planted_df):
    with pytest.raises(ValueError, match="never"):
        write_synthetic_csv(
            planted_df, analysis.ROOT / "results" / "SYNTHETIC_should_never_land_here.csv"
        )
    with pytest.raises(ValueError, match="SYNTHETIC"):
        write_synthetic_csv(planted_df, analysis.ROOT / "data" / "innocuous_name.csv")


def test_no_synthetic_files_under_results():
    hits = [p for p in (analysis.ROOT / "results").rglob("*") if "synthetic" in p.name.lower()]
    assert hits == [], f"synthetic fixtures leaked into results/: {hits}"


# ---------------------------------------------------------------------------
# H1 dose-response.
# ---------------------------------------------------------------------------

def test_h1_recovers_planted_drop(planted_df):
    r = analysis.h1_dose_response(planted_df)
    assert 18.0 <= r["drop_pp"] <= 32.0          # planted 25 pp
    assert r["or_per_level"] <= analysis.H1_MAX_OR_PER_LEVEL
    assert r["p_value"] < analysis.ALPHA
    assert r["supported"] is True
    assert "GEE" in r["method"]                  # the documented fallback is on the record


def test_h1_reports_null(null_df):
    r = analysis.h1_dose_response(null_df)
    assert abs(r["drop_pp"]) < 6.0               # nothing planted
    assert 0.90 <= r["or_per_level"] <= 1.10
    assert r["supported"] is False


# ---------------------------------------------------------------------------
# H2 failure rate at (L5, tier3).
# ---------------------------------------------------------------------------

def test_h2_recovers_planted_failure(planted_df):
    r = analysis.h2_failure_rate(planted_df, n_boot=N_BOOT)
    assert 0.25 <= r["failure_rate"] <= 0.42     # planted 0.35 (A4's two chances dilute it)
    assert r["ci95"][0] >= analysis.H2_MIN_CI_LOWER
    assert r["supported"] is True
    assert r["positive_null"] is False
    assert r["bootstrap_unit"] == "script (variant)"


def test_h2_reports_null_and_positive_null(null_df):
    r = analysis.h2_failure_rate(null_df, n_boot=N_BOOT)
    assert r["failure_rate"] == 0.0              # failure prob planted at exactly 0
    assert r["supported"] is False
    assert r["positive_null"] is True            # <= 5% in every model: the reportable null


# ---------------------------------------------------------------------------
# H3 cohort gap.
# ---------------------------------------------------------------------------

def test_h3_recovers_planted_gap(planted_df):
    r = analysis.h3_cohort_gap(planted_df, n_boot=N_BOOT)
    assert 10.0 <= r["gap_pp"] <= 26.0           # planted ~17 pp
    assert r["ci95_pp"][0] > 0.0
    assert r["supported"] is True


def test_h3_reports_null(null_df):
    r = analysis.h3_cohort_gap(null_df, n_boot=N_BOOT)
    assert abs(r["gap_pp"]) < 6.0
    assert r["supported"] is False


def test_h3_requires_both_cohorts(planted_df):
    local_only = planted_df[planted_df["cohort"] == "local"]
    with pytest.raises(ValueError, match="both cohorts"):
        analysis.h3_cohort_gap(local_only, n_boot=10)


# ---------------------------------------------------------------------------
# H4a entrenchment.
# ---------------------------------------------------------------------------

def test_h4a_recovers_planted_entrenchment(planted_df):
    r = analysis.h4a_entrenchment(planted_df, n_boot=N_BOOT)
    assert 12.0 <= r["diff_pp"] <= 28.0          # planted 20 pp
    assert r["acr_by_variant"]["A3"] < min(
        r["acr_by_variant"]["A1"], r["acr_by_variant"]["A2"]
    )
    assert r["supported"] is True


def test_h4a_reports_null(null_df):
    r = analysis.h4a_entrenchment(null_df, n_boot=N_BOOT)
    assert abs(r["diff_pp"]) < 8.0
    assert r["supported"] is False


# ---------------------------------------------------------------------------
# H4b zigzag / retreat reset.
# ---------------------------------------------------------------------------

def test_h4b_turn_constants_match_frozen_scripts():
    analysis.verify_h4b_turn_constants()  # raises on any drift from benchmark/scripts.json
    assert analysis.A4_SECOND_TIER3_TURN == 12  # 1-based turn 12 == 0-based index 11
    assert analysis.BASELINE_TIER3_TURN == 7


def test_h4b_recovers_planted_gap(planted_df):
    r = analysis.h4b_zigzag(planted_df, n_boot=N_BOOT)
    assert 15.0 <= r["diff_pp"] <= 35.0          # planted 25 pp
    assert r["rate_a4_turn12"] < r["rate_baseline_turn7"]
    assert r["supported"] is True
    assert r["turns"]["a4_second_tier3_turn_1based"] == 12


def test_h4b_reports_null(null_df):
    r = analysis.h4b_zigzag(null_df, n_boot=N_BOOT)
    assert abs(r["diff_pp"]) < 8.0
    assert r["supported"] is False


# ---------------------------------------------------------------------------
# run_all orchestration + CLI.
# ---------------------------------------------------------------------------

def test_run_all_planted_full_report(planted_df):
    report = analysis.run_all(planted_df, n_boot=N_BOOT)
    assert {"H1", "H2", "H3", "H4a", "H4b"} <= set(report)
    for key in ("H1", "H2", "H3", "H4a", "H4b"):
        assert report[key].get("supported") is True, key


def test_run_all_null_reports_nulls(null_df):
    report = analysis.run_all(null_df, n_boot=N_BOOT)
    for key in ("H1", "H2", "H3", "H4a", "H4b"):
        assert report[key].get("supported") is False, key
    assert report["H2"]["positive_null"] is True


def test_run_all_without_acr_skips_judge_dependent_tests(planted_df):
    no_judge = planted_df.drop(columns=["acr"])
    report = analysis.run_all(no_judge, n_boot=N_BOOT)
    for key in ("H1", "H3", "H4a"):  # ACR requires the validated judge back-fill
        assert report[key]["skipped"] is True
    for key in ("H2", "H4b"):        # deterministic-regex tests always run
        assert "supported" in report[key]


def test_cli_writes_json_report(planted_df, tmp_path, capsys):
    csv_path = tmp_path / "SYNTHETIC_combined.csv"
    planted_df.to_csv(csv_path, index=False)
    out_path = tmp_path / "SYNTHETIC_report.json"
    rcode = analysis.main(
        ["--combined-csv", str(csv_path), "--out", str(out_path), "--n-boot", "100"]
    )
    assert rcode == 0
    assert out_path.exists()
    import json

    report = json.loads(out_path.read_text())
    assert report["H2"]["supported"] is True
