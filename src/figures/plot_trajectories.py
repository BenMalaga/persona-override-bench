"""Show-off figure: the three escalation TRAJECTORY shapes in the benchmark.

This is a pure DESIGN SCHEMATIC. It reads the scripted user turns straight from
`benchmark/scripts.json` and plots each turn's pre-registered severity tier against
its turn number. No model is ever invoked; nothing here is an outcome metric. The
point of the picture is the *shape* of the conversation the benchmark feeds a model:

  - baseline-short   (A1): a short monotonic climb to the crisis peak,
  - deep-entrenchment (A3): ten persona-reinforcing turns before any disclosure,
  - zigzag           (A4): an advance / retreat / advance "test-the-waters" path.

A small inset shows the L0-L5 persona-pressure ladder (the manipulated variable),
read from `benchmark/pressure_ladder.json`.

Run:  python -m src.figures.plot_trajectories
Output: docs/figures/escalation_trajectories.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / reproducible
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "benchmark" / "scripts.json"
LADDER = ROOT / "benchmark" / "pressure_ladder.json"
OUT = ROOT / "docs" / "figures" / "escalation_trajectories.png"

# Severity tiers as an ordinal y-axis (benign -> tier1 -> tier2 -> tier3).
# "post" is the small-talk pivot back down to benign after the peak.
TIER_Y = {"benign": 0, "tier1": 1, "tier2": 2, "tier3": 3, "post": 0}
TIER_LABELS = ["benign", "tier 1\nlow mood", "tier 2\nwithdrawal", "tier 3\npeak"]

# Colorblind-friendly (Okabe-Ito subset).
COLORS = {
    "baseline-short": "#0072B2",      # blue
    "deep-entrenchment": "#D55E00",   # vermillion
    "zigzag": "#009E73",              # bluish green
}
# Plot one representative variant per trajectory shape.
SHOWN_VARIANTS = {
    "baseline-short": "A1",
    "deep-entrenchment": "A3",
    "zigzag": "A4",
}
SHAPE_ORDER = ["baseline-short", "deep-entrenchment", "zigzag"]


def load_track_a_variants():
    data = json.loads(SCRIPTS.read_text())
    track_a = next(t for t in data["tracks"] if t["id"] == "A")
    return {v["id"]: v for v in track_a["variants"]}


def main() -> None:
    variants = load_track_a_variants()

    fig, ax = plt.subplots(figsize=(10.0, 5.6), dpi=150)

    peak_handles = []
    for shape in SHAPE_ORDER:
        vid = SHOWN_VARIANTS[shape]
        turns = variants[vid]["turns"]
        xs = list(range(1, len(turns) + 1))
        ys = [TIER_Y[t["tier"]] for t in turns]
        color = COLORS[shape]

        ax.plot(
            xs, ys,
            marker="o", markersize=5.5, linewidth=2.4,
            color=color, alpha=0.95, zorder=3,
            label=f"{shape}  ({vid}, {len(turns)} turns)",
        )

        # Mark each tier-3 crisis peak, the turn where the law requires a referral.
        for x, t in zip(xs, turns):
            if t["tier"] == "tier3":
                ax.scatter(
                    [x], [3], s=190, facecolors="none",
                    edgecolors=color, linewidths=2.2, zorder=4,
                )

    # Annotate the regulatory expectation at the peak tier.
    ax.axhline(3, color="#999999", linestyle=":", linewidth=1.1, zorder=1)
    ax.text(
        16.9, 3.45,
        "tier 3 = explicit active ideation\n(no method/means content): the turn\na crisis referral is legally required\n(CA SB 243 § 22602(b)(1))",
        va="bottom", ha="center", fontsize=8.0, color="#444444",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="#cccccc", lw=0.8),
    )

    # Highlight the zigzag retreat (the H4(b) "reset" probe): turn 9 peak -> retreat -> turn 12 peak.
    zz = variants["A4"]["turns"]
    if zz[8]["tier"] == "tier3" and zz[11]["tier"] == "tier3":
        arrow = FancyArrowPatch(
            (9, 3), (12, 3),
            connectionstyle="arc3,rad=-0.45",
            arrowstyle="-|>", mutation_scale=14,
            color=COLORS["zigzag"], lw=1.4, alpha=0.7, zorder=2,
        )
        ax.add_patch(arrow)
        ax.text(
            10.5, 3.62, "retreat then re-escalate",
            ha="center", va="bottom", fontsize=7.8,
            color=COLORS["zigzag"], style="italic",
        )

    ax.set_xlim(0.3, 22.5)
    ax.set_ylim(-0.45, 4.05)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(TIER_LABELS, fontsize=9)
    ax.set_xlabel("conversation turn", fontsize=11)
    ax.set_ylabel("disclosed severity (ordinal)", fontsize=11)
    ax.set_title(
        "Three escalation trajectories in PersonaOverride-Bench",
        fontsize=13.5, fontweight="bold", pad=12,
    )
    ax.text(
        0.0, 1.015,
        "Scripted user turns (Track A, crisis) read from benchmark/scripts.json, a design schematic, not model output.",
        transform=ax.transAxes, fontsize=8.2, color="#666666",
    )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", color="#eaeaea", linewidth=0.8, zorder=0)
    ax.legend(loc="upper left", frameon=True, fontsize=8.6, framealpha=0.95)

    # ---- Inset: the L0-L5 persona-pressure ladder (the manipulated variable) ----
    ladder = json.loads(LADDER.read_text())["levels"]
    iax = fig.add_axes([0.655, 0.135, 0.31, 0.34])
    short = {
        "L0": "no persona (control)",
        "L1": "role-play request",
        "L2": "+ stay-in-character",
        "L3": "system prompt + do-not-break",
        "L4": "absolute lock-in",
        "L5": "lock-in + per-turn reinforce",
    }
    n = len(ladder)
    cmap = plt.get_cmap("viridis")
    for i, lvl in enumerate(ladder):
        y = n - 1 - i
        shade = cmap(0.12 + 0.7 * (i / (n - 1)))
        iax.barh(y, i + 0.6, height=0.7, color=shade, edgecolor="white", linewidth=0.6)
        iax.text(0.12, y, f"{lvl['id']}  {short[lvl['id']]}",
                 va="center", ha="left", fontsize=7.0, color="#222222")
    iax.set_xlim(0, n + 0.3)
    iax.set_ylim(-0.6, n - 0.4)
    iax.set_xticks([])
    iax.set_yticks([])
    for s in iax.spines.values():
        s.set_visible(False)
    iax.set_title("persona-pressure ladder\n(applied to every trajectory)",
                  fontsize=8.0, color="#333333", pad=3)
    iax.annotate("", xy=(0.06, -0.55), xytext=(0.06, n - 0.45),
                 xycoords=("axes fraction", "data"),
                 arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.0))
    iax.text(-0.02, n / 2 - 0.5, "weaker  →  stronger override pressure",
             rotation=90, va="center", ha="right", fontsize=6.6,
             color="#777777", transform=iax.get_yaxis_transform())

    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.12)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}  ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
