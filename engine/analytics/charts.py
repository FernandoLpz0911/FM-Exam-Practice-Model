"""Matplotlib charts: readiness over time, per-category mastery."""
from __future__ import annotations

import io
from datetime import datetime

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # must be called before importing pyplot
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def _fig_to_png(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def readiness_over_time(snapshots: list[dict], pass_threshold: float = 0.70) -> bytes:
    """Line chart of readiness score vs time.

    Args:
        snapshots: list of {taken_at, score} dicts (oldest first)
        pass_threshold: horizontal reference line (default 0.70)
    """
    if not snapshots:
        fig, ax = plt.subplots(figsize=(8, 3), facecolor="#0f1117")
        ax.set_facecolor("#1a1d27")
        ax.text(0.5, 0.5, "No readiness data yet", transform=ax.transAxes,
                ha="center", va="center", color="#8892a4", fontsize=13)
        ax.set_axis_off()
        return _fig_to_png(fig)

    dates = [datetime.fromisoformat(s["taken_at"]) for s in snapshots]
    scores = [s["score"] for s in snapshots]

    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor="#0f1117")
    ax.set_facecolor("#1a1d27")

    ax.plot(dates, scores, color="#6c8ef5", linewidth=2, marker="o",
            markersize=4, zorder=3)
    ax.axhline(pass_threshold, color="#4ade80", linewidth=1.2,
               linestyle="--", alpha=0.7, label=f"Pass target ({pass_threshold:.0%})")
    ax.fill_between(dates, scores, alpha=0.15, color="#6c8ef5")

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Readiness", color="#8892a4", fontsize=11)
    ax.tick_params(colors="#8892a4", labelsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30, ha="right")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2e3245")
    ax.legend(fontsize=9, framealpha=0.2, labelcolor="#e2e8f0")
    ax.set_title("Readiness Over Time", color="#e2e8f0", fontsize=12, pad=10)
    fig.tight_layout()
    return _fig_to_png(fig)


def category_mastery(detail: dict) -> bytes:
    """Horizontal bar chart of per-category mastery scores.

    Args:
        detail: {category: {score, band_weight, concepts}} from compute_readiness
    """
    if not detail:
        fig, ax = plt.subplots(figsize=(6, 2.5), facecolor="#0f1117")
        ax.set_facecolor("#1a1d27")
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color="#8892a4")
        ax.set_axis_off()
        return _fig_to_png(fig)

    cats = list(detail.keys())
    scores = [detail[c]["score"] for c in cats]
    colors = ["#6c8ef5" if s >= 0.70 else "#fbbf24" if s >= 0.50 else "#f87171"
              for s in scores]

    fig, ax = plt.subplots(figsize=(6, max(2.5, len(cats) * 0.7)), facecolor="#0f1117")
    ax.set_facecolor("#1a1d27")

    bars = ax.barh(cats, scores, color=colors, height=0.5)
    ax.set_xlim(0, 1.05)
    ax.axvline(0.70, color="#4ade80", linewidth=1.2, linestyle="--",
               alpha=0.7, label="70%")
    ax.set_xlabel("Mastery", color="#8892a4", fontsize=10)
    ax.tick_params(colors="#8892a4", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2e3245")

    for bar, score in zip(bars, scores):
        ax.text(min(score + 0.02, 0.98), bar.get_y() + bar.get_height() / 2,
                f"{score:.0%}", va="center", color="#e2e8f0", fontsize=9)

    ax.set_title("Per-Category Mastery", color="#e2e8f0", fontsize=12, pad=10)
    ax.legend(fontsize=9, framealpha=0.2, labelcolor="#e2e8f0")
    fig.tight_layout()
    return _fig_to_png(fig)


def ablation_comparison(fsrs_log: list[dict], dkt_log: list[dict],
                        pass_threshold: float = 0.70) -> bytes:
    """Line chart comparing FSRS-only vs DKT-hybrid readiness curves."""
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0f1117")
    ax.set_facecolor("#1a1d27")

    if fsrs_log:
        steps_f = [d["step"] for d in fsrs_log]
        scores_f = [d["readiness"] for d in fsrs_log]
        ax.plot(steps_f, scores_f, color="#8892a4", linewidth=2, label="FSRS-only")

    if dkt_log:
        steps_d = [d["step"] for d in dkt_log]
        scores_d = [d["readiness"] for d in dkt_log]
        ax.plot(steps_d, scores_d, color="#6c8ef5", linewidth=2, label="FSRS+DKT")

    ax.axhline(pass_threshold, color="#4ade80", linewidth=1.2,
               linestyle="--", alpha=0.7, label=f"Pass ({pass_threshold:.0%})")

    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Study steps", color="#8892a4", fontsize=10)
    ax.set_ylabel("Readiness", color="#8892a4", fontsize=10)
    ax.tick_params(colors="#8892a4", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2e3245")
    ax.legend(fontsize=10, framealpha=0.2, labelcolor="#e2e8f0")
    ax.set_title("Ablation: FSRS-only vs FSRS+DKT", color="#e2e8f0", fontsize=12, pad=10)
    fig.tight_layout()
    return _fig_to_png(fig)


def predicted_vs_actual(sample_exams: list[dict]) -> bytes:
    """Scatter of predicted readiness vs actual sample-exam score.

    Args:
        sample_exams: list of {taken_at, score, predicted} dicts
    """
    valid = [e for e in sample_exams if e.get("predicted") is not None]

    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0f1117")
    ax.set_facecolor("#1a1d27")

    if not valid:
        ax.text(0.5, 0.5, "No sample exam data yet", transform=ax.transAxes,
                ha="center", va="center", color="#8892a4", fontsize=12)
        ax.set_axis_off()
        return _fig_to_png(fig)

    xs = [e["predicted"] for e in valid]
    ys = [e["score"] for e in valid]
    ax.scatter(xs, ys, color="#6c8ef5", s=60, zorder=3)
    ax.plot([0, 1], [0, 1], color="#2e3245", linewidth=1, linestyle="--")
    ax.axhline(0.70, color="#4ade80", linewidth=1, linestyle=":", alpha=0.6)
    ax.axvline(0.70, color="#4ade80", linewidth=1, linestyle=":", alpha=0.6)

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Predicted readiness", color="#8892a4", fontsize=10)
    ax.set_ylabel("Actual score", color="#8892a4", fontsize=10)
    ax.tick_params(colors="#8892a4", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2e3245")
    ax.set_title("Predicted vs Actual", color="#e2e8f0", fontsize=12, pad=10)
    fig.tight_layout()
    return _fig_to_png(fig)
