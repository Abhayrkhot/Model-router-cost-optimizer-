"""README figures, rendered once per colour scheme (GitHub picks one via <picture>)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from modelrouter.eval import metrics as M

# Reference data-viz palette: three categorical slots validated (CVD + normal-vision) in both
# modes; neutrals for references.  Light aqua is < 3:1 on the surface, so every series is
# also direct-labelled.
THEMES: dict[str, dict[str, Any]] = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "text2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#f0efec",
        "text2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70"],
    },
}

# name -> (label, x offset pt, y offset pt); placed by hand to avoid collisions.
LABELLED_MODELS = {
    "gpt-4-1106-preview": ("GPT-4 Turbo", -8, 9, "right"),
    "claude-v2": ("Claude 2", 6, -11, "left"),
    "gpt-3.5-turbo-1106": ("GPT-3.5", 6, -3, "left"),
    "yi-34b": ("Yi 34B", 2, -13, "left"),
    "mixtral-8x7b": ("Mixtral 8x7B", 6, -10, "left"),
    "mistral-7b": ("Mistral 7B", 6, -10, "left"),
    "wizardlm-13b": ("WizardLM 13B", 6, -10, "left"),
}


def _setup(theme: dict[str, Any]) -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": theme["axis"],
            "axes.labelcolor": theme["text2"],
            "axes.titlecolor": theme["text"],
            "xtick.color": theme["muted"],
            "ytick.color": theme["muted"],
            "axes.facecolor": theme["surface"],
            "figure.facecolor": theme["surface"],
            "savefig.facecolor": theme["surface"],
            "axes.grid": True,
            "grid.color": theme["grid"],
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.labelcolor": theme["text2"],
        }
    )
    return plt


def _hull_xy(c: M.Curve) -> tuple[np.ndarray, np.ndarray]:
    hc, hq = M.upper_hull(c.cost, c.quality)
    return hc, hq


def cost_quality(
    out: Path,
    curves: dict[str, M.Curve],
    router: tuple[str, str],
    single: dict[str, dict[str, float]],
    title: str,
    mode: str,
    ref_model: str = "gpt-4-1106-preview",
    budget_frac: float = 0.1,
) -> None:
    """Cost per 1k prompts (log x) vs quality: single models and their best mix, the oracle,
    one learned router, and the original length rule.  Marks the router's gain at one budget."""
    theme = THEMES[mode]
    plt = _setup(theme)
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=150)
    per_k = 1000.0

    xs = np.array([v["cost"] for v in single.values()]) * per_k
    ys = np.array([v["quality"] for v in single.values()])
    zc, zq = _hull_xy(curves["single-model"])
    ax.plot(
        zc * per_k, zq, color=theme["muted"], lw=1.5, ls="--", label="Best mix of single models"
    )
    ax.scatter(xs, ys, s=36, color=theme["muted"], zorder=3, edgecolor=theme["surface"], lw=1.5)
    for name, x, y in zip(single, xs, ys, strict=True):
        if name in LABELLED_MODELS:
            label, dx, dy, ha = LABELLED_MODELS[name]
            ax.annotate(
                label,
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8,
                color=theme["text2"],
                ha=ha,
            )

    oc, oq = _hull_xy(curves["oracle"])
    ax.plot(oc * per_k, oq, color=theme["text2"], lw=1.0, ls=":", label="Oracle (upper bound)")

    key, label = router
    hc, hq = _hull_xy(curves[key])
    color = theme["series"][0]
    ax.plot(hc * per_k, hq, color=color, lw=2.2, label=label, solid_capstyle="round", zorder=4)

    if "length-rule" in curves:
        lc = curves["length-rule"]
        ax.scatter(
            lc.cost * per_k,
            lc.quality,
            marker="D",
            s=40,
            color=theme["series"][1],
            zorder=5,
            edgecolor=theme["surface"],
            lw=1.5,
            label="Old length rule",
        )

    # Gain at a fixed budget (fraction of the reference model's cost).
    ref_cost = single[ref_model]["cost"] * per_k
    x_b = budget_frac * ref_cost
    q_r = float(M.hull_quality_at(hc * per_k, hq, np.array([x_b]))[0])
    q_z = float(M.hull_quality_at(zc * per_k, zq, np.array([x_b]))[0])
    ax.plot([x_b, x_b], [q_z, q_r], color=theme["text"], lw=1.0)
    ax.annotate(
        f"+{(q_r - q_z) * 100:.1f} pts at {budget_frac:.0%} of\nGPT-4 Turbo's cost",
        (x_b, (q_r + q_z) / 2),
        xytext=(8, -4),
        textcoords="offset points",
        fontsize=8.5,
        color=theme["text"],
        va="center",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Cost per 1,000 prompts (USD, log scale)")
    ax.set_ylabel("Mean quality (share answered correctly)")
    ax.set_title(title, loc="left", fontsize=11.5, pad=12)
    lo = min(float(xs.min()), float(oc.min() * per_k)) * 0.8
    ax.set_xlim(lo, float(xs.max()) * 2.0)
    ax.set_ylim(max(0.0, float(ys.min()) - 0.05), min(1.0, float(oq.max()) + 0.03))
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def pgr_curve(
    out: Path,
    curves: dict[str, M.Curve],
    router: tuple[str, str],
    strong_label: str,
    weak_label: str,
    mode: str,
) -> None:
    """Share of the weak->strong quality gap recovered vs share of calls to the strong model."""
    theme = THEMES[mode]
    plt = _setup(theme)
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150)
    key, label = router
    c = curves[key]
    q_w, q_s = float(c.quality[0]), float(c.quality[-1])
    y = M.pgr(c, q_w, q_s) * 100

    ax.plot([0, 100], [0, 100], color=theme["muted"], lw=1.5, ls="--", label="Random")
    if "oracle" in curves:
        o = curves["oracle"]
        ax.plot(
            o.params * 100,
            M.pgr(o, q_w, q_s) * 100,
            color=theme["text2"],
            lw=1.0,
            ls=":",
            label="Oracle",
        )
    ax.plot(c.params * 100, y, color=theme["series"][0], lw=2.2, label=label)
    for target in (50, 80):
        p = float(c.params[np.flatnonzero(y >= target - 1e-9)[0]] * 100)
        ax.plot([p, p], [0, target], color=theme["axis"], lw=1.0)
        ax.annotate(
            f"{target}% of the gap with\n{p:.0f}% of calls",
            (p, target),
            xytext=(-8, 6),
            textcoords="offset points",
            fontsize=8,
            color=theme["text"],
            ha="right",
        )
    ax.set_xlim(0, 100)
    ax.set_ylim(-3, 110)
    ax.set_xlabel(f"Calls sent to {strong_label} (%)")
    ax.set_ylabel(f"Quality gap recovered, {weak_label} -> {strong_label} (%)")
    ax.set_title(
        f"{weak_label} vs {strong_label}: which prompts need the strong model?",
        loc="left",
        fontsize=11,
        pad=10,
    )
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def outage_timeline(
    out: Path,
    runs: list[tuple[str, dict[str, Any]]],
    window: tuple[float, float],
    title: str,
    mode: str,
) -> None:
    """Per-second share of requests answered for load runs sharing a fault window (shaded).

    Each line is labelled once with its success rate over the whole fault window."""
    theme = THEMES[mode]
    plt = _setup(theme)
    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=150)
    ax.axvspan(window[0], window[1], color=theme["grid"], alpha=0.9, lw=0, zorder=0)
    ax.annotate(
        "GPT-4 Turbo returns 503",
        ((window[0] + window[1]) / 2, 4),
        ha="center",
        fontsize=8.5,
        color=theme["text2"],
    )
    mid = (window[0] + window[1]) / 2
    # label anchors: (x, y offset in points, vertical alignment) per series
    anchors = [(mid, 8, "bottom"), (window[0] + 2.5, -9, "top"), (mid, -14, "top")]
    for i, ((label, run), color) in enumerate(zip(runs, theme["series"], strict=False)):
        tl = run["timeline"]
        t = np.array([b["t"] for b in tl], dtype=float) + 0.5
        ok = np.array([100.0 * b["ok"] / max(1, b["sent"]) for b in tl])
        ax.plot(t, ok, color=color, lw=2.0, label=label, zorder=3 - i * 0.1)
        inside = [b for b in tl if window[0] <= b["t"] < window[1]]
        share = 100.0 * sum(b["ok"] for b in inside) / max(1, sum(b["sent"] for b in inside))
        x, dy, va = anchors[min(i, len(anchors) - 1)]
        y = float(np.interp(x, t, ok)) if i != 0 else 100.0
        text = f"{label}: {share:.1f}% answered" if share < 99.95 else f"{label}: 100% answered"
        ax.annotate(
            text,
            (x, y),
            xytext=(0 if i != 1 else 4, dy),
            textcoords="offset points",
            fontsize=8.5,
            color=theme["text"],
            ha="center" if i != 1 else "left",
            va=va,
        )
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlim(0, max(b["t"] for _, r in runs for b in r["timeline"]) + 1)
    ax.set_xlabel("Seconds into the run")
    ax.set_ylabel("Requests answered (%)")
    ax.set_title(title, loc="left", fontsize=11, pad=10)
    ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
