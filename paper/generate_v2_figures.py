"""Generate v2 publication-quality figures for PhaseBreak paper.

Produces:
1. fig4_v2_benchmark_heatmap.pdf  — quality scores heatmap (all domains × episodes)
2. fig5_forward_predictions.pdf   — April 2026 predictions bar chart (13 assets)
3. fig6_baselines_comparison.pdf  — grouped bar: 6 methods × Precision/Recall/F1
4. fig7_hurst_mfdfa_scatter.pdf   — scatter Hurst vs MFDFA Δα with separation lines

All saved as both PDF (vector) and PNG (150 dpi) in paper/figures/.
Run from repo root OR from paper/ directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# WHY: allow running from either repo root or paper/ subdirectory
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

# Canonical output dir (paper/figures/)
OUTPUT_DIR = _HERE / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data paths
BENCHMARK_JSON = _ROOT / "data" / "v2_results" / "benchmark_results.json"
BASELINES_JSON = _ROOT / "data" / "v2_results" / "baselines.json"
PREDICTIONS_JSON = _ROOT / "data" / "predictions" / "april_2026_predictions.json"

# ─── Global rcParams for publication style ───────────────────────────────────
plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,  # WHY: embeds fonts in PDF for journal submission
        "ps.fonttype": 42,
    }
)

# Colorblind-friendly palette (Wong 2011)
CB_RED = "#D55E00"
CB_BLUE = "#0072B2"
CB_GREEN = "#009E73"
CB_ORANGE = "#E69F00"
CB_PURPLE = "#CC79A7"
CB_SKYBLUE = "#56B4E9"
CB_VERMILION = "#D55E00"
CB_YELLOW = "#F0E442"


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _save(fig: plt.Figure, stem: str) -> None:
    """Save figure as PDF and PNG with consistent naming."""
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    png_path = OUTPUT_DIR / f"{stem}.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {stem}.pdf + .png")


# ─── Figure 4: Benchmark heatmap ─────────────────────────────────────────────

DOMAIN_DISPLAY = {
    "finance": "Finance",
    "commodities": "Commodities",
    "housing": "Housing (FHFA)",
    "housing_monthly": "Housing (Zillow)",
    "adversarial": "Adversarial",
    "forward": "Forward 2024-25",
}

VERDICT_SHORT = {
    "BUBBLE": "B",
    "POSSIBLE": "P",
    "NO_BUBBLE": "—",
}

VERDICT_MARKER = {
    "BUBBLE": "★",
    "POSSIBLE": "◆",
    "NO_BUBBLE": "·",
}


def fig4_benchmark_heatmap(data: dict[str, Any]) -> None:
    """Heatmap: rows = domains, columns = episodes within domain.

    Color encodes quality score (0=blue, 1=red).
    Text overlay shows verdict abbreviation + correct/incorrect marker.
    """
    domains_order = [
        "finance",
        "commodities",
        "housing",
        "housing_monthly",
        "adversarial",
        "forward",
    ]
    domain_episodes: list[tuple[str, list[dict]]] = []
    for d in domains_order:
        if d in data["domains"]:
            episodes = data["domains"][d]["episodes"]
            domain_episodes.append((d, episodes))

    max_eps = max(len(eps) for _, eps in domain_episodes)
    n_rows = len(domain_episodes)

    # Build matrix
    quality_matrix = np.full((n_rows, max_eps), np.nan)
    verdict_matrix: list[list[str]] = [[""] * max_eps for _ in range(n_rows)]
    correct_matrix: list[list[bool | None]] = [[None] * max_eps for _ in range(n_rows)]
    ep_names: list[list[str]] = [[""] * max_eps for _ in range(n_rows)]

    for r, (domain, episodes) in enumerate(domain_episodes):
        for c, ep in enumerate(episodes):
            quality_matrix[r, c] = ep["quality"]
            verdict_matrix[r][c] = ep["verdict"]
            correct_matrix[r][c] = ep.get("correct")
            # WHY: shorten episode names for readability inside small cells
            ep_names[r][c] = ep["name"].replace("_", "\n")

    # Custom colormap: blue (0) → white (0.4) → red (1.0)
    cmap = LinearSegmentedColormap.from_list(
        "quality",
        [(0.0, CB_BLUE), (0.4, "#f5f5f5"), (0.6, CB_ORANGE), (1.0, CB_RED)],
    )
    cmap.set_bad(color="#e0e0e0")  # NaN cells

    fig_width = max(14, max_eps * 0.75)
    fig, ax = plt.subplots(figsize=(fig_width, n_rows * 1.1 + 1.2))

    im = ax.imshow(quality_matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.02)
    cbar.set_label("Quality score", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # Y-axis: domain names
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(
        [DOMAIN_DISPLAY[d] for d, _ in domain_episodes],
        fontsize=10,
    )

    # X-axis: episode index (no labels — too many; we put text in cells)
    ax.set_xticks(range(max_eps))
    ax.set_xticklabels([str(i + 1) for i in range(max_eps)], fontsize=8)
    ax.set_xlabel("Episode index within domain", fontsize=10)
    ax.set_title("PhaseBreak v2 — Benchmark Quality Scores (all 58 episodes)", fontsize=12, pad=8)

    # Cell annotations
    for r in range(n_rows):
        for c in range(max_eps):
            if np.isnan(quality_matrix[r, c]):
                continue
            verdict = verdict_matrix[r][c]
            correct = correct_matrix[r][c]
            q = quality_matrix[r, c]

            # Text color: dark if background is light, light if dark
            text_color = "white" if q > 0.55 or q < 0.1 else "black"

            # Verdict letter
            vshort = VERDICT_SHORT.get(verdict, "?")
            # Correctness indicator
            check = "✓" if correct else "✗" if correct is False else ""

            label = f"{vshort}\n{check}"
            ax.text(
                c,
                r,
                label,
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
                fontweight="bold" if verdict == "BUBBLE" else "normal",
            )

    # Domain separator lines
    for r in range(1, n_rows):
        ax.axhline(r - 0.5, color="white", linewidth=1.5)

    # Legend patches
    legend_elements = [
        mpatches.Patch(facecolor=CB_RED, label="Bubble (score→1)"),
        mpatches.Patch(facecolor=CB_ORANGE, label="Possible (score~0.5)"),
        mpatches.Patch(facecolor=CB_BLUE, label="No bubble (score→0)"),
        mpatches.Patch(facecolor="#e0e0e0", label="N/A"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(1.0, -0.06),
        ncol=4,
        fontsize=8,
        frameon=False,
    )

    fig.tight_layout()
    _save(fig, "fig4_v2_benchmark_heatmap")


# ─── Figure 5: Forward predictions bar chart ─────────────────────────────────


def fig5_forward_predictions(pred_data: dict[str, Any]) -> None:
    """Horizontal bar chart of April 2026 predictions (13 assets).

    Color encodes verdict: BUBBLE=red, POSSIBLE=orange, NO_BUBBLE=green.
    Horizontal dashed line at threshold=0.4.
    """
    predictions = pred_data["predictions"][:13]  # first 13 as specified

    assets = [p["asset"] for p in predictions]
    scores = [p["quality_score"] for p in predictions]
    verdicts = [p["verdict"] for p in predictions]

    # Shorten asset names for x-axis
    short_names = []
    for name in assets:
        # Extract ticker from parentheses: "Gold (GC=F)" → "GC=F"
        if "(" in name and ")" in name:
            ticker = name[name.index("(") + 1 : name.index(")")]
            short_names.append(ticker)
        else:
            short_names.append(name[:8])

    verdict_color = {
        "BUBBLE": CB_RED,
        "POSSIBLE": CB_ORANGE,
        "NO_BUBBLE": CB_GREEN,
    }
    bar_colors = [verdict_color.get(v, "gray") for v in verdicts]

    fig, ax = plt.subplots(figsize=(11, 4.5))

    x = np.arange(len(assets))
    bars = ax.bar(x, scores, color=bar_colors, edgecolor="black", linewidth=0.4, width=0.65)

    # Threshold line
    ax.axhline(
        0.4, color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="Threshold (0.4)"
    )

    # Value labels on bars
    for bar, score, verdict in zip(bars, scores, verdicts):
        if score > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{score:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold" if verdict == "BUBBLE" else "normal",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("LPPLS Quality Score", fontsize=11)
    ax.set_title("PhaseBreak v2 — April 2026 Forward Predictions (13 assets)", fontsize=12, pad=8)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.6, len(assets) - 0.4)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=CB_RED, label="BUBBLE"),
        mpatches.Patch(facecolor=CB_ORANGE, label="POSSIBLE"),
        mpatches.Patch(facecolor=CB_GREEN, label="NO BUBBLE"),
        plt.Line2D([0], [0], color="black", linestyle="--", label="Threshold (0.4)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9, frameon=True)

    fig.tight_layout()
    _save(fig, "fig5_forward_predictions")


# ─── Figure 6: Baselines comparison ──────────────────────────────────────────

# WHY: read from benchmark JSON instead of hardcoding — keeps figures in sync with data
import json as _json
from pathlib import Path as _Path

_bench_path = _Path(__file__).parent.parent / "data" / "v2_results" / "benchmark_results.json"
if _bench_path.exists():
    with open(_bench_path) as _f:
        _bench = _json.load(_f)
    _tp = sum(d["tp"] for d in _bench["domains"].values())
    _fp = sum(d["fp"] for d in _bench["domains"].values())
    _fn = sum(d["fn"] for d in _bench["domains"].values())
    _prec = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0
    _rec = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0
    _f1 = 2 * _prec * _rec / (_prec + _rec) if (_prec + _rec) > 0 else 0
    LPPLS_V2_METRICS = {"precision": round(_prec, 2), "recall": round(_rec, 2), "f1": round(_f1, 2)}
else:
    LPPLS_V2_METRICS = {"precision": 0.76, "recall": 0.61, "f1": 0.68}  # fallback

METHOD_DISPLAY = {
    "CAGR>50%": "CAGR>50%",
    "Z-score>2": "Z-score>2",
    "Trend>2\u03c3": "Trend>2σ",
    "Vol spike>2\u00d7": "VolSpike>2×",
    "BOCPD": "BOCPD",
    "LPPLS v2": "LPPLS v2\n(ours)",
}


def fig6_baselines_comparison(baselines_data: dict[str, Any]) -> None:
    """Grouped bar chart: methods × metrics (Precision, Recall, F1).

    LPPLS v2 column is highlighted distinctly to show model's standing.
    """
    # Merge baselines with LPPLS v2
    all_methods_raw = list(baselines_data.keys()) + ["LPPLS v2"]
    all_methods_display = [METHOD_DISPLAY.get(m, m) for m in all_methods_raw]

    prec_vals = [baselines_data[m]["precision"] for m in baselines_data] + [
        LPPLS_V2_METRICS["precision"]
    ]
    rec_vals = [baselines_data[m]["recall"] for m in baselines_data] + [LPPLS_V2_METRICS["recall"]]
    f1_vals = [baselines_data[m]["f1"] for m in baselines_data] + [LPPLS_V2_METRICS["f1"]]

    n_methods = len(all_methods_raw)
    x = np.arange(n_methods)
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 4.8))

    # WHY: use slightly different alpha for LPPLS v2 to visually emphasize it
    def bar_alpha(vals: list[float]) -> list[float]:
        return [1.0 if i == n_methods - 1 else 0.75 for i in range(len(vals))]

    bars_p = ax.bar(
        x - width,
        prec_vals,
        width,
        label="Precision",
        color=CB_BLUE,
        edgecolor="black",
        linewidth=0.4,
        alpha=0.85,
    )
    bars_r = ax.bar(
        x,
        rec_vals,
        width,
        label="Recall",
        color=CB_GREEN,
        edgecolor="black",
        linewidth=0.4,
        alpha=0.85,
    )
    bars_f = ax.bar(
        x + width,
        f1_vals,
        width,
        label="F1",
        color=CB_ORANGE,
        edgecolor="black",
        linewidth=0.4,
        alpha=0.85,
    )

    # Highlight LPPLS v2 column with a background band
    ax.axvspan(
        n_methods - 1 - 0.45,
        n_methods - 1 + 0.45,
        alpha=0.08,
        color=CB_RED,
        label="_nolegend_",
    )

    # Value labels on all bars
    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.012,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    rotation=90 if h > 0.85 else 0,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(all_methods_display, fontsize=9)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.set_title(
        "PhaseBreak v2 vs. Baselines — Precision / Recall / F1 (Finance domain, n=42)",
        fontsize=11,
        pad=8,
    )
    ax.legend(fontsize=9, loc="upper left", frameon=True)

    # Annotation: LPPLS unique value
    ax.annotate(
        "Only method\npredicting tc (when)",
        xy=(n_methods - 1, LPPLS_V2_METRICS["precision"] + 0.05),
        xytext=(n_methods - 1.6, 1.05),
        fontsize=7.5,
        color=CB_RED,
        arrowprops=dict(arrowstyle="->", color=CB_RED, lw=1.0),
        ha="center",
    )

    fig.tight_layout()
    _save(fig, "fig6_baselines_comparison")


# ─── Figure 7: Hurst vs MFDFA scatter ────────────────────────────────────────


def _hurst_rs(ts: np.ndarray) -> float:
    """Hurst exponent via R/S analysis (Hurst 1951).

    WHY: R/S is the classical estimator, stable for financial time series of
    200-500 points. Faster than DFA for our use case.
    """
    ts = np.log(ts / ts[0] + 1e-12)  # log-returns relative to first value
    n = len(ts)
    min_chunk = 16
    max_chunk = n // 2
    if max_chunk < min_chunk:
        return 0.5  # fallback: insufficient data

    scales = np.unique(np.logspace(np.log10(min_chunk), np.log10(max_chunk), 20).astype(int))
    rs_vals = []
    scale_vals = []
    for scale in scales:
        if scale < 2:
            continue
        n_chunks = n // scale
        if n_chunks < 1:
            continue
        rs_list = []
        for i in range(n_chunks):
            chunk = ts[i * scale : (i + 1) * scale]
            mean_c = chunk.mean()
            dev = np.cumsum(chunk - mean_c)
            r = dev.max() - dev.min()
            s = chunk.std(ddof=1)
            if s > 1e-12:
                rs_list.append(r / s)
        if rs_list:
            rs_vals.append(np.mean(rs_list))
            scale_vals.append(scale)

    if len(scale_vals) < 4:
        return 0.5

    log_scales = np.log10(scale_vals)
    log_rs = np.log10(rs_vals)
    # WHY: linear fit in log-log space gives Hurst exponent as slope
    coeffs = np.polyfit(log_scales, log_rs, 1)
    return float(np.clip(coeffs[0], 0.0, 1.5))


def _mfdfa_delta_alpha(ts: np.ndarray, q_values: np.ndarray | None = None) -> float:
    """Multifractal width Δα via MFDFA (Kantelhardt et al. 2002).

    Δα = α_max - α_min (singularity spectrum width).
    Larger Δα → stronger multifractality → more complex dynamics (bubble signal).

    WHY: we use a simplified MFDFA (DFA for multiple q-orders) which runs fast
    without requiring the full chhabra-jensen method.
    """
    if q_values is None:
        q_values = np.array([-4, -2, -1, 0, 1, 2, 4], dtype=float)

    # WHY: work on log-return increments which are stationary
    log_prices = np.log(ts + 1e-12)
    returns = np.diff(log_prices)
    n = len(returns)

    if n < 64:
        return 0.0

    # Profile (cumulative sum of demeaned returns)
    profile = np.cumsum(returns - returns.mean())

    scales = np.unique(np.logspace(np.log10(16), np.log10(n // 4), 15).astype(int))
    scales = scales[scales >= 8]

    hq_vals = []
    for q in q_values:
        fq_list = []
        for s in scales:
            n_segs = n // s
            if n_segs < 4:
                continue
            f2_segs = []
            for seg in range(n_segs):
                seg_data = profile[seg * s : (seg + 1) * s]
                x = np.arange(s)
                # Detrend each segment with a linear polynomial
                poly = np.polyfit(x, seg_data, 1)
                trend = np.polyval(poly, x)
                f2 = np.mean((seg_data - trend) ** 2)
                f2_segs.append(f2)
            f2_arr = np.array(f2_segs)
            f2_arr = np.clip(f2_arr, 1e-20, None)
            if q == 0:
                # WHY: q=0 limit: F_q = exp(0.5 * mean(ln(F^2)))
                fq = np.exp(0.5 * np.mean(np.log(f2_arr)))
            else:
                fq = (np.mean(f2_arr ** (q / 2))) ** (1.0 / q)
            fq_list.append(fq)

        if len(fq_list) < 4:
            hq_vals.append(0.5)
            continue

        log_s = np.log10(scales[: len(fq_list)])
        log_f = np.log10(fq_list)
        coeffs = np.polyfit(log_s, log_f, 1)
        hq_vals.append(float(coeffs[0]))

    hq_arr = np.array(hq_vals)
    # Singularity exponent α(q) ≈ H(q) (simplified, valid for monofractal limit)
    # Δα = max(H) - min(H) as proxy for spectrum width
    delta_alpha = float(np.max(hq_arr) - np.min(hq_arr))
    return float(np.clip(delta_alpha, 0.0, 2.0))


def _load_ticker(ticker: str, start: str, end: str) -> np.ndarray:
    """Download daily closing prices for a ticker. Returns close price array."""
    import yfinance as yf  # WHY: lazy import to keep startup fast if yfinance unavailable

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for {ticker} [{start}:{end}]")
    close = df["Close"].dropna().values
    if hasattr(close, "flatten"):
        close = close.flatten()
    return close.astype(float)


# Known episodes for fig7
BUBBLE_EPISODES = [
    ("BTC-USD", "2017-01-01", "2017-12-17", "BTC 2017"),
    ("NVDA", "2023-01-01", "2024-06-20", "NVDA 2024"),
    ("BTC-USD", "2024-01-01", "2024-03-14", "BTC 2024"),
    ("^IXIC", "1998-01-01", "2000-03-10", "Dotcom"),
]

CONTROL_EPISODES = [
    ("^GSPC", "2024-01-01", "2024-12-31", "S&P500 2024"),
    ("^IXIC", "2016-01-01", "2016-12-31", "NASDAQ 2016"),
    ("KO", "2019-01-01", "2019-12-31", "KO 2019"),
    ("TSLA", "2023-01-01", "2023-12-31", "TSLA 2023"),
]


def fig7_hurst_mfdfa_scatter() -> None:
    """Scatter: Hurst exponent (x) vs MFDFA Δα (y).

    Bubbles in red, controls in blue. Dashed threshold lines at H=0.85, Δα=0.3.
    """
    print("  Computing Hurst + MFDFA (downloading data)…")

    points_bubble: list[tuple[float, float, str]] = []
    points_control: list[tuple[float, float, str]] = []

    for ticker, start, end, label in BUBBLE_EPISODES:
        try:
            prices = _load_ticker(ticker, start, end)
            h = _hurst_rs(prices)
            da = _mfdfa_delta_alpha(prices)
            points_bubble.append((h, da, label))
            print(f"    {label}: H={h:.3f}, Δα={da:.3f}")
        except Exception as exc:
            print(f"    WARNING: {label} failed — {exc}")

    for ticker, start, end, label in CONTROL_EPISODES:
        try:
            prices = _load_ticker(ticker, start, end)
            h = _hurst_rs(prices)
            da = _mfdfa_delta_alpha(prices)
            points_control.append((h, da, label))
            print(f"    {label}: H={h:.3f}, Δα={da:.3f}")
        except Exception as exc:
            print(f"    WARNING: {label} failed — {exc}")

    if not points_bubble and not points_control:
        print("  ERROR: no data downloaded, fig7 skipped")
        return

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Threshold lines
    ax.axvline(0.85, color="gray", linestyle="--", linewidth=1.0, alpha=0.6, label="H = 0.85")
    ax.axhline(0.30, color="gray", linestyle=":", linewidth=1.0, alpha=0.6, label="Δα = 0.30")

    # Shading for "bubble zone" (H>0.85 AND Δα>0.3) — subtle background
    ax.axvspan(0.85, 1.5, alpha=0.05, color=CB_RED)
    ax.axhspan(0.30, 2.0, alpha=0.05, color=CB_RED)

    # Plot bubbles
    if points_bubble:
        hx = [p[0] for p in points_bubble]
        hy = [p[1] for p in points_bubble]
        labels_b = [p[2] for p in points_bubble]
        ax.scatter(
            hx,
            hy,
            c=CB_RED,
            s=120,
            marker="o",
            edgecolors="black",
            linewidth=0.6,
            zorder=5,
            label=f"Bubble (n={len(points_bubble)})",
        )
        for xi, yi, lbl in zip(hx, hy, labels_b):
            ax.annotate(
                lbl,
                (xi, yi),
                textcoords="offset points",
                xytext=(7, 4),
                fontsize=8,
                color="#8B0000",
            )

    # Plot controls
    if points_control:
        cx = [p[0] for p in points_control]
        cy = [p[1] for p in points_control]
        labels_c = [p[2] for p in points_control]
        ax.scatter(
            cx,
            cy,
            c=CB_BLUE,
            s=100,
            marker="s",
            edgecolors="black",
            linewidth=0.6,
            zorder=5,
            label=f"Control (n={len(points_control)})",
        )
        for xi, yi, lbl in zip(cx, cy, labels_c):
            ax.annotate(
                lbl,
                (xi, yi),
                textcoords="offset points",
                xytext=(7, -10),
                fontsize=8,
                color="#003060",
            )

    ax.set_xlabel("Hurst Exponent $H$", fontsize=11)
    ax.set_ylabel("MFDFA Multifractal Width $\\Delta\\alpha$", fontsize=11)
    ax.set_title(
        "Separation Power: Hurst vs. MFDFA Δα\n(Bubbles vs. Controls)",
        fontsize=12,
        pad=8,
    )
    ax.legend(fontsize=9, loc="upper left", frameon=True)
    ax.set_xlim(0.4, 1.1)
    ax.set_ylim(-0.02, 1.0)

    # Zone label
    ax.text(
        0.92,
        0.85,
        "Bubble\nzone",
        fontsize=8,
        color=CB_RED,
        alpha=0.7,
        ha="center",
        transform=ax.transData,
    )

    fig.tight_layout()
    _save(fig, "fig7_hurst_mfdfa_scatter")


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    """Generate all four v2 figures."""
    print(f"Output directory: {OUTPUT_DIR}")

    # Load JSON data
    with BENCHMARK_JSON.open() as f:
        benchmark_data = json.load(f)
    with BASELINES_JSON.open() as f:
        baselines_data = json.load(f)
    with PREDICTIONS_JSON.open() as f:
        pred_data = json.load(f)

    print("\n[fig4] Benchmark heatmap…")
    fig4_benchmark_heatmap(benchmark_data)

    print("\n[fig5] Forward predictions bar chart…")
    fig5_forward_predictions(pred_data)

    print("\n[fig6] Baselines comparison…")
    fig6_baselines_comparison(baselines_data)

    print("\n[fig7] Hurst vs MFDFA scatter…")
    fig7_hurst_mfdfa_scatter()

    print(f"\nDone. All figures in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
