"""Generate all figures for PhaseBreak paper.

Produces:
1. fig1_m_omega_scatter.pdf — (m, ω) scatter by domain
2. fig2_ablation_bar.pdf — ablation precision/recall bars
3. fig3_tc_accuracy.pdf — tc prediction error for known bubbles
"""

import sys

sys.path.insert(0, "..")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "figures"

# ============================================================
# Data from verified test runs
# ============================================================

# Finance (m, ω) — multi-window means from real yfinance data
FIN_LABELS = ["BTC 2017", "BTC 2021", "Dotcom", "China 2015"]
FIN_M = np.array([0.282, 0.519, 0.600, 0.632])
FIN_OMEGA = np.array([8.59, 7.50, 9.50, 8.33])

# Geology (m, ω) — from real Sentinel-2 fits (R² > 0.5)
GEO_LABELS = [
    "NDVI-1",
    "NDVI-2",
    "Clay",
    "BSI-1",
    "NDVI-3",
    "Clay-2",
    "NDVI-4",
    "Clay-3",
    "Iron",
    "BSI-2",
    "BSI-3",
    "Iron-2",
    "Iron-3",
]
GEO_M = np.array(
    [0.762, 0.456, 0.633, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.189, 0.367, 0.900, 0.900]
)
GEO_OMEGA = np.array(
    [6.47, 15.67, 13.33, 13.33, 8.67, 6.33, 6.33, 6.33, 13.33, 8.67, 18.00, 8.67, 8.67]
)

# Ablation data — from verified test run
ABLATION_LAYERS = [
    "LPPLS\n(loose)",
    "+ Tight\nfilters",
    "+ Multi-\nwindow",
    "+ HMM\ngate",
    "+ EWS\nconfirm",
]
ABLATION_PRECISION = [54.5, 100.0, 66.7, 80.0, 100.0]
ABLATION_RECALL = [100.0, 66.7, 66.7, 66.7, 50.0]
ABLATION_FP_RATE = [83.3, 0.0, 33.3, 16.7, 0.0]

# tc accuracy
TC_BUBBLES = ["BTC\n2017", "BTC\n2021", "Dotcom\n2000", "China\n2015"]
TC_ERRORS = [1, 19, 10, 16]
TC_THRESHOLD = 30


def fig1_scatter():
    """(m, ω) scatter plot by domain."""
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(
        FIN_M,
        FIN_OMEGA,
        c="#e74c3c",
        s=120,
        marker="o",
        label=f"Finance (n={len(FIN_M)})",
        zorder=5,
        edgecolors="black",
        linewidth=0.5,
    )
    ax.scatter(
        GEO_M,
        GEO_OMEGA,
        c="#3498db",
        s=80,
        marker="s",
        label=f"Geology (n={len(GEO_M)})",
        zorder=4,
        edgecolors="black",
        linewidth=0.5,
    )

    # Annotate finance points
    for i, label in enumerate(FIN_LABELS):
        ax.annotate(
            label,
            (FIN_M[i], FIN_OMEGA[i]),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=7,
            color="#c0392b",
        )

    ax.set_xlabel("Power law exponent $m$", fontsize=12)
    ax.set_ylabel("Log-frequency $\\omega$", fontsize=12)
    ax.set_title("Cross-Domain LPPLS Parameters $(m, \\omega)$", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(3, 20)
    ax.axhline(y=6, color="gray", linestyle="--", alpha=0.3, label="_")
    ax.axhline(y=13, color="gray", linestyle="--", alpha=0.3, label="_")
    ax.axvline(x=0.1, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(x=0.87, color="gray", linestyle="--", alpha=0.3)
    ax.text(0.5, 5.5, "Sornette bounds", fontsize=8, color="gray", ha="center")

    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig1_m_omega_scatter.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUTPUT_DIR}/fig1_m_omega_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("fig1 saved")


def fig2_ablation():
    """Ablation bar chart."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.arange(len(ABLATION_LAYERS))
    width = 0.25

    bars1 = ax.bar(
        x - width,
        ABLATION_PRECISION,
        width,
        label="Precision",
        color="#2ecc71",
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x, ABLATION_RECALL, width, label="Recall", color="#3498db", edgecolor="black", linewidth=0.5
    )
    bars3 = ax.bar(
        x + width,
        ABLATION_FP_RATE,
        width,
        label="FP Rate",
        color="#e74c3c",
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_title("Ablation Study: Layer-by-Layer Performance", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(ABLATION_LAYERS, fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 110)
    ax.axhline(y=80, color="gray", linestyle=":", alpha=0.4)

    # Value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 1,
                    f"{h:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig2_ablation_bar.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUTPUT_DIR}/fig2_ablation_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("fig2 saved")


def fig3_tc_accuracy():
    """tc prediction error bar chart."""
    fig, ax = plt.subplots(figsize=(6, 4))

    colors = ["#2ecc71" if e <= TC_THRESHOLD else "#e74c3c" for e in TC_ERRORS]
    bars = ax.bar(TC_BUBBLES, TC_ERRORS, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(
        y=TC_THRESHOLD, color="red", linestyle="--", alpha=0.5, label=f"Threshold ({TC_THRESHOLD}d)"
    )

    for bar, err in zip(bars, TC_ERRORS):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{err}d",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("$t_c$ Prediction Error (days)", fontsize=11)
    ax.set_title("Critical Time Prediction Accuracy (Multi-Window CI)", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 40)

    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig3_tc_accuracy.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUTPUT_DIR}/fig3_tc_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("fig3 saved")


if __name__ == "__main__":
    import os

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig1_scatter()
    fig2_ablation()
    fig3_tc_accuracy()
    print("All figures generated.")
