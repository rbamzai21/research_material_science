"""
Evidence 3: Correlation between P(τ) and DFT decomposition enthalpy.
Reproduces Figure 2 panel D from Bartel et al. Sci. Adv. 2019.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = Path(__file__).parent.parent / "perovskite-stability"

ANION_STYLE = {
    "Cl": {"color": "#e65100", "marker": "s", "label": "Cl⁻"},
    "Br": {"color": "#e65100", "marker": "^", "label": "Br⁻"},
    "I": {"color": "#e65100", "marker": "v", "label": "I⁻"},
    "O": {"color": "#1976d2", "marker": "D", "label": "O²⁻"},
    "S": {"color": "#1976d2", "marker": "o", "label": "S²⁻"},
    "Se": {"color": "#1976d2", "marker": "<", "label": "Se²⁻"},
}


def load_data():
    df = pd.read_csv(DATA_DIR / "TableS2.csv")
    df.rename(
        columns={
            "dHdec (meV/atom)": "dHd",
            "rA (Ang)": "rA",
            "rB1 (Ang)": "rB1",
            "rB2 (Ang)": "rB2",
            "rX (Ang)": "rX",
        },
        inplace=True,
    )
    return df


def print_metrics(df):
    print("=" * 60)
    print("Evidence 3: τ vs DFT decomposition enthalpy (73 compounds)")
    print("=" * 60)

    agree = ((df["dHd"] > 0) & (df["tau_pred"] == 1)) | ((df["dHd"] <= 0) & (df["tau_pred"] == -1))
    print(f"  τ agrees with DFT sign: {agree.sum()}/{len(df)} ({agree.mean() * 100:.0f}%)")

    # Separate R² for halides vs chalcogenides
    halides = df[df["X"].isin(["Cl", "Br", "I"])]
    chalco = df[df["X"].isin(["O", "S", "Se"])]
    # Exclude CaZrO3, CaHfO3 from chalcogenide fit (as in the paper)
    chalco_fit = chalco[~chalco["compound"].isin(["CaZrO3", "CaHfO3"])]

    for label, sub in [
        ("Halides (Cl, Br, I)", halides),
        ("Chalcogenides (O, S, Se) excl. CaZrO₃/CaHfO₃", chalco_fit),
    ]:
        if len(sub) < 3:
            continue
        slope, intercept, r, p, se = stats.linregress(sub["dHd"], sub["tau_prob"])
        print(f"\n  {label}:")
        print(f"    R² = {r**2:.2f},  slope = {slope:.4f},  intercept = {intercept:.2f}")


def plot_panel_d(df):
    halides = df[df["X"].isin(["Cl", "Br", "I"])]
    chalco = df[df["X"].isin(["O", "S", "Se"])]
    chalco_fit = chalco[~chalco["compound"].isin(["CaZrO3", "CaHfO3"])]

    fig, ax = plt.subplots(figsize=(6.5, 5))

    # Shaded regions: green = agreement, white = disagreement
    ax.axvspan(0, 250, color="#c8e6c9", alpha=0.4, zorder=0)
    ax.axhspan(0.5, 1.05, xmin=0, xmax=0.5, color="#c8e6c9", alpha=0.0, zorder=0)

    # Plot each anion group
    for anion, style in ANION_STYLE.items():
        sub = df[df["X"] == anion]
        if len(sub) == 0:
            continue
        disagree = ((sub["dHd"] > 0) & (sub["tau_pred"] == -1)) | (
            (sub["dHd"] <= 0) & (sub["tau_pred"] == 1)
        )
        agree_mask = ~disagree

        ax.scatter(
            sub.loc[agree_mask, "dHd"],
            sub.loc[agree_mask, "tau_prob"],
            marker=style["marker"],
            c=style["color"],
            s=50,
            alpha=0.8,
            label=style["label"],
            edgecolors="none",
            zorder=3,
        )
        if disagree.any():
            ax.scatter(
                sub.loc[disagree, "dHd"],
                sub.loc[disagree, "tau_prob"],
                marker=style["marker"],
                c=style["color"],
                s=50,
                alpha=0.8,
                edgecolors="red",
                linewidths=1.5,
                zorder=4,
            )

    # Linear fits
    for sub, color in [
        (halides, "#e65100"),
        (chalco_fit, "#1976d2"),
    ]:
        if len(sub) < 3:
            continue
        slope, intercept, r, _, _ = stats.linregress(sub["dHd"], sub["tau_prob"])
        x_fit = np.linspace(sub["dHd"].min() - 20, sub["dHd"].max() + 20, 100)
        ax.plot(x_fit, slope * x_fit + intercept, "--", color=color, linewidth=1.5, alpha=0.7)
        mid_x = sub["dHd"].median()
        mid_y = slope * mid_x + intercept
        ax.annotate(
            f"$R^2 = {r**2:.2f}$",
            (mid_x, mid_y),
            fontsize=10,
            color=color,
            fontweight="bold",
            xytext=(-50, -18),
            textcoords="offset points",
        )

    # Label CaZrO3 and CaHfO3
    for name in ["CaZrO3", "CaHfO3"]:
        row = df[df["compound"] == name]
        if len(row) > 0:
            row = row.iloc[0]
            display = name.replace("3", "₃")
            ax.annotate(
                display,
                (row["dHd"], row["tau_prob"]),
                fontsize=9,
                ha="center",
                xytext=(0, 12),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="-", lw=0.8),
            )

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.set_xlabel(r"$\Delta H_\mathrm{d}$ (meV/atom)", fontsize=13)
    ax.set_ylabel(r"$P(\tau)$", fontsize=13)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=9, loc="lower right", ncol=2)
    ax.set_title("D", fontsize=14, fontweight="bold", loc="left")

    fig.tight_layout()
    out = Path(__file__).parent / "fig2_d.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    df = load_data()
    print_metrics(df)
    plot_panel_d(df)
