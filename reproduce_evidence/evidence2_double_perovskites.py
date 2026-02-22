"""
Evidence 2: Generalization to 918 A2BB'X6 double perovskites from ICSD.
Tests τ on compounds entirely excluded from training.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "perovskite-stability"


def load_data():
    df = pd.read_csv(DATA_DIR / "icsd_A2BBX6.csv")
    return df


def print_accuracy_metrics(df):
    print("=" * 60)
    print("Evidence 2: Generalization to 918 ICSD double perovskites")
    print("=" * 60)

    valid = df.dropna(subset=["icsd_label", "tau_pred"])
    total = len(valid)

    for name, pred_col in [("Goldschmidt t", "t_pred"), ("New τ", "tau_pred")]:
        correct = (valid["icsd_label"] == valid[pred_col]).sum()
        acc = correct / total * 100

        tp = ((valid["icsd_label"] == 1) & (valid[pred_col] == 1)).sum()
        fp = ((valid["icsd_label"] == -1) & (valid[pred_col] == 1)).sum()
        tn = ((valid["icsd_label"] == -1) & (valid[pred_col] == -1)).sum()
        fn = ((valid["icsd_label"] == 1) & (valid[pred_col] == -1)).sum()

        print(f"\n--- {name} ---")
        print(f"  Overall accuracy: {acc:.1f}% ({correct}/{total})")
        print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
        if tp + fn > 0:
            print(f"  Perovskite recall:    {tp / (tp + fn) * 100:.1f}% ({tp}/{tp + fn})")
        if tn + fp > 0:
            print(f"  Nonperovskite recall: {tn / (tn + fp) * 100:.1f}% ({tn}/{tn + fp})")

    # Per-anion breakdown
    anion_map = {"O": "O²⁻", "F": "F⁻", "Cl": "Cl⁻", "Br": "Br⁻", "I": "I⁻"}
    print("\n--- Per-anion accuracy (τ) ---")
    print(f"  {'Anion':<6} {'accuracy':>10} {'count':>6}")
    for x in ["O", "F", "Cl", "Br", "I"]:
        sub = valid[valid["X"] == x]
        if len(sub) == 0:
            continue
        acc = (sub["icsd_label"] == sub["tau_pred"]).mean() * 100
        print(f"  {anion_map.get(x, x):<6} {acc:>9.0f}% {len(sub):>6}")


def plot_double_perovskite_overview(df):
    valid = df.dropna(subset=["icsd_label", "tau"])

    perov = valid[valid["icsd_label"] == 1]
    nonperov = valid[valid["icsd_label"] == -1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Panel A: Histogram of tau for double perovskites ---
    ax = axes[0]
    tau_max = 13
    bins = np.arange(0, tau_max + 0.5, 0.5)
    ax.axvspan(0, 4.18, color="#c8e6c9", zorder=0)
    ax.axvspan(4.18, tau_max, color="#ffcdd2", zorder=0)
    ax.hist(
        perov["tau"].clip(upper=tau_max),
        bins=bins,
        alpha=0.7,
        color="#1976d2",
        label="Perovskite (ICSD)",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.hist(
        nonperov["tau"].clip(upper=tau_max),
        bins=bins,
        alpha=0.7,
        color="#e65100",
        label="Nonperovskite (ICSD)",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axvline(4.18, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"$\tau$", fontsize=13)
    ax.set_ylabel("Counts", fontsize=13)
    ax.set_xlim(0, tau_max)
    ax.legend(fontsize=9)
    ax.set_title("A₂BB′X₆: τ distribution", fontsize=12)

    # --- Panel B: tau_prob for perovskite vs nonperovskite ---
    ax = axes[1]
    ax.hist(
        perov["tau_prob"],
        bins=30,
        alpha=0.7,
        color="#1976d2",
        label="Perovskite (ICSD)",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.hist(
        nonperov["tau_prob"],
        bins=30,
        alpha=0.7,
        color="#e65100",
        label="Nonperovskite (ICSD)",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axvline(0.5, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"$P(\tau)$", fontsize=13)
    ax.set_ylabel("Counts", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_title("A₂BB′X₆: probability distribution", fontsize=12)

    fig.tight_layout()
    out = Path(__file__).parent / "fig_evidence2.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    df = load_data()
    print_accuracy_metrics(df)
    plot_double_perovskite_overview(df)
