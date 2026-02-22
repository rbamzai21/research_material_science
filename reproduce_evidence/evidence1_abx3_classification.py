"""
Evidence 1: Classification accuracy on 576 experimental ABX3 compounds.
Reproduces Figure 2 panels A, B, C from Bartel et al. Sci. Adv. 2019.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "perovskite-stability"


def load_data():
    df = pd.read_csv(DATA_DIR / "TableS1.csv")
    df.rename(columns={"rA (Ang)": "rA", "rB (Ang)": "rB", "rX (Ang)": "rX"}, inplace=True)
    return df


def print_accuracy_metrics(df):
    print("=" * 60)
    print("Evidence 1: Classification on 576 ABX3 compounds")
    print("=" * 60)

    for name, pred_col in [("Goldschmidt t", "t_pred"), ("New τ", "tau_pred")]:
        correct = (df["exp_label"] == df[pred_col]).sum()
        total = len(df)
        acc = correct / total * 100

        tp = ((df["exp_label"] == 1) & (df[pred_col] == 1)).sum()
        fp = ((df["exp_label"] == -1) & (df[pred_col] == 1)).sum()
        tn = ((df["exp_label"] == -1) & (df[pred_col] == -1)).sum()
        fn = ((df["exp_label"] == 1) & (df[pred_col] == -1)).sum()
        fpr = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0

        print(f"\n--- {name} ---")
        print(f"  Overall accuracy: {acc:.1f}% ({correct}/{total})")
        print(f"  True positives:  {tp}  |  False positives: {fp}")
        print(f"  True negatives:  {tn}  |  False negatives: {fn}")
        print(f"  False positive rate: {fpr:.1f}%")
        print(f"  Perovskite recall:    {tp / (tp + fn) * 100:.1f}%")
        print(f"  Nonperovskite recall: {tn / (tn + fp) * 100:.1f}%")

    # Per-anion accuracy
    anion_map = {"O": "O²⁻", "F": "F⁻", "Cl": "Cl⁻", "Br": "Br⁻", "I": "I⁻"}
    print("\n--- Per-anion accuracy ---")
    print(f"  {'Anion':<6} {'t accuracy':>12} {'τ accuracy':>12} {'count':>6}")
    for x in ["O", "F", "Cl", "Br", "I"]:
        sub = df[df["X"] == x]
        if len(sub) == 0:
            continue
        t_acc = (sub["exp_label"] == sub["t_pred"]).mean() * 100
        tau_acc = (sub["exp_label"] == sub["tau_pred"]).mean() * 100
        print(f"  {anion_map[x]:<6} {t_acc:>11.0f}% {tau_acc:>11.0f}% {len(sub):>6}")

    # Train/test split
    test = df[df["is_train"] == -1]
    test_acc = (test["exp_label"] == test["tau_pred"]).mean() * 100
    print(f"\n  τ accuracy on test set (20%): {test_acc:.0f}% ({len(test)} compounds)")


def plot_panels_abc(df):
    perov = df[df["exp_label"] == 1]
    nonperov = df[df["exp_label"] == -1]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # --- Panel A: Histogram of t ---
    ax = axes[0]
    bins_t = np.arange(0.7, 1.35, 0.02)
    ax.axvspan(0.825, 1.059, color="#c8e6c9", zorder=0, label="Predicted perovskite")
    ax.axvspan(0.7, 0.825, color="#ffcdd2", zorder=0, label="Predicted nonperovskite")
    ax.axvspan(1.059, 1.35, color="#ffcdd2", zorder=0)
    ax.hist(
        perov["t"],
        bins=bins_t,
        alpha=0.7,
        color="#1976d2",
        label="Perovskite",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.hist(
        nonperov["t"],
        bins=bins_t,
        alpha=0.7,
        color="#e65100",
        label="Nonperovskite",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axvline(0.825, color="k", linestyle="--", linewidth=0.8)
    ax.axvline(1.059, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("$t$", fontsize=13)
    ax.set_ylabel("Counts", fontsize=13)
    ax.set_xlim(0.7, 1.3)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("A", fontsize=14, fontweight="bold", loc="left")

    # --- Panel B: Histogram of tau ---
    ax = axes[1]
    tau_perov = perov["tau"]
    tau_nonperov = nonperov["tau"]
    tau_max_plot = 13
    bins_tau = np.arange(0, tau_max_plot + 0.5, 0.5)
    ax.axvspan(0, 4.18, color="#c8e6c9", zorder=0, label="Predicted perovskite")
    ax.axvspan(4.18, tau_max_plot, color="#ffcdd2", zorder=0, label="Predicted nonperovskite")
    ax.hist(
        tau_perov[tau_perov <= tau_max_plot],
        bins=bins_tau,
        alpha=0.7,
        color="#1976d2",
        label="Perovskite",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.hist(
        tau_nonperov[tau_nonperov <= tau_max_plot],
        bins=bins_tau,
        alpha=0.7,
        color="#e65100",
        label="Nonperovskite",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axvline(4.18, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"$\tau$", fontsize=13)
    ax.set_ylabel("Counts", fontsize=13)
    ax.set_xlim(0, tau_max_plot)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("B", fontsize=14, fontweight="bold", loc="left")

    # --- Panel C: P(tau) vs t ---
    ax = axes[2]
    ax.axvline(0.825, color="k", linestyle="--", linewidth=0.8)
    ax.axvline(1.059, color="k", linestyle="--", linewidth=0.8)
    ax.scatter(
        perov["t"],
        perov["tau_prob"],
        s=20,
        alpha=0.5,
        c="#1976d2",
        marker="o",
        label="Perovskite",
        edgecolors="none",
    )
    ax.scatter(
        nonperov["t"],
        nonperov["tau_prob"],
        s=20,
        alpha=0.5,
        c="#e65100",
        marker="^",
        label="Nonperovskite",
        edgecolors="none",
    )

    # Highlight LaAlO3 and NaBeCl3
    for name in ["LaAlO3", "NaBeCl3"]:
        row = df[df["ABX3"] == name]
        if len(row) > 0:
            row = row.iloc[0]
            ax.annotate(
                name.replace("3", "₃"),
                (row["t"], row["tau_prob"]),
                fontsize=9,
                ha="center",
                xytext=(0, 12),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="-", lw=0.8),
            )

    ax.set_xlabel("$t$", fontsize=13)
    ax.set_ylabel(r"$P(\tau)$", fontsize=13)
    ax.set_xlim(0.7, 1.3)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("C", fontsize=14, fontweight="bold", loc="left")

    fig.tight_layout()
    out = Path(__file__).parent / "fig2_abc.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    df = load_data()
    print_accuracy_metrics(df)
    plot_panels_abc(df)
