"""Evaluate candidate descriptor formulas on the ABX3 dataset."""

import logging
import signal
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

log = logging.getLogger(__name__)

TIMEOUT_SECONDS = 10


class EvalTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise EvalTimeout("Function execution timed out")


@dataclass
class EvalResult:
    accuracy: float = 0.0
    train_accuracy: float = 0.0
    test_accuracy: float = 0.0
    false_positive_rate: float = 0.0
    per_anion_accuracy: dict = field(default_factory=dict)
    thresholds: list = field(default_factory=list)
    plot_path: str = ""
    metrics_summary: str = ""
    error: str = ""
    descriptor_values: list = field(default_factory=list)


def load_dataset(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df.rename(
        columns={"rA (Ang)": "rA", "rB (Ang)": "rB", "rX (Ang)": "rX"},
        inplace=True,
    )
    return df


def _exec_descriptor(func_code: str, df: pd.DataFrame) -> np.ndarray:
    namespace = {"np": np, "math": __import__("math")}
    exec(func_code, namespace)  # noqa: S102
    descriptor_fn = namespace["descriptor"]

    values = []
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)
    # try/finally approved: must restore signal handler even if descriptor crashes
    try:
        for _, row in df.iterrows():
            val = descriptor_fn(
                rA=row["rA"],
                rB=row["rB"],
                rX=row["rX"],
                nA=row["nA"],
                nB=row["nB"],
                nX=row["nX"],
            )
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                raise ValueError(f"descriptor returned {val} for {row['ABX3']}")
            values.append(float(val))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    return np.array(values)


def _classify(
    values: np.ndarray, labels: np.ndarray, train_mask: np.ndarray, max_depth: int
) -> tuple[list[float], float, np.ndarray, DecisionTreeClassifier]:
    X_train = values[train_mask].reshape(-1, 1)
    y_train = labels[train_mask]

    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X_train, y_train)

    preds_encoded = clf.predict(values.reshape(-1, 1))
    accuracy = float((preds_encoded == labels).mean())

    tree = clf.tree_
    thresholds = sorted(
        tree.threshold[i] for i in range(tree.node_count) if tree.threshold[i] != -2.0
    )

    return thresholds, accuracy, preds_encoded, clf


def _compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    df: pd.DataFrame,
) -> dict:
    test_mask = ~train_mask

    tp = ((labels == 1) & (preds == 1)).sum()
    fp = ((labels == -1) & (preds == 1)).sum()
    tn = ((labels == -1) & (preds == -1)).sum()
    fn = ((labels == 1) & (preds == -1)).sum()

    per_anion = {}
    for x in ["O", "F", "Cl", "Br", "I"]:
        mask = df["X"].values == x
        if mask.any():
            per_anion[x] = float((preds[mask] == labels[mask]).mean())

    return {
        "train_accuracy": float((preds[train_mask] == labels[train_mask]).mean()),
        "test_accuracy": float((preds[test_mask] == labels[test_mask]).mean()),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "false_positive_rate": float(fp / (fp + tn)),
        "per_anion_accuracy": per_anion,
    }


def _format_metrics_summary(accuracy: float, metrics: dict, thresholds: list[float]) -> str:
    thresh_str = ", ".join(f"{t:.4f}" for t in thresholds)
    lines = [
        f"Overall accuracy: {accuracy:.1%}",
        f"Train accuracy: {metrics['train_accuracy']:.1%}",
        f"Test accuracy: {metrics['test_accuracy']:.1%}",
        f"False positive rate: {metrics['false_positive_rate']:.1%}",
        f"Decision thresholds: [{thresh_str}]",
        f"TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}",
        "Per-anion accuracy:",
    ]
    for anion, acc in metrics["per_anion_accuracy"].items():
        lines.append(f"  {anion}: {acc:.1%}")
    return "\n".join(lines)


def _generate_plot(
    values: np.ndarray,
    labels: np.ndarray,
    thresholds: list[float],
    clf: DecisionTreeClassifier,
    plot_path: Path,
) -> None:
    perov_vals = values[labels == 1]
    nonperov_vals = values[labels == -1]

    margin = max(1.0, (values.max() - values.min()) * 0.05)
    vmin = min(np.percentile(values, 1), min(thresholds, default=0) - margin)
    vmax = max(np.percentile(values, 99), max(thresholds, default=0) + margin)
    bins = np.linspace(vmin, vmax, 40)

    fig, ax = plt.subplots(figsize=(7, 4))

    xs = np.linspace(vmin, vmax, 500)
    preds = clf.predict(xs.reshape(-1, 1))
    prev_x = vmin
    prev_pred = preds[0]
    for x, pred in zip(xs[1:], preds[1:], strict=True):
        if pred != prev_pred or x == xs[-1]:
            color = "#c8e6c9" if prev_pred == 1 else "#ffcdd2"
            ax.axvspan(prev_x, x, color=color, zorder=0, alpha=0.5)
            prev_x = x
            prev_pred = pred
    color = "#c8e6c9" if prev_pred == 1 else "#ffcdd2"
    ax.axvspan(prev_x, vmax, color=color, zorder=0, alpha=0.5)

    ax.hist(
        perov_vals, bins=bins, alpha=0.7, color="#1976d2", label="Perovskite", edgecolor="white"
    )
    ax.hist(
        nonperov_vals,
        bins=bins,
        alpha=0.7,
        color="#e65100",
        label="Nonperovskite",
        edgecolor="white",
    )
    for t in thresholds:
        ax.axvline(t, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Descriptor value", fontsize=12)
    ax.set_ylabel("Counts", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate_candidate(
    func_code: str,
    df: pd.DataFrame,
    plot_dir: Path,
    node_id: str,
    decision_tree_max_depth: int = 2,
    train_split_label: int = 1,
) -> EvalResult:
    # try-catch approved: exec runs untrusted LLM-generated code that can crash arbitrarily
    try:
        values = _exec_descriptor(func_code, df)
    except Exception as e:
        return EvalResult(accuracy=0.0, error=str(e))

    labels = df["exp_label"].values
    train_mask = df["is_train"].values == train_split_label

    thresholds, accuracy, preds, clf = _classify(
        values, labels, train_mask, decision_tree_max_depth
    )
    metrics = _compute_metrics(preds, labels, train_mask, df)

    plot_path = plot_dir / f"{node_id}.png"
    _generate_plot(values, labels, thresholds, clf, plot_path)

    return EvalResult(
        accuracy=accuracy,
        train_accuracy=metrics["train_accuracy"],
        test_accuracy=metrics["test_accuracy"],
        false_positive_rate=metrics["false_positive_rate"],
        per_anion_accuracy=metrics["per_anion_accuracy"],
        thresholds=thresholds,
        plot_path=str(plot_path),
        metrics_summary=_format_metrics_summary(accuracy, metrics, thresholds),
        descriptor_values=values.tolist(),
    )
