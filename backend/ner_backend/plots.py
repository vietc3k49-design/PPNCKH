"""Plot helper for judge metric reports."""

from __future__ import annotations

import json
from pathlib import Path


def plot_judge_reports(report_paths: list[Path | str], output_path: Path | str | None = None) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    labels = []
    strict_p, strict_r, strict_f1 = [], [], []
    relaxed_p, relaxed_r, relaxed_f1 = [], [], []

    for path in report_paths:
        report_path = Path(path)
        with report_path.open("r", encoding="utf-8") as f:
            report = json.load(f)

        labels.append(report_path.stem)
        strict = report["metrics"]["strict"]
        relaxed = report["metrics"]["relaxed"]
        strict_p.append(strict["precision"])
        strict_r.append(strict["recall"])
        strict_f1.append(strict["f1"])
        relaxed_p.append(relaxed["precision"])
        relaxed_r.append(relaxed["recall"])
        relaxed_f1.append(relaxed["f1"])

    x = np.arange(len(labels))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, title, values in (
        (axes[0], "Strict metrics", (strict_p, strict_r, strict_f1)),
        (axes[1], "Relaxed metrics", (relaxed_p, relaxed_r, relaxed_f1)),
    ):
        p_vals, r_vals, f1_vals = values
        ax.bar(x - width, p_vals, width, label="Precision", color="#3498db")
        ax.bar(x, r_vals, width, label="Recall", color="#e67e22")
        ax.bar(x + width, f1_vals, width, label="F1", color="#2ecc71")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=160)
    else:
        plt.show()
