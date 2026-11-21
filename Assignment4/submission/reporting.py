"""Small helpers for saving tables and figures, shared by Parts 2 and 3."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def load_pyplot():
    """Import pyplot with the non-interactive Agg backend.

    The import stays inside a function because the backend has to be chosen
    before pyplot is first imported, and doing that at module level would
    change the backend of any program that merely imports this file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def write_csv(path: str | Path, header: list[str], rows: list[list]) -> None:
    """Write a small results table as CSV."""
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def plot_heatmap(values: NDArray, row_names: list[str], column_names: list[str],
                 title: str, output_path: str | Path, fmt: str, cmap: str,
                 symmetric: bool = False, xlabel: str = "", ylabel: str = "") -> None:
    """Save an annotated heatmap with one number printed in every cell.

    symmetric=True centres the colour scale on zero (for z-scores); otherwise
    the scale runs from zero to the largest value (for counts).
    """
    plt = load_pyplot()
    values = np.asarray(values)
    fig, ax = plt.subplots(figsize=(1.1 + 0.62 * len(column_names),
                                    1.3 + 0.42 * len(row_names)))
    limit = np.abs(values).max()
    image = ax.imshow(values, cmap=cmap, aspect="auto",
                      vmin=-limit if symmetric else 0, vmax=limit)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            # White text on the dark ends of the colour scale, black elsewhere.
            dark = abs(values[i, j]) > 0.6 * limit
            ax.text(j, i, format(values[i, j], fmt), ha="center", va="center",
                    fontsize=7, color="white" if dark else "black")
    ax.set_xticks(range(len(column_names)), column_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_names)), row_names, fontsize=8)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, fontsize=10)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
