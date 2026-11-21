"""Part 3: SVM classification with noisy labels and k-NN label repair (45 points).

Trains an RBF SVM on clean, corrupted and repaired training labels of the Dry
Bean dataset and compares the three on one clean test set. Test labels are
never corrupted and are used only for evaluation.
"""

from __future__ import annotations

import argparse
import numbers
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.svm import SVC

from dataset import DEFAULT_DATA_PATH, load_dry_beans, standardize, stratified_split
from reporting import load_pyplot, plot_heatmap, write_csv

C_GRID = (0.1, 1.0, 10.0, 100.0)
GAMMA_GRID = (0.01, 0.03, 0.1, 0.3)
# (kind, rate). "uniform" flips to any other class with equal probability;
# "nearest" flips only to the class whose centroid is closest to the true class.
NOISE_SETTINGS = (("uniform", 0.20), ("uniform", 0.40), ("nearest", 0.20), ("nearest", 0.40))
REPAIR_NEIGHBORS_GRID = (5, 15, 45, 135)
REPAIR_THRESHOLD_GRID = (0.5, 0.6, 0.7)
N_TRIALS = 10
SELECTION_MODES = ("clean", "noisy")    # which validation labels tune the repair


def _check_int(value: int, name: str, minimum: int) -> None:
    """Raise unless value is an integer >= minimum (bool is rejected explicitly)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")


def _as_label_vector(labels: ArrayLike, name: str, n_classes: int) -> NDArray[np.int_]:
    """Return labels as a 1-D integer array with values in [0, n_classes), or raise."""
    array = np.asarray(labels)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must hold integer class indices, got dtype {array.dtype}")
    if array.size and (array.min() < 0 or array.max() >= n_classes):
        raise ValueError(f"{name} must lie in [0, {n_classes})")
    return array


def _as_2d_float(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """Return values as a finite (rows, columns) float array, or raise."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric array-like, "
                        f"got {type(values).__name__}") from exc
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D (rows, features), got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def uniform_transition(n_classes: int, rate: float) -> NDArray[np.float64]:
    """Return the transition matrix of symmetric noise.

    Entry [c, c'] is P(noisy label = c' | clean label = c). A label is kept
    with probability 1 - rate and otherwise moved to one of the other classes,
    each with probability rate / (n_classes - 1).
    """
    off_diagonal = np.full((n_classes, n_classes), rate / (n_classes - 1))
    np.fill_diagonal(off_diagonal, 1 - rate)
    return off_diagonal


def nearest_class_transition(nearest: ArrayLike, rate: float) -> NDArray[np.float64]:
    """Return the transition matrix of class-dependent "look-alike" noise.

    A label of class c is kept with probability 1 - rate and otherwise becomes
    nearest[c], the class most similar to c. The matrix is not symmetric: the
    look-alike of c need not have c as its own look-alike.
    """
    nearest = np.asarray(nearest, dtype=int)
    transition = np.eye(len(nearest)) * (1 - rate)
    transition[np.arange(len(nearest)), nearest] += rate
    return transition


def nearest_classes(X: NDArray, labels: NDArray, n_classes: int) -> NDArray[np.int_]:
    """Return, for each class, the other class with the closest centroid.

    This is part of the noise DESIGN (it decides which varieties get confused)
    and uses clean labels. The repair method never sees it.
    """
    centroids = np.array([X[labels == c].mean(axis=0) for c in range(n_classes)])
    sq_dist = ((centroids[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(sq_dist, np.inf)      # a class is not its own look-alike
    return np.argmin(sq_dist, axis=1)


def corrupt_labels(y: ArrayLike, transition: ArrayLike,
                   rng: np.random.Generator) -> NDArray[np.int_]:
    """Return a noisy copy of y drawn from a class-transition matrix.

    transition[c, c'] is the probability that a label whose ORIGINAL value is c
    is reported as c'. Every example is resampled exactly once from the row of
    its original label, so a label cannot be flipped and then flipped again.
    y itself is left unchanged.

    Raises ValueError if transition is not a square matrix of probabilities
    whose rows sum to 1, TypeError/ValueError if y is not a 1-D array of class
    indices, and TypeError if rng is not a NumPy random generator.
    """
    transition = _as_2d_float(transition, "transition")
    n_classes = transition.shape[0]
    if transition.shape[1] != n_classes:
        raise ValueError(f"transition must be square, got shape {transition.shape}")
    if (transition < 0).any() or not np.allclose(transition.sum(axis=1), 1):
        raise ValueError("every row of transition must be non-negative and sum to 1")
    y = _as_label_vector(y, "y", n_classes)
    if not callable(getattr(rng, "random", None)):
        raise TypeError("rng must be a NumPy random generator, e.g. "
                        f"np.random.default_rng(seed), got {type(rng).__name__}")

    # Inverse-CDF sampling with one uniform draw per example. cumulative[y] has
    # one row per example holding the running sum of its class's transition
    # probabilities; the noisy label is the first class whose running sum
    # exceeds the draw, which is the number of running sums at or below it.
    cumulative = np.cumsum(transition, axis=1)
    draws = rng.random(y.shape[0])
    noisy = (draws[:, None] >= cumulative[y]).sum(axis=1)
    # Guard against a running sum that ends at 0.999... through rounding.
    return np.minimum(noisy, n_classes - 1).astype(int)


def nearest_neighbors(X: ArrayLike, k: int, chunk_size: int = 1024) -> NDArray[np.int_]:
    """Return the (n, k) indices of each row's k nearest OTHER rows (Euclidean).

    Neighbors depend on the features only, so one call serves every noise
    setting and trial.

    Raises TypeError/ValueError for badly shaped input or k outside [1, n - 1].
    """
    X = _as_2d_float(X, "X")
    _check_int(k, "k", 1)
    n = X.shape[0]
    if k > n - 1:
        raise ValueError(f"k must be at most n - 1 = {n - 1} (a point is not its own "
                         f"neighbor), got {k}")
    sq_norms = (X * X).sum(axis=1)
    index = np.empty((n, k), dtype=int)
    # Work through the rows in chunks so the distance matrix in memory is
    # (chunk_size, n) rather than (n, n).
    for start in range(0, n, chunk_size):
        rows = np.arange(start, min(start + chunk_size, n))
        # Squared distances from one matrix product, using
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b. Only the ordering matters, so
        # the square root is skipped.
        sq_dist = sq_norms[rows, None] + sq_norms[None, :] - 2 * X[rows] @ X.T
        # A point must not vote on its own label: put it infinitely far away.
        sq_dist[np.arange(len(rows)), rows] = np.inf
        # Stable sort, so neighbors at exactly equal distance are taken in row
        # order (lowest row index first).
        index[rows] = np.argsort(sq_dist, axis=1, kind="stable")[:, :k]
    return index


def vote_shares(neighbor_index: NDArray, labels: NDArray, n_classes: int) -> NDArray[np.float64]:
    """Return the (n, n_classes) share of each point's neighbors per class."""
    neighbor_labels = labels[neighbor_index]              # (n, k) labels of neighbors
    counts = np.zeros((neighbor_index.shape[0], n_classes))
    for c in range(n_classes):
        counts[:, c] = (neighbor_labels == c).sum(axis=1)
    return counts / neighbor_index.shape[1]


def repair_labels(
    X: ArrayLike, y_noisy: ArrayLike, neighbors: int = 15, threshold: float = 0.6,
    n_classes: int | None = None, neighbor_index: NDArray | None = None,
) -> NDArray[np.int_]:
    """Return y_noisy with confidently outvoted labels replaced (k-NN repair).

    For each training point, take its `neighbors` nearest other training points
    and compute the share of them carrying each noisy label. The point's label
    is replaced by the most common neighbor label only if that label (1) differs
    from the point's own, (2) holds at least `threshold` of the votes, and
    (3) holds strictly more votes than the point's own label. Otherwise the
    label is kept. Ties for the most common label go to the lowest class index.

    All votes read the ORIGINAL noisy labels, never a repaired one, so the
    result does not depend on row order. The method sees training features and
    noisy training labels only. neighbor_index may pass a precomputed result of
    nearest_neighbors(X, k) with k >= neighbors to avoid recomputing distances.

    Raises TypeError/ValueError for badly shaped input, a neighbor count
    outside [1, n - 1], or a threshold outside (0, 1].
    """
    X = _as_2d_float(X, "X")
    _check_int(neighbors, "neighbors", 1)
    if isinstance(threshold, bool) or not isinstance(threshold, numbers.Real):
        raise TypeError(f"threshold must be a number, got {type(threshold).__name__}")
    if not 0 < threshold <= 1:
        raise ValueError(f"threshold must lie in (0, 1], got {threshold}")
    if n_classes is None:
        n_classes = int(np.max(y_noisy)) + 1
    y_noisy = _as_label_vector(y_noisy, "y_noisy", n_classes)
    if y_noisy.shape[0] != X.shape[0]:
        raise ValueError(f"got {X.shape[0]} feature rows but {y_noisy.shape[0]} labels")
    if neighbor_index is None:
        neighbor_index = nearest_neighbors(X, neighbors)
    elif neighbor_index.shape[0] != X.shape[0] or neighbor_index.shape[1] < neighbors:
        raise ValueError(f"neighbor_index must have shape ({X.shape[0]}, >= {neighbors}), "
                         f"got {neighbor_index.shape}")

    # Columns are sorted by distance, so the first `neighbors` are the nearest.
    shares = vote_shares(neighbor_index[:, :neighbors], y_noisy, n_classes)
    rows = np.arange(X.shape[0])
    winner = np.argmax(shares, axis=1)
    winner_share = shares[rows, winner]
    own_share = shares[rows, y_noisy]
    change = (winner != y_noisy) & (winner_share >= threshold) & (winner_share > own_share)
    return np.where(change, winner, y_noisy).astype(int)


def train_and_evaluate(train_X: NDArray, train_y: NDArray, eval_X: NDArray,
                       eval_y: NDArray, C: float, gamma: float) -> dict:
    """Train an RBF SVM and score it on (eval_X, eval_y).

    Returns accuracy, macro-averaged F1 (every variety counts equally, however
    many beans it has), per-class recall and the predictions.
    """
    model = SVC(kernel="rbf", C=C, gamma=gamma)
    model.fit(train_X, train_y)
    predicted = model.predict(eval_X)
    classes = np.unique(eval_y)
    return {"accuracy": float(accuracy_score(eval_y, predicted)),
            "macro_f1": float(f1_score(eval_y, predicted, average="macro", labels=classes)),
            "recall": recall_score(eval_y, predicted, average=None, labels=classes,
                                   zero_division=0),
            "predicted": predicted, "n_support": int(model.n_support_.sum())}


def tune_svm(train_X: NDArray, train_y: NDArray, val_X: NDArray,
             val_y: NDArray) -> tuple[float, float, list[list]]:
    """Grid-search C and gamma on the validation set; return (C, gamma, rows).

    The pair with the highest validation accuracy wins. Ties go to the pair
    tried first, i.e. the smallest C and then the smallest gamma (the smoother
    model). The test set plays no part.
    """
    best, rows = None, []
    for C in C_GRID:
        for gamma in GAMMA_GRID:
            score = train_and_evaluate(train_X, train_y, val_X, val_y, C, gamma)
            rows.append([C, gamma, f"{score['accuracy']:.4f}", f"{score['macro_f1']:.4f}",
                         score["n_support"]])
            if best is None or score["accuracy"] > best[0]:
                best = (score["accuracy"], C, gamma)
    return best[1], best[2], rows


def select_repair_parameters(
    train_X: NDArray, noisy_train_y: NDArray, val_X: NDArray, val_y: NDArray,
    neighbor_index: NDArray, n_classes: int, C: float, gamma: float,
) -> tuple[int, float, list[list]]:
    """Choose the repair's neighbor count and threshold on the validation set.

    Each candidate pair repairs the noisy training labels; an SVM trained on
    the result is scored against val_y, and the pair with the highest accuracy
    wins. Ties go to the pair tried first (fewest neighbors, lowest threshold).
    The caller decides whether val_y is a trusted clean sample or as noisy as
    the training labels; the repair itself only ever sees noisy_train_y.
    """
    best, rows = None, []
    for neighbors in REPAIR_NEIGHBORS_GRID:
        for threshold in REPAIR_THRESHOLD_GRID:
            repaired = repair_labels(train_X, noisy_train_y, neighbors, threshold,
                                     n_classes, neighbor_index)
            accuracy = train_and_evaluate(train_X, repaired, val_X, val_y,
                                          C, gamma)["accuracy"]
            rows.append([neighbors, threshold, accuracy])
            if best is None or accuracy > best[0]:
                best = (accuracy, neighbors, threshold)
    return best[1], best[2], rows


def setting_name(kind: str, rate: float) -> str:
    """Return a short identifier such as "nearest_40"."""
    return f"{kind}_{round(100 * rate):02d}"


def plot_accuracy(summary: dict, clean: dict, output_path: Path) -> None:
    """Save grouped bars of test accuracy and macro-F1 for every noise setting."""
    plt = load_pyplot()
    names = list(summary)
    positions = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, metric, title in ((axes[0], "accuracy", "Clean test accuracy"),
                              (axes[1], "macro_f1", "Clean test macro-F1")):
        for offset, condition, color in ((-0.2, "noisy", "tab:orange"),
                                         (0.2, "repaired", "tab:blue")):
            values = np.array([summary[name][condition][metric] for name in names])
            # Bar height = mean over trials, error bar = sample standard deviation.
            ax.bar(positions + offset, values.mean(axis=1), 0.38, yerr=values.std(axis=1, ddof=1),
                   color=color, capsize=3,
                   label="noisy labels" if condition == "noisy" else "after k-NN repair")
        ax.axhline(clean[metric], color="black", linestyle="--", linewidth=1,
                   label="clean labels")
        ax.set_xticks(positions, names)
        ax.set(title=title, xlabel="noise setting")
        ax.grid(axis="y", alpha=0.3)
    # Zoom the shared y-axis onto the range the bars actually occupy.
    lowest = min(summary[name][c][m].min() for name in names
                 for c in ("noisy", "repaired") for m in ("accuracy", "macro_f1"))
    axes[0].set_ylim(np.floor(lowest * 50) / 50 - 0.01, 0.96)
    # One legend for both panels, placed under them so it covers no bars.
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_label_error(summary: dict, output_path: Path) -> None:
    """Save the share of wrong training labels before and after repair.

    The repaired bar is split by where its errors come from: flipped labels the
    repair did not fix, and correct labels the repair changed to a wrong class.
    """
    plt = load_pyplot()
    names = list(summary)
    positions = np.arange(len(names))
    noisy = np.array([summary[name]["noisy"]["label_error"].mean() for name in names])
    unfixed = np.array([summary[name]["repaired"]["unfixed_share"].mean() for name in names])
    broken = np.array([summary[name]["repaired"]["broken_share"].mean() for name in names])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(positions - 0.2, noisy, 0.38, color="tab:orange", label="noisy labels")
    ax.bar(positions + 0.2, unfixed, 0.38, color="tab:blue",
           label="after repair: flips left unfixed")
    ax.bar(positions + 0.2, broken, 0.38, bottom=unfixed, color="tab:red",
           label="after repair: correct labels made wrong")
    ax.set_xticks(positions, names)
    ax.set(title="Share of wrong training labels (mean over trials)", xlabel="noise setting")
    ax.grid(axis="y", alpha=0.3)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_class_recall(summary: dict, clean: dict, class_names: list[str],
                      output_path: Path) -> None:
    """Save per-variety test recall for clean, noisy and repaired training."""
    plt = load_pyplot()
    names = list(summary)
    positions = np.arange(len(class_names))
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.6), sharey=True)
    for ax, name in zip(axes, names):
        ax.bar(positions - 0.27, clean["recall"], 0.26, color="0.4", label="clean labels")
        ax.bar(positions, summary[name]["noisy"]["recall"].mean(axis=0), 0.26,
               color="tab:orange", label="noisy labels")
        ax.bar(positions + 0.27, summary[name]["repaired"]["recall"].mean(axis=0), 0.26,
               color="tab:blue", label="after k-NN repair")
        ax.set_xticks(positions, class_names, rotation=45, ha="right", fontsize=7)
        ax.set_title(name, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("test recall")
    axes[0].set_ylim(0.75, 1.0)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_repair_selection(selection: dict, mode: str, output_path: Path) -> None:
    """Save validation accuracy over the repair grid, one panel per setting."""
    plt = load_pyplot()
    fig, axes = plt.subplots(1, len(selection), figsize=(3.1 * len(selection), 3.0))
    for ax, (name, (chosen_k, chosen_t, rows)) in zip(axes, selection.items()):
        grid = np.array([row[2] for row in rows]).reshape(len(REPAIR_NEIGHBORS_GRID),
                                                          len(REPAIR_THRESHOLD_GRID))
        ax.imshow(grid, cmap="Blues", aspect="auto")
        for i, neighbors in enumerate(REPAIR_NEIGHBORS_GRID):
            for j, threshold in enumerate(REPAIR_THRESHOLD_GRID):
                picked = neighbors == chosen_k and threshold == chosen_t
                # The selected pair is printed in bold with a box around it.
                ax.text(j, i, f"{grid[i, j]:.4f}", ha="center", va="center", fontsize=7,
                        fontweight="bold" if picked else "normal",
                        color="white" if grid[i, j] > grid.min() + 0.6 * np.ptp(grid) else "black",
                        bbox=dict(boxstyle="round", fc="none", ec="crimson") if picked else None)
        ax.set_xticks(range(len(REPAIR_THRESHOLD_GRID)), REPAIR_THRESHOLD_GRID)
        ax.set_yticks(range(len(REPAIR_NEIGHBORS_GRID)), REPAIR_NEIGHBORS_GRID)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("vote threshold")
    axes[0].set_ylabel("neighbors k")
    fig.suptitle(f"Accuracy against {mode} validation labels (boxed = selected)", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_comparison(
    output_dir: str | Path, seed: int = 0, n_trials: int = N_TRIALS,
    data_path: str | Path = DEFAULT_DATA_PATH, selection_mode: str = "clean",
) -> dict:
    """Run the whole Part 3 study and save its tables and figures.

    One clean train/validation/test split is drawn once. Every noise setting is
    then repeated n_trials times with fresh label noise; within a trial the
    noisy and the repaired classifier share the same corrupted labels.

    selection_mode says which validation labels tune the repair parameters:
    "clean" treats the validation split as a small trusted sample; "noisy"
    corrupts it with the same noise as the training labels, so that no clean
    label is used anywhere before the final test.

    Raises TypeError/ValueError if n_trials is not an integer >= 2 (a standard
    deviation needs at least two trials) or selection_mode is unknown.
    """
    _check_int(n_trials, "n_trials", 2)
    if selection_mode not in SELECTION_MODES:
        raise ValueError(f"selection_mode must be one of {SELECTION_MODES}, "
                         f"got {selection_mode!r}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----- 3.1 data, split, clean baseline -----
    features, labels, _, class_names = load_dry_beans(data_path)
    n_classes = len(class_names)
    train, val, test = stratified_split(labels, np.random.default_rng(seed))
    # Scaling statistics come from the training rows only.
    train_X, val_X, test_X = standardize(features[train], features[val], features[test])
    train_y, val_y, test_y = labels[train], labels[val], labels[test]
    print(f"split: {len(train)} train / {len(val)} validation / {len(test)} test, seed {seed}")

    C, gamma, grid_rows = tune_svm(train_X, train_y, val_X, val_y)
    write_csv(output_dir / "svm_grid.csv",
              ["C", "gamma", "val_accuracy", "val_macro_f1", "n_support_vectors"], grid_rows)
    clean = train_and_evaluate(train_X, train_y, test_X, test_y, C, gamma)
    print(f"selected C = {C}, gamma = {gamma} on clean validation data")
    print(f"clean baseline: test accuracy {clean['accuracy']:.4f}, "
          f"macro-F1 {clean['macro_f1']:.4f}")
    confusion = confusion_matrix(test_y, clean["predicted"])
    plot_heatmap(confusion, class_names, class_names,
                 "Clean baseline: test confusion matrix",
                 output_dir / "confusion_clean.png", "d", "Blues",
                 xlabel="predicted variety", ylabel="true variety")
    write_csv(output_dir / "confusion_clean.csv", ["true_variety", *class_names],
              [[class_names[c], *row] for c, row in enumerate(confusion.tolist())])

    # ----- 3.2 noise design -----
    nearest = nearest_classes(train_X, train_y, n_classes)
    write_csv(output_dir / "nearest_class.csv", ["class", "look_alike_class"],
              [[class_names[c], class_names[nearest[c]]] for c in range(n_classes)])
    print("look-alike classes: " + ", ".join(
        f"{class_names[c]}->{class_names[nearest[c]]}" for c in range(n_classes)))

    # Neighbors depend on features only: compute them once, for the largest k.
    neighbor_index = nearest_neighbors(train_X, max(REPAIR_NEIGHBORS_GRID))

    # Cost of repairing labels that are already clean (reported in 3.5). The
    # repair parameters are chosen exactly as below, here on clean validation
    # labels because this "setting" has no noise.
    clean_k, clean_t, _ = select_repair_parameters(
        train_X, train_y, val_X, val_y, neighbor_index, n_classes, C, gamma)
    clean_repaired_y = repair_labels(train_X, train_y, clean_k, clean_t, n_classes,
                                     neighbor_index)
    clean_repaired = train_and_evaluate(train_X, clean_repaired_y, test_X, test_y, C, gamma)
    clean_repaired["label_error"] = float(np.mean(clean_repaired_y != train_y))
    print(f"repair applied to clean labels (k = {clean_k}, threshold = {clean_t}): "
          f"test accuracy {clean_repaired['accuracy']:.4f}, "
          f"{clean_repaired['label_error']:.4f} of labels made wrong")

    summary, selection, trial_rows, selection_rows = {}, {}, [], []
    for setting_id, (kind, rate) in enumerate(NOISE_SETTINGS):
        name = setting_name(kind, rate)
        transition = (uniform_transition(n_classes, rate) if kind == "uniform"
                      else nearest_class_transition(nearest, rate))

        # ----- 3.3 choose the repair parameters on a separate noise draw -----
        # Generators are seeded with [seed, setting, trial]: trial 0 is used for
        # selection only, trials 1..n_trials for the reported comparison.
        rng = np.random.default_rng([seed, setting_id, 0])
        selection_train_y = corrupt_labels(train_y, transition, rng)
        selection_val_y = (val_y if selection_mode == "clean"
                           else corrupt_labels(val_y, transition, rng))
        neighbors, threshold, rows = select_repair_parameters(
            train_X, selection_train_y, val_X, selection_val_y, neighbor_index,
            n_classes, C, gamma)
        selection[name] = (neighbors, threshold, rows)
        selection_rows += [[name, k, t, f"{a:.4f}", int(k == neighbors and t == threshold)]
                           for k, t, a in rows]
        print(f"[{name}] repair parameters ({selection_mode} validation labels): "
              f"k = {neighbors}, threshold = {threshold}")

        # ----- 3.4 controlled comparison -----
        scores = {condition: {"accuracy": [], "macro_f1": [], "label_error": [], "recall": []}
                  for condition in ("noisy", "repaired")}
        scores["repaired"].update(unfixed_share=[], broken_share=[])
        for trial in range(1, n_trials + 1):
            rng = np.random.default_rng([seed, setting_id, trial])
            noisy_y = corrupt_labels(train_y, transition, rng)
            # The repair sees features and noisy labels only: no clean labels.
            repaired_y = repair_labels(train_X, noisy_y, neighbors, threshold, n_classes,
                                       neighbor_index)
            # Diagnostics computed with the clean training labels AFTER the
            # fact: which flipped labels the repair fixed or left alone, and
            # which correct labels it changed to a wrong class.
            flipped = noisy_y != train_y
            wrong_after = repaired_y != train_y
            fixed = int((flipped & ~wrong_after).sum())
            broken = int((~flipped & wrong_after).sum())
            diagnostics = {"unfixed_share": float((flipped & wrong_after).mean()),
                           "broken_share": float((~flipped & wrong_after).mean())}
            for condition, y in (("noisy", noisy_y), ("repaired", repaired_y)):
                result = train_and_evaluate(train_X, y, test_X, test_y, C, gamma)
                result["label_error"] = float(np.mean(y != train_y))
                result.update(diagnostics)
                for key in scores[condition]:
                    scores[condition][key].append(result[key])
            trial_rows.append(
                [name, trial]
                + [f"{scores[c][m][-1]:.4f}" for m in ("accuracy", "macro_f1", "label_error")
                   for c in ("noisy", "repaired")]
                + [int(flipped.sum()), fixed, broken])
            print(f"[{name}] trial {trial:2d}: noisy {scores['noisy']['accuracy'][-1]:.4f}  "
                  f"repaired {scores['repaired']['accuracy'][-1]:.4f}")
        summary[name] = {c: {k: np.array(v) for k, v in s.items()} for c, s in scores.items()}

    # ----- tables -----
    write_csv(output_dir / "repair_selection.csv",
              ["setting", "neighbors", "threshold", "noisy_val_accuracy", "selected"],
              selection_rows)
    write_csv(output_dir / "trials.csv",
              ["setting", "trial", "noisy_accuracy", "repaired_accuracy", "noisy_macro_f1",
               "repaired_macro_f1", "noisy_label_error", "repaired_label_error",
               "flipped_labels", "flipped_labels_fixed", "correct_labels_made_wrong"],
              trial_rows)
    summary_rows = [["clean", "clean", 1, f"{clean['accuracy']:.4f}", "", f"{clean['macro_f1']:.4f}",
                     "", "0.0000", "", "", ""],
                    ["clean", "repaired", 1, f"{clean_repaired['accuracy']:.4f}", "",
                     f"{clean_repaired['macro_f1']:.4f}", "",
                     f"{clean_repaired['label_error']:.4f}", "", clean_k, clean_t]]
    recall_rows = [["clean", "clean", *(f"{r:.4f}" for r in clean["recall"])],
                   ["clean", "repaired", *(f"{r:.4f}" for r in clean_repaired["recall"])]]
    print(f"\n{'setting':<12}{'condition':<10}{'accuracy':>18}{'macro-F1':>18}{'label error':>13}")
    print(f"{'clean':<12}{'clean':<10}{clean['accuracy']:>18.4f}{clean['macro_f1']:>18.4f}"
          f"{0:>13.4f}")
    for name, conditions in summary.items():
        # Number of trials in which the repaired classifier beat the noisy one.
        wins = int((conditions["repaired"]["accuracy"] > conditions["noisy"]["accuracy"]).sum())
        for condition, s in conditions.items():
            acc, f1 = s["accuracy"], s["macro_f1"]
            summary_rows.append([name, condition, n_trials, f"{acc.mean():.4f}",
                                 f"{acc.std(ddof=1):.4f}", f"{f1.mean():.4f}",
                                 f"{f1.std(ddof=1):.4f}", f"{s['label_error'].mean():.4f}",
                                 *((wins, *selection[name][:2]) if condition == "repaired"
                                   else ("", "", ""))])
            recall_rows.append([name, condition, *(f"{r:.4f}" for r in s["recall"].mean(axis=0))])
            print(f"{name:<12}{condition:<10}"
                  f"{acc.mean():>10.4f} +/- {acc.std(ddof=1):.4f}"
                  f"{f1.mean():>10.4f} +/- {f1.std(ddof=1):.4f}{s['label_error'].mean():>13.4f}")
        difference = conditions["repaired"]["accuracy"] - conditions["noisy"]["accuracy"]
        print(f"{'':<12}repaired - noisy: {difference.mean():+.4f} +/- "
              f"{difference.std(ddof=1):.4f}, better in {wins} of {n_trials} trials")
    write_csv(output_dir / "summary.csv",
              ["setting", "condition", "trials", "accuracy_mean", "accuracy_std",
               "macro_f1_mean", "macro_f1_std", "train_label_error_mean",
               "trials_repaired_beats_noisy", "repair_neighbors", "repair_threshold"],
              summary_rows)
    write_csv(output_dir / "per_class_recall.csv", ["setting", "condition", *class_names],
              recall_rows)

    # ----- figures -----
    plot_accuracy(summary, clean, output_dir / "accuracy_by_setting.png")
    plot_label_error(summary, output_dir / "label_error.png")
    plot_class_recall(summary, clean, class_names, output_dir / "per_class_recall.png")
    plot_repair_selection(selection, selection_mode, output_dir / "repair_selection.png")
    print(f"Saved tables and plots to {output_dir}")
    return {"clean": clean, "clean_repaired": clean_repaired, "summary": summary,
            "selection": selection, "svm": (C, gamma), "nearest": nearest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=N_TRIALS,
                        help="Independent label-noise draws per noise setting.")
    parser.add_argument("--selection", choices=SELECTION_MODES, default="clean",
                        help="Validation labels used to tune the repair parameters.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: results/noisy_labels, or "
                             "results/noisy_labels_noisy_selection with --selection noisy.")
    args = parser.parse_args()
    if args.output_dir is None:
        folder = "noisy_labels" if args.selection == "clean" else "noisy_labels_noisy_selection"
        args.output_dir = Path(__file__).resolve().parent / "results" / folder
    run_comparison(args.output_dir, args.seed, args.trials, args.data, args.selection)


if __name__ == "__main__":
    main()
