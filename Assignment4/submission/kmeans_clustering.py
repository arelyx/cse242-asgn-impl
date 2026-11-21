"""Part 2: K-means from scratch on the Dry Bean dataset (35 points).

Fits K-means for several values of K, scores every K with three diagnostics
(within-cluster sum of squares, silhouette, stability across random starts) and
saves the tables and figures used in the report.

The variety labels are never used to fit a model. They are read only after
clustering, to interpret what the clusters contain.
"""

from __future__ import annotations

import argparse
import numbers
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import adjusted_rand_score, silhouette_samples

from dataset import DEFAULT_DATA_PATH, load_dry_beans, standardize
from reporting import load_pyplot, plot_heatmap, write_csv

K_VALUES = (2, 3, 4, 5, 6, 7, 8, 9, 10)
# K examined in detail (profiles, variety table, ambiguity). Section 2.3 of the
# report explains the choice; change it with --chosen-k.
CHOSEN_K = 5
N_INIT = 10                 # independent random starts per K
AMBIGUITY_RATIO = 0.8       # nearest / second-nearest centroid distance


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


def _check_int(value: int, name: str, minimum: int) -> None:
    """Raise unless value is an integer >= minimum (bool is rejected explicitly)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")


def _check_data_and_centroids(
    X: ArrayLike, centroids: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate the (n, d) data and (k, d) centroids and return them as arrays."""
    X = _as_2d_float(X, "X")
    centroids = _as_2d_float(centroids, "centroids")
    if centroids.shape[0] == 0:
        raise ValueError("centroids must contain at least one row")
    if centroids.shape[1] != X.shape[1]:
        raise ValueError(f"X has {X.shape[1]} features but centroids have "
                         f"{centroids.shape[1]}")
    return X, centroids


def squared_distances(X: NDArray, centroids: NDArray) -> NDArray[np.float64]:
    """Return the (n, k) matrix of squared distances from points to centroids."""
    # Broadcasting builds every point-to-centroid difference at once:
    # X[:, None, :] has shape (n, 1, d) and centroids[None, :, :] has shape
    # (1, k, d), so the difference has shape (n, k, d). Squaring and summing over
    # the feature axis leaves entry [i, j] = squared distance from point i to
    # centroid j.
    return ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)


def update_assignments(X: ArrayLike, centroids: ArrayLike) -> NDArray[np.int_]:
    """Return the index of the closest centroid for each observation.

    X has shape (n, d); centroids has shape (k, d); the result has shape (n,).

    Raises TypeError for non-numeric input and ValueError for input with the
    wrong shape, mismatched feature counts, or NaN/infinite values.
    """
    X, centroids = _check_data_and_centroids(X, centroids)
    # Squared distances are compared because they order the centroids the same
    # way Euclidean distances do, with no square root needed.
    # Tie rule: np.argmin returns the FIRST minimum, so a point equidistant from
    # several centroids goes to the one with the lowest index.
    return np.argmin(squared_distances(X, centroids), axis=1).astype(int)


def update_centroids(
    X: ArrayLike, assignments: ArrayLike, centroids: ArrayLike
) -> NDArray[np.float64]:
    """Return updated centroids: the mean of the points assigned to each cluster.

    centroids holds the current (k, d) centroids. It fixes k and supplies the
    position of any cluster that received no points; it is not modified.

    Raises TypeError if assignments are not integers and ValueError if they do
    not hold one index in [0, k) per row of X.
    """
    X, centroids = _check_data_and_centroids(X, centroids)
    assignments = np.asarray(assignments)
    if not np.issubdtype(assignments.dtype, np.integer):
        raise TypeError(f"assignments must be integers, got dtype {assignments.dtype}")
    if assignments.shape != (X.shape[0],):
        raise ValueError(f"assignments must have shape ({X.shape[0]},), "
                         f"got {assignments.shape}")
    k = centroids.shape[0]
    if assignments.size and (assignments.min() < 0 or assignments.max() >= k):
        raise ValueError(f"assignments must lie in [0, {k})")

    # Start from a copy so the caller's centroid array is never modified.
    new_centroids = centroids.copy()
    for j in range(k):
        members = X[assignments == j]
        # Empty-cluster policy: a cluster that received no points keeps its
        # previous centroid, so it can win points back in a later iteration.
        if len(members):
            new_centroids[j] = members.mean(axis=0)
    return new_centroids


def initialize_centroids(
    X: ArrayLike, k: int, rng: np.random.Generator, method: str = "kmeans++"
) -> NDArray[np.float64]:
    """Return k starting centroids chosen from the rows of X.

    "kmeans++" picks the first centroid uniformly at random and every later
    one with probability proportional to its squared distance from the nearest
    centroid chosen so far, which spreads the starts over the data.
    "random" picks k distinct rows uniformly at random.

    Raises ValueError for an unknown method or if k exceeds the number of rows.
    """
    X = _as_2d_float(X, "X")
    _check_int(k, "k", 1)
    n = X.shape[0]
    if k > n:
        raise ValueError(f"k = {k} exceeds the number of observations ({n})")
    if method == "random":
        return X[rng.choice(n, size=k, replace=False)].copy()
    if method != "kmeans++":
        raise ValueError(f"method must be 'kmeans++' or 'random', got {method!r}")

    chosen = [int(rng.integers(n))]
    # nearest_sq[i] = squared distance from point i to its closest chosen centroid.
    nearest_sq = ((X - X[chosen[0]]) ** 2).sum(axis=1)
    while len(chosen) < k:
        total = nearest_sq.sum()
        if total > 0:
            # A chosen row has distance 0 and therefore probability 0, so no row
            # is picked twice.
            pick = int(rng.choice(n, p=nearest_sq / total))
        else:
            # Every remaining point coincides with a chosen centroid (fewer than
            # k distinct rows): fall back to a uniform draw.
            pick = int(rng.integers(n))
        chosen.append(pick)
        nearest_sq = np.minimum(nearest_sq, ((X - X[pick]) ** 2).sum(axis=1))
    return X[chosen].copy()


def within_cluster_ss(X: NDArray, centroids: NDArray, assignments: NDArray) -> float:
    """Return the K-means objective: summed squared distance to own centroid."""
    # centroids[assignments] lines each point up with its own centroid.
    return float(((X - centroids[assignments]) ** 2).sum())


def kmeans(
    X: ArrayLike, k: int, max_iterations: int = 100, tol: float = 1e-6,
    rng: np.random.Generator | None = None, init: str = "kmeans++",
) -> tuple[NDArray[np.float64], NDArray[np.int_], dict]:
    """Fit K-means from scratch and return (centroids, assignments, history).

    Stopping criterion: stop as soon as an iteration changes no assignment, or
    the largest centroid movement is at most tol; otherwise stop after
    max_iterations. history records the objective after every iteration
    ("wcss"), the largest centroid movement ("shift"), the number of iterations
    run, whether the run converged, and how often a cluster came up empty.

    Raises TypeError/ValueError for a bad k, max_iterations or tol, plus the
    checks made by initialize_centroids and the two update functions.
    """
    X = _as_2d_float(X, "X")
    _check_int(k, "k", 1)
    _check_int(max_iterations, "max_iterations", 1)
    if isinstance(tol, bool) or not isinstance(tol, numbers.Real) or tol < 0:
        raise ValueError(f"tol must be a non-negative number, got {tol!r}")
    rng = np.random.default_rng() if rng is None else rng

    centroids = initialize_centroids(X, k, rng, init)
    assignments = None
    history = {"wcss": [], "shift": [], "n_iterations": 0, "converged": False,
               "empty_cluster_events": 0}
    for _ in range(max_iterations):
        new_assignments = update_assignments(X, centroids)
        new_centroids = update_centroids(X, new_assignments, centroids)
        # Largest distance moved by any single centroid in this iteration.
        shift = float(np.sqrt(((new_centroids - centroids) ** 2).sum(axis=1)).max())
        unchanged = assignments is not None and np.array_equal(new_assignments, assignments)
        centroids, assignments = new_centroids, new_assignments

        history["wcss"].append(within_cluster_ss(X, centroids, assignments))
        history["shift"].append(shift)
        history["n_iterations"] += 1
        history["empty_cluster_events"] += int(k - len(np.unique(assignments)))
        if unchanged or shift <= tol:
            history["converged"] = True
            break
    # The loop ends on a centroid update, so assign once more to return labels
    # that match the returned centroids.
    assignments = update_assignments(X, centroids)
    return centroids, assignments, history


def pca_project(
    X: ArrayLike, n_components: int = 2
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return (scores, components, explained_variance_ratio) of a PCA of X.

    Used for visualization only: K-means always runs on the full feature set.
    """
    X = _as_2d_float(X, "X")
    centered = X - X.mean(axis=0)
    # SVD of the centred data: the rows of vt are the principal directions and
    # the squared singular values are proportional to the variance along each.
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    explained = singular ** 2 / (singular ** 2).sum()
    components = vt[:n_components]
    return centered @ components.T, components, explained[:n_components]


def contingency_table(assignments: NDArray, labels: NDArray, k: int,
                      n_classes: int) -> NDArray[np.int_]:
    """Return counts[cluster, class]. Interpretation only; needs the labels."""
    table = np.zeros((k, n_classes), dtype=int)
    np.add.at(table, (assignments, labels), 1)
    return table


def ambiguity_ratio(X: NDArray, centroids: NDArray) -> NDArray[np.float64]:
    """Return, per point, nearest / second-nearest centroid distance (in [0, 1]).

    A ratio near 1 means the point lies almost on the border between two
    clusters, so its hard assignment is close to arbitrary.
    """
    ordered = np.sort(np.sqrt(squared_distances(X, centroids)), axis=1)
    return ordered[:, 0] / ordered[:, 1]


def evaluate_k(X: NDArray, k: int, n_init: int, max_iterations: int,
               rng: np.random.Generator) -> dict:
    """Run n_init independent K-means fits for one K and score them.

    Returns the best run (lowest objective) with its silhouette values, the
    spread of the objective over the starts, and a stability score: the mean
    adjusted Rand index between every pair of runs (1 = all starts agree).
    """
    runs = [kmeans(X, k, max_iterations=max_iterations, rng=rng) for _ in range(n_init)]
    wcss = np.array([history["wcss"][-1] for _, _, history in runs])
    best = int(np.argmin(wcss))
    centroids, assignments, history = runs[best]
    pair_ari = [adjusted_rand_score(runs[a][1], runs[b][1])
                for a in range(n_init) for b in range(a + 1, n_init)]
    return {"k": k, "centroids": centroids, "assignments": assignments,
            "wcss": float(wcss[best]), "wcss_worst": float(wcss.max()),
            "silhouette_values": silhouette_samples(X, assignments),
            "stability": float(np.mean(pair_ari)) if pair_ari else 1.0,
            "sizes": np.bincount(assignments, minlength=k),
            "n_iterations": history["n_iterations"],
            "converged": all(h["converged"] for _, _, h in runs),
            "empty_cluster_events": sum(h["empty_cluster_events"] for _, _, h in runs)}


def plot_k_diagnostics(results: dict[int, dict], chosen_k: int, output_path: Path) -> None:
    """Save the three K-selection diagnostics side by side."""
    plt = load_pyplot()
    ks = sorted(results)
    panels = (
        ("Within-cluster sum of squares", [results[k]["wcss"] for k in ks]),
        ("Mean silhouette", [float(results[k]["silhouette_values"].mean()) for k in ks]),
        ("Stability (mean pairwise ARI of starts)", [results[k]["stability"] for k in ks]),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (title, values) in zip(axes, panels):
        ax.plot(ks, values, "o-", color="tab:blue")
        ax.axvline(chosen_k, color="crimson", linestyle="--", linewidth=1,
                   label=f"chosen K = {chosen_k}")
        ax.set(title=title, xlabel="K", xticks=ks)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_pca_scatter(scores: NDArray, colors: NDArray, title: str, output_path: Path,
                     names: list[str], explained: NDArray,
                     centroids_2d: NDArray | None = None) -> None:
    """Save a PCA scatter plot coloured by an integer code per point."""
    plt = load_pyplot()
    fig, ax = plt.subplots(figsize=(6.4, 5))
    # vmin/vmax pin the tab10 colour scale so a code has the same colour in
    # every figure, whatever the number of groups.
    ax.scatter(scores[:, 0], scores[:, 1], c=colors, cmap="tab10", s=3, alpha=0.6,
               vmin=0, vmax=9, linewidths=0)
    # Empty scatters only create one legend entry per group.
    cmap = plt.get_cmap("tab10")
    for code, name in enumerate(names):
        ax.scatter([], [], color=cmap(code / 9), s=25, label=name)
    if centroids_2d is not None:
        ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c="black", marker="X", s=140,
                   edgecolors="white", linewidths=1.5, label="centroids")
    ax.set(title=title, xlabel=f"PC1 ({explained[0]:.0%} of variance)",
           ylabel=f"PC2 ({explained[1]:.0%} of variance)")
    ax.legend(loc="best", fontsize=7, markerscale=1.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_ambiguity(scores: NDArray, ratio: NDArray, threshold: float, explained: NDArray,
                   output_path: Path) -> None:
    """Save a PCA scatter that highlights points lying between two clusters."""
    plt = load_pyplot()
    ambiguous = ratio > threshold
    fig, ax = plt.subplots(figsize=(6.4, 5))
    ax.scatter(scores[~ambiguous, 0], scores[~ambiguous, 1], c="0.75", s=3, linewidths=0,
               label=f"ratio <= {threshold} ({(~ambiguous).mean():.1%})")
    ax.scatter(scores[ambiguous, 0], scores[ambiguous, 1], c="crimson", s=4, linewidths=0,
               label=f"ratio > {threshold} ({ambiguous.mean():.1%})")
    ax.set(title="Points almost equidistant from two centroids",
           xlabel=f"PC1 ({explained[0]:.0%} of variance)",
           ylabel=f"PC2 ({explained[1]:.0%} of variance)")
    ax.legend(loc="best", fontsize=8, markerscale=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_experiments(
    output_dir: str | Path, seed: int = 0, max_iterations: int = 100,
    n_init: int = N_INIT, chosen_k: int = CHOSEN_K,
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> dict[int, dict]:
    """Fit every K, save the diagnostics, and analyse chosen_k in detail.

    Raises ValueError if chosen_k is not one of K_VALUES, and TypeError or
    ValueError if n_init or max_iterations is not a positive integer.
    """
    _check_int(n_init, "n_init", 1)
    _check_int(max_iterations, "max_iterations", 1)
    if chosen_k not in K_VALUES:
        raise ValueError(f"chosen_k must be one of {K_VALUES}, got {chosen_k}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features, labels, feature_names, class_names = load_dry_beans(data_path)
    # Z-score every feature: the raw columns range from about 0.003 (shape
    # factors) to about 250,000 (pixel areas), and unscaled Euclidean distance
    # would be decided by the area columns alone.
    (X,) = standardize(features)
    rng = np.random.default_rng(seed)
    scores, components, explained = pca_project(X)
    print(f"{X.shape[0]} beans, {X.shape[1]} standardized features, seed {seed}, "
          f"{n_init} starts per K")

    plot_pca_scatter(scores, labels, "Reference: the seven registered varieties",
                     output_dir / "pca_varieties.png", class_names, explained)

    results, table, previous_wcss = {}, [], None
    for k in K_VALUES:
        result = evaluate_k(X, k, n_init, max_iterations, rng)
        counts = contingency_table(result["assignments"], labels, k, len(class_names))
        # The two label-based scores are computed AFTER clustering, for
        # interpretation only. Purity = share of beans in their cluster's
        # majority variety.
        result["counts"] = counts
        result["purity"] = float(counts.max(axis=1).sum() / counts.sum())
        result["ari_varieties"] = float(adjusted_rand_score(labels, result["assignments"]))
        results[k] = result

        silhouette = float(result["silhouette_values"].mean())
        # Relative drop of the objective from the previous K: the "elbow" is
        # where this drop becomes small.
        drop = "" if previous_wcss is None else f"{1 - result['wcss'] / previous_wcss:.4f}"
        previous_wcss = result["wcss"]
        table.append([k, f"{result['wcss']:.2f}", drop, f"{result['wcss_worst']:.2f}",
                      f"{silhouette:.4f}", f"{result['stability']:.4f}",
                      result["n_iterations"], result["empty_cluster_events"],
                      f"{result['purity']:.4f}", f"{result['ari_varieties']:.4f}"])
        cluster_names = [f"cluster {j}" for j in range(k)]
        # PCA is linear, so centroids are projected exactly like the points.
        centroids_2d = (result["centroids"] - X.mean(axis=0)) @ components.T
        plot_pca_scatter(scores, result["assignments"], f"K-means, K = {k}",
                         output_dir / f"pca_clusters_k{k}.png", cluster_names, explained,
                         centroids_2d)
        # Centroids are already in z-score units: an entry of +1 means the
        # cluster averages one standard deviation above the whole-dataset mean.
        plot_heatmap(result["centroids"], cluster_names, feature_names,
                     f"Cluster centroids at K = {k} (z-scores)",
                     output_dir / f"cluster_profiles_k{k}.png", ".1f", "RdBu_r",
                     symmetric=True)
        plot_heatmap(counts, cluster_names, class_names,
                     f"Beans per cluster and variety at K = {k}",
                     output_dir / f"cluster_variety_k{k}.png", "d", "Blues")
        write_csv(output_dir / f"cluster_variety_k{k}.csv", ["cluster", *class_names],
                  [[j, *row] for j, row in enumerate(counts.tolist())])
        write_csv(output_dir / f"cluster_profiles_k{k}.csv", ["cluster", *feature_names],
                  [[j, *(f"{v:.4f}" for v in row)] for j, row in enumerate(result["centroids"])])
        print(f"K={k:2d}  WCSS={result['wcss']:9.1f}  silhouette={silhouette:.3f}  "
              f"stability={result['stability']:.3f}  sizes={result['sizes'].tolist()}")
        if not result["converged"]:
            print(f"      warning: a start hit max_iterations = {max_iterations}")

    write_csv(output_dir / "k_diagnostics.csv",
              ["k", "wcss_best", "wcss_drop_from_previous_k", "wcss_worst_start",
               "mean_silhouette", "stability_ari",
               "iterations_best", "empty_cluster_events", "purity", "ari_vs_varieties"],
              table)
    plot_k_diagnostics(results, chosen_k, output_dir / "k_diagnostics.png")

    # ----- ambiguity of the hard assignments at the chosen K -----
    chosen = results[chosen_k]
    ratio = ambiguity_ratio(X, chosen["centroids"])
    ambiguous = ratio > AMBIGUITY_RATIO
    negative = chosen["silhouette_values"] < 0
    plot_ambiguity(scores, ratio, AMBIGUITY_RATIO, explained,
                   output_dir / f"ambiguity_k{chosen_k}.png")
    write_csv(output_dir / f"ambiguity_k{chosen_k}.csv",
              ["cluster", "size", "share_ratio_above_threshold", "share_negative_silhouette",
               "mean_silhouette"],
              [[j, int((chosen["assignments"] == j).sum()),
                f"{ambiguous[chosen['assignments'] == j].mean():.4f}",
                f"{negative[chosen['assignments'] == j].mean():.4f}",
                f"{chosen['silhouette_values'][chosen['assignments'] == j].mean():.4f}"]
               for j in range(chosen_k)])
    chosen["ambiguity_ratio"] = ratio
    print(f"Chosen K = {chosen_k}: purity {chosen['purity']:.3f}, "
          f"ARI vs varieties {chosen['ari_varieties']:.3f}, "
          f"{ambiguous.mean():.1%} of beans have distance ratio > {AMBIGUITY_RATIO}, "
          f"{negative.mean():.1%} have a negative silhouette")
    print(f"Saved tables and plots to {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--n-init", type=int, default=N_INIT,
                        help="Independent random starts per K.")
    parser.add_argument("--chosen-k", type=int, default=CHOSEN_K, choices=K_VALUES,
                        help="K to analyse in detail.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results" / "kmeans")
    args = parser.parse_args()
    run_experiments(args.output_dir, args.seed, args.max_iterations, args.n_init,
                    args.chosen_k, args.data)


if __name__ == "__main__":
    main()
