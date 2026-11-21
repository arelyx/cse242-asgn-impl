"""Dry Bean dataset: loading, splitting and scaling shared by Parts 2 and 3.

Source: Koklu, M. and Ozkan, I.A. (2020), "Multiclass Classification of Dry
Beans Using Computer Vision and Machine Learning Techniques", Computers and
Electronics in Agriculture 174, 105507. Distributed by the UCI Machine
Learning Repository (dataset 602) under a CC BY 4.0 licence.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

DATA_URL = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
ARFF_MEMBER = "DryBeanDataset/Dry_Bean_Dataset.arff"
DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "Dry_Bean_Dataset.arff"
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)       # train, validation, test


def download_dataset(path: str | Path = DEFAULT_DATA_PATH) -> Path:
    """Fetch the ARFF file from UCI into path and return it.

    Only needed if data/Dry_Bean_Dataset.arff is missing; the submission ships
    with the file, so the experiments run without network access.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATA_URL, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    path.write_bytes(archive.read(ARFF_MEMBER))
    return path


def load_dry_beans(
    path: str | Path = DEFAULT_DATA_PATH, drop_duplicates: bool = True
) -> tuple[NDArray[np.float64], NDArray[np.int_], list[str], list[str]]:
    """Return (features, labels, feature_names, class_names).

    features has shape (n, 16); labels holds one integer per row that indexes
    class_names, which is sorted alphabetically so the coding does not depend
    on the order of rows in the file.

    Raises ValueError if the file is not the expected ARFF table.
    """
    path = Path(path)
    if not path.exists():
        download_dataset(path)

    # ARFF layout: "%" comment lines, one "@ATTRIBUTE name type" line per column,
    # then "@DATA" followed by comma-separated rows. The last column is the class.
    names, rows, in_data = [], [], False
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            keyword = line.split()[0].upper()
            if keyword == "@ATTRIBUTE":
                names.append(line.split()[1])
            elif keyword == "@DATA":
                in_data = True
            elif in_data:
                rows.append(line.split(","))
    if not rows or len(names) < 2 or any(len(row) != len(names) for row in rows):
        raise ValueError(f"{path} is not a rectangular ARFF table")

    feature_names = names[:-1]
    features = np.array([row[:-1] for row in rows], dtype=float)
    class_names, labels = np.unique([row[-1] for row in rows], return_inverse=True)

    if drop_duplicates:
        # The file contains exact repeats of some beans. Left in, a repeated bean
        # could sit in both the training and the test split, so each distinct
        # (features, label) row is kept once, at its first position in the file.
        table = np.column_stack((features, labels))
        _, first_seen = np.unique(table, axis=0, return_index=True)
        keep = np.sort(first_seen)
        features, labels = features[keep], labels[keep]
    return features, labels.astype(int), feature_names, [str(c) for c in class_names]


def stratified_split(
    labels: ArrayLike, rng: np.random.Generator,
    fractions: tuple[float, float, float] = SPLIT_FRACTIONS,
) -> tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]]:
    """Return shuffled (train, validation, test) row indices.

    The split is stratified: every class is divided in the same proportions, so
    the small classes are represented in all three parts.

    Raises ValueError if labels is not 1-D or the fractions do not sum to 1.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {labels.shape}")
    if len(fractions) != 3 or min(fractions) <= 0 or abs(sum(fractions) - 1) > 1e-9:
        raise ValueError(f"fractions must be three positive numbers summing to 1, "
                         f"got {fractions}")

    parts = ([], [], [])
    for cls in np.unique(labels):
        members = rng.permutation(np.flatnonzero(labels == cls))
        first_cut = int(round(fractions[0] * len(members)))
        second_cut = int(round((fractions[0] + fractions[1]) * len(members)))
        for part, chunk in zip(parts, np.split(members, (first_cut, second_cut))):
            part.extend(chunk.tolist())
    # Shuffle each part so its rows are not grouped by class.
    return tuple(rng.permutation(np.array(part, dtype=int)) for part in parts)


def standardize(reference: ArrayLike, *others: ArrayLike) -> tuple[NDArray[np.float64], ...]:
    """Z-score every array with the column means and deviations of reference.

    The statistics come from reference alone (the training rows in Part 3), so
    nothing about the validation or test rows leaks into the scaling.

    Raises ValueError if a reference column is constant.
    """
    reference = np.asarray(reference, dtype=float)
    mean = reference.mean(axis=0)
    scale = reference.std(axis=0)
    if (scale == 0).any():
        raise ValueError("cannot standardize a constant feature column")
    return tuple((np.asarray(a, dtype=float) - mean) / scale for a in (reference, *others))
