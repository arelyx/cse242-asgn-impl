# CSE 242 Assignment 4: Clustering and Learning with Noisy Data

Written answers, figures and tables are in `report.pdf`. This file explains how to
run the code and where the data comes from.

## Contents

| Path | What it is |
| --- | --- |
| `report.pdf` | The report: Part 1 derivations, Part 2 and Part 3 write-ups. |
| `report/` | LaTeX source of the report (`report.tex`, `acknowledgment.tex`). |
| `dataset.py` | Loads the Dry Bean data, removes duplicate rows, splits and standardizes. |
| `kmeans_clustering.py` | Part 2: K-means from scratch and the K-selection experiment. |
| `noisy_classification.py` | Part 3: label noise, k-NN label repair, SVM comparison. |
| `reporting.py` | Shared helpers for saving CSV tables and heatmaps. |
| `data/Dry_Bean_Dataset.arff` | The dataset, unmodified from the UCI archive. |
| `results/` | Every table (`.csv`) and figure (`.png`) the report uses. |
| `logs/` | Console output of the three runs below. |

## Setup

Python 3.9 or newer. From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Running

```bash
python kmeans_clustering.py --seed 0
python noisy_classification.py --seed 0 --trials 10
python noisy_classification.py --seed 0 --trials 10 --selection noisy
```

- The first command writes `results/kmeans/` (under a minute).
- The second writes `results/noisy_labels/` (a few minutes). It is the main Part 3
  experiment: the repair parameters are tuned on the clean validation split.
- The third writes `results/noisy_labels_noisy_selection/` (a few minutes). It is
  the variant discussed in Section 3.5, where the validation labels used for tuning
  are corrupted as well.

All three scripts accept `--help`. With the same seed they reproduce the numbers in
the report. The test labels are never corrupted and are used only for the final
evaluation.

To rebuild the report after rerunning the experiments:

```bash
cd report && pdflatex report.tex && pdflatex report.tex && cp report.pdf ..
```

## Dataset

Dry Bean Dataset: 13,611 images of dry bean grains from seven registered varieties,
each described by 16 shape measurements extracted from the image.

- Source: UCI Machine Learning Repository, dataset 602,
  <https://archive.ics.uci.edu/dataset/602/dry+bean+dataset>
- Licence: CC BY 4.0
- Citation: Koklu, M. and Ozkan, I.A. (2020). Multiclass classification of dry beans
  using computer vision and machine learning techniques. *Computers and Electronics
  in Agriculture*, 174, 105507. <https://doi.org/10.1016/j.compag.2020.105507>

The file in `data/` is the `Dry_Bean_Dataset.arff` from the UCI zip archive. If it is
missing, `dataset.py` downloads it again. The loader removes 68 exact duplicate rows,
which leaves 13,543 beans.

## Acknowledgment of outside help

See the last section of `report.pdf`.
