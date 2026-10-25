#!/usr/bin/env python3
"""Training script for the bargaining table classifier.

The model follows the specification provided by the user:
* Random Forest with two hyper-parameters (number of trees and max depth).
* Five-fold cross validation, each fold containing 76 observations (380 total).
* Hyper-parameter search over depths within [1, 50] and trees within [1, 300].
* Classification threshold fixed to 50% (probability >= 0.5 -> class 1).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold


CLASSIFICATION_THRESHOLD = 0.5
DEFAULT_DATA_PATH = Path("./../data/Dor-humans/bargaining_games_player_blue.csv")
DEFAULT_MODEL_PATH = Path("./../logs/bargaining_table_rf.joblib")
DEFAULT_SAMPLE_SIZE = 380
DEFAULT_RANDOM_STATE = 42
DEFAULT_SEARCH_ITERATIONS = 200
MIN_DEPTH = 1
MIN_TREES = 1
MAX_DEPTH = 50
MAX_TREES = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Random Forest classifier for bargaining outcomes."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to the CSV dataset (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Where to store the best trained model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=(
            "Number of random observations to consider for cross validation. "
            "Must be divisible by 5 to keep folds at 76 samples."
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for sampling and model reproducibility.",
    )
    parser.add_argument(
        "--search-iterations",
        type=int,
        default=DEFAULT_SEARCH_ITERATIONS,
        help="How many RandomizedSearchCV iterations to run over the hyper-parameter grid.",
    )
    parser.add_argument(
        "--min-depth",
        type=int,
        default=MIN_DEPTH,
        help="Minimum max_depth value to explore.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=MAX_DEPTH,
        help="Maximum max_depth value to explore.",
    )
    parser.add_argument(
        "--min-trees",
        type=int,
        default=MIN_TREES,
        help="Minimum n_estimators value to explore.",
    )
    parser.add_argument(
        "--max-trees",
        type=int,
        default=MAX_TREES,
        help="Maximum n_estimators value to explore.",
    )
    return parser.parse_args()


def probabilities_to_labels(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Convert prediction probabilities to labels with the configured threshold."""
    probs = np.asarray(probabilities)
    if probs.ndim == 1:
        scores = probs
    else:
        scores = probs[:, -1]  # last column always corresponds to the positive class in scikit-learn
    return (scores >= threshold).astype(int)


def load_dataset(
    path: Path, sample_size: int, random_state: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Load the dataset from disk and sample observations for CV."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    target_col = "y"
    drop_cols = ["game_number"]  # columns to exclude from features

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {path}")

    # Drop non-feature columns if present
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError("Sample size must be positive.")
        if sample_size % 5 != 0:
            raise ValueError(
                "Sample size must be divisible by 5 to ensure folds of equal size."
            )
        if sample_size < len(df):
            df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    features = df.drop(columns=[target_col]).to_numpy()
    target = df[target_col].to_numpy(dtype=int)

    return features, target


def build_search(
    random_state: int,
    search_iterations: int,
    depth_bounds: Tuple[int, int],
    tree_bounds: Tuple[int, int],
) -> RandomizedSearchCV:
    """Create the RandomizedSearchCV object with the requested configuration."""
    min_depth, max_depth = depth_bounds
    min_trees, max_trees = tree_bounds

    if min_depth < MIN_DEPTH or max_depth > MAX_DEPTH or min_depth > max_depth:
        raise ValueError(
            f"Depth bounds must be within [{MIN_DEPTH}, {MAX_DEPTH}] and min <= max."
        )
    if min_trees < MIN_TREES or max_trees > MAX_TREES or min_trees > max_trees:
        raise ValueError(
            f"Tree bounds must be within [{MIN_TREES}, {MAX_TREES}] and min <= max."
        )

    max_depth_values = np.arange(min_depth, max_depth + 1)
    tree_values = np.arange(min_trees, max_trees + 1)

    base_estimator = RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1,
        criterion="gini",
        class_weight="balanced",
        bootstrap=True,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions={
            "max_depth": max_depth_values,
            "n_estimators": tree_values,
        },
        n_iter=min(search_iterations, len(max_depth_values) * len(tree_values)),
        scoring="accuracy",  # accuracy relies on predict(), which uses a 50% threshold internally
        n_jobs=-1,
        cv=cv,
        verbose=1,
        refit=True,
        random_state=random_state,
    )

    return search


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Saved trained model to {path}")


def main() -> None:
    args = parse_args()

    X, y = load_dataset(args.dataset, args.sample_size, args.random_state)
    print(f"Loaded features shape: {X.shape}, target distribution: {np.bincount(y)}")

    search = build_search(
        random_state=args.random_state,
        search_iterations=args.search_iterations,
        depth_bounds=(args.min_depth, args.max_depth),
        tree_bounds=(args.min_trees, args.max_trees),
    )

    print("Starting cross-validated hyper-parameter search...")
    search.fit(X, y)
    best_model = search.best_estimator_

    print("\nBest hyper-parameters found:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"Best cross-validated accuracy (threshold=50%): {search.best_score_:.4f}")

    predicted_proba = best_model.predict_proba(X)
    predicted_labels = probabilities_to_labels(
        predicted_proba, threshold=CLASSIFICATION_THRESHOLD
    )
    train_accuracy = accuracy_score(y, predicted_labels)
    print(f"\nAccuracy on the sampled training set: {train_accuracy:.4f}")
    print("\nClassification report on the sampled training set:")
    print(classification_report(y, predicted_labels, digits=4))

    if args.output_model:
        save_model(best_model, args.output_model)


if __name__ == "__main__":
    main()
