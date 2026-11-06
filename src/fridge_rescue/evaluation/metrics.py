"""Ranking metrics and validation-based hybrid weight tuning."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from itertools import product

import numpy as np
import pandas as pd


def precision_at_k(
    recommended: Iterable,
    relevant: Iterable,
    k: int = 10,
) -> float:
    """Calculate Precision@K."""

    if k < 1:
        raise ValueError("k must be positive")

    recommended = list(recommended)[:k]
    relevant = set(relevant)

    if not recommended:
        return 0.0

    hits = sum(item in relevant for item in recommended)
    return hits / k


def recall_at_k(
    recommended: Iterable,
    relevant: Iterable,
    k: int = 10,
) -> float:
    """Calculate Recall@K."""

    if k < 1:
        raise ValueError("k must be positive")

    relevant = set(relevant)
    if not relevant:
        return 0.0

    hits = sum(item in relevant for item in list(recommended)[:k])
    return hits / len(relevant)


def ndcg_at_k(
    recommended: Iterable,
    relevant: Iterable,
    k: int = 10,
) -> float:
    """Calculate binary NDCG@K."""

    if k < 1:
        raise ValueError("k must be positive")

    relevant = set(relevant)
    ranked = list(recommended)[:k]

    if not relevant or not ranked:
        return 0.0

    dcg = sum(1.0 / np.log2(rank + 2) for rank, item in enumerate(ranked) if item in relevant)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))

    return float(dcg / idcg) if idcg else 0.0


def evaluate_recommendations(
    recommendations: dict,
    ground_truth: dict,
    k: int = 10,
) -> dict:
    """Calculate macro Precision@K, Recall@K and NDCG@K.

    Both arguments map user_id to a list/set of recipe_ids.
    Users without relevant test recipes are excluded.
    """

    users = [user_id for user_id, relevant in ground_truth.items() if len(relevant) > 0]

    if not users:
        return {
            f"precision@{k}": 0.0,
            f"recall@{k}": 0.0,
            f"ndcg@{k}": 0.0,
            "evaluated_users": 0,
        }

    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    for user_id in users:
        ranked = recommendations.get(user_id, [])
        relevant = ground_truth[user_id]

        precision_scores.append(precision_at_k(ranked, relevant, k))
        recall_scores.append(recall_at_k(ranked, relevant, k))
        ndcg_scores.append(ndcg_at_k(ranked, relevant, k))

    return {
        f"precision@{k}": float(np.mean(precision_scores)),
        f"recall@{k}": float(np.mean(recall_scores)),
        f"ndcg@{k}": float(np.mean(ndcg_scores)),
        "evaluated_users": len(users),
    }


def tune_hybrid_weights(
    validation_function: Callable[[float, float, float], dict],
    *,
    step: float = 0.1,
    metric: str = "ndcg@10",
) -> tuple[dict, pd.DataFrame]:
    """Find weights with the best validation metric.

    validation_function(alpha, beta, gamma) must return a metric
    dictionary, for example evaluate_recommendations() output.
    """

    if not 0 < step <= 1:
        raise ValueError("step must be in (0, 1]")

    values = np.round(np.arange(0.0, 1.0 + step / 2, step), 10)

    experiments = []

    for alpha, beta, gamma in product(values, repeat=3):
        if not np.isclose(alpha + beta + gamma, 1.0):
            continue

        scores = validation_function(float(alpha), float(beta), float(gamma))

        if metric not in scores:
            raise KeyError(f"Metric '{metric}' not found in validation result")

        experiments.append(
            {
                "alpha": float(alpha),
                "beta": float(beta),
                "gamma": float(gamma),
                **scores,
            }
        )

    results = pd.DataFrame(experiments)

    if results.empty:
        raise ValueError("No valid weight combinations were evaluated")

    results = results.sort_values(
        [metric, "alpha", "beta", "gamma"],
        ascending=[False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)

    best = results.iloc[0].to_dict()
    return best, results
