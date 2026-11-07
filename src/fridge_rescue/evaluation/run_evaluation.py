"""Protocol A evaluation on a shared, full eligible candidate catalog.

- Candidates are all Food.com recipes meeting the train-only activity threshold.
- Validation tunes the hybrid content/CF weights.
- Test is used once for the final comparison.
- Recipes rated by a user in train (and validation for test) are excluded.
- Pantry proxies use only train recipes rated 4 or 5.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from fridge_rescue.config import load_config
from fridge_rescue.evaluation.metrics import evaluate_recommendations
from fridge_rescue.recommenders.item_knn_optimized import ItemKNNRecommender


def _as_ingredient_set(value: object) -> set[str]:
    if isinstance(value, np.ndarray):
        values = value.tolist()
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        return set()
    return {str(item).strip() for item in values if str(item).strip()}


def _build_pantry(user_id, train: pd.DataFrame, recipe_ingredients: dict) -> list[str]:
    liked_ids = train.loc[
        (train["user_id"] == user_id) & (train["rating"] >= 4), "recipe_id"
    ].tolist()
    counts: dict[str, int] = {}
    for recipe_id in liked_ids:
        for ingredient in recipe_ingredients.get(recipe_id, set()):
            counts[ingredient] = counts.get(ingredient, 0) + 1
    return [
        ingredient
        for ingredient, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:10]
    ]


def _content_scores(pantry: list[str], candidate_ids: list, recipe_ingredients: dict):
    query = set(pantry)
    scores = np.zeros(len(candidate_ids), dtype=np.float32)
    coverage = np.zeros(len(candidate_ids), dtype=np.float32)
    missing_count = np.zeros(len(candidate_ids), dtype=np.int32)
    for position, recipe_id in enumerate(candidate_ids):
        ingredients = recipe_ingredients.get(recipe_id, set())
        if not query or not ingredients:
            continue
        overlap = len(query & ingredients)
        scores[position] = overlap / len(query | ingredients)
        coverage[position] = overlap / len(ingredients)
        missing_count[position] = len(ingredients - query)
    return scores, coverage, missing_count


def _knn_scores(model: ItemKNNRecommender, user_id, candidate_ids: list) -> np.ndarray:
    """Score every catalog candidate; unsupported items receive the user's mean rating."""
    if user_id not in model.user_to_index:
        return np.full(len(candidate_ids), 3.0, dtype=np.float32)

    user_idx = model.user_to_index[user_id]
    rated_items = model.user_rated_items[user_idx]
    user_mean = float(model.user_means.loc[user_id])
    numerators: dict[int, float] = {}
    denominators: dict[int, float] = {}

    for rated_idx in rated_items:
        centered_rating = float(model.user_item_matrix[user_idx, rated_idx])
        for candidate_idx, similarity in model._neighbors_for_item(rated_idx):
            if candidate_idx in rated_items:
                continue
            numerators[candidate_idx] = (
                numerators.get(candidate_idx, 0.0) + similarity * centered_rating
            )
            denominators[candidate_idx] = denominators.get(candidate_idx, 0.0) + abs(similarity)

    scores = np.full(len(candidate_ids), user_mean, dtype=np.float32)
    for position, recipe_id in enumerate(candidate_ids):
        recipe_idx = model.recipe_to_index.get(recipe_id)
        if recipe_idx is None or recipe_idx not in numerators:
            continue
        denominator = denominators[recipe_idx]
        if denominator > 0:
            prediction = user_mean + numerators[recipe_idx] / denominator
            scores[position] = float(np.clip(prediction, 1.0, 5.0))
    return scores


def _rank(ids, scores, *, coverage=None, missing=None, titles=None):
    frame = pd.DataFrame(
        {
            "recipe_id": ids,
            "score": scores,
            "recipe_sort": [str(value) for value in ids],
        }
    )
    if coverage is not None:
        frame["coverage"] = coverage
        frame["missing"] = missing
        frame["title_sort"] = [str(titles.get(value, "")).casefold() for value in ids]
        frame = frame.sort_values(
            ["score", "coverage", "missing", "title_sort", "recipe_sort"],
            ascending=[False, False, True, True, True],
            kind="stable",
        )
    else:
        frame = frame.sort_values(["score", "recipe_sort"], ascending=[False, True], kind="stable")
    return frame["recipe_id"].tolist()


def _metrics(recommendations: dict, relevant: dict, k: int) -> dict:
    return evaluate_recommendations(recommendations, relevant, k=k)


def _relevant_by_user(split: pd.DataFrame, eligible_ids: set, seen: dict) -> dict:
    positives = split[split["rating"] >= 4]
    result = {}
    for user_id, group in positives.groupby("user_id"):
        blocked = seen.get(user_id, set())
        relevant = set(group["recipe_id"]) & eligible_ids
        relevant -= blocked
        if relevant:
            result[user_id] = relevant
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=100, help="0 evaluates all eligible users")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--neighbors", type=int, default=50)
    parser.add_argument("--weight-step", type=float, default=0.1)
    args = parser.parse_args()
    if args.k < 1 or args.users < 0 or args.neighbors < 1:
        parser.error("--k and --neighbors must be positive; --users must be >= 0")
    if not 0 < args.weight_step <= 1:
        parser.error("--weight-step must be in (0, 1]")

    config = load_config()
    base = config.processed_data_dir / "foodcom"
    train = pd.read_parquet(base / "splits" / "train.parquet")
    validation = pd.read_parquet(base / "splits" / "validation.parquet")
    test = pd.read_parquet(base / "splits" / "test.parquet")
    recipes = pd.read_parquet(base / "recipes.parquet")

    required = {"recipe_id", "title", "normalized_ingredients"}
    missing = required - set(recipes.columns)
    if missing:
        raise ValueError(f"Recipe catalog is missing columns: {sorted(missing)}")

    # Candidate eligibility is computed from TRAIN only, as specified in the proposal.
    recipe_activity = train.groupby("recipe_id").size()
    eligible_ids = set(
        recipe_activity[recipe_activity >= config.min_train_recipe_interactions].index
    )
    recipe_catalog = recipes[recipes["recipe_id"].isin(eligible_ids)].drop_duplicates("recipe_id")
    candidate_ids = sorted(recipe_catalog["recipe_id"].tolist(), key=lambda value: str(value))
    recipe_ingredients = {
        row.recipe_id: _as_ingredient_set(row.normalized_ingredients)
        for row in recipe_catalog[["recipe_id", "normalized_ingredients"]].itertuples(index=False)
    }
    titles = dict(
        zip(
            recipe_catalog["recipe_id"],
            recipe_catalog["title"].astype(str),
            strict=True,
        )
    )

    train_seen = train.groupby("user_id")["recipe_id"].apply(set).to_dict()
    validation_seen = validation.groupby("user_id")["recipe_id"].apply(set).to_dict()
    validation_relevant = _relevant_by_user(validation, eligible_ids, train_seen)
    test_seen = {
        user_id: train_seen.get(user_id, set()) | validation_seen.get(user_id, set())
        for user_id in set(train_seen) | set(validation_seen)
    }
    test_relevant = _relevant_by_user(test, eligible_ids, test_seen)
    train_activity = train.groupby("user_id").size()

    # Tune and test on the same users who have relevant labels in both splits.
    cohort = [
        user_id
        for user_id in test_relevant
        if user_id in validation_relevant
        and train_activity.get(user_id, 0) >= config.min_train_user_interactions
    ]
    cohort = sorted(cohort, key=lambda value: str(value))
    if args.users and len(cohort) > args.users:
        cohort = pd.Series(cohort).sample(n=args.users, random_state=config.random_seed).tolist()

    if not candidate_ids:
        raise ValueError("No candidate recipes meet the train-only activity threshold")
    if not cohort:
        raise ValueError("No users have eligible positive targets in both validation and test")

    print(f"Eligible candidate catalog: {len(candidate_ids):,} recipes")
    print(f"Evaluation cohort: {len(cohort)} users")
    print("Scoring the full eligible catalog; no top-200 pre-truncation.")

    knn = ItemKNNRecommender(train, recipes, n_neighbors=args.neighbors)
    cached = {}
    validation_recs = {"Content-Based": {}, "Item-KNN": {}}
    test_recs = {"Content-Based": {}, "Item-KNN": {}}

    for number, user_id in enumerate(cohort, start=1):
        pantry = _build_pantry(user_id, train, recipe_ingredients)
        content_score, coverage, missing = _content_scores(
            pantry, candidate_ids, recipe_ingredients
        )
        cf_score = _knn_scores(knn, user_id, candidate_ids)
        cached[user_id] = (content_score, cf_score)

        val_blocked = train_seen.get(user_id, set())
        test_blocked = test_seen.get(user_id, set())
        val_positions = [i for i, rid in enumerate(candidate_ids) if rid not in val_blocked]
        test_positions = [i for i, rid in enumerate(candidate_ids) if rid not in test_blocked]

        def ranked_for(
            positions,
            scores,
            coverage_values=None,
            missing_values=None,
            use_content_ties=False,
        ):
            ids = [candidate_ids[i] for i in positions]
            vals = scores[positions]
            if use_content_ties:
                return _rank(
                    ids,
                    vals,
                    coverage=coverage_values[positions],
                    missing=missing_values[positions],
                    titles=titles,
                )[: args.k]
            return _rank(ids, vals)[: args.k]

        validation_recs["Content-Based"][user_id] = ranked_for(
            val_positions, content_score, coverage, missing, True
        )
        validation_recs["Item-KNN"][user_id] = ranked_for(val_positions, cf_score)
        test_recs["Content-Based"][user_id] = ranked_for(
            test_positions, content_score, coverage, missing, True
        )
        test_recs["Item-KNN"][user_id] = ranked_for(test_positions, cf_score)

        if number % 20 == 0 or number == len(cohort):
            print(f"Prepared full-catalog scores for {number}/{len(cohort)} users")

    # Grid-search alpha/beta on VALIDATION only; gamma=0 because no category input is used.
    weights = np.round(np.arange(0.0, 1.0 + args.weight_step / 2, args.weight_step), 10)
    weight_rows = []
    best_alpha = 0.5
    best_score = -1.0
    for alpha in weights:
        beta = 1.0 - alpha
        trial = {}
        for user_id in cohort:
            content_score, cf_score = cached[user_id]
            combined = alpha * content_score + beta * ((cf_score - 1.0) / 4.0)
            blocked = train_seen.get(user_id, set())
            positions = [i for i, rid in enumerate(candidate_ids) if rid not in blocked]
            trial[user_id] = _rank([candidate_ids[i] for i in positions], combined[positions])[
                : args.k
            ]
        score = _metrics(trial, {u: validation_relevant[u] for u in cohort}, args.k)[
            f"ndcg@{args.k}"
        ]
        weight_rows.append(
            {
                "alpha_content": float(alpha),
                "beta_knn": float(beta),
                f"validation_ndcg@{args.k}": score,
            }
        )
        if score > best_score:
            best_score, best_alpha = score, float(alpha)

    best_beta = 1.0 - best_alpha
    for user_id in cohort:
        content_score, cf_score = cached[user_id]
        combined = best_alpha * content_score + best_beta * ((cf_score - 1.0) / 4.0)
        blocked = test_seen.get(user_id, set())
        positions = [i for i, rid in enumerate(candidate_ids) if rid not in blocked]
        test_recs.setdefault("Hybrid", {})[user_id] = _rank(
            [candidate_ids[i] for i in positions], combined[positions]
        )[: args.k]

    final_relevant = {user_id: test_relevant[user_id] for user_id in cohort}
    rows = []
    for model_name, recommendations in test_recs.items():
        metrics = _metrics(recommendations, final_relevant, args.k)
        hits = {
            user_id: len(set(recommendations.get(user_id, [])) & final_relevant[user_id])
            for user_id in cohort
        }
        rows.append(
            {
                "model": model_name,
                **metrics,
                "users_with_recommendations": sum(
                    bool(recommendations.get(user_id)) for user_id in cohort
                ),
                "users_with_hits": sum(value > 0 for value in hits.values()),
                "total_hits": sum(hits.values()),
            }
        )

    result = pd.DataFrame(rows)
    result["hybrid_alpha_content"] = best_alpha
    result["hybrid_beta_knn"] = best_beta
    result["validation_ndcg_at_selected_weights"] = best_score

    output = config.output_dir / "tables"
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "model_comparison.csv"
    weights_path = output / "hybrid_weight_search.csv"
    result.to_csv(result_path, index=False)
    pd.DataFrame(weight_rows).to_csv(weights_path, index=False)

    print("\nFinal Protocol A comparison (test set):")
    print(result.to_string(index=False))
    print(
        f"\nSelected validation weights: alpha={best_alpha:.2f}, beta={best_beta:.2f}, gamma=0.00"
    )
    print(f"Saved: {result_path}")
    print(f"Saved: {weights_path}")
    print(f"Candidate catalog: {len(candidate_ids):,} train-eligible recipes")
    print("Excluded train/validation ratings before test ranking.")
    print("Pantry proxy uses only ingredients from train recipes rated 4/5.")
    print("Note: this is a sampled cohort when --users is less than the eligible cohort size.")


if __name__ == "__main__":
    main()
