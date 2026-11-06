"""Protocol B: partial-ingredient retrieval evaluation."""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import pandas as pd

from fridge_rescue.config import load_config
from fridge_rescue.data.preprocessing import load_ingredient_synonyms
from fridge_rescue.recommenders.content_based import ContentBasedRecommender

SEEDS = (42, 43, 44)
RETENTION_RATES = (0.5, 0.7, 0.9)
TOP_K = 10


def ingredient_list(value):
    if isinstance(value, (list, tuple)):
        return list(dict.fromkeys(str(x) for x in value if str(x).strip()))
    if hasattr(value, "tolist"):
        return ingredient_list(value.tolist())
    if isinstance(value, str):
        import ast

        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return ingredient_list(parsed)
        except (ValueError, SyntaxError):
            pass
        return [x.strip() for x in value.split(",") if x.strip()]
    return []


def load_data(root: Path):
    food_path = root / "foodcom" / "recipes.parquet"
    train_path = root / "foodcom" / "splits" / "train.parquet"
    test_path = root / "foodcom" / "splits" / "test.parquet"
    external_path = root / "recipenlg" / "external_test_recipes.parquet"

    food = pd.read_parquet(food_path)
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    external = pd.read_parquet(external_path)

    return food, train, test, external


def build_distractor_pool(food: pd.DataFrame, train: pd.DataFrame) -> list[str]:
    """Build ingredient frequencies using Food.com training recipes only."""
    train_recipe_ids = set(train["recipe_id"].astype(str))
    train_recipes = food[food["recipe_id"].astype(str).isin(train_recipe_ids)]
    counts = Counter()
    for value in train_recipes["normalized_ingredients"]:
        counts.update(set(ingredient_list(value)))
    return [name for name, _ in counts.most_common()]


def make_query(target_ingredients, retention, seed, distractor_pool):
    rng = random.Random(seed)
    ingredients = list(dict.fromkeys(target_ingredients))

    if len(ingredients) < 2:
        return None

    keep_count = round(len(ingredients) * retention)
    keep_count = max(1, min(len(ingredients) - 1, keep_count))
    kept = rng.sample(ingredients, keep_count)

    distractors = [
        item for item in distractor_pool if item not in set(ingredients) and item not in set(kept)
    ]
    if len(distractors) < 2:
        return None

    kept.extend(rng.sample(distractors, min(rng.randint(2, 3), len(distractors))))
    return kept


def evaluate_dataset(
    name,
    targets,
    candidates,
    recommender,
    distractor_pool,
    targets_per_dataset,
):
    valid_targets = targets.copy()
    valid_targets["_ingredients"] = valid_targets["normalized_ingredients"].map(ingredient_list)
    valid_targets = valid_targets[valid_targets["_ingredients"].map(len) >= 2]

    target_count = min(targets_per_dataset, len(valid_targets))
    valid_targets = valid_targets.sample(n=target_count, random_state=42).reset_index(drop=True)

    rows = []

    for seed in SEEDS:
        for retention in RETENTION_RATES:
            hits = []
            reciprocal_ranks = []

            import zlib

            for _, target in valid_targets.iterrows():
                target_id = str(target["recipe_id"])
                stable_seed = seed + zlib.crc32(target_id.encode("utf-8"))
                query = make_query(
                    target["_ingredients"],
                    retention,
                    stable_seed,
                    distractor_pool,
                )
                if not query:
                    continue

                ranked = recommender.recommend(query, top_k=TOP_K)
                ranked_ids = ranked["recipe_id"].astype(str).tolist()

                if target_id in ranked_ids:
                    rank = ranked_ids.index(target_id) + 1
                    hits.append(1)
                    reciprocal_ranks.append(1.0 / rank)
                else:
                    hits.append(0)
                    reciprocal_ranks.append(0.0)

            rows.append(
                {
                    "dataset": name,
                    "seed": seed,
                    "ingredient_retention": retention,
                    "targets_attempted": len(valid_targets),
                    "queries_evaluated": len(hits),
                    "hit@10": sum(hits) / len(hits) if hits else 0.0,
                    "mrr@10": (
                        sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
                    ),
                    "candidate_catalog_size": recommender.recipe_count,
                    "full_catalog_size": len(candidates),
                }
            )

            print(
                f"{name} | seed={seed} | retention={retention:.0%} "
                f"| Hit@10={rows[-1]['hit@10']:.4f} "
                f"| MRR@10={rows[-1]['mrr@10']:.4f}"
            )

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets-per-dataset", type=int, default=100)
    parser.add_argument("--recipenlg-candidates", type=int, default=100000)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    root = config.processed_data_dir
    food, train, test, external = load_data(root)

    synonyms = load_ingredient_synonyms(config.ingredient_synonyms_path)
    distractor_pool = build_distractor_pool(food, train)

    print(f"Food.com recipe catalog: {len(food):,}")
    print(f"Food.com test targets: {len(test):,}")
    print(f"RecipeNLG full external catalog: {len(external):,}")

    food_candidates = food[["recipe_id", "title", "normalized_ingredients"]].drop_duplicates(
        "recipe_id"
    )

    food_recommender = ContentBasedRecommender(
        food_candidates,
        synonyms=synonyms,
        max_top_k=max(TOP_K, config.max_top_k),
    )

    # Use a deterministic candidate sample for memory-safe external evaluation.
    external_targets = external.sample(
        n=min(args.targets_per_dataset, len(external)),
        random_state=42,
    )
    external_candidates = external.sample(
        n=min(args.recipenlg_candidates, len(external)),
        random_state=42,
    )
    external_candidates = pd.concat(
        [external_candidates, external_targets], ignore_index=True
    ).drop_duplicates("recipe_id")

    external_candidates = external_candidates[["recipe_id", "title", "normalized_ingredients"]]

    external_recommender = ContentBasedRecommender(
        external_candidates,
        synonyms=synonyms,
        max_top_k=max(TOP_K, config.max_top_k),
    )

    # Attach recipe ingredients to test interactions, then evaluate unique recipes.
    food_test_targets = (
        test[["recipe_id"]]
        .drop_duplicates()
        .merge(
            food[["recipe_id", "normalized_ingredients"]],
            on="recipe_id",
            how="inner",
        )
    )

    results = []
    results.extend(
        evaluate_dataset(
            "Food.com Test",
            food_test_targets,
            food_candidates,
            food_recommender,
            distractor_pool,
            args.targets_per_dataset,
        )
    )
    results.extend(
        evaluate_dataset(
            "RecipeNLG External",
            external_targets,
            external_candidates,
            external_recommender,
            distractor_pool,
            args.targets_per_dataset,
        )
    )

    output_dir = Path("outputs/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    detail = pd.DataFrame(results)
    detail.to_csv(output_dir / "protocol_b_results.csv", index=False)

    summary = detail.groupby(["dataset", "ingredient_retention"], as_index=False).agg(
        hit_at_10=("hit@10", "mean"),
        mrr_at_10=("mrr@10", "mean"),
        mean_queries=("queries_evaluated", "mean"),
        candidate_catalog_size=("candidate_catalog_size", "first"),
        full_catalog_size=("full_catalog_size", "first"),
    )
    summary.to_csv(output_dir / "protocol_b_summary.csv", index=False)

    print("\nProtocol B summary:")
    print(summary.to_string(index=False))
    print("\nSaved outputs/tables/protocol_b_results.csv")
    print("Saved outputs/tables/protocol_b_summary.csv")


if __name__ == "__main__":
    main()
