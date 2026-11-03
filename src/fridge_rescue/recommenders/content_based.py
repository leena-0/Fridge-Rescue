"""Interpretable ingredient-based recipe retrieval using Jaccard similarity."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from fridge_rescue.config import AppConfig, load_config
from fridge_rescue.data.preprocessing import (
    load_ingredient_synonyms,
    normalize_ingredient_list,
    parse_ingredient_list,
)

RECOMMENDATION_COLUMNS = [
    "recipe_id",
    "title",
    "content_score",
    "ingredient_coverage",
    "matched_ingredients",
    "missing_ingredients",
    "missing_count",
    "minutes",
    "source",
]


@dataclass(frozen=True, slots=True)
class _RecipeCandidate:
    recipe_id: Any
    title: str
    ingredients: tuple[str, ...]
    ingredient_set: frozenset[str]
    minutes: float | None
    source: str


def _ingredient_values(value: object) -> list[str] | None:
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return parse_ingredient_list(value)


def _valid_minutes(value: object) -> float | None:
    try:
        minutes = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(minutes) or minutes <= 0:
        return None
    return minutes


def _display_minutes(value: float | None) -> int | float | None:
    if value is None:
        return None
    return int(value) if value.is_integer() else value


class ContentBasedRecommender:
    """Rank recipes by set-based ingredient similarity with deterministic explanations."""

    def __init__(
        self,
        recipes: pd.DataFrame,
        *,
        synonyms: dict[str, str] | None = None,
        max_top_k: int = 50,
    ) -> None:
        if max_top_k <= 0:
            raise ValueError("max_top_k must be a positive integer")
        required = {"recipe_id", "title", "normalized_ingredients"}
        missing = sorted(required - set(recipes.columns))
        if missing:
            raise ValueError(f"Recipe data is missing required columns: {', '.join(missing)}")
        if recipes["recipe_id"].duplicated().any():
            raise ValueError("recipe_id must be unique")

        self.synonyms = synonyms or {}
        self.max_top_k = max_top_k
        self._recipes: list[_RecipeCandidate] = []
        self._ingredient_index: dict[str, list[int]] = {}

        selected_columns = [
            column
            for column in [
                "recipe_id",
                "title",
                "normalized_ingredients",
                "minutes",
                "source",
                "dataset",
            ]
            if column in recipes.columns
        ]
        column_index = {column: index for index, column in enumerate(selected_columns)}
        for row in recipes.loc[:, selected_columns].itertuples(index=False, name=None):
            parsed = _ingredient_values(row[column_index["normalized_ingredients"]])
            if parsed is None:
                continue
            ingredients = tuple(dict.fromkeys(item for item in parsed if item))
            if not ingredients:
                continue
            source_value = row[column_index["source"]] if "source" in column_index else None
            if source_value is None or pd.isna(source_value) or not str(source_value).strip():
                source_value = (
                    row[column_index["dataset"]] if "dataset" in column_index else "unknown"
                )
            candidate = _RecipeCandidate(
                recipe_id=row[column_index["recipe_id"]],
                title=str(row[column_index["title"]]),
                ingredients=ingredients,
                ingredient_set=frozenset(ingredients),
                minutes=_valid_minutes(
                    row[column_index["minutes"]] if "minutes" in column_index else None
                ),
                source=str(source_value),
            )
            candidate_index = len(self._recipes)
            self._recipes.append(candidate)
            for ingredient in candidate.ingredient_set:
                self._ingredient_index.setdefault(ingredient, []).append(candidate_index)

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        *,
        synonyms: dict[str, str] | None = None,
        max_top_k: int = 50,
    ) -> ContentBasedRecommender:
        """Load processed recipes and build a recommender index."""

        parquet_path = Path(path)
        if not parquet_path.is_file():
            raise FileNotFoundError(f"Processed recipe file not found: {parquet_path}")
        available_columns = set(pq.read_schema(parquet_path).names)
        columns = [
            column
            for column in [
                "recipe_id",
                "title",
                "normalized_ingredients",
                "minutes",
                "source",
                "dataset",
            ]
            if column in available_columns
        ]
        return cls(
            pd.read_parquet(parquet_path, columns=columns),
            synonyms=synonyms,
            max_top_k=max_top_k,
        )

    @property
    def recipe_count(self) -> int:
        """Number of non-empty recipes retained in the search index."""

        return len(self._recipes)

    def normalize_query(self, ingredients: list[str] | tuple[str, ...] | str) -> list[str]:
        """Normalize typed or comma-separated query ingredients with pipeline rules."""

        if isinstance(ingredients, str):
            values = ingredients.split(",")
        else:
            values = list(ingredients)
        normalized = normalize_ingredient_list(values, self.synonyms)
        if not normalized:
            raise ValueError("At least one valid ingredient is required")
        return normalized

    def recommend(
        self,
        ingredients: list[str] | tuple[str, ...] | str,
        *,
        top_k: int = 10,
        max_minutes: float | None = None,
    ) -> pd.DataFrame:
        """Return Top-K recipes with Jaccard, coverage, and missing-ingredient explanations."""

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if top_k > self.max_top_k:
            raise ValueError(f"top_k must not exceed {self.max_top_k}")
        if max_minutes is not None:
            try:
                max_minutes = float(max_minutes)
            except (TypeError, ValueError) as exc:
                raise ValueError("max_minutes must be a positive number") from exc
            if not math.isfinite(max_minutes) or max_minutes <= 0:
                raise ValueError("max_minutes must be a positive number")

        normalized_query = self.normalize_query(ingredients)
        query_set = frozenset(normalized_query)
        candidate_indices: set[int] = set()
        for ingredient in query_set:
            candidate_indices.update(self._ingredient_index.get(ingredient, []))

        rows: list[dict[str, Any]] = []
        for candidate_index in candidate_indices:
            candidate = self._recipes[candidate_index]
            if max_minutes is not None and (
                candidate.minutes is None or candidate.minutes > max_minutes
            ):
                continue

            matched_set = query_set & candidate.ingredient_set
            if not matched_set:
                continue
            union_size = len(query_set | candidate.ingredient_set)
            matched = [item for item in candidate.ingredients if item in matched_set]
            missing = [item for item in candidate.ingredients if item not in query_set]
            rows.append(
                {
                    "recipe_id": candidate.recipe_id,
                    "title": candidate.title,
                    "content_score": len(matched_set) / union_size,
                    "ingredient_coverage": len(matched_set) / len(candidate.ingredient_set),
                    "matched_ingredients": matched,
                    "missing_ingredients": missing,
                    "missing_count": len(missing),
                    "minutes": _display_minutes(candidate.minutes),
                    "source": candidate.source,
                    "_title_sort": candidate.title.casefold(),
                    "_recipe_id_sort": str(candidate.recipe_id),
                }
            )

        if not rows:
            return pd.DataFrame(columns=RECOMMENDATION_COLUMNS)
        ranked = pd.DataFrame.from_records(rows)
        ranked = ranked.sort_values(
            [
                "content_score",
                "ingredient_coverage",
                "missing_count",
                "_title_sort",
                "_recipe_id_sort",
            ],
            ascending=[False, False, True, True, True],
            kind="stable",
        )
        return ranked.head(top_k).loc[:, RECOMMENDATION_COLUMNS].reset_index(drop=True)


def load_content_recommender(
    config: AppConfig,
    *,
    dataset: str = "foodcom",
) -> ContentBasedRecommender:
    """Build a configured Food.com or RecipeNLG content recommender."""

    if dataset == "foodcom":
        recipe_path = config.processed_data_dir / "foodcom" / "recipes.parquet"
    elif dataset == "recipenlg":
        recipe_path = config.processed_data_dir / "recipenlg" / "external_test_recipes.parquet"
    else:
        raise ValueError("dataset must be one of: foodcom, recipenlg")
    synonyms = load_ingredient_synonyms(config.ingredient_synonyms_path)
    return ContentBasedRecommender.from_parquet(
        recipe_path,
        synonyms=synonyms,
        max_top_k=config.max_top_k,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ingredients", nargs="+", help="Available ingredients")
    parser.add_argument("--dataset", choices=("foodcom", "recipenlg"), default="foodcom")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--config", type=Path, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = load_config(args.config)
    recommender = load_content_recommender(config, dataset=args.dataset)
    recommendations = recommender.recommend(
        args.ingredients,
        top_k=args.top_k,
        max_minutes=args.max_minutes,
    )
    if recommendations.empty:
        print("No matching recipes found.")
        return
    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
