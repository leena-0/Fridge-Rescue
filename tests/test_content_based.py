import math

import pandas as pd
import pytest

from fridge_rescue.recommenders.content_based import (
    RECOMMENDATION_COLUMNS,
    ContentBasedRecommender,
)


def _recipes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "recipe_id": [1, 2, 3, 4],
            "title": ["Quick Omelet", "Fried Egg", "Unknown Time Egg", "Empty"],
            "normalized_ingredients": [
                ["egg", "milk", "salt"],
                ["egg", "oil"],
                ["egg"],
                [],
            ],
            "minutes": [10, 25, None, 5],
            "dataset": ["foodcom"] * 4,
        }
    )


def test_recommend_returns_jaccard_coverage_and_explanations() -> None:
    recommender = ContentBasedRecommender(_recipes(), synonyms={"eggs": "egg"})

    result = recommender.recommend(["Eggs", "milk"], top_k=3)

    assert result.columns.tolist() == RECOMMENDATION_COLUMNS
    assert result["recipe_id"].tolist() == [1, 3, 2]
    assert math.isclose(result.loc[0, "content_score"], 2 / 3)
    assert math.isclose(result.loc[0, "ingredient_coverage"], 2 / 3)
    assert result.loc[0, "matched_ingredients"] == ["egg", "milk"]
    assert result.loc[0, "missing_ingredients"] == ["salt"]
    assert result.loc[0, "missing_count"] == 1
    assert result.loc[0, "source"] == "foodcom"


def test_time_limit_is_a_hard_filter_before_ranking() -> None:
    recommender = ContentBasedRecommender(_recipes())

    result = recommender.recommend(["egg"], top_k=10, max_minutes=20)

    assert result["recipe_id"].tolist() == [1]
    assert result.loc[0, "minutes"] == 10


def test_tie_break_is_deterministic_independent_of_input_order() -> None:
    recipes = pd.DataFrame(
        {
            "recipe_id": [20, 10, 30],
            "title": ["Beta", "Alpha", "Alpha"],
            "normalized_ingredients": [["egg", "milk"]] * 3,
            "minutes": [10, 10, 10],
            "source": ["test"] * 3,
        }
    )

    first = ContentBasedRecommender(recipes).recommend(["egg"], top_k=3)
    second = ContentBasedRecommender(recipes.sample(frac=1, random_state=42)).recommend(
        ["egg"], top_k=3
    )

    assert first["recipe_id"].tolist() == [10, 30, 20]
    assert second["recipe_id"].tolist() == first["recipe_id"].tolist()


def test_empty_recipes_and_zero_overlap_are_not_returned() -> None:
    recommender = ContentBasedRecommender(_recipes())

    assert recommender.recipe_count == 3
    result = recommender.recommend(["rice"], top_k=10)

    assert result.empty
    assert result.columns.tolist() == RECOMMENDATION_COLUMNS


@pytest.mark.parametrize(
    ("ingredients", "top_k", "max_minutes", "message"),
    [
        (" , ", 10, None, "At least one valid ingredient"),
        (["egg"], 0, None, "top_k must be a positive integer"),
        (["egg"], 51, None, "top_k must not exceed 50"),
        (["egg"], 10, 0, "max_minutes must be a positive number"),
    ],
)
def test_invalid_queries_raise_clear_errors(
    ingredients: list[str] | str,
    top_k: int,
    max_minutes: float | None,
    message: str,
) -> None:
    recommender = ContentBasedRecommender(_recipes())

    with pytest.raises(ValueError, match=message):
        recommender.recommend(ingredients, top_k=top_k, max_minutes=max_minutes)
