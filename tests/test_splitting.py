from dataclasses import replace
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from fridge_rescue.config import load_config
from fridge_rescue.data.splitting import (
    build_external_recipenlg_catalog,
    canonicalize_source_url,
    clean_explicit_interactions,
    prepare_interaction_splits,
    run_data_split,
)


def _interaction_rows(user_id: int, count: int, recipe_offset: int = 0) -> list[dict[str, object]]:
    return [
        {
            "user_id": user_id,
            "recipe_id": recipe_offset + index,
            "date": f"2026-01-{index + 1:02d}",
            "rating": 5,
            "review": "good",
        }
        for index in range(count)
    ]


def test_clean_explicit_interactions_keeps_latest_valid_duplicate() -> None:
    interactions = pd.DataFrame(
        [
            {"user_id": 1, "recipe_id": 10, "date": "2026-01-01", "rating": 4},
            {"user_id": 1, "recipe_id": 10, "date": "2026-02-01", "rating": 5},
            {"user_id": 1, "recipe_id": 11, "date": "2026-01-01", "rating": 0},
            {"user_id": 1, "recipe_id": 12, "date": "2026-01-01", "rating": 6},
            {"user_id": 1, "recipe_id": 13, "date": "2026-01-01", "rating": None},
            {"user_id": None, "recipe_id": 14, "date": "2026-01-01", "rating": 5},
        ]
    )

    cleaned, report = clean_explicit_interactions(interactions)

    assert cleaned[["user_id", "recipe_id", "rating"]].astype(int).values.tolist() == [[1, 10, 5]]
    assert report["duplicate_user_recipe_records_removed"] == 1
    assert report["invalid_removal_reasons"] == {
        "missing_user_or_recipe_id": 1,
        "missing_rating": 1,
        "zero_rating": 1,
        "invalid_rating": 1,
    }


def test_prepare_interaction_splits_is_deterministic_and_disjoint() -> None:
    interactions = pd.DataFrame(
        _interaction_rows(1, 10) + _interaction_rows(2, 2, 100) + _interaction_rows(3, 6, 200)
    )

    first = prepare_interaction_splits(
        interactions,
        random_seed=42,
        min_user_interactions=5,
        min_recipe_interactions=1,
    )
    second = prepare_interaction_splits(
        interactions.sample(frac=1, random_state=7),
        random_seed=42,
        min_user_interactions=5,
        min_recipe_interactions=1,
    )

    assert_frame_equal(first.train, second.train)
    assert_frame_equal(first.validation, second.validation)
    assert_frame_equal(first.test, second.test)
    assert len(first.train) == 14
    assert len(first.validation) == 2
    assert len(first.test) == 2
    assert first.report["leakage_checks"]["user_recipe_pairs_disjoint"] is True
    user_eligibility = first.users.set_index("user_id")
    assert bool(user_eligibility.loc[1, "eligible"]) is True
    assert bool(user_eligibility.loc[2, "eligible"]) is False
    assert bool(user_eligibility.loc[3, "eligible"]) is False


def test_build_external_catalog_removes_id_content_and_internal_duplicates() -> None:
    foodcom = pd.DataFrame(
        {
            "recipe_id": [100, 200],
            "title": ["Tomato Soup", "Egg Dish"],
            "normalized_ingredients": [["tomato", "salt"], ["egg", "milk"]],
        }
    )
    recipenlg = pd.DataFrame(
        {
            "recipe_id": [
                "recipenlg:0",
                "recipenlg:1",
                "recipenlg:2",
                "recipenlg:3",
                "recipenlg:4",
            ],
            "title": ["Other", "Egg Dish", "Rice Bowl", "Different", "Rice Bowl"],
            "normalized_ingredients": [
                ["flour"],
                ["milk", "egg"],
                ["rice", "egg"],
                ["beans"],
                ["egg", "rice"],
            ],
            "link": [
                "https://www.food.com/recipe/tomato-soup-100?servings=4&utm_source=test",
                "https://example.com/egg",
                "https://example.com/rice",
                "https://example.com/rice#duplicate",
                "https://example.com/other-rice",
            ],
        }
    )

    result = build_external_recipenlg_catalog(foodcom, recipenlg)

    assert result.recipes["recipe_id"].tolist() == ["recipenlg:2"]
    assert result.report["removal_reasons"] == {
        "foodcom_recipe_id_overlap": 1,
        "foodcom_content_overlap": 1,
        "duplicate_recipenlg_source_url": 1,
        "duplicate_recipenlg_content": 1,
    }


def test_canonicalize_source_url_preserves_recipe_identity_query() -> None:
    first = canonicalize_source_url(
        "www.cookbooks.com/Recipe-Details.aspx?id=44874&utm_source=newsletter"
    )
    second = canonicalize_source_url("https://cookbooks.com/recipe-details.aspx?id=699419")

    assert first == "cookbooks.com/recipe-details.aspx?id=44874"
    assert second == "cookbooks.com/recipe-details.aspx?id=699419"
    assert first != second


def test_run_data_split_writes_all_outputs(tmp_path: Path) -> None:
    interactions_path = tmp_path / "RAW_interactions.csv"
    pd.DataFrame(_interaction_rows(1, 10)).to_csv(interactions_path, index=False)

    processed_dir = tmp_path / "processed"
    foodcom_dir = processed_dir / "foodcom"
    recipenlg_dir = processed_dir / "recipenlg"
    foodcom_dir.mkdir(parents=True)
    recipenlg_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "recipe_id": list(range(10)),
            "title": [f"Recipe {index}" for index in range(10)],
            "normalized_ingredients": [["ingredient", str(index)] for index in range(10)],
        }
    ).to_parquet(foodcom_dir / "recipes.parquet", index=False)
    pd.DataFrame(
        {
            "recipe_id": ["recipenlg:0"],
            "title": ["External recipe"],
            "normalized_ingredients": [["external"]],
            "link": ["https://example.com/external"],
        }
    ).to_parquet(recipenlg_dir / "recipes.parquet", index=False)

    config = replace(
        load_config(),
        processed_data_dir=processed_dir,
        output_dir=tmp_path / "outputs",
        foodcom_interactions_path=interactions_path,
        min_train_user_interactions=5,
        min_train_recipe_interactions=1,
    )
    report = run_data_split(config)

    assert report["foodcom_catalog_alignment"]["interactions_removed_without_processed_recipe"] == 0
    assert report["foodcom_interactions"]["split"]["train_interactions"] == 8
    assert (foodcom_dir / "splits" / "train.parquet").is_file()
    assert (foodcom_dir / "splits" / "validation.parquet").is_file()
    assert (foodcom_dir / "splits" / "test.parquet").is_file()
    assert (recipenlg_dir / "external_test_recipes.parquet").is_file()
    assert (config.output_dir / "tables" / "data_split_report.json").is_file()
