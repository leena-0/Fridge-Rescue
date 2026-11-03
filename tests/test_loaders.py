from pathlib import Path

import pandas as pd
import pytest

from fridge_rescue.data.loaders import DataValidationError, load_foodcom, load_recipenlg


def _write_foodcom_files(directory: Path) -> tuple[Path, Path]:
    recipes_path = directory / "RAW_recipes.csv"
    interactions_path = directory / "RAW_interactions.csv"
    pd.DataFrame(
        {
            "name": ["Rice bowl", "Soup"],
            "id": [1, 2],
            "minutes": [20, 35],
            "tags": ["['quick']", "['soup']"],
            "ingredients": ["['rice', 'egg']", "['water', 'onion']"],
            "n_ingredients": [2, 2],
        }
    ).to_csv(recipes_path, index=False)
    pd.DataFrame(
        {
            "user_id": [10, 10],
            "recipe_id": [1, 2],
            "date": ["2026-01-01", "2026-01-02"],
            "rating": [5, 4],
        }
    ).to_csv(interactions_path, index=False)
    return recipes_path, interactions_path


def test_load_foodcom_validates_and_coerces_types(tmp_path: Path) -> None:
    recipes_path, interactions_path = _write_foodcom_files(tmp_path)

    data = load_foodcom(recipes_path, interactions_path)

    assert len(data.recipes) == 2
    assert len(data.interactions) == 2
    assert str(data.recipes["id"].dtype) == "Int64"
    assert data.interactions["rating"].tolist() == [5, 4]


def test_load_foodcom_reports_missing_columns(tmp_path: Path) -> None:
    recipes_path = tmp_path / "recipes.csv"
    interactions_path = tmp_path / "interactions.csv"
    pd.DataFrame({"id": [1]}).to_csv(recipes_path, index=False)
    pd.DataFrame({"user_id": [1], "recipe_id": [1], "date": ["2026-01-01"], "rating": [5]}).to_csv(
        interactions_path, index=False
    )

    with pytest.raises(DataValidationError, match="missing required columns"):
        load_foodcom(recipes_path, interactions_path)


def test_load_recipenlg_drops_exported_index(tmp_path: Path) -> None:
    recipes_path = tmp_path / "full_dataset.csv"
    pd.DataFrame(
        {
            "Unnamed: 0": [0],
            "title": ["Rice bowl"],
            "ingredients": ['["1 cup rice"]'],
            "directions": ['["Cook rice"]'],
            "link": ["https://example.com/rice"],
            "source": ["Gathered"],
            "NER": ['["rice"]'],
        }
    ).to_csv(recipes_path, index=False)

    recipes = load_recipenlg(recipes_path)

    assert "Unnamed: 0" not in recipes.columns
    assert recipes.loc[0, "title"] == "Rice bowl"
