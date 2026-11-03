from dataclasses import replace
from pathlib import Path

import pandas as pd

from fridge_rescue.config import load_config
from fridge_rescue.data.preprocessing import (
    normalize_ingredient,
    normalize_ingredient_list,
    parse_ingredient_list,
    preprocess_foodcom_recipes,
    preprocess_recipenlg_recipes,
    run_preprocessing,
)

SYNONYMS = {
    "eggs": "egg",
    "garlic cloves": "garlic",
    "scallions": "green onion",
}


def test_normalize_ingredient_removes_measurements_and_cleans_text() -> None:
    assert normalize_ingredient("2 Tbsp. Extra-Virgin Olive Oil, divided") == (
        "extra virgin olive oil"
    )
    assert normalize_ingredient("1 (8 oz.) pkg. Cream Cheese, cubed") == "cream cheese"
    assert normalize_ingredient("orange, juice of") == "orange juice"


def test_normalize_ingredient_uses_only_explicit_synonyms_and_deduplicates() -> None:
    normalized = normalize_ingredient_list(
        ["Eggs", "egg", "2 eggs", "Scallions", "scallions"],
        SYNONYMS,
    )

    assert normalized == ["egg", "green onion"]
    assert normalize_ingredient("butter or margarine", SYNONYMS) == "butter or margarine"


def test_parse_ingredient_list_distinguishes_invalid_and_empty() -> None:
    assert parse_ingredient_list("['egg', 'milk']") == ["egg", "milk"]
    assert parse_ingredient_list("[]") == []
    assert parse_ingredient_list("not a list") is None


def test_preprocess_foodcom_reports_exclusive_removal_reasons() -> None:
    recipes = pd.DataFrame(
        {
            "name": ["Omelet", "Broken", "Empty", "Duplicate"],
            "id": pd.Series([1, 2, 3, 1], dtype="Int64"),
            "minutes": [10, 20, 30, 40],
            "tags": ["[]"] * 4,
            "ingredients": ["['eggs', 'egg']", "not a list", "[]", "['milk']"],
            "n_ingredients": [2, 1, 0, 1],
        }
    )

    result = preprocess_foodcom_recipes(recipes, SYNONYMS)

    assert result.recipes["normalized_ingredients"].tolist() == [["egg"]]
    assert result.report["input_records"] == 4
    assert result.report["output_records"] == 1
    assert result.report["removal_reasons"]["invalid_ingredient_list"] == 1
    assert result.report["removal_reasons"]["empty_normalized_ingredients"] == 1
    assert result.report["removal_reasons"]["duplicate_recipe_id"] == 1
    assert result.report["duplicate_ingredient_entries_removed"] == 1


def test_preprocess_recipenlg_uses_ner_with_shared_rules() -> None:
    recipes = pd.DataFrame(
        {
            "title": ["Soup", "Invalid"],
            "ingredients": ['["1 c. milk"]', "[]"],
            "directions": ['["Heat"]', "[]"],
            "link": ["https://example.com/soup", "https://example.com/invalid"],
            "source": ["Gathered", "Gathered"],
            "NER": ['["Garlic Cloves", "garlic cloves", "milk"]', "invalid"],
        }
    )

    result = preprocess_recipenlg_recipes(recipes, SYNONYMS)

    assert result.recipes.loc[0, "recipe_id"] == "recipenlg:0"
    assert result.recipes.loc[0, "normalized_ingredients"] == ["garlic", "milk"]
    assert result.report["removal_reasons"]["invalid_ingredient_list"] == 1


def test_run_preprocessing_writes_parquet_and_audit_files(tmp_path: Path) -> None:
    foodcom_path = tmp_path / "RAW_recipes.csv"
    recipenlg_path = tmp_path / "full_dataset.csv"
    synonym_path = tmp_path / "synonyms.yaml"
    pd.DataFrame(
        {
            "name": ["Omelet"],
            "id": [1],
            "minutes": [10],
            "tags": ["['breakfast']"],
            "ingredients": ["['eggs', 'milk']"],
            "n_ingredients": [2],
        }
    ).to_csv(foodcom_path, index=False)
    pd.DataFrame(
        {
            "title": ["Egg soup"],
            "ingredients": ['["1 egg"]'],
            "directions": ['["Cook"]'],
            "link": ["https://example.com/egg-soup"],
            "source": ["Gathered"],
            "NER": ['["Eggs"]'],
        }
    ).to_csv(recipenlg_path, index=False)
    synonym_path.write_text("version: 1\nsynonyms:\n  eggs: egg\n", encoding="utf-8")

    config = replace(
        load_config(),
        processed_data_dir=tmp_path / "processed",
        output_dir=tmp_path / "outputs",
        foodcom_recipes_path=foodcom_path,
        recipenlg_recipes_path=recipenlg_path,
        ingredient_synonyms_path=synonym_path,
    )
    report = run_preprocessing(config, max_rows=10)

    assert report["datasets"]["foodcom"]["output_records"] == 1
    assert report["datasets"]["recipenlg"]["output_records"] == 1
    assert (config.processed_data_dir / "foodcom" / "recipes.parquet").is_file()
    assert (config.processed_data_dir / "recipenlg" / "recipes.parquet").is_file()
    assert (config.output_dir / "tables" / "preprocessing_report.json").is_file()
    audit_path = config.output_dir / "tables" / "ingredient_normalization_examples.csv"
    assert audit_path.is_file()
    assert set(pd.read_csv(audit_path)["dataset"]) == {"foodcom", "recipenlg"}
