from dataclasses import replace
from pathlib import Path

import pandas as pd

from fridge_rescue.config import load_config
from fridge_rescue.data.eda import run_eda


def test_run_eda_creates_summaries_and_figures(tmp_path: Path) -> None:
    foodcom_dir = tmp_path / "foodcom"
    recipenlg_dir = tmp_path / "recipenlg"
    foodcom_dir.mkdir()
    recipenlg_dir.mkdir()

    recipes_path = foodcom_dir / "RAW_recipes.csv"
    interactions_path = foodcom_dir / "RAW_interactions.csv"
    recipenlg_path = recipenlg_dir / "full_dataset.csv"

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
    pd.DataFrame(
        {
            "title": ["Rice bowl"],
            "ingredients": ['["1 cup rice"]'],
            "directions": ['["Cook rice"]'],
            "link": ["https://example.com/rice"],
            "source": ["Gathered"],
            "NER": ['["rice"]'],
        }
    ).to_csv(recipenlg_path, index=False)

    config = replace(
        load_config(),
        output_dir=tmp_path / "outputs",
        foodcom_recipes_path=recipes_path,
        foodcom_interactions_path=interactions_path,
        recipenlg_recipes_path=recipenlg_path,
    )

    summary = run_eda(config, max_rows=10)

    assert summary["datasets"]["foodcom"]["recipes"]["rows_loaded"] == 2
    assert summary["datasets"]["foodcom"]["recipes"]["cooking_time_minutes"]["max"] == 35
    assert summary["datasets"]["recipenlg"]["rows_loaded"] == 1
    assert (config.output_dir / "tables" / "eda_summary.json").is_file()
    assert (config.output_dir / "tables" / "eda_missing_values.csv").is_file()
    assert (config.output_dir / "figures" / "foodcom_rating_distribution.png").is_file()
    assert (config.output_dir / "figures" / "recipenlg_ingredient_count.png").is_file()
