"""Validated loaders for the Food.com and RecipeNLG raw datasets."""

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

FOODCOM_RECIPE_COLUMNS = {
    "name",
    "id",
    "minutes",
    "tags",
    "ingredients",
    "n_ingredients",
}
FOODCOM_INTERACTION_COLUMNS = {"user_id", "recipe_id", "date", "rating"}
RECIPENLG_COLUMNS = {"title", "ingredients", "directions", "link", "source", "NER"}


class DataValidationError(ValueError):
    """Raised when a raw dataset does not match the expected schema."""


@dataclass(frozen=True)
class FoodComData:
    """Food.com recipes and interactions loaded with a shared row limit."""

    recipes: pd.DataFrame
    interactions: pd.DataFrame


def _read_csv(
    path: str | Path,
    required_columns: Collection[str],
    *,
    max_rows: int | None = None,
    usecols: Collection[str] | None = None,
) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {csv_path}. Check configs/default.yaml and README.md."
        )

    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be a positive integer or None")

    try:
        frame = pd.read_csv(csv_path, nrows=max_rows, usecols=usecols, low_memory=False)
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        raise DataValidationError(f"Failed to read CSV file {csv_path}: {exc}") from exc

    missing_columns = sorted(set(required_columns) - set(frame.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise DataValidationError(f"{csv_path} is missing required columns: {missing}")

    unnamed_columns = [column for column in frame.columns if column.startswith("Unnamed:")]
    if unnamed_columns:
        frame = frame.drop(columns=unnamed_columns)

    return frame


def load_foodcom(
    recipes_path: str | Path,
    interactions_path: str | Path,
    *,
    max_rows: int | None = None,
) -> FoodComData:
    """Load Food.com recipe and interaction CSV files with schema checks."""

    recipes = load_foodcom_recipes(recipes_path, max_rows=max_rows)
    interactions = load_foodcom_interactions(interactions_path, max_rows=max_rows)

    return FoodComData(recipes=recipes, interactions=interactions)


def load_foodcom_recipes(
    recipes_path: str | Path,
    *,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Load Food.com recipes with schema checks and numeric type coercion."""

    recipes = _read_csv(recipes_path, FOODCOM_RECIPE_COLUMNS, max_rows=max_rows)
    recipes["id"] = pd.to_numeric(recipes["id"], errors="coerce").astype("Int64")
    recipes["minutes"] = pd.to_numeric(recipes["minutes"], errors="coerce")
    recipes["n_ingredients"] = pd.to_numeric(recipes["n_ingredients"], errors="coerce")
    return recipes


def load_foodcom_interactions(
    interactions_path: str | Path,
    *,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Load Food.com interactions with schema checks and numeric type coercion."""

    interactions = _read_csv(
        interactions_path,
        FOODCOM_INTERACTION_COLUMNS,
        max_rows=max_rows,
    )
    interactions["user_id"] = pd.to_numeric(interactions["user_id"], errors="coerce").astype(
        "Int64"
    )
    interactions["recipe_id"] = pd.to_numeric(interactions["recipe_id"], errors="coerce").astype(
        "Int64"
    )
    interactions["rating"] = pd.to_numeric(interactions["rating"], errors="coerce")
    return interactions


def load_recipenlg(
    recipes_path: str | Path,
    *,
    max_rows: int | None = None,
    usecols: Collection[str] | None = None,
) -> pd.DataFrame:
    """Load RecipeNLG recipes, dropping CSV index columns such as ``Unnamed: 0``."""

    selected_columns = usecols or RECIPENLG_COLUMNS
    return _read_csv(
        recipes_path,
        RECIPENLG_COLUMNS.intersection(selected_columns),
        max_rows=max_rows,
        usecols=selected_columns,
    )
