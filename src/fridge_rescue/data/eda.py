"""Generate bounded, reproducible EDA summaries and figures for both datasets."""

from __future__ import annotations

import argparse
import ast
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fridge-rescue-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "fridge-rescue-cache"))

import matplotlib
import pandas as pd

from fridge_rescue.config import AppConfig, load_config
from fridge_rescue.data.loaders import FoodComData, load_foodcom, load_recipenlg

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _list_length(value: object) -> float:
    if isinstance(value, list):
        return float(len(value))
    if not isinstance(value, str) or not value.strip():
        return float("nan")
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return float("nan")
    return float(len(parsed)) if isinstance(parsed, list) else float("nan")


def _base_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows_loaded": int(len(frame)),
        "columns": list(frame.columns),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_values": {key: int(value) for key, value in frame.isna().sum().items()},
    }


def _save_histogram(
    values: pd.Series,
    *,
    title: str,
    xlabel: str,
    output_path: Path,
    bins: int,
) -> None:
    clean_values = pd.to_numeric(values, errors="coerce").dropna()
    fig, axis = plt.subplots(figsize=(8, 5))
    if clean_values.empty:
        axis.text(0.5, 0.5, "No valid values", ha="center", va="center")
    else:
        axis.hist(clean_values, bins=bins, color="#4f46a5", edgecolor="white")
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_bar_chart(
    values: pd.Series,
    *,
    title: str,
    xlabel: str,
    output_path: Path,
) -> None:
    counts = values.dropna().astype(str).value_counts().sort_index()
    fig, axis = plt.subplots(figsize=(8, 5))
    if counts.empty:
        axis.text(0.5, 0.5, "No valid values", ha="center", va="center")
    else:
        counts.plot.bar(ax=axis, color="#167c68")
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def analyze_foodcom(
    data: FoodComData,
    *,
    figure_dir: Path,
    histogram_bins: int,
) -> dict[str, Any]:
    """Summarize Food.com recipes/interactions and save core EDA plots."""

    recipes = data.recipes.copy()
    interactions = data.interactions.copy()
    if "n_ingredients" not in recipes or recipes["n_ingredients"].isna().all():
        recipes["n_ingredients"] = recipes["ingredients"].map(_list_length)

    cooking_minutes = pd.to_numeric(recipes["minutes"], errors="coerce").dropna()
    if cooking_minutes.empty:
        cooking_time_plot = cooking_minutes
        cooking_time_statistics: dict[str, int | float | None] = {
            "valid_count": 0,
            "min": None,
            "median": None,
            "p99": None,
            "max": None,
            "values_above_p99": 0,
        }
    else:
        p99_minutes = float(cooking_minutes.quantile(0.99))
        cooking_time_plot = cooking_minutes[cooking_minutes <= p99_minutes]
        cooking_time_statistics = {
            "valid_count": int(len(cooking_minutes)),
            "min": float(cooking_minutes.min()),
            "median": float(cooking_minutes.median()),
            "p99": p99_minutes,
            "max": float(cooking_minutes.max()),
            "values_above_p99": int((cooking_minutes > p99_minutes).sum()),
        }

    _save_histogram(
        cooking_time_plot,
        title="Food.com Cooking Time Distribution (<= 99th percentile)",
        xlabel="Minutes",
        output_path=figure_dir / "foodcom_cooking_time.png",
        bins=histogram_bins,
    )
    _save_histogram(
        recipes["n_ingredients"],
        title="Food.com Ingredient Count Distribution",
        xlabel="Number of ingredients",
        output_path=figure_dir / "foodcom_ingredient_count.png",
        bins=histogram_bins,
    )
    _save_bar_chart(
        interactions["rating"],
        title="Food.com Rating Distribution",
        xlabel="Rating",
        output_path=figure_dir / "foodcom_rating_distribution.png",
    )

    return {
        "recipes": {
            **_base_summary(recipes),
            "unique_recipe_ids": int(recipes["id"].nunique(dropna=True)),
            "duplicate_recipe_ids": int(recipes["id"].duplicated().sum()),
            "cooking_time_minutes": cooking_time_statistics,
        },
        "interactions": {
            **_base_summary(interactions),
            "unique_users": int(interactions["user_id"].nunique(dropna=True)),
            "unique_recipe_ids": int(interactions["recipe_id"].nunique(dropna=True)),
            "duplicate_user_recipe_pairs": int(
                interactions.duplicated(subset=["user_id", "recipe_id"]).sum()
            ),
            "rating_counts": {
                str(key): int(value)
                for key, value in interactions["rating"].value_counts().sort_index().items()
            },
        },
    }


def analyze_recipenlg(
    recipes: pd.DataFrame,
    *,
    figure_dir: Path,
    histogram_bins: int,
) -> dict[str, Any]:
    """Summarize RecipeNLG and save ingredient/source distributions."""

    recipes = recipes.copy()
    recipes["ingredient_count"] = recipes["NER"].map(_list_length)

    _save_histogram(
        recipes["ingredient_count"],
        title="RecipeNLG Ingredient Count Distribution",
        xlabel="Number of normalized ingredients",
        output_path=figure_dir / "recipenlg_ingredient_count.png",
        bins=histogram_bins,
    )
    top_sources = recipes["source"].value_counts().head(10)
    _save_bar_chart(
        recipes.loc[recipes["source"].isin(top_sources.index), "source"],
        title="RecipeNLG Source Distribution",
        xlabel="Source",
        output_path=figure_dir / "recipenlg_source_distribution.png",
    )

    return {
        **_base_summary(recipes),
        "unique_titles": int(recipes["title"].nunique(dropna=True)),
        "duplicate_title_ingredient_pairs": int(
            recipes.duplicated(subset=["title", "ingredients"]).sum()
        ),
        "source_counts": {str(key): int(value) for key, value in top_sources.items()},
        "invalid_ingredient_lists": int(recipes["ingredient_count"].isna().sum()),
    }


def _missing_values_table(summary: dict[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for dataset, dataset_summary in summary["datasets"].items():
        tables = dataset_summary if dataset == "foodcom" else {"recipes": dataset_summary}
        for table, table_summary in tables.items():
            for column, missing_count in table_summary["missing_values"].items():
                records.append(
                    {
                        "dataset": dataset,
                        "table": table,
                        "column": column,
                        "missing_count": missing_count,
                        "rows_loaded": table_summary["rows_loaded"],
                    }
                )
    return pd.DataFrame.from_records(records)


def run_eda(
    config: AppConfig,
    *,
    dataset: str = "all",
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Run EDA for one or both datasets and persist machine-readable outputs."""

    if dataset not in {"all", "foodcom", "recipenlg"}:
        raise ValueError("dataset must be one of: all, foodcom, recipenlg")
    if max_rows is not None and max_rows <= 0:
        max_rows = None

    figure_dir = config.output_dir / "figures"
    table_dir = config.output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "random_seed": config.random_seed,
        "max_rows_per_table": max_rows,
        "datasets": {},
    }

    if dataset in {"all", "foodcom"}:
        foodcom = load_foodcom(
            config.foodcom_recipes_path,
            config.foodcom_interactions_path,
            max_rows=max_rows,
        )
        summary["datasets"]["foodcom"] = analyze_foodcom(
            foodcom,
            figure_dir=figure_dir,
            histogram_bins=config.eda_histogram_bins,
        )

    if dataset in {"all", "recipenlg"}:
        recipenlg = load_recipenlg(
            config.recipenlg_recipes_path,
            max_rows=max_rows,
            usecols={"title", "ingredients", "directions", "link", "source", "NER"},
        )
        summary["datasets"]["recipenlg"] = analyze_recipenlg(
            recipenlg,
            figure_dir=figure_dir,
            histogram_bins=config.eda_histogram_bins,
        )

    summary_path = table_dir / "eda_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _missing_values_table(summary).to_csv(table_dir / "eda_missing_values.csv", index=False)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("all", "foodcom", "recipenlg"),
        default="all",
        help="Dataset to analyze (default: all)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum rows loaded per table. Use 0 for a full pass.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config path")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = load_config(args.config)
    max_rows = config.eda_default_max_rows if args.max_rows is None else args.max_rows
    summary = run_eda(config, dataset=args.dataset, max_rows=max_rows)

    for dataset_name, dataset_summary in summary["datasets"].items():
        if dataset_name == "foodcom":
            recipe_rows = dataset_summary["recipes"]["rows_loaded"]
            interaction_rows = dataset_summary["interactions"]["rows_loaded"]
            print(f"Food.com: {recipe_rows:,} recipes and {interaction_rows:,} interactions loaded")
        else:
            print(f"RecipeNLG: {dataset_summary['rows_loaded']:,} recipes loaded")
    print(f"EDA outputs saved under: {config.output_dir}")


if __name__ == "__main__":
    main()
