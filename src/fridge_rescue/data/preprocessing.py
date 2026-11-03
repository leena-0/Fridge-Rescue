"""Shared ingredient normalization and recipe preprocessing for both datasets."""

from __future__ import annotations

import argparse
import ast
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from fridge_rescue.config import AppConfig, load_config
from fridge_rescue.data.loaders import load_foodcom_recipes, load_recipenlg

NORMALIZATION_VERSION = 1

_UNICODE_FRACTIONS = {
    "¼": " 1/4 ",
    "½": " 1/2 ",
    "¾": " 3/4 ",
    "⅓": " 1/3 ",
    "⅔": " 2/3 ",
    "⅛": " 1/8 ",
    "⅜": " 3/8 ",
    "⅝": " 5/8 ",
    "⅞": " 7/8 ",
}
_NUMBER = r"(?:(?:\d+\s+)?\d+/\d+|\d+(?:\.\d+)?)"
_LEADING_QUANTITY = re.compile(
    rf"^\s*{_NUMBER}(?:\s*(?:-|to)\s*{_NUMBER})?\s*",
    flags=re.IGNORECASE,
)
_MEASUREMENT_PARENTHESIS = re.compile(
    r"\([^)]*(?:\d|cups?|ounces?|oz|pounds?|lbs?|grams?|kilograms?|kg|ml|liters?)[^)]*\)",
    flags=re.IGNORECASE,
)
_LEADING_UNIT = re.compile(
    r"^\s*(?:"
    r"cups?|c|tablespoons?|tbsp|tbs|teaspoons?|tsp|"
    r"ounces?|oz|pounds?|lbs?|grams?|g|kilograms?|kg|"
    r"milliliters?|ml|liters?|l|pints?|quarts?|gallons?"
    r")\.?\b\s*(?:of\s+)?",
    flags=re.IGNORECASE,
)
_LEADING_CONTAINER = re.compile(
    r"^\s*(?:packages?|pkg|cans?|jars?|cartons?|boxes?|bags?|bunch(?:es)?)\.?\b\s*(?:of\s+)?",
    flags=re.IGNORECASE,
)
_REVERSED_PART = re.compile(
    r"^\s*(?P<ingredient>[^,]+),\s*(?P<part>juice|zest|rind)\s+of\s*$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class PreprocessingResult:
    """Processed recipe table and its record-level audit summary."""

    recipes: pd.DataFrame
    report: dict[str, Any]


def parse_ingredient_list(value: object) -> list[str] | None:
    """Parse a serialized list while distinguishing invalid input from an empty list."""

    if isinstance(value, (list, tuple)):
        parsed = list(value)
    elif not isinstance(value, str) or not value.strip():
        return None
    else:
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None
        if not isinstance(parsed, (list, tuple)):
            return None

    return [str(item).strip() for item in parsed if item is not None and str(item).strip()]


def _normalize_surface(value: object) -> str:
    if not isinstance(value, str):
        return ""

    text = unicodedata.normalize("NFKC", value).strip().lower()
    for fraction, replacement in _UNICODE_FRACTIONS.items():
        text = text.replace(fraction, replacement)

    reversed_part = _REVERSED_PART.match(text)
    if reversed_part:
        text = f"{reversed_part.group('ingredient')} {reversed_part.group('part')}"
    else:
        text = text.split(",", maxsplit=1)[0]

    text = _MEASUREMENT_PARENTHESIS.sub(" ", text)
    for _ in range(4):
        previous = text
        text = _LEADING_QUANTITY.sub("", text)
        text = _LEADING_UNIT.sub("", text)
        text = _LEADING_CONTAINER.sub("", text)
        if text == previous:
            break

    text = re.sub(r"^\s*of\s+", "", text)
    text = text.replace("&", " and ").replace("_", " ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def normalize_ingredient(value: object, synonyms: dict[str, str] | None = None) -> str:
    """Normalize one ingredient using conservative, explicit synonym mappings."""

    normalized = _normalize_surface(value)
    if not normalized:
        return ""
    return (synonyms or {}).get(normalized, normalized)


def normalize_ingredient_list(
    ingredients: list[str],
    synonyms: dict[str, str] | None = None,
) -> list[str]:
    """Normalize ingredients and remove duplicates while preserving first-seen order."""

    normalized: list[str] = []
    seen: set[str] = set()
    for ingredient in ingredients:
        item = normalize_ingredient(ingredient, synonyms)
        if item and item not in seen:
            seen.add(item)
            normalized.append(item)
    return normalized


def load_ingredient_synonyms(path: str | Path) -> dict[str, str]:
    """Load and validate the reviewable synonym mapping from YAML."""

    synonym_path = Path(path)
    if not synonym_path.is_file():
        raise FileNotFoundError(f"Ingredient synonym file not found: {synonym_path}")

    with synonym_path.open(encoding="utf-8") as synonym_file:
        raw = yaml.safe_load(synonym_file)
    if not isinstance(raw, dict) or not isinstance(raw.get("synonyms"), dict):
        raise ValueError("Ingredient synonym file must contain a 'synonyms' mapping")

    synonyms: dict[str, str] = {}
    for source, target in raw["synonyms"].items():
        normalized_source = _normalize_surface(source)
        normalized_target = _normalize_surface(target)
        if not normalized_source or not normalized_target:
            raise ValueError("Ingredient synonym keys and values must be non-empty strings")
        if normalized_source in synonyms and synonyms[normalized_source] != normalized_target:
            raise ValueError(f"Conflicting ingredient synonym: {normalized_source}")
        synonyms[normalized_source] = normalized_target
    return synonyms


def _assign_reason(reasons: pd.Series, mask: pd.Series, reason: str) -> None:
    reasons.loc[reasons.isna() & mask.fillna(False)] = reason


def _build_report(
    dataset: str,
    working: pd.DataFrame,
    retained: pd.DataFrame,
    reasons: pd.Series,
) -> dict[str, Any]:
    reason_names = [
        "missing_recipe_id",
        "missing_title",
        "invalid_ingredient_list",
        "empty_normalized_ingredients",
        "duplicate_recipe_id",
    ]
    reason_counts = reasons.value_counts()
    retained_original_count = int(retained["ingredients_original"].map(len).sum())
    retained_normalized_count = int(retained["normalized_ingredients"].map(len).sum())
    changed_count = int(
        sum(
            original != normalized
            for original, normalized in zip(
                retained["ingredients_original"],
                retained["normalized_ingredients"],
                strict=True,
            )
        )
    )
    return {
        "dataset": dataset,
        "input_records": int(len(working)),
        "output_records": int(len(retained)),
        "removed_records": int(len(working) - len(retained)),
        "removal_reasons": {name: int(reason_counts.get(name, 0)) for name in reason_names},
        "retained_original_ingredient_entries": retained_original_count,
        "retained_normalized_ingredient_entries": retained_normalized_count,
        "duplicate_ingredient_entries_removed": retained_original_count - retained_normalized_count,
        "recipes_with_changed_ingredients": changed_count,
    }


def preprocess_foodcom_recipes(
    recipes: pd.DataFrame,
    synonyms: dict[str, str],
) -> PreprocessingResult:
    """Normalize Food.com recipes and report every removed record."""

    working = recipes.copy().reset_index(drop=True)
    parsed = working["ingredients"].map(parse_ingredient_list)
    normalized = parsed.map(
        lambda items: normalize_ingredient_list(items, synonyms) if items is not None else []
    )
    reasons = pd.Series(pd.NA, index=working.index, dtype="string")

    _assign_reason(reasons, working["id"].isna(), "missing_recipe_id")
    _assign_reason(
        reasons,
        working["name"].isna() | working["name"].astype("string").str.strip().eq(""),
        "missing_title",
    )
    _assign_reason(reasons, parsed.isna(), "invalid_ingredient_list")
    _assign_reason(reasons, normalized.map(len).eq(0), "empty_normalized_ingredients")
    _assign_reason(reasons, working["id"].duplicated(keep="first"), "duplicate_recipe_id")

    working["dataset"] = "foodcom"
    working["recipe_id"] = working["id"]
    working["title"] = working["name"]
    working["ingredients_original"] = parsed
    working["normalized_ingredients"] = normalized
    working["normalized_ingredient_count"] = normalized.map(len)
    retained = working.loc[reasons.isna()].reset_index(drop=True)
    return PreprocessingResult(
        recipes=retained,
        report=_build_report("foodcom", working, retained, reasons),
    )


def preprocess_recipenlg_recipes(
    recipes: pd.DataFrame,
    synonyms: dict[str, str],
) -> PreprocessingResult:
    """Normalize RecipeNLG NER ingredients with the same Food.com rules."""

    working = recipes.copy().reset_index(drop=True)
    working["source_row_number"] = working.index.astype("int64")
    parsed = working["NER"].map(parse_ingredient_list)
    normalized = parsed.map(
        lambda items: normalize_ingredient_list(items, synonyms) if items is not None else []
    )
    reasons = pd.Series(pd.NA, index=working.index, dtype="string")

    _assign_reason(
        reasons,
        working["title"].isna() | working["title"].astype("string").str.strip().eq(""),
        "missing_title",
    )
    _assign_reason(reasons, parsed.isna(), "invalid_ingredient_list")
    _assign_reason(reasons, normalized.map(len).eq(0), "empty_normalized_ingredients")

    working["dataset"] = "recipenlg"
    working["recipe_id"] = working["source_row_number"].map(lambda value: f"recipenlg:{value}")
    working["ingredients_original"] = parsed
    working["normalized_ingredients"] = normalized
    working["normalized_ingredient_count"] = normalized.map(len)
    retained = working.loc[reasons.isna()].reset_index(drop=True)
    return PreprocessingResult(
        recipes=retained,
        report=_build_report("recipenlg", working, retained, reasons),
    )


def _audit_examples(results: list[PreprocessingResult], sample_size: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for result in results:
        dataset_records = 0
        for row in result.recipes.itertuples(index=False):
            if row.ingredients_original == row.normalized_ingredients:
                continue
            records.append(
                {
                    "dataset": row.dataset,
                    "recipe_id": row.recipe_id,
                    "title": row.title,
                    "ingredients_original": json.dumps(
                        row.ingredients_original, ensure_ascii=False
                    ),
                    "normalized_ingredients": json.dumps(
                        row.normalized_ingredients, ensure_ascii=False
                    ),
                }
            )
            dataset_records += 1
            if dataset_records >= sample_size:
                break
    return pd.DataFrame.from_records(
        records,
        columns=[
            "dataset",
            "recipe_id",
            "title",
            "ingredients_original",
            "normalized_ingredients",
        ],
    )


def run_preprocessing(
    config: AppConfig,
    *,
    dataset: str = "all",
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Preprocess selected raw recipe datasets and persist Parquet outputs and audits."""

    if dataset not in {"all", "foodcom", "recipenlg"}:
        raise ValueError("dataset must be one of: all, foodcom, recipenlg")
    if max_rows is not None and max_rows <= 0:
        max_rows = None

    synonyms = load_ingredient_synonyms(config.ingredient_synonyms_path)
    table_dir = config.output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    results: list[PreprocessingResult] = []

    if dataset in {"all", "foodcom"}:
        foodcom = load_foodcom_recipes(config.foodcom_recipes_path, max_rows=max_rows)
        result = preprocess_foodcom_recipes(foodcom, synonyms)
        output_dir = config.processed_data_dir / "foodcom"
        output_dir.mkdir(parents=True, exist_ok=True)
        result.recipes.to_parquet(output_dir / "recipes.parquet", index=False)
        results.append(result)

    if dataset in {"all", "recipenlg"}:
        recipenlg = load_recipenlg(config.recipenlg_recipes_path, max_rows=max_rows)
        result = preprocess_recipenlg_recipes(recipenlg, synonyms)
        output_dir = config.processed_data_dir / "recipenlg"
        output_dir.mkdir(parents=True, exist_ok=True)
        result.recipes.to_parquet(output_dir / "recipes.parquet", index=False)
        results.append(result)

    report = {
        "normalization_version": NORMALIZATION_VERSION,
        "random_seed": config.random_seed,
        "max_rows_per_dataset": max_rows,
        "synonym_count": len(synonyms),
        "audit_sample_size_per_dataset": config.preprocessing_audit_sample_size,
        "datasets": {result.report["dataset"]: result.report for result in results},
    }
    (table_dir / "preprocessing_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _audit_examples(results, config.preprocessing_audit_sample_size).to_csv(
        table_dir / "ingredient_normalization_examples.csv",
        index=False,
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("all", "foodcom", "recipenlg"),
        default="all",
        help="Dataset to preprocess (default: all)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum rows loaded per dataset. Use 0 for a full pass.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config path")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = load_config(args.config)
    max_rows = config.preprocessing_default_max_rows if args.max_rows is None else args.max_rows
    report = run_preprocessing(config, dataset=args.dataset, max_rows=max_rows)

    for dataset_name, dataset_report in report["datasets"].items():
        print(
            f"{dataset_name}: {dataset_report['input_records']:,} input -> "
            f"{dataset_report['output_records']:,} retained"
        )
    print(f"Processed data saved under: {config.processed_data_dir}")
    print(f"Preprocessing audit saved under: {config.output_dir / 'tables'}")


if __name__ == "__main__":
    main()
