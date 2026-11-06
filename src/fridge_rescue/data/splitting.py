"""Create leakage-safe Food.com splits and a deduplicated RecipeNLG external catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit

import numpy as np
import pandas as pd

from fridge_rescue.config import AppConfig, load_config
from fridge_rescue.data.loaders import load_foodcom_interactions


@dataclass(frozen=True)
class InteractionSplits:
    """Leakage-safe interaction splits and train-only activity tables."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    users: pd.DataFrame
    recipes: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class ExternalCatalog:
    """RecipeNLG external-test recipes after Food.com overlap removal."""

    recipes: pd.DataFrame
    removals: pd.DataFrame
    report: dict[str, Any]


def clean_explicit_interactions(
    interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep ratings 1-5 and retain the latest duplicate user-recipe record."""

    working = interactions.copy().reset_index(drop=True)
    working["_input_order"] = np.arange(len(working), dtype=np.int64)
    working["user_id"] = pd.to_numeric(working["user_id"], errors="coerce").astype("Int64")
    working["recipe_id"] = pd.to_numeric(working["recipe_id"], errors="coerce").astype("Int64")
    working["rating"] = pd.to_numeric(working["rating"], errors="coerce")
    working["date"] = pd.to_datetime(working["date"], errors="coerce")

    reasons = pd.Series(pd.NA, index=working.index, dtype="string")
    missing_id = working["user_id"].isna() | working["recipe_id"].isna()
    reasons.loc[missing_id] = "missing_user_or_recipe_id"
    reasons.loc[reasons.isna() & working["rating"].isna()] = "missing_rating"
    reasons.loc[reasons.isna() & working["rating"].eq(0)] = "zero_rating"
    valid_ratings = working["rating"].isin([1, 2, 3, 4, 5])
    reasons.loc[reasons.isna() & ~valid_ratings] = "invalid_rating"

    valid = working.loc[reasons.isna()].copy()
    valid = valid.sort_values(
        ["user_id", "recipe_id", "date", "_input_order"],
        kind="stable",
        na_position="first",
    )
    duplicate_mask = valid.duplicated(subset=["user_id", "recipe_id"], keep="last")
    duplicate_count = int(duplicate_mask.sum())
    cleaned = valid.loc[~duplicate_mask].copy()
    cleaned = cleaned.sort_values(["user_id", "recipe_id"], kind="stable").reset_index(drop=True)

    reason_counts = reasons.value_counts()
    report = {
        "input_interactions": int(len(working)),
        "removed_invalid_interactions": int(reasons.notna().sum()),
        "invalid_removal_reasons": {
            name: int(reason_counts.get(name, 0))
            for name in [
                "missing_user_or_recipe_id",
                "missing_rating",
                "zero_rating",
                "invalid_rating",
            ]
        },
        "duplicate_user_recipe_records_removed": duplicate_count,
        "valid_deduplicated_interactions": int(len(cleaned)),
        "duplicate_rule": "keep the latest date; break date ties by last source-row order",
    }
    return cleaned, report


def _stable_user_seed(random_seed: int, user_id: object) -> int:
    payload = f"{random_seed}:{user_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def split_interactions_per_user(
    interactions: pd.DataFrame,
    *,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split each user 80/10/10; users with fewer than three records remain train-only."""

    assignments: list[pd.DataFrame] = []
    train_only_users = 0
    train_only_interactions = 0

    for user_id, group in interactions.groupby("user_id", sort=True):
        ordered = group.sort_values(["recipe_id", "date", "_input_order"], kind="stable").copy()
        size = len(ordered)
        labels = np.full(size, "train", dtype=object)
        if size < 3:
            train_only_users += 1
            train_only_interactions += size
        else:
            holdout_size = max(1, int(np.floor(size * 0.1 + 0.5)))
            while 2 * holdout_size >= size:
                holdout_size -= 1
            rng = np.random.default_rng(_stable_user_seed(random_seed, user_id))
            permutation = rng.permutation(size)
            labels[permutation[:holdout_size]] = "test"
            labels[permutation[holdout_size : 2 * holdout_size]] = "validation"
        ordered["split"] = labels
        assignments.append(ordered)

    if assignments:
        assigned = pd.concat(assignments, ignore_index=True)
    else:
        assigned = interactions.assign(split=pd.Series(dtype="string"))

    output_columns = [column for column in assigned.columns if column != "_input_order"]

    def select(split_name: str) -> pd.DataFrame:
        return (
            assigned.loc[assigned["split"].eq(split_name), output_columns]
            .sort_values(["user_id", "recipe_id"], kind="stable")
            .reset_index(drop=True)
        )

    train = select("train")
    validation = select("validation")
    test = select("test")
    report = {
        "strategy": "fixed-seed per-user 80/10/10 with validation/test retention",
        "random_seed": random_seed,
        "train_interactions": int(len(train)),
        "validation_interactions": int(len(validation)),
        "test_interactions": int(len(test)),
        "train_only_users_with_fewer_than_3_interactions": train_only_users,
        "train_only_interactions_for_sparse_users": train_only_interactions,
    }
    return train, validation, test, report


def _pair_set(frame: pd.DataFrame) -> set[tuple[int, int]]:
    return set(zip(frame["user_id"].astype(int), frame["recipe_id"].astype(int), strict=True))


def _assert_disjoint_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    train_pairs = _pair_set(train)
    validation_pairs = _pair_set(validation)
    test_pairs = _pair_set(test)
    if train_pairs & validation_pairs or train_pairs & test_pairs or validation_pairs & test_pairs:
        raise ValueError("A user-recipe pair appears in more than one split")


def _activity_tables(
    all_interactions: pd.DataFrame,
    train: pd.DataFrame,
    *,
    min_user_interactions: int,
    min_recipe_interactions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    user_counts = train["user_id"].value_counts()
    recipe_counts = train["recipe_id"].value_counts()

    users = pd.DataFrame(
        {"user_id": sorted(all_interactions["user_id"].dropna().astype(int).unique())}
    )
    users["train_interactions"] = users["user_id"].map(user_counts).fillna(0).astype("int64")
    users["eligible"] = users["train_interactions"].ge(min_user_interactions)

    recipes = pd.DataFrame(
        {"recipe_id": sorted(all_interactions["recipe_id"].dropna().astype(int).unique())}
    )
    recipes["train_interactions"] = (
        recipes["recipe_id"].map(recipe_counts).fillna(0).astype("int64")
    )
    recipes["eligible"] = recipes["train_interactions"].ge(min_recipe_interactions)
    return users, recipes


def prepare_interaction_splits(
    interactions: pd.DataFrame,
    *,
    random_seed: int,
    min_user_interactions: int,
    min_recipe_interactions: int,
) -> InteractionSplits:
    """Clean interactions, split per user, and compute train-only activity eligibility."""

    cleaned, cleaning_report = clean_explicit_interactions(interactions)
    train, validation, test, split_report = split_interactions_per_user(
        cleaned,
        random_seed=random_seed,
    )
    _assert_disjoint_splits(train, validation, test)
    users, recipes = _activity_tables(
        cleaned,
        train,
        min_user_interactions=min_user_interactions,
        min_recipe_interactions=min_recipe_interactions,
    )
    eligible_users = set(users.loc[users["eligible"], "user_id"])
    eligible_recipes = set(recipes.loc[recipes["eligible"], "recipe_id"])

    def eligible_interactions(frame: pd.DataFrame) -> int:
        return int(
            (
                frame["user_id"].isin(eligible_users) & frame["recipe_id"].isin(eligible_recipes)
            ).sum()
        )

    eligible_train_count = eligible_interactions(train)
    eligible_validation_count = eligible_interactions(validation)
    eligible_test_count = eligible_interactions(test)

    report = {
        "cleaning": cleaning_report,
        "split": split_report,
        "activity_thresholds": {
            "computed_from": "train_only",
            "min_train_user_interactions": min_user_interactions,
            "min_train_recipe_interactions": min_recipe_interactions,
            "total_users": int(len(users)),
            "eligible_users": int(users["eligible"].sum()),
            "excluded_users": int((~users["eligible"]).sum()),
            "total_recipes": int(len(recipes)),
            "eligible_recipes": int(recipes["eligible"].sum()),
            "excluded_recipes": int((~recipes["eligible"]).sum()),
            "eligible_train_interactions": eligible_train_count,
            "excluded_train_interactions": int(len(train) - eligible_train_count),
            "eligible_validation_interactions": eligible_validation_count,
            "excluded_validation_interactions": int(len(validation) - eligible_validation_count),
            "eligible_test_interactions": eligible_test_count,
            "excluded_test_interactions": int(len(test) - eligible_test_count),
        },
        "leakage_checks": {
            "user_recipe_pairs_disjoint": True,
            "activity_thresholds_use_train_only": True,
        },
    }
    return InteractionSplits(train, validation, test, users, recipes, report)


def canonicalize_source_url(value: object) -> str:
    """Canonicalize a source URL while retaining identity-bearing query parameters."""

    if not isinstance(value, str) or not value.strip():
        return ""
    raw = value.strip()
    if "://" not in raw:
        raw = f"https://{raw}"
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return ""
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = re.sub(r"/+", "/", unquote(parsed.path)).rstrip("/").lower()
    tracking_parameters = {
        "fbclid",
        "gclid",
        "ref",
        "referrer",
        "utm_campaign",
        "utm_content",
        "utm_medium",
        "utm_source",
        "utm_term",
    }
    query_items = sorted(
        (key.lower(), value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in tracking_parameters
    )
    query = urlencode(query_items, doseq=True)
    canonical = f"{host}{path}" if host else ""
    return f"{canonical}?{query}" if canonical and query else canonical


def _foodcom_id_from_url(canonical_url: str) -> int | None:
    if not canonical_url.startswith("food.com/"):
        return None
    url_path = canonical_url.split("?", maxsplit=1)[0]
    match = re.search(r"(?:-|/)(\d+)$", url_path)
    return int(match.group(1)) if match else None


def _canonical_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKC", value).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _content_fingerprint(title: object, ingredients: object) -> str:
    canonical_title = _canonical_text(title)
    if isinstance(ingredients, np.ndarray):
        ingredient_values = ingredients.tolist()
    elif isinstance(ingredients, (list, tuple)):
        ingredient_values = list(ingredients)
    else:
        return ""
    canonical_ingredients = sorted(
        {_canonical_text(item) for item in ingredient_values if _canonical_text(item)}
    )
    if not canonical_title or not canonical_ingredients:
        return ""
    payload = json.dumps(
        [canonical_title, canonical_ingredients],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def build_external_recipenlg_catalog(
    foodcom_recipes: pd.DataFrame,
    recipenlg_recipes: pd.DataFrame,
) -> ExternalCatalog:
    """Remove Food.com overlap and exact RecipeNLG duplicates for external testing."""

    required = {"recipe_id", "title", "normalized_ingredients"}
    for name, frame in {"foodcom": foodcom_recipes, "recipenlg": recipenlg_recipes}.items():
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} processed recipes are missing columns: {', '.join(missing)}")
    if "link" not in recipenlg_recipes.columns:
        raise ValueError("recipenlg processed recipes are missing column: link")

    foodcom = foodcom_recipes.copy()
    recipenlg = recipenlg_recipes.copy().reset_index(drop=True)
    foodcom_ids = set(pd.to_numeric(foodcom["recipe_id"], errors="coerce").dropna().astype(int))
    foodcom_fingerprints = {
        fingerprint
        for fingerprint in (
            _content_fingerprint(row.title, row.normalized_ingredients)
            for row in foodcom.itertuples(index=False)
        )
        if fingerprint
    }

    recipenlg["canonical_source_url"] = recipenlg["link"].map(canonicalize_source_url)
    recipenlg["foodcom_source_recipe_id"] = recipenlg["canonical_source_url"].map(
        _foodcom_id_from_url
    )
    recipenlg["content_fingerprint"] = [
        _content_fingerprint(row.title, row.normalized_ingredients)
        for row in recipenlg.itertuples(index=False)
    ]

    reasons = pd.Series(pd.NA, index=recipenlg.index, dtype="string")
    id_overlap = recipenlg["foodcom_source_recipe_id"].isin(foodcom_ids)
    reasons.loc[id_overlap] = "foodcom_recipe_id_overlap"
    content_overlap = recipenlg["content_fingerprint"].isin(foodcom_fingerprints) & recipenlg[
        "content_fingerprint"
    ].ne("")
    reasons.loc[reasons.isna() & content_overlap] = "foodcom_content_overlap"
    duplicate_url = recipenlg["canonical_source_url"].ne("") & recipenlg[
        "canonical_source_url"
    ].duplicated(keep="first")
    reasons.loc[reasons.isna() & duplicate_url] = "duplicate_recipenlg_source_url"
    duplicate_content = recipenlg["content_fingerprint"].ne("") & recipenlg[
        "content_fingerprint"
    ].duplicated(keep="first")
    reasons.loc[reasons.isna() & duplicate_content] = "duplicate_recipenlg_content"

    retained = recipenlg.loc[reasons.isna()].reset_index(drop=True)
    removals = recipenlg.loc[reasons.notna(), ["recipe_id", "title", "canonical_source_url"]].copy()
    removals["removal_reason"] = reasons.loc[reasons.notna()].to_numpy()
    reason_counts = reasons.value_counts()
    report = {
        "input_recipenlg_recipes": int(len(recipenlg)),
        "retained_external_test_recipes": int(len(retained)),
        "removed_recipes": int(reasons.notna().sum()),
        "removal_reasons": {
            name: int(reason_counts.get(name, 0))
            for name in [
                "foodcom_recipe_id_overlap",
                "foodcom_content_overlap",
                "duplicate_recipenlg_source_url",
                "duplicate_recipenlg_content",
            ]
        },
        "dataset_role": "content-based external retrieval test only; no user-rating evaluation",
        "ids_remain_dataset_local": True,
    }
    return ExternalCatalog(retained, removals, report)


def _read_processed_recipes(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Processed recipe file not found: {path}. Run the preprocessing pipeline first."
        )
    return pd.read_parquet(path)


def run_data_split(
    config: AppConfig,
    *,
    max_interactions: int | None = None,
) -> dict[str, Any]:
    """Run interaction splitting and RecipeNLG external-catalog construction."""

    if max_interactions is not None and max_interactions <= 0:
        max_interactions = None
    interactions = load_foodcom_interactions(
        config.foodcom_interactions_path,
        max_rows=max_interactions,
    )
    foodcom_recipes = _read_processed_recipes(
        config.processed_data_dir / "foodcom" / "recipes.parquet"
    )
    recipenlg_path = config.processed_data_dir / "recipenlg" / "recipes.parquet"
    recipenlg_recipes = (
        _read_processed_recipes(recipenlg_path)
        if recipenlg_path.is_file()
        else None
    )

    processed_foodcom_ids = set(
        pd.to_numeric(foodcom_recipes["recipe_id"], errors="coerce").dropna().astype(int)
    )
    interaction_recipe_ids = pd.to_numeric(interactions["recipe_id"], errors="coerce")
    catalog_match = interaction_recipe_ids.isin(processed_foodcom_ids)
    catalog_alignment = {
        "processed_foodcom_recipes": int(len(processed_foodcom_ids)),
        "input_interactions": int(len(interactions)),
        "interactions_with_processed_recipes": int(catalog_match.sum()),
        "interactions_removed_without_processed_recipe": int((~catalog_match).sum()),
    }
    interactions = interactions.loc[catalog_match].reset_index(drop=True)

    splits = prepare_interaction_splits(
        interactions,
        random_seed=config.random_seed,
        min_user_interactions=config.min_train_user_interactions,
        min_recipe_interactions=config.min_train_recipe_interactions,
    )
    external = (
        build_external_recipenlg_catalog(foodcom_recipes, recipenlg_recipes)
        if recipenlg_recipes is not None
        else None
    )

    split_dir = config.processed_data_dir / "foodcom" / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    splits.train.to_parquet(split_dir / "train.parquet", index=False)
    splits.validation.to_parquet(split_dir / "validation.parquet", index=False)
    splits.test.to_parquet(split_dir / "test.parquet", index=False)
    splits.users.to_parquet(split_dir / "user_activity.parquet", index=False)
    splits.recipes.to_parquet(split_dir / "recipe_activity.parquet", index=False)

    table_dir = config.output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    if external is not None:
        external_dir = config.processed_data_dir / "recipenlg"
        external_dir.mkdir(parents=True, exist_ok=True)
        external.recipes.to_parquet(
            external_dir / "external_test_recipes.parquet", index=False
        )
        external.removals.to_csv(
            table_dir / "recipenlg_overlap_removals.csv", index=False
        )
    report = {
        "max_interactions": max_interactions,
        "foodcom_catalog_alignment": catalog_alignment,
        "foodcom_interactions": splits.report,
        "recipenlg_external_catalog": (
            external.report
            if external is not None
            else {
                "status": "skipped",
                "reason": "RecipeNLG data not available",
            }
        ),
    }
    (table_dir / "data_split_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-interactions",
        type=int,
        default=None,
        help="Maximum Food.com interactions to load. Use 0 for all interactions (default).",
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config path")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = load_config(args.config)
    report = run_data_split(config, max_interactions=args.max_interactions)
    split = report["foodcom_interactions"]["split"]
    print(
        "Food.com split: "
        f"{split['train_interactions']:,} train / "
        f"{split['validation_interactions']:,} validation / "
        f"{split['test_interactions']:,} test"
    )
    external = report["recipenlg_external_catalog"]
    if external.get("status") == "skipped":
        print("RecipeNLG external catalog: skipped (data not available)")
    else:
        print(
            "RecipeNLG external catalog: "
            f"{external['input_recipenlg_recipes']:,} input -> "
            f"{external['retained_external_test_recipes']:,} retained"
        )
    print(f"Split audit saved under: {config.output_dir / 'tables'}")


if __name__ == "__main__":
    main()
