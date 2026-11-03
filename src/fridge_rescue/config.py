"""Load and validate shared project configuration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class AppConfig:
    """Validated application settings used across pipelines and the UI."""

    project_name: str
    random_seed: int
    raw_data_dir: Path
    processed_data_dir: Path
    model_dir: Path
    output_dir: Path
    foodcom_recipes_path: Path
    foodcom_interactions_path: Path
    recipenlg_recipes_path: Path
    eda_default_max_rows: int
    eda_histogram_bins: int
    default_top_k: int
    max_top_k: int


def _required(mapping: dict[str, Any], key: str, section: str) -> Any:
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValueError(f"Missing required config value: {section}.{key}") from exc


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load YAML settings and resolve configured paths from the project root."""

    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)

    if not isinstance(raw_config, dict):
        raise ValueError("Config root must be a mapping")

    project = _required(raw_config, "project", "root")
    paths = _required(raw_config, "paths", "root")
    datasets = _required(raw_config, "datasets", "root")
    foodcom = _required(datasets, "foodcom", "datasets")
    recipenlg = _required(datasets, "recipenlg", "datasets")
    eda = _required(raw_config, "eda", "root")
    recommendation = _required(raw_config, "recommendation", "root")

    random_seed = _required(project, "random_seed", "project")
    eda_default_max_rows = _required(eda, "default_max_rows", "eda")
    eda_histogram_bins = _required(eda, "histogram_bins", "eda")
    default_top_k = _required(recommendation, "default_top_k", "recommendation")
    max_top_k = _required(recommendation, "max_top_k", "recommendation")

    if not isinstance(random_seed, int) or random_seed < 0:
        raise ValueError("project.random_seed must be a non-negative integer")
    if not isinstance(eda_default_max_rows, int) or eda_default_max_rows <= 0:
        raise ValueError("eda.default_max_rows must be a positive integer")
    if not isinstance(eda_histogram_bins, int) or eda_histogram_bins <= 0:
        raise ValueError("eda.histogram_bins must be a positive integer")
    if not isinstance(default_top_k, int) or default_top_k <= 0:
        raise ValueError("recommendation.default_top_k must be a positive integer")
    if not isinstance(max_top_k, int) or max_top_k < default_top_k:
        raise ValueError("recommendation.max_top_k must be at least default_top_k")

    return AppConfig(
        project_name=str(_required(project, "name", "project")),
        random_seed=random_seed,
        raw_data_dir=PROJECT_ROOT / _required(paths, "raw_data_dir", "paths"),
        processed_data_dir=PROJECT_ROOT / _required(paths, "processed_data_dir", "paths"),
        model_dir=PROJECT_ROOT / _required(paths, "model_dir", "paths"),
        output_dir=PROJECT_ROOT / _required(paths, "output_dir", "paths"),
        foodcom_recipes_path=PROJECT_ROOT / _required(foodcom, "recipes", "datasets.foodcom"),
        foodcom_interactions_path=PROJECT_ROOT
        / _required(foodcom, "interactions", "datasets.foodcom"),
        recipenlg_recipes_path=PROJECT_ROOT / _required(recipenlg, "recipes", "datasets.recipenlg"),
        eda_default_max_rows=eda_default_max_rows,
        eda_histogram_bins=eda_histogram_bins,
        default_top_k=default_top_k,
        max_top_k=max_top_k,
    )
