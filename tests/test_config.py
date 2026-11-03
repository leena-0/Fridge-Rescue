from pathlib import Path

import pytest

from fridge_rescue.config import PROJECT_ROOT, load_config


def test_default_config_loads() -> None:
    config = load_config()

    assert config.project_name == "Fridge Rescue"
    assert config.random_seed == 42
    assert config.default_top_k == 10
    assert config.max_top_k >= config.default_top_k
    assert config.eda_default_max_rows == 100_000
    assert config.eda_histogram_bins == 40
    assert config.preprocessing_default_max_rows == 100_000
    assert config.preprocessing_audit_sample_size == 50
    assert config.raw_data_dir == PROJECT_ROOT / "data" / "raw"
    assert (
        config.foodcom_recipes_path == PROJECT_ROOT / "data" / "raw" / "foodcom" / "RAW_recipes.csv"
    )


def test_missing_config_raises_clear_error(tmp_path: Path) -> None:
    missing_config = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(missing_config)
