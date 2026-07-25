"""Tests for the strict YAML config loaders."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sentinel.config import (
    DEFAULT_DATA_CONFIG,
    DEFAULT_MODEL_CONFIG,
    ConfigError,
    load_data_config,
    load_model_config,
)
from sentinel.schema import ATTACK_TYPES, CONFOUNDER_TYPES, DETECTOR_NAMES, ENTITY_TYPES


def test_shipped_data_config_loads() -> None:
    cfg = load_data_config()
    assert cfg.entities.total == 800 + 150 + 350
    assert set(cfg.attacks) == set(ATTACK_TYPES)
    assert set(cfg.confounders) == set(CONFOUNDER_TYPES)
    assert set(cfg.events_per_entity_per_day) == set(ENTITY_TYPES)
    assert cfg.time.total_days == 42 + 7 + 21
    assert list(cfg.time.split_boundaries) == ["train", "val", "test"]
    for spec in cfg.attacks.values():
        assert 0.005 <= spec.rate <= 0.03


def test_shipped_model_config_loads() -> None:
    cfg = load_model_config()
    assert set(cfg.fusion.weights) == set(DETECTOR_NAMES)
    assert cfg.alerting.budget_pcts == [0.5, 1.0, 2.0]
    assert cfg.profile.cohort_shrinkage_k == 20
    assert cfg.drift.page_hinkley.lambda_ == 50.0
    assert cfg.torch.enabled == "auto"
    assert isinstance(cfg.torch.resolve(), bool)


def test_missing_gru_weight_is_redistributed() -> None:
    cfg = load_model_config()
    without_gru = cfg.fusion.active_weights(["profile", "isolation", "sequence", "graph"])
    assert set(without_gru) == {"profile", "isolation", "sequence", "graph"}
    assert sum(without_gru.values()) == pytest.approx(sum(cfg.fusion.weights.values()))


def test_unknown_key_raises(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_DATA_CONFIG.read_text(encoding="utf-8"))
    raw["seeed"] = 1
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="seeed"):
        load_data_config(path)


def test_missing_required_key_raises(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_MODEL_CONFIG.read_text(encoding="utf-8"))
    del raw["fusion"]["bias"]
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="bias"):
        load_model_config(path)


def test_unknown_attack_type_raises(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_DATA_CONFIG.read_text(encoding="utf-8"))
    raw["attacks"]["sql_injection"] = {"rate": 0.01}
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="sql_injection"):
        load_data_config(path)
