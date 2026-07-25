"""Typed loaders for ``configs/data.yaml`` and ``configs/model.yaml``.

The loaders are deliberately strict: an unrecognised key raises
:class:`ConfigError` rather than being silently ignored. A typo in a YAML key
that quietly falls back to a default is the classic way an experiment ends up
measuring something other than what the config file appears to say.

Usage::

    from sentinel.config import load_data_config, load_model_config

    data = load_data_config()          # defaults to configs/data.yaml
    model = load_model_config()        # defaults to configs/model.yaml
    k = model.profile.cohort_shrinkage_k
"""

from __future__ import annotations

import types
import typing
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, ClassVar

import yaml

from sentinel.schema import (
    ATTACK_TYPES,
    CONFOUNDER_TYPES,
    DETECTOR_NAMES,
    EDGE_CASE_TYPES,
    ENTITY_TYPES,
    RISK_THRESHOLD_CRITICAL,
    RISK_THRESHOLD_HIGH,
    RISK_THRESHOLD_MEDIUM,
)

__all__ = [
    "DEFAULT_DATA_CONFIG",
    "DEFAULT_MODEL_CONFIG",
    "AlertingConfig",
    "AttackSpec",
    "ClassifierConfig",
    "ColdStartConfig",
    "ConfigError",
    "DataConfig",
    "DriftDataConfig",
    "DriftModelConfig",
    "EntityCounts",
    "ExplainConfig",
    "FusionConfig",
    "GraphConfig",
    "GruConfig",
    "IsolationConfig",
    "ModelConfig",
    "NoiseConfig",
    "PageHinkleyConfig",
    "PoisoningGuardConfig",
    "ProfileConfig",
    "RebaselineConfig",
    "RiskThresholds",
    "SequenceConfig",
    "ServingConfig",
    "StreamingConfig",
    "TimeConfig",
    "TorchConfig",
    "VolumeRange",
    "load_data_config",
    "load_model_config",
    "project_root",
]


class ConfigError(ValueError):
    """A config file is malformed, incomplete, or contains unknown keys."""


def project_root() -> Path:
    """Repository root, resolved from this file's location (editable install)."""
    return Path(__file__).resolve().parents[2]


DEFAULT_DATA_CONFIG: Path = project_root() / "configs" / "data.yaml"
DEFAULT_MODEL_CONFIG: Path = project_root() / "configs" / "model.yaml"


# --------------------------------------------------------------------------- #
# data.yaml
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class EntityCounts:
    users: int
    service_accounts: int
    edge_devices: int

    @property
    def total(self) -> int:
        return self.users + self.service_accounts + self.edge_devices

    def as_dict(self) -> dict[str, int]:
        """Keyed by :data:`~sentinel.schema.ENTITY_TYPES`."""
        return {
            "user": self.users,
            "service_account": self.service_accounts,
            "edge_device": self.edge_devices,
        }


@dataclass(frozen=True, slots=True)
class TimeConfig:
    start_date: date
    train_days: int
    val_days: int
    test_days: int
    timezone: str = "UTC"

    def __post_init__(self) -> None:
        for name in ("train_days", "val_days", "test_days"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"time.{name} must be positive")

    @property
    def total_days(self) -> int:
        return self.train_days + self.val_days + self.test_days

    @property
    def split_boundaries(self) -> dict[str, tuple[date, date]]:
        """Half-open ``[start, end)`` date range per split."""
        train_end = self.start_date + timedelta(days=self.train_days)
        val_end = train_end + timedelta(days=self.val_days)
        test_end = val_end + timedelta(days=self.test_days)
        return {
            "train": (self.start_date, train_end),
            "val": (train_end, val_end),
            "test": (val_end, test_end),
        }


@dataclass(frozen=True, slots=True)
class VolumeRange:
    min: int
    max: int

    def __post_init__(self) -> None:
        if self.min < 0 or self.max < self.min:
            raise ConfigError(f"invalid volume range min={self.min} max={self.max}")


@dataclass(frozen=True, slots=True)
class AttackSpec:
    """Injection rate (fraction of sessions) plus injector-owned parameters."""

    rate: float
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.rate <= 1.0:
            raise ConfigError(f"injection rate must be in [0, 1], got {self.rate}")


@dataclass(frozen=True, slots=True)
class DriftDataConfig:
    drifting_entities: int
    drift_start_day: int
    ramp_days: int
    magnitude: float


@dataclass(frozen=True, slots=True)
class NoiseConfig:
    baseline_auth_failure_rate: float
    off_hours_fraction: float
    rare_resource_fraction: float


@dataclass(frozen=True, slots=True)
class DataConfig:
    seed: int
    output_dir: Path
    entities: EntityCounts
    cold_start_entities: int
    time: TimeConfig
    events_per_entity_per_day: dict[str, VolumeRange]
    privileged_session_rate: float
    attacks: dict[str, AttackSpec]
    edge_cases: dict[str, AttackSpec]
    confounders: dict[str, float]
    drift: DriftDataConfig
    noise: NoiseConfig

    def __post_init__(self) -> None:
        _exact_keys(self.events_per_entity_per_day, ENTITY_TYPES, "events_per_entity_per_day")
        _exact_keys(self.attacks, ATTACK_TYPES, "attacks")
        _exact_keys(self.edge_cases, EDGE_CASE_TYPES, "edge_cases")
        _exact_keys(self.confounders, CONFOUNDER_TYPES, "confounders")
        if self.cold_start_entities >= self.entities.total:
            raise ConfigError("cold_start_entities must be smaller than the total population")
        if not 0 <= self.drift.drift_start_day <= self.time.total_days:
            raise ConfigError("drift.drift_start_day falls outside the generated time window")


# --------------------------------------------------------------------------- #
# model.yaml
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class FusionConfig:
    weights: dict[str, float]
    bias: float
    logit_clip: float
    redistribute_missing_weight: bool

    def __post_init__(self) -> None:
        _exact_keys(self.weights, DETECTOR_NAMES, "fusion.weights")

    def active_weights(self, available: list[str]) -> dict[str, float]:
        """Weights restricted to ``available`` detectors, renormalised if asked.

        Keeps risk scores comparable between a torch-enabled run and a
        numpy-only run, where the ``gru`` detector is simply absent.
        """
        subset = {name: self.weights[name] for name in available}
        if not self.redistribute_missing_weight:
            return subset
        total, kept = sum(self.weights.values()), sum(subset.values())
        if kept <= 0:
            raise ConfigError("no active detector carries a positive fusion weight")
        scale = total / kept
        return {name: weight * scale for name, weight in subset.items()}


@dataclass(frozen=True, slots=True)
class PoisoningGuardConfig:
    enabled: bool
    max_risk_for_update: float


@dataclass(frozen=True, slots=True)
class ProfileConfig:
    cohort_shrinkage_k: int
    ewma_half_life_days: float
    min_variance: float
    poisoning_guard: PoisoningGuardConfig


@dataclass(frozen=True, slots=True)
class IsolationConfig:
    n_estimators: int
    max_samples: int
    contamination: str | float
    random_state: int


@dataclass(frozen=True, slots=True)
class SequenceConfig:
    order: int
    smoothing: float
    min_sequence_length: int


@dataclass(frozen=True, slots=True)
class GraphConfig:
    peer_window_days: int
    min_peers: int
    novelty_decay: float


@dataclass(frozen=True, slots=True)
class GruConfig:
    hidden_size: int
    num_layers: int
    epochs: int
    batch_size: int
    learning_rate: float


@dataclass(frozen=True, slots=True)
class TorchConfig:
    """``enabled`` is ``True``, ``False`` or the string ``"auto"``."""

    enabled: bool | str = "auto"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool) and self.enabled != "auto":
            raise ConfigError(f"torch.enabled must be true, false or 'auto', got {self.enabled!r}")

    def resolve(self) -> bool:
        """Decide whether the torch detector should be used in this process.

        Raises:
            ConfigError: when torch is explicitly required but not importable.
        """
        import importlib.util

        available = importlib.util.find_spec("torch") is not None
        if self.enabled is True and not available:
            raise ConfigError("torch.enabled is true but torch is not installed")
        return available if self.enabled == "auto" else bool(self.enabled)


@dataclass(frozen=True, slots=True)
class StreamingConfig:
    quantile_window: int
    min_window_for_pvalue: int
    warmup_pvalue: float


@dataclass(frozen=True, slots=True)
class AlertingConfig:
    budget_pcts: list[float]
    default_budget_pct: float
    analyst_minutes_per_alert: float

    def __post_init__(self) -> None:
        if self.default_budget_pct not in self.budget_pcts:
            raise ConfigError("alerting.default_budget_pct must be one of alerting.budget_pcts")


@dataclass(frozen=True, slots=True)
class RiskThresholds:
    medium: float
    high: float
    critical: float

    def __post_init__(self) -> None:
        expected = (RISK_THRESHOLD_MEDIUM, RISK_THRESHOLD_HIGH, RISK_THRESHOLD_CRITICAL)
        if (self.medium, self.high, self.critical) != expected:
            raise ConfigError(
                "risk_thresholds must match sentinel.schema.RISK_BANDS "
                f"{expected}; change both or neither"
            )


@dataclass(frozen=True, slots=True)
class ColdStartConfig:
    min_events: int
    provisional_threshold_widening: float
    fall_back_to_cohort: bool


@dataclass(frozen=True, slots=True)
class PageHinkleyConfig:
    """Page-Hinkley change detector parameters.

    ``lambda`` is a Python keyword, so the YAML key ``lambda`` maps onto the
    attribute ``lambda_``.
    """

    __yaml_aliases__: ClassVar[dict[str, str]] = {"lambda": "lambda_"}

    delta: float
    lambda_: float
    min_instances: int


@dataclass(frozen=True, slots=True)
class RebaselineConfig:
    fast_half_life_days: float
    recovery_days: int


@dataclass(frozen=True, slots=True)
class DriftModelConfig:
    page_hinkley: PageHinkleyConfig
    rebaseline: RebaselineConfig


@dataclass(frozen=True, slots=True)
class ClassifierConfig:
    max_iter: int
    learning_rate: float
    max_depth: int
    class_weight: str
    min_confidence: float
    random_state: int


@dataclass(frozen=True, slots=True)
class ExplainConfig:
    top_k_contributions: int
    counterfactual_top_k: int
    shap_max_samples: int


@dataclass(frozen=True, slots=True)
class ServingConfig:
    artifacts_dir: Path
    max_alerts_in_memory: int
    sse_heartbeat_s: float
    replay_speed: float


@dataclass(frozen=True, slots=True)
class ModelConfig:
    fusion: FusionConfig
    profile: ProfileConfig
    isolation: IsolationConfig
    sequence: SequenceConfig
    graph: GraphConfig
    gru: GruConfig
    torch: TorchConfig
    streaming: StreamingConfig
    alerting: AlertingConfig
    risk_thresholds: RiskThresholds
    cold_start: ColdStartConfig
    drift: DriftModelConfig
    classifier: ClassifierConfig
    explain: ExplainConfig
    serving: ServingConfig


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_data_config(path: str | Path | None = None) -> DataConfig:
    """Load and validate ``configs/data.yaml`` (or ``path``)."""
    return _load(DataConfig, path or DEFAULT_DATA_CONFIG)


def load_model_config(path: str | Path | None = None) -> ModelConfig:
    """Load and validate ``configs/model.yaml`` (or ``path``)."""
    return _load(ModelConfig, path or DEFAULT_MODEL_CONFIG)


def _load(cls: type[Any], path: str | Path) -> Any:
    target = Path(path)
    if not target.is_file():
        raise ConfigError(f"config file not found: {target}")
    with target.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError(f"{target} must contain a YAML mapping at the top level")
    return _build(cls, raw, target.name)


def _build(cls: type[Any], data: Any, path: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(
            f"{path}: expected a mapping for {cls.__name__}, got {type(data).__name__}"
        )

    aliases: dict[str, str] = getattr(cls, "__yaml_aliases__", {})
    supplied = {aliases.get(key, key): value for key, value in data.items()}

    hints = typing.get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(supplied) - known)
    if unknown:
        raise ConfigError(
            f"{path}: unknown key(s) {unknown} for {cls.__name__}; expected {sorted(known)}"
        )

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name in supplied:
            kwargs[f.name] = _coerce(hints[f.name], supplied[f.name], f"{path}.{f.name}")
        elif f.default is MISSING and f.default_factory is MISSING:  # type: ignore[misc]
            raise ConfigError(f"{path}: missing required key '{f.name}' for {cls.__name__}")
    return cls(**kwargs)


def _coerce(annotation: Any, value: Any, path: str) -> Any:
    if annotation is Any:
        return value

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin in (typing.Union, types.UnionType):
        return _coerce_union(args, value, path)
    if origin in (dict, typing.Dict):  # noqa: UP006
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
        return {str(k): _coerce(args[1], v, f"{path}.{k}") for k, v in value.items()}
    if origin in (list, typing.List):  # noqa: UP006
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        return [_coerce(args[0], v, f"{path}[{i}]") for i, v in enumerate(value)]

    if is_dataclass(annotation):
        return _build(annotation, value, path)
    if annotation is Path:
        return Path(str(value))
    if annotation is date:
        return value if isinstance(value, date) else date.fromisoformat(str(value))
    if annotation is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected a boolean, got {value!r}")
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value
    return value


def _coerce_union(args: tuple[Any, ...], value: Any, path: str) -> Any:
    if value is None and type(None) in args:
        return None
    for candidate in args:
        if candidate is type(None):
            continue
        try:
            return _coerce(candidate, value, path)
        except (ConfigError, ValueError, TypeError):
            continue
    raise ConfigError(f"{path}: {value!r} does not match any of {args}")


def _exact_keys(mapping: dict[str, Any], expected: list[str], name: str) -> None:
    missing = sorted(set(expected) - set(mapping))
    unknown = sorted(set(mapping) - set(expected))
    if missing or unknown:
        raise ConfigError(f"{name}: missing={missing} unexpected={unknown}; expected {expected}")
