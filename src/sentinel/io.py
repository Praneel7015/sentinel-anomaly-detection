"""Schema-enforcing parquet readers and writers.

The only sanctioned way to touch ``events.parquet`` and ``labels.parquet``.
Both directions validate against :mod:`sentinel.schema`, and the readers and
writers refuse to let ground truth leak into the observable events table --
that check is the mechanical guarantee behind "labels are hidden at inference".
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sentinel.schema import (
    EVENT_FIELDS,
    LABEL_FIELDS,
    LABEL_ONLY_FIELDS,
    events_arrow_schema,
    labels_arrow_schema,
)

__all__ = [
    "LabelLeakageError",
    "SchemaError",
    "read_events",
    "read_labels",
    "write_events",
    "write_labels",
]


class SchemaError(ValueError):
    """A frame or file does not conform to the SENTINEL schema."""


class LabelLeakageError(SchemaError):
    """Ground-truth columns were found in (or on their way into) an events table."""


def write_events(df: pd.DataFrame, path: str | Path) -> Path:
    """Write ``df`` to ``path`` as ``events.parquet``.

    Rejects any ground-truth column, reorders to :data:`~sentinel.schema.EVENT_FIELDS`
    and casts to the canonical Arrow schema.

    Raises:
        LabelLeakageError: if a label column is present.
        SchemaError: on missing/unknown columns or an uncastable dtype.
    """
    _assert_no_labels(df.columns, context="events frame being written")
    frame = _project(df, EVENT_FIELDS, "events")
    frame = _normalise_events(frame)
    return _write(frame, events_arrow_schema(), path, "events")


def read_events(path: str | Path) -> pd.DataFrame:
    """Read ``events.parquet`` and validate it.

    Raises:
        LabelLeakageError: if the file contains any of
            :data:`~sentinel.schema.LABEL_ONLY_FIELDS`. This is a hard failure,
            not a warning: silently dropping the columns would let a leaky
            generator ship undetected.
        SchemaError: if the column set does not match the events schema.
    """
    target = Path(path)
    file_schema = pq.read_schema(target)
    _assert_no_labels(file_schema.names, context=f"events file {target}")

    table = pq.read_table(target)
    _assert_columns(table.schema.names, EVENT_FIELDS, "events", target)
    table = table.select(EVENT_FIELDS).cast(events_arrow_schema())
    frame = table.to_pandas()
    _restore_nulls(frame, ["episode_id"])
    frame["command_sequence"] = [list(v) for v in frame["command_sequence"]]
    return frame


def write_labels(df: pd.DataFrame, path: str | Path) -> Path:
    """Write ``df`` to ``path`` as ``labels.parquet``."""
    frame = _project(df, LABEL_FIELDS, "labels")
    frame = frame.assign(is_anomaly=frame["is_anomaly"].astype(bool))
    return _write(frame, labels_arrow_schema(), path, "labels")


def read_labels(path: str | Path) -> pd.DataFrame:
    """Read and validate ``labels.parquet``."""
    target = Path(path)
    table = pq.read_table(target)
    _assert_columns(table.schema.names, LABEL_FIELDS, "labels", target)
    table = table.select(LABEL_FIELDS).cast(labels_arrow_schema())
    frame = table.to_pandas()
    _restore_nulls(frame, ["episode_id", "confounder_type", "attack_stage"])
    return frame


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #


def _assert_no_labels(columns: object, *, context: str) -> None:
    present = [c for c in LABEL_ONLY_FIELDS if c in set(columns)]  # type: ignore[arg-type]
    if present:
        raise LabelLeakageError(
            f"ground-truth columns {present} found in {context}. Labels belong in "
            "labels.parquet, joined on event_id -- never in the events table."
        )


def _assert_columns(actual: list[str], expected: list[str], kind: str, path: Path) -> None:
    missing = [c for c in expected if c not in actual]
    unknown = [c for c in actual if c not in expected]
    if missing or unknown:
        raise SchemaError(
            f"{path} is not a valid {kind} table: missing={missing} unexpected={unknown}"
        )


def _project(df: pd.DataFrame, fields: list[str], kind: str) -> pd.DataFrame:
    _assert_columns(list(df.columns), fields, kind, Path("<in-memory frame>"))
    return df.loc[:, fields].copy()


def _normalise_events(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce the two columns pandas most often gets subtly wrong."""
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = ts
    if df["command_sequence"].isna().any():
        df["command_sequence"] = df["command_sequence"].apply(lambda v: [] if _is_missing(v) else v)
    return df


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and value != value)


def _restore_nulls(df: pd.DataFrame, columns: list[str]) -> None:
    """Turn pandas' ``NaN`` back into ``None`` for nullable string columns.

    Without this, ``AccessEvent(**row)`` fails on a null ``episode_id`` because
    pandas represents the missing value as a float. Callers get real ``None``.
    """
    for column in columns:
        values = [None if _is_missing(v) else v for v in df[column].tolist()]
        df[column] = pd.Series(values, index=df.index, dtype=object)


def _write(df: pd.DataFrame, schema: pa.Schema, path: str | Path, kind: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError) as exc:
        raise SchemaError(f"{kind} frame does not match the SENTINEL schema: {exc}") from exc
    pq.write_table(table, target, compression="zstd")
    return target
