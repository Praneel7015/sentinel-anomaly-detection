"""``sentinel`` command line entry point.

Every command is currently a stub: it prints what it will do, reports that it
is not implemented and exits 1. The option surface is real and stable -- later
phases fill in the bodies without changing the flags.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from sentinel import __version__
from sentinel.config import DEFAULT_DATA_CONFIG, DEFAULT_MODEL_CONFIG

app = typer.Typer(
    name="sentinel",
    help="SENTINEL - explainable behavioural anomaly detection over access logs.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()

DATA_CONFIG_OPT = Annotated[
    Path, typer.Option("--config", "-c", help="Path to the data generator config (YAML).")
]
MODEL_CONFIG_OPT = Annotated[
    Path, typer.Option("--config", "-c", help="Path to the model/detector config (YAML).")
]
DATA_DIR_OPT = Annotated[
    Path, typer.Option("--data-dir", help="Directory holding events.parquet and labels.parquet.")
]
ARTIFACTS_OPT = Annotated[
    Path, typer.Option("--artifacts", help="Directory for fitted models and calibrators.")
]
REPORTS_OPT = Annotated[Path, typer.Option("--reports", help="Directory for metrics and plots.")]


def _not_implemented(command: str, detail: str) -> None:
    console.print(f"[bold yellow]sentinel {command}[/]: not implemented")
    console.print(f"[dim]{detail}[/]")
    raise typer.Exit(code=1)


@app.callback()
def main(
    version: Annotated[bool, typer.Option("--version", help="Print the version and exit.")] = False,
) -> None:
    if version:
        console.print(f"sentinel {__version__}")
        raise typer.Exit()


@app.command()
def gen(
    config: DATA_CONFIG_OPT = DEFAULT_DATA_CONFIG,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Override the config's output_dir.")
    ] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Override the config's seed.")] = None,
) -> None:
    """Generate the synthetic corpus: events.parquet + labels.parquet."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from sentinel.datagen.generate import generate

    events_df, labels_df = generate(
        config_path=config,
        out_dir=out,
        seed_override=seed,
    )
    n_events = len(events_df)
    n_anomaly = int(labels_df["is_anomaly"].sum())
    console.print(f"[bold green]sentinel gen[/]: {n_events:,} events written")
    console.print(f"  anomaly rate = {n_anomaly / n_events:.3%} ({n_anomaly:,} labelled)")
    splits = events_df["split"].value_counts().to_dict()
    for spl in ("train", "val", "test"):
        console.print(f"  {spl:<6} = {splits.get(spl, 0):,} events")


@app.command()
def train(
    config: MODEL_CONFIG_OPT = DEFAULT_MODEL_CONFIG,
    data_config: Annotated[
        Path, typer.Option("--data-config", help="Data config, for split boundaries.")
    ] = DEFAULT_DATA_CONFIG,
    data_dir: DATA_DIR_OPT = Path("data"),
    artifacts: ARTIFACTS_OPT = Path("artifacts"),
) -> None:
    """Fit the detector stack, fusion calibration and attack-type classifier."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    events_path = data_dir / "events.parquet"
    labels_path = data_dir / "labels.parquet"

    if not events_path.exists():
        console.print(
            f"[bold red]sentinel train[/]: {events_path} not found.\n"
            "Run [bold]sentinel gen[/] first to generate the synthetic dataset."
        )
        raise typer.Exit(code=1)

    import time

    import pandas as pd

    from sentinel.config import load_model_config
    from sentinel.features.pipeline import batch_features
    from sentinel.models.classifier import AttackClassifier
    from sentinel.models.registry import ModelRegistry

    console.print(f"[bold green]sentinel train[/]: loading data from {data_dir}...")
    t0 = time.perf_counter()

    cfg = load_model_config(config)
    events_df = pd.read_parquet(events_path)
    labels_df = pd.read_parquet(labels_path) if labels_path.exists() else None

    n_train = int((events_df["split"] == "train").sum())
    console.print(
        f"  Loaded {len(events_df):,} events  ({n_train:,} train)  "
        f"{'+ labels' if labels_df is not None else 'no labels file'}"
    )

    console.print("  Computing features (batch pipeline)...")
    feature_df = batch_features(events_df, config=cfg.get("profile", {}))

    # Attach entity_id, cohort, split to feature_df
    for col in ("entity_id", "cohort", "split", "event_id"):
        if col in events_df.columns:
            feature_df[col] = events_df[col].values

    console.print("  Fitting detectors...")
    registry = ModelRegistry.from_config(cfg)
    registry.fit_all(events_df, labels_df, feature_df)

    console.print("  Fitting attack-type classifier...")
    classifier = AttackClassifier(
        max_iter=cfg.get("classifier", {}).get("max_iter", 300),
        learning_rate=cfg.get("classifier", {}).get("learning_rate", 0.08),
        max_depth=cfg.get("classifier", {}).get("max_depth", 6),
        min_confidence=cfg.get("classifier", {}).get("min_confidence", 0.35),
        random_state=cfg.get("classifier", {}).get("random_state", 20260725),
    )
    if labels_df is not None:
        classifier.fit(feature_df, labels_df)

    console.print(f"  Saving artifacts to {artifacts}/...")
    artifacts.mkdir(parents=True, exist_ok=True)
    registry.save(artifacts / "model_registry.pkl")

    import pickle

    with open(artifacts / "classifier.pkl", "wb") as f:
        pickle.dump(classifier.to_dict(), f)

    elapsed = time.perf_counter() - t0
    console.print(f"[bold green]sentinel train[/]: done in {elapsed:.1f}s")
    console.print(f"  Artifacts: {artifacts}/model_registry.pkl, classifier.pkl")
    if registry.gru._fitted:
        console.print("  GRU autoencoder: fitted ✓")
    else:
        console.print("  GRU autoencoder: not fitted (torch unavailable or no data)")


@app.command("eval")
def evaluate(
    data_dir: DATA_DIR_OPT = Path("data"),
    artifacts: ARTIFACTS_OPT = Path("artifacts"),
    reports: REPORTS_OPT = Path("reports"),
    max_train: Annotated[
        int, typer.Option("--max-train", help="Cap train events for faster eval (0 = all).")
    ] = 50_000,
    max_val: Annotated[
        int, typer.Option("--max-val", help="Cap val events (0 = all).")
    ] = 10_000,
    max_test: Annotated[
        int, typer.Option("--max-test", help="Cap test events (0 = all).")
    ] = 30_000,
) -> None:
    """Train on the train split, calibrate on val, score on test, write metrics + plots."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    events_path = data_dir / "events.parquet"
    if not events_path.exists():
        console.print(
            f"[bold red]sentinel eval[/]: {events_path} not found.\n"
            "Run [bold]sentinel gen[/] first to generate the synthetic dataset."
        )
        raise typer.Exit(code=1)

    from sentinel.eval.run_eval import run_eval

    console.print("[bold green]sentinel eval[/]: starting evaluation pipeline…")
    results = run_eval(
        data_dir=data_dir,
        artifacts_dir=artifacts,
        reports_dir=reports,
        max_train_events=max_train or None,
        max_val_events=max_val or None,
        max_test_events=max_test or None,
    )

    console.print(f"\n[bold]Results[/]")
    console.print(f"  PR-AUC        = {results['pr_auc']:.4f}")
    console.print(f"  ROC-AUC       = {results['roc_auc']:.4f}")
    console.print(f"  Anomaly rate  = {results['anomaly_rate_pct']:.2f}%  ({results['n_test_anomalies']:,} / {results['n_test_events']:,})")
    console.print(f"\n  Alert-budget precision/recall:")
    for b in results["budget_curve"]:
        console.print(
            f"    @{b['budget_pct']:.1f}%  P={b['precision']:.3f}  R={b['recall']:.3f}  "
            f"({b['alerts']:,} alerts, {b['analyst_hours']:.1f}h analyst)"
        )
    lat = results["latency_ms"]
    console.print(f"\n  Latency  p50={lat['p50']:.1f}ms  p95={lat['p95']:.1f}ms  p99={lat['p99']:.1f}ms")
    console.print(f"\n  FP rate — confounders={results['fp_rate_confounders']:.3f}  insider_drift={results['fp_rate_insider_drift']:.3f}")
    console.print(f"\n  Plots -> {reports}/")
    console.print(f"  JSON  -> {artifacts}/eval_results.json")


@app.command()
def serve(
    config: MODEL_CONFIG_OPT = DEFAULT_MODEL_CONFIG,
    artifacts: ARTIFACTS_OPT = Path("artifacts"),
    data_dir: DATA_DIR_OPT = Path("data"),
    host: Annotated[str, typer.Option("--host", help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", "-p", help="Bind port.")] = 8000,
    reload: Annotated[bool, typer.Option("--reload", help="Auto-reload on code change.")] = False,
) -> None:
    """Run the FastAPI scoring service (REST + SSE) under uvicorn."""
    model_registry = artifacts / "model_registry.pkl"
    if model_registry.exists():
        console.print(f"[bold green]sentinel serve[/]: model artifacts found at {artifacts}")
    else:
        console.print(
            f"[bold yellow]sentinel serve[/]: no model_registry.pkl found in {artifacts}; "
            "running in stub mode (scores are synthetic)"
        )

    from sentinel.serving.app import run_server

    run_server(
        host=host,
        port=port,
        reload=reload,
        artifacts_dir=artifacts,
        data_dir=data_dir,
    )


@app.command()
def replay(
    config: MODEL_CONFIG_OPT = DEFAULT_MODEL_CONFIG,
    data_dir: DATA_DIR_OPT = Path("data"),
    api: Annotated[str, typer.Option("--api", help="Base URL of a running service.")] = (
        "http://127.0.0.1:8000/api"
    ),
    speed: Annotated[
        float, typer.Option("--speed", help="Replay rate in events per second.")
    ] = 50.0,
    split: Annotated[str, typer.Option("--split", help="Which split to replay.")] = "test",
) -> None:
    """Drive the live demo by replaying a split into the running service."""
    _not_implemented("replay", f"would replay split={split} at {speed} ev/s into {api}")


@app.command()
def ablate(
    config: MODEL_CONFIG_OPT = DEFAULT_MODEL_CONFIG,
    data_dir: DATA_DIR_OPT = Path("data"),
    artifacts: ARTIFACTS_OPT = Path("artifacts"),
    reports: REPORTS_OPT = Path("reports"),
    holdout_attack: Annotated[
        str | None,
        typer.Option("--holdout-attack", help="Attack type withheld from classifier training."),
    ] = None,
) -> None:
    """Run per-detector ablations and the held-out-attack generalisation test."""
    _not_implemented("ablate", f"would ablate into {reports} (holdout={holdout_attack})")


if __name__ == "__main__":  # pragma: no cover
    app()
