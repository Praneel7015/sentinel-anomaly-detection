# SENTINEL

> Explainable behavioural anomaly detection over synthetic enterprise access logs.

SENTINEL is a complete, end-to-end ML system built for the **Scoping Hackathon** challenge:
ingest streams of access-log events, profile every entity's normal behaviour, detect and
rank anomalies, explain *why* each alert was raised, and present everything through a
live SOC analyst dashboard.

---

## Architecture at a glance

```
 Raw events (parquet / SSE stream)
         │
         ▼
 ┌───────────────────┐   batch_features / EntityState
 │  Feature Pipeline  │──────────────────────────────────────────────────────┐
 └───────────────────┘   leakage-free, streaming-equivalent                  │
                                                                              │
                                   ┌──────────────────────────────────────┐  │
                                   │         Detector Stack               │  │
                                   │  ┌─────────────┐  ┌──────────────┐  │  │
                                   │  │ Statistical  │  │ IsolationF.  │  │  │
                                   │  │ Profiler     │  │ (per-cohort) │  │  │
                                   │  └─────────────┘  └──────────────┘  │  │
                                   │  ┌─────────────┐  ┌──────────────┐  │  │
                                   │  │  Markov /   │  │  Graph peer- │  │  │
                                   │  │ n-gram seq. │  │  deviation   │  │  │
                                   │  └─────────────┘  └──────────────┘  │  │
                                   │  ┌─────────────────────────────────┐ │  │
                                   │  │  GRU Autoencoder (optional      │ │  │
                                   │  │  PyTorch, degrades gracefully)  │ │  │
                                   │  └─────────────────────────────────┘ │  │
                                   └──────────────────────────────────────┘  │
                                                  │                           │
                                   ┌──────────────▼──────────────────┐       │
                                   │  Additive Log-Odds Fusion        │       │
                                   │  + Isotonic Calibration          │       │
                                   │  → risk score 0-100              │       │
                                   └─────────────┬───────────────────┘       │
                                                 │                            │
                        ┌────────────────────────▼───────────────────┐       │
                        │   Attack-Type Classifier + Explainer        │◄──────┘
                        │   (HistGBM + transparent signature rules)   │
                        │   Contributions · Narrative · Counterfact.  │
                        └────────────────────┬───────────────────────┘
                                             │
                         ┌───────────────────▼─────────────────────────┐
                         │  FastAPI service (:8000)                      │
                         │  REST: /score /alerts /entity /feedback       │
                         │  SSE:  /stream (live ranked alert queue)      │
                         └───────────────────┬─────────────────────────┘
                                             │
                         ┌───────────────────▼─────────────────────────┐
                         │  React + Vite + TypeScript + Tailwind CSS    │
                         │  SOC Dashboard  (dark theme, live updates)   │
                         │  Triage queue · Alert detail · Entity profile│
                         │  PR curve · Confusion matrix · Latency stats │
                         └─────────────────────────────────────────────┘
```

---

## Repository layout

```
sentinel/
├── configs/
│   ├── data.yaml              # data-generation parameters
│   └── model.yaml             # detector / fusion / alerting config
├── src/sentinel/
│   ├── schema.py              # single source of truth for AccessEvent + Label
│   ├── io.py                  # leakage-free parquet read/write
│   ├── config.py              # typed config loaders
│   ├── cli.py                 # `sentinel` CLI entry point
│   ├── datagen/               # synthetic corpus generator
│   ├── features/              # incremental feature extractors + pipeline
│   ├── models/                # detectors, fusion, calibration, classifier
│   ├── explain/               # waterfall, narrative, counterfactuals, SHAP
│   ├── drift/                 # EWMA adaptive baselines, Page-Hinkley drift
│   ├── serving/               # FastAPI app + SSE stream replay
│   └── eval/                  # evaluation harness (run_eval.py)
├── dashboard/                 # Vite + React + TS + Tailwind SOC dashboard
├── tests/                     # 126 tests across all layers
├── docs/
│   └── CONTRACTS.md           # REST/SSE API + schema contract
├── reports/                   # generated plots (pr_curve, ablation, …)
├── artifacts/                 # saved model artefacts
└── data/                      # events.parquet + labels.parquet
```

---

## Quickstart

### Prerequisites

- Python 3.12 and [`uv`](https://github.com/astral-sh/uv)
- Node 18+ (for the dashboard)

### 1 — Install Python dependencies

```powershell
uv venv --python 3.12 .venv
uv pip install -e ".[dev]" --python .venv\Scripts\python.exe
```

Optional PyTorch (the stack degrades gracefully without it):

```powershell
uv pip install -e ".[torch]" --python .venv\Scripts\python.exe
```

### 2 — Generate the synthetic corpus

```powershell
.venv\Scripts\sentinel.exe gen
# writes data/events.parquet (~4.8 M rows) and data/labels.parquet
# train / val / test split: 42 / 7 / 21 days
```

### 3 — Run the evaluation harness

```powershell
.venv\Scripts\sentinel.exe eval
# trains all detectors on train split, calibrates on val, evaluates on test
# writes artifacts/eval_results.json and reports/*.png
```

### 4 — Start the API server

```powershell
.venv\Scripts\sentinel.exe serve
# FastAPI at http://127.0.0.1:8000
# Interactive docs: http://127.0.0.1:8000/docs
```

### 5 — Start the SOC dashboard

```powershell
cd dashboard
npm install
npm run dev
# Opens http://localhost:5173
```

### 6 — Run the test suite

```powershell
.venv\Scripts\pytest.exe -q
# 126 tests, all passing
```

---

## CLI reference

```
sentinel --help

Commands:
  gen     Generate synthetic corpus (events.parquet + labels.parquet)
  train   Fit the detector stack, fusion calibration and attack classifier
  eval    Train → calibrate → evaluate; write metrics + plots
  serve   Run the FastAPI scoring service under uvicorn
  replay  Drive the live demo by replaying a split into the running service
  ablate  Per-detector ablations and held-out-attack generalisation test
```

---

## Configuration

| File | Purpose |
|------|---------|
| `configs/data.yaml` | Entity counts, time windows, attack rates, confounder rates, drift settings |
| `configs/model.yaml` | Detector hyperparameters, fusion weights, alert budgets, GRU config |

---

## Key design decisions

| Decision | Rationale |
|----------|-----------|
| **Leakage-free feature pipeline** | All features are computed from *past* events only via `EntityState`; batch and streaming paths produce identical output |
| **Additive log-odds fusion** | Enables exact per-detector contribution attribution without post-hoc approximation |
| **Isotonic calibration** | Converts raw risk scores to well-calibrated probabilities; fitted on the val split |
| **Cohort-shrunk profiler** | Solves cold-start: new entities borrow a prior from their cohort |
| **Page-Hinkley drift detection** | Triggers per-entity re-baselining after concept drift; EWMA guards against poisoning |
| **Transparent signature rules + HistGBM** | Dual-path classifier; disagreement surfaces as `unknown_novel` |
| **SHAP for the classifier** | Provides feature-level explanations for attack classification |

---

## API summary

See [`docs/CONTRACTS.md`](docs/CONTRACTS.md) for the full contract.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/score` | POST | Score a single event; returns risk score + contributions + narrative |
| `/api/alerts` | GET | Ranked alert queue (paginated) |
| `/api/entity/{id}` | GET | Entity profile + recent events |
| `/api/feedback` | POST | Analyst triage feedback (true_positive / false_positive / escalate) |
| `/api/metrics` | GET | Service and model health metrics |
| `/api/stream` | GET | SSE stream of live scored events |

---

## Licence

MIT
