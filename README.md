# SENTINEL

Explainable behavioural anomaly detection over synthetic access logs.

> Scaffold stage. The full quickstart and write-up are produced by the reporting phase.
> The binding cross-component contracts live in [`docs/CONTRACTS.md`](docs/CONTRACTS.md).

## Quickstart

```powershell
uv venv --python 3.12 .venv
.venv\Scripts\python.exe -m pip --version
uv pip install -e ".[dev]" --python .venv\Scripts\python.exe
.venv\Scripts\sentinel.exe --help
```

Optional PyTorch upgrade (the stack degrades gracefully without it):

```powershell
uv pip install -e ".[torch]" --python .venv\Scripts\python.exe
```
