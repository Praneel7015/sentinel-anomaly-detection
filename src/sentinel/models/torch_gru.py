"""GRU autoencoder detector (optional PyTorch upgrade).

Degrades gracefully when torch is not installed.

Architecture:
  Encoder: 2-layer GRU → bottleneck linear → ReLU
  Decoder: linear → 2-layer GRU → linear projection to input_dim
  Loss: MSE reconstruction error on the input sequence

Input: (batch, GRU_WINDOW, GRU_FEATURE_DIM) float32 tensor
Output: same shape; anomaly score = MSE between input and reconstruction

Trained only on normal (label == "normal") events from the train split.
Score normalised to [0, 1] using training reconstruction error distribution.

If torch is unavailable, fit() is a no-op and score() returns None.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sentinel.features.sequence import GRU_FEATURE_DIM, GRU_WINDOW, gru_encode

__all__ = ["GRUAutoencoder", "TORCH_AVAILABLE"]

logger = logging.getLogger(__name__)

_ARTIFACTS_DIR = Path("artifacts")


# --------------------------------------------------------------------------- #
# Model definition (only defined when torch is available)
# --------------------------------------------------------------------------- #

if TORCH_AVAILABLE:

    class _GRUAEModel(nn.Module):  # type: ignore[misc]
        """GRU encoder-decoder autoencoder."""

        def __init__(
            self,
            input_dim: int,
            hidden_size: int,
            num_layers: int,
            bottleneck: int,
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            # Encoder
            self.enc_gru = nn.GRU(
                input_dim,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
            self.enc_fc = nn.Linear(hidden_size, bottleneck)

            # Decoder
            self.dec_fc = nn.Linear(bottleneck, hidden_size)
            self.dec_gru = nn.GRU(
                hidden_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
            self.dec_out = nn.Linear(hidden_size, input_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x: (batch, seq_len, input_dim) -> reconstruction of same shape."""
            batch, seq_len, _ = x.shape

            # Encode
            _, h_n = self.enc_gru(x)  # h_n: (num_layers, batch, hidden)
            last_hidden = h_n[-1]  # (batch, hidden)
            bottleneck_vec = torch.relu(self.enc_fc(last_hidden))  # (batch, bottleneck)

            # Decode: expand bottleneck across seq_len
            dec_input_0 = torch.relu(self.dec_fc(bottleneck_vec))  # (batch, hidden)
            # Use the bottleneck as the initial hidden state for decoder GRU
            h_0 = dec_input_0.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
            # Decoder input: zeros (teacher forcing not used for AE)
            dec_input = torch.zeros(batch, seq_len, self.hidden_size, device=x.device)
            dec_out, _ = self.dec_gru(dec_input, h_0)  # (batch, seq_len, hidden)
            recon = self.dec_out(dec_out)  # (batch, seq_len, input_dim)
            return recon

        def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
            """Return per-sample MSE. Shape: (batch,)."""
            recon = self.forward(x)
            return ((x - recon) ** 2).mean(dim=(1, 2))


# --------------------------------------------------------------------------- #
# Public wrapper
# --------------------------------------------------------------------------- #


class GRUAutoencoder:
    """GRU autoencoder anomaly detector.

    Usage::

        gru = GRUAutoencoder(hidden_size=64, num_layers=1)
        gru.fit(feature_df, labels_df, epochs=8, batch_size=256)
        score = gru.score(fv, entity_state)
        # Returns None if torch unavailable or not fitted.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 1,
        bottleneck_ratio: float = 0.5,
        learning_rate: float = 1e-3,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bottleneck = max(1, int(hidden_size * bottleneck_ratio))
        self.learning_rate = learning_rate

        self._model: Any | None = None
        self._train_errors: np.ndarray | None = None
        self._fitted = False

    # ---------------------------------------------------------------------- #
    # Fitting
    # ---------------------------------------------------------------------- #

    def fit(
        self,
        feature_df: Any,  # pd.DataFrame
        labels_df: Any,  # pd.DataFrame with event_id and label
        epochs: int = 8,
        batch_size: int = 256,
    ) -> None:
        """Train on normal train-split events only.

        Args:
            feature_df: DataFrame with ``entity_id``, ``split``, and
                        ``gru_window`` column (or entity states for encoding).
                        If the gru_window encoding is not available, falls back
                        to zero tensors (still trains, just with noise).
            labels_df: DataFrame with ``event_id`` and ``label`` columns.
            epochs: training epochs.
            batch_size: mini-batch size.
        """
        if not TORCH_AVAILABLE:
            logger.debug("torch not available, skipping GRU fit")
            return

        import pandas as pd

        # Join to get labels
        if labels_df is not None and len(labels_df) > 0:
            merged = feature_df.merge(labels_df[["event_id", "label"]], on="event_id", how="left")
        else:
            merged = feature_df.copy()
            merged["label"] = "normal"

        # Filter: train split, normal label only
        mask = merged["split"] == "train"
        if "label" in merged.columns:
            mask = mask & (merged["label"] == "normal")
        train_df = merged[mask]

        if len(train_df) == 0:
            logger.warning("No normal train events for GRU training")
            self._fitted = True
            return

        # Build tensors from gru_window data if available, else use zeros
        sequences = self._build_sequences(train_df)
        if len(sequences) == 0:
            logger.warning("No sequences built for GRU training")
            self._fitted = True
            return

        X = torch.tensor(sequences, dtype=torch.float32)  # (N, seq_len, feat)

        # Build and train model
        model = _GRUAEModel(
            input_dim=GRU_FEATURE_DIM,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bottleneck=self.bottleneck,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)  # type: ignore[attr-defined]
        criterion = nn.MSELoss()  # type: ignore[attr-defined]

        dataset = torch.utils.data.TensorDataset(X)  # type: ignore[attr-defined]
        loader = torch.utils.data.DataLoader(  # type: ignore[attr-defined]
            dataset, batch_size=batch_size, shuffle=True
        )

        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for (batch_x,) in loader:
                optimizer.zero_grad()
                recon = model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_x)
            avg_loss = total_loss / max(len(X), 1)
            logger.debug("GRU epoch %d/%d  loss=%.4f", epoch + 1, epochs, avg_loss)

        model.eval()
        self._model = model

        # Compute training error distribution for normalisation
        with torch.no_grad():
            errors = model.reconstruction_error(X).numpy()
        self._train_errors = np.sort(errors)
        self._fitted = True
        logger.info("GRU autoencoder fitted on %d normal events", len(X))

    def _build_sequences(self, df: Any) -> np.ndarray:
        """Build (N, GRU_WINDOW, GRU_FEATURE_DIM) array from dataframe."""
        # If we have a gru_window column (list of tuples), use it
        if "gru_window" in df.columns:
            seqs = []
            for row in df.itertuples(index=False):
                entries = list(row.gru_window) if row.gru_window else []
                arr = gru_encode(entries)
                seqs.append(arr)
            return np.stack(seqs, axis=0) if seqs else np.zeros((0, GRU_WINDOW, GRU_FEATURE_DIM), dtype=np.float32)

        # Fallback: build from feature columns that map to GRU dims
        n = len(df)
        if n == 0:
            return np.zeros((0, GRU_WINDOW, GRU_FEATURE_DIM), dtype=np.float32)

        # Use command_surprisal as the surprisal channel, zeros elsewhere
        seqs = np.zeros((n, GRU_WINDOW, GRU_FEATURE_DIM), dtype=np.float32)
        if "command_surprisal" in df.columns:
            surprisal_vals = df["command_surprisal"].values.astype(np.float32)
            # Fill the surprisal channel (column 1) of the last time step
            seqs[:, -1, 1] = np.clip(surprisal_vals, 0.0, 20.0)
        return seqs

    # ---------------------------------------------------------------------- #
    # Scoring
    # ---------------------------------------------------------------------- #

    def score(self, fv: Any, entity_state: Any | None = None) -> float | None:
        """Return normalised anomaly score in [0, 1], or None if unavailable.

        Args:
            fv: FeatureVector (used to build the GRU input sequence)
            entity_state: EntityState with gru_window (optional)

        Returns:
            float in [0, 1] if torch is available and model is fitted,
            None otherwise.
        """
        if not TORCH_AVAILABLE or self._model is None or self._train_errors is None:
            return None

        # Build sequence from entity_state.gru_window if available
        if entity_state is not None and hasattr(entity_state, "gru_window"):
            entries = list(entity_state.gru_window)
        else:
            entries = []

        seq = gru_encode(entries)  # (GRU_WINDOW, GRU_FEATURE_DIM)
        x = torch.tensor(seq[None], dtype=torch.float32)  # (1, seq, feat)

        self._model.eval()
        with torch.no_grad():
            error = float(self._model.reconstruction_error(x)[0].item())

        # Rank-normalise against training distribution
        rank = float(
            np.searchsorted(self._train_errors, error, side="right")
        ) / len(self._train_errors)
        return float(max(0.0, min(1.0, rank)))

    # ---------------------------------------------------------------------- #
    # Save / load weights
    # ---------------------------------------------------------------------- #

    def save(self, path: str | Path | None = None) -> None:
        """Save model weights to ``artifacts/gru_autoencoder.pt``."""
        if not TORCH_AVAILABLE or self._model is None:
            return
        dest = Path(path) if path else _ARTIFACTS_DIR / "gru_autoencoder.pt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._model.state_dict(),
                "config": {
                    "input_dim": GRU_FEATURE_DIM,
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                    "bottleneck": self.bottleneck,
                },
                "train_errors": self._train_errors.tolist()
                if self._train_errors is not None
                else [],
            },
            dest,
        )

    def load(self, path: str | Path | None = None) -> bool:
        """Load model weights. Returns True on success, False otherwise."""
        if not TORCH_AVAILABLE:
            return False
        src = Path(path) if path else _ARTIFACTS_DIR / "gru_autoencoder.pt"
        if not src.exists():
            return False
        checkpoint = torch.load(src, map_location="cpu", weights_only=False)
        cfg = checkpoint["config"]
        model = _GRUAEModel(
            input_dim=cfg["input_dim"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            bottleneck=cfg["bottleneck"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        self._model = model
        errors = checkpoint.get("train_errors", [])
        self._train_errors = np.sort(np.array(errors, dtype=np.float32)) if errors else None
        self._fitted = True
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "bottleneck": self.bottleneck,
            "learning_rate": self.learning_rate,
            "fitted": self._fitted,
            "torch_available": TORCH_AVAILABLE,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GRUAutoencoder:
        obj = cls(
            hidden_size=d["hidden_size"],
            num_layers=d["num_layers"],
            learning_rate=d["learning_rate"],
        )
        obj.bottleneck = d["bottleneck"]
        obj._fitted = d.get("fitted", False)
        return obj
