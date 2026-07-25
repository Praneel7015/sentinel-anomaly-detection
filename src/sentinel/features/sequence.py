"""Command-sequence modelling: per-entity and per-cohort Markov + n-gram.

Surprisal model
---------------
For event with ``command_sequence`` = [c_0, c_1, ..., c_{n-1}]:

  surprisal = mean_i(-log2 P(c_i | c_{i-1}))

where the transition probability is estimated with additive smoothing and
**backoff**:

    P_e(b|a) = (count_entity(a→b) + smooth) /
               (count_entity(a) + smooth * V)

Backed off to cohort if the entity bigram count is 0:

    P(b|a) = interp(P_e, P_c, P_global)

where the interpolation weight is the entity event count shrinkage weight.

Empty sequences
---------------
If ``command_sequence = []`` (non-privileged session), surprisal is 0.0
— a session with no commands is NOT anomalous for the absence of commands.

GRU tensor encoding
-------------------
``gru_encode(entity_state, vocab)`` returns a fixed-width 2-D numpy array
of shape ``(GRU_WINDOW, GRU_FEATURE_DIM)``, ready for the GRU autoencoder.

Columns:
  0: command token index (0 = padding)
  1: log(1 + bigram_surprisal)  — float32
  2: is_privileged (1 or 0)
  3: hour_of_day / 23            — float32 in [0, 1]

Shape: ``(16, 4)``.  The GRU agent reads ``GRU_WINDOW`` and
``GRU_FEATURE_DIM`` from this module.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

__all__ = [
    "SequenceModel",
    "GRU_WINDOW",
    "GRU_FEATURE_DIM",
    "gru_encode",
]

GRU_WINDOW: int = 16
GRU_FEATURE_DIM: int = 4  # (cmd_idx, surprisal, is_privileged, hour_norm)

_DEFAULT_SMOOTH: float = 0.5
_DEFAULT_ORDER: int = 2


class SequenceModel:
    """Markov transition counts with additive smoothing and entity→cohort→global backoff.

    This object is **not** tied to a specific entity; it is used for both
    cohort-level and global models.  Per-entity transition counts live inside
    ``EntityState.markov_counts`` and are accessed directly by extractors.
    """

    def __init__(self, smoothing: float = _DEFAULT_SMOOTH) -> None:
        self.smoothing: float = smoothing
        # bigram counts: (prev, curr) -> count
        self.bigram_counts: Counter[tuple[str, str]] = Counter()
        self.unigram_counts: Counter[str] = Counter()
        self.total_tokens: int = 0

    def update(self, sequence: list[str]) -> None:
        for cmd in sequence:
            self.unigram_counts[cmd] += 1
            self.total_tokens += 1
        for prev_cmd, curr_cmd in zip(sequence, sequence[1:], strict=False):
            self.bigram_counts[(prev_cmd, curr_cmd)] += 1

    def transition_prob(self, prev: str, curr: str) -> float:
        """Additive-smoothed P(curr | prev)."""
        vocab_size = max(len(self.unigram_counts), 1)
        numerator = self.bigram_counts[(prev, curr)] + self.smoothing
        denominator = self.unigram_counts[prev] + self.smoothing * vocab_size
        return numerator / denominator

    def unigram_prob(self, cmd: str) -> float:
        """Additive-smoothed unigram probability."""
        vocab_size = max(len(self.unigram_counts), 1)
        return (self.unigram_counts[cmd] + self.smoothing) / (
            self.total_tokens + self.smoothing * vocab_size
        )

    def vocab_size(self) -> int:
        return len(self.unigram_counts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "smoothing": self.smoothing,
            "bigram_counts": {f"{k[0]}\x00{k[1]}": v for k, v in self.bigram_counts.items()},
            "unigram_counts": dict(self.unigram_counts),
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SequenceModel:
        obj = cls(smoothing=d["smoothing"])
        obj.bigram_counts = Counter(
            {(k.split("\x00")[0], k.split("\x00")[1]): v for k, v in d["bigram_counts"].items()}
        )
        obj.unigram_counts = Counter(d["unigram_counts"])
        obj.total_tokens = d["total_tokens"]
        return obj


# ---------------------------------------------------------------------------
# Surprisal calculation
# ---------------------------------------------------------------------------


def sequence_surprisal(
    sequence: list[str],
    entity_bigrams: Counter[tuple[str, str]],
    entity_unigrams: Counter[str],
    cohort_model: SequenceModel | None,
    global_model: SequenceModel | None,
    entity_n: int,
    shrinkage_k: int = 20,
    smoothing: float = _DEFAULT_SMOOTH,
) -> float:
    """Length-normalised surprisal (bits per token).

    Returns 0.0 for empty sequences (non-privileged sessions are not penalised
    for having no commands).

    Backoff order: entity → cohort → global.
    """
    if not sequence:
        return 0.0

    # Build bigrams from the sequence
    pairs = list(zip(sequence, sequence[1:], strict=False))
    if not pairs:
        # Single-token sequence: use unigram surprisal from cohort/global
        cmd = sequence[0]
        prob = _unigram_prob_blended(
            cmd, entity_unigrams, entity_n, cohort_model, global_model, shrinkage_k, smoothing
        )
        return -math.log2(max(prob, 1e-10))

    surprisal_total = 0.0
    for prev_cmd, curr_cmd in pairs:
        prob = _bigram_prob_blended(
            prev_cmd,
            curr_cmd,
            entity_bigrams,
            entity_unigrams,
            entity_n,
            cohort_model,
            global_model,
            shrinkage_k,
            smoothing,
        )
        surprisal_total += -math.log2(max(prob, 1e-10))

    return surprisal_total / len(pairs)


def unseen_bigram_count(
    sequence: list[str],
    entity_bigrams: Counter[tuple[str, str]],
) -> int:
    """Count bigrams in ``sequence`` that have never been seen for this entity."""
    return sum(
        1 for prev, curr in zip(sequence, sequence[1:], strict=False) if entity_bigrams[(prev, curr)] == 0
    )


def sequence_length_zscore(
    length: int, cohort_model: SequenceModel | None, entity_n: int
) -> float:
    """Z-score of sequence length vs cohort distribution.

    Falls back to 0.0 when insufficient data.
    """
    if cohort_model is None or cohort_model.total_tokens == 0:
        return 0.0
    # Use cohort total_tokens / number of sequences as the mean
    # We approximate from what we have: total_tokens / max(1, events with commands)
    mean = cohort_model.total_tokens / max(1, cohort_model.vocab_size())
    if mean <= 0:
        return 0.0
    # approximation: std ≈ mean (geometric assumption for command sequences)
    std = max(1.0, math.sqrt(mean))
    return (length - mean) / std


# ---------------------------------------------------------------------------
# GRU encoding
# ---------------------------------------------------------------------------


def gru_encode(
    gru_window_entries: list[tuple[int, ...]],
    pad_value: float = 0.0,
) -> Any:
    """Return a (GRU_WINDOW, GRU_FEATURE_DIM) float32 numpy array.

    ``gru_window_entries`` is a list of (cmd_idx, surprisal_q, is_priv, hour_norm)
    tuples from ``EntityState.gru_window``.  Padded with zeros at the start
    when fewer than GRU_WINDOW entries are available.

    Shape: ``(16, 4)`` = ``(GRU_WINDOW, GRU_FEATURE_DIM)``.
    """
    import numpy as np

    out = np.full((GRU_WINDOW, GRU_FEATURE_DIM), pad_value, dtype=np.float32)
    n = len(gru_window_entries)
    start = GRU_WINDOW - n
    for i, entry in enumerate(gru_window_entries):
        row = list(entry)
        for j in range(min(len(row), GRU_FEATURE_DIM)):
            out[start + i, j] = float(row[j])
    return out


# ---------------------------------------------------------------------------
# Private backoff helpers
# ---------------------------------------------------------------------------


def _entity_bigram_prob(
    prev: str,
    curr: str,
    entity_bigrams: Counter[tuple[str, str]],
    entity_unigrams: Counter[str],
    smoothing: float,
) -> float:
    vocab_size = max(len(entity_unigrams), 1)
    numerator = entity_bigrams[(prev, curr)] + smoothing
    denominator = entity_unigrams[prev] + smoothing * vocab_size
    return numerator / denominator


def _bigram_prob_blended(
    prev: str,
    curr: str,
    entity_bigrams: Counter[tuple[str, str]],
    entity_unigrams: Counter[str],
    entity_n: int,
    cohort_model: SequenceModel | None,
    global_model: SequenceModel | None,
    shrinkage_k: int,
    smoothing: float,
) -> float:
    """Shrinkage-blended bigram probability."""
    w = entity_n / (entity_n + shrinkage_k)
    p_entity = _entity_bigram_prob(prev, curr, entity_bigrams, entity_unigrams, smoothing)

    # Backoff: cohort or global
    if cohort_model is not None and cohort_model.total_tokens > 0:
        p_back = cohort_model.transition_prob(prev, curr)
    elif global_model is not None and global_model.total_tokens > 0:
        p_back = global_model.transition_prob(prev, curr)
    else:
        p_back = smoothing / max(len(entity_unigrams) + 1, 2)

    return w * p_entity + (1.0 - w) * p_back


def _unigram_prob_blended(
    cmd: str,
    entity_unigrams: Counter[str],
    entity_n: int,
    cohort_model: SequenceModel | None,
    global_model: SequenceModel | None,
    shrinkage_k: int,
    smoothing: float,
) -> float:
    w = entity_n / (entity_n + shrinkage_k)
    vocab_size = max(len(entity_unigrams), 1)
    p_entity = (entity_unigrams[cmd] + smoothing) / (
        sum(entity_unigrams.values()) + smoothing * vocab_size
    )

    if cohort_model is not None and cohort_model.total_tokens > 0:
        p_back = cohort_model.unigram_prob(cmd)
    elif global_model is not None and global_model.total_tokens > 0:
        p_back = global_model.unigram_prob(cmd)
    else:
        p_back = 1.0 / vocab_size

    return w * p_entity + (1.0 - w) * p_back
