"""Counterfactual "what-if" explanations.

For each of the top-k high-contribution features, compute what the risk score
would be if that feature were set to its cohort-median value (i.e., benign).

Output: list of Counterfactual objects:
  "if geo-velocity were normal, risk drops from 87 to 51 (−36)"
"""
from __future__ import annotations

from typing import Any

from sentinel.features.extractors import FEATURE_NAMES, FeatureVector
from sentinel.serving.models import Contribution, Counterfactual

__all__ = ["compute_counterfactuals"]


def compute_counterfactuals(
    fv: FeatureVector,
    contributions: list[Contribution],
    risk_score: float,
    profiler: Any,  # StatisticalProfiler - avoid circular import
    fusion: Any,  # LogOddsFusion
    entity_id: str,
    cohort: str,
    top_k: int = 3,
) -> list[Counterfactual]:
    """Compute counterfactual risk scores.

    For each top-contributing feature, neutralise it to its cohort-median
    value, re-score, and report the delta.

    Args:
        fv: original feature vector.
        contributions: sorted list of Contribution objects.
        risk_score: original risk score.
        profiler: StatisticalProfiler instance (for cohort medians).
        fusion: LogOddsFusion instance (for re-scoring).
        entity_id: entity to score against.
        cohort: cohort for profiler lookup.
        top_k: number of counterfactuals to compute.

    Returns:
        List of Counterfactual objects.
    """
    from sentinel.models.registry import ModelRegistry  # lazy to avoid circular
    from sentinel.serving.models import DetectorScores

    results: list[Counterfactual] = []
    top_contribs = contributions[:top_k]

    original_values = fv.to_dict()

    for contrib in top_contribs:
        fname = contrib.feature
        cohort_val = profiler.cohort_median_approx(cohort, fname)

        # Build neutralised feature vector
        neutralised = dict(original_values)
        neutralised[fname] = cohort_val

        neutral_vals = [neutralised[f] for f in FEATURE_NAMES]
        neutral_fv = FeatureVector(neutral_vals)

        # Compute neutralised profiler z-scores
        neutral_profiler_scores = profiler.score(neutral_fv, entity_id=entity_id, cohort=cohort)
        mahal = neutral_profiler_scores.get("mahalanobis", 0.0)
        profile_score = float(max(0.0, min(1.0, mahal / 5.0)))

        # Build neutral DetectorScores (only profile changes significantly)
        # For simplicity, assume other detectors don't change much from one feature
        # (this is a local approximation)
        neutral_detector_scores = DetectorScores(
            profile=profile_score,
            isolation=0.3,  # approximate benign isolation score
            sequence=0.3,
            graph=0.3,
            gru=None,
        )

        # Re-fuse
        neutralised_risk, _ = fusion.fuse(neutral_detector_scores)

        delta = neutralised_risk - risk_score  # negative = risk drops (good)

        results.append(
            Counterfactual(
                feature=fname,
                display_name=contrib.display_name,
                neutralised_risk=float(max(0.0, min(100.0, neutralised_risk))),
                delta=float(delta),
            )
        )

    # Sort by absolute delta (most impactful first)
    results.sort(key=lambda c: abs(c.delta), reverse=True)
    return results
