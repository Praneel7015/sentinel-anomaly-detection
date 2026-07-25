"""Incrementally maintained entity-resource bipartite graph.

The graph answers four questions per event, all in O(1) / O(peers):
1. Is this resource new for the entity?
2. How many peers in the cohort have ever accessed this resource?
3. Jaccard of the entity's resource set against the cohort centroid resource set.
4. Entity degree deviation vs cohort mean degree.

No full recomputation per event — all statistics are maintained as running
counters updated once per event.
"""

from __future__ import annotations

from typing import Any

__all__ = ["GraphState"]


class GraphState:
    """Bipartite entity-resource graph.

    Maintained inside ``GlobalState`` or the pipeline; passed as a context
    object to the graph feature extractors.

    All per-cohort statistics live in the CohortState (resource_entity_sets,
    n_entities).  GraphState augments that with per-entity degree tracking.
    """

    def __init__(self) -> None:
        # entity_id -> number of distinct resources seen
        self.entity_degree: dict[str, int] = {}

        # cohort_id -> running mean + M2 of entity degree (Welford over entities)
        self._cohort_degree_n: dict[str, int] = {}
        self._cohort_degree_mean: dict[str, float] = {}
        self._cohort_degree_M2: dict[str, float] = {}

        # Global: total distinct resources seen
        self.global_resource_set: set[str] = set()

    def update(
        self,
        entity_id: str,
        cohort_id: str,
        resource: str,
        entity_resource_set: set[str],
    ) -> None:
        """Record that ``entity_id`` accessed ``resource``.

        Call **after** emitting features (past-only guarantee).
        ``entity_resource_set`` is the entity's resource set **before** adding
        the current resource (the caller decides when to add it).
        """
        new_degree = len(entity_resource_set)
        old_degree = self.entity_degree.get(entity_id, 0)
        self.entity_degree[entity_id] = new_degree

        self.global_resource_set.add(resource)

        # Update cohort degree Welford only when the degree changed
        if new_degree != old_degree:
            n = self._cohort_degree_n.get(cohort_id, 0) + 1
            mean = self._cohort_degree_mean.get(cohort_id, 0.0)
            M2 = self._cohort_degree_M2.get(cohort_id, 0.0)
            delta = new_degree - mean
            mean += delta / n
            delta2 = new_degree - mean
            M2 += delta * delta2
            self._cohort_degree_n[cohort_id] = n
            self._cohort_degree_mean[cohort_id] = mean
            self._cohort_degree_M2[cohort_id] = M2

    def entity_degree_deviation(self, entity_id: str, cohort_id: str) -> float:
        """Z-score of entity degree vs cohort mean degree.

        Returns 0 when fewer than 2 data points are available.
        """
        degree = self.entity_degree.get(entity_id, 0)
        n = self._cohort_degree_n.get(cohort_id, 0)
        if n < 2:
            return 0.0
        mean = self._cohort_degree_mean[cohort_id]
        M2 = self._cohort_degree_M2[cohort_id]
        variance = M2 / n
        if variance <= 0:
            return 0.0
        import math
        std = math.sqrt(variance)
        return (degree - mean) / std

    def jaccard_vs_cohort_centroid(
        self,
        entity_resource_set: set[str],
        cohort_resource_counts: dict[str, int],
        cohort_n_entities: int,
        threshold_fraction: float = 0.1,
    ) -> float:
        """Jaccard similarity between entity's resource set and the cohort centroid.

        The cohort centroid is defined as the set of resources accessed by at
        least ``threshold_fraction`` of cohort entities.  Returns 0 when both
        sets are empty.
        """
        if cohort_n_entities == 0:
            return 0.0
        min_peers = max(1, int(threshold_fraction * cohort_n_entities))
        # cohort_resource_counts here is the CohortState.resource_entity_sets peer count dict
        # We accept it as {resource: peer_count}
        centroid: set[str] = {r for r, cnt in cohort_resource_counts.items() if cnt >= min_peers}
        if not entity_resource_set and not centroid:
            return 0.0
        intersection = len(entity_resource_set & centroid)
        union = len(entity_resource_set | centroid)
        return intersection / union if union > 0 else 0.0

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_degree": dict(self.entity_degree),
            "_cohort_degree_n": dict(self._cohort_degree_n),
            "_cohort_degree_mean": dict(self._cohort_degree_mean),
            "_cohort_degree_M2": dict(self._cohort_degree_M2),
            "global_resource_set": list(self.global_resource_set),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GraphState:
        obj = cls()
        obj.entity_degree = dict(d["entity_degree"])
        obj._cohort_degree_n = dict(d["_cohort_degree_n"])
        obj._cohort_degree_mean = dict(d["_cohort_degree_mean"])
        obj._cohort_degree_M2 = dict(d["_cohort_degree_M2"])
        obj.global_resource_set = set(d["global_resource_set"])
        return obj
