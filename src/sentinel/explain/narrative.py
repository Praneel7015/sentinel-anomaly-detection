"""Human-readable narrative generation.

Selects the top-3 contributing features and generates a concise sentence for
the SOC analyst, interpolating display values into description templates.

Output: a string ≤ ~200 chars.
"""
from __future__ import annotations

from sentinel.serving.models import Contribution

__all__ = ["build_narrative"]

_MAX_CHARS = 200


def build_narrative(
    contributions: list[Contribution],
    risk_score: float,
    top_k: int = 3,
) -> str:
    """Generate a human-readable risk narrative.

    Args:
        contributions: list of Contribution objects, sorted by |contribution|.
        risk_score: final risk score 0-100.
        top_k: number of top factors to include.

    Returns:
        A narrative string ≤ ~200 chars.
    """
    if not contributions:
        return f"Risk score {risk_score:.0f}/100. No significant factors identified."

    top = contributions[:top_k]

    # Build per-factor phrases
    phrases: list[str] = []
    for c in top:
        phrase = _make_phrase(c)
        if phrase:
            phrases.append(phrase)

    if not phrases:
        return f"Risk score {risk_score:.0f}/100. No significant factors identified."

    joined = " + ".join(f"[{p}]" for p in phrases)
    narrative = f"Risk elevated by {joined}."

    # Truncate gracefully if too long
    if len(narrative) > _MAX_CHARS:
        # Reduce to fewer phrases
        narrative = f"Risk elevated by [{phrases[0]}] and {len(phrases) - 1} other factors."
        if len(narrative) > _MAX_CHARS:
            narrative = narrative[:_MAX_CHARS - 3] + "..."

    return narrative


def _make_phrase(contribution: Contribution) -> str:
    """Build a short descriptive phrase for one contribution."""
    fname = contribution.feature
    display_val = contribution.display_value
    display_name = contribution.display_name

    # Template-based descriptions per feature
    templates: dict[str, str] = {
        "geo_velocity_kmh": f"geo-velocity {display_val}",
        "geo_centroid_distance_km": f"unusual location ({display_val} from home)",
        "new_country_flag": "first-ever access from this country",
        "new_subnet_flag": "new source IP subnet",
        "entity_novel_resource": "first-ever access to this resource",
        "cohort_novel_resource": "resource never seen in peer cohort",
        "graph_entity_novel_resource": "new entity-resource access edge",
        "graph_peer_resource_count": f"resource accessed by only {display_val} peers",
        "graph_jaccard_vs_cohort": f"resource set diverges from cohort (Jaccard {display_val})",
        "entity_failure_count_5m": f"{display_val} in last 5 min",
        "entity_failure_count_1h": f"{display_val} in last hour",
        "ip_failure_count_5m": f"{display_val} from this IP in 5 min",
        "fingerprint_unknown": "unknown device fingerprint",
        "os_mismatch": "operating system never seen for this entity",
        "mac_oui_change": "new MAC OUI (possible device change)",
        "success_after_failures": "success following repeated failures",
        "distinct_entities_per_ip": f"{display_val} distinct entities from same IP",
        "offhours_bytes_rolling_7d": f"{display_val} off-hours bytes (7 d rolling)",
        "transfer_to_duration_ratio": f"high transfer rate ({display_val})",
        "bytes_zscore": f"anomalous bytes transferred ({display_val})",
        "command_surprisal": f"unusual command sequence ({display_val} bits/token)",
        "unseen_bigram_count": f"{display_val} unseen command bigrams",
        "is_offhours": "off-hours access",
        "protocol_novelty": "new protocol for this entity",
        "auth_method_novelty": "new authentication method",
    }

    if fname in templates:
        return templates[fname]

    # Generic fallback using display_name and value
    return f"{display_name}: {display_val}"
