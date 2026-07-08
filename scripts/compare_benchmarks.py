"""Benchmark comparison utilities for cross-checkpoint evaluation."""


def compute_common_subset(results: list[dict]) -> dict:
    """Compute the intersection of successfully-evaluated materials across results.

    Args:
        results: List of dicts, each with:
            - label: checkpoint/run label
            - materials: dict of {material_id: {"status": "OK"|...}}

    Returns:
        dict with:
            - common: sorted list of material IDs in all results with OK status
            - excluded: sorted list of material IDs excluded from any result
    """
    all_ids: list[set[str]] = []
    for r in results:
        ok_ids = {mid for mid, info in r["materials"].items() if info.get("status") == "OK"}
        all_ids.append(ok_ids)

    if not all_ids:
        return {"common": [], "excluded": []}

    common = set.intersection(*all_ids) if len(all_ids) > 1 else all_ids[0]
    union = set.union(*all_ids)
    excluded = sorted(union - common)
    return {"common": sorted(common), "excluded": excluded}
