from __future__ import annotations

import logging

from src.training.data_audit import REQUIRED_TASKS

logger = logging.getLogger(__name__)


def generate_coverage_report(normalizer=None) -> dict:
    report = {}
    for task in REQUIRED_TASKS:
        has_normalizer = task in normalizer.stats if normalizer else False
        report[task] = {
            "n_total": 0,
            "n_labeled": 0,
            "coverage_pct": 100.0 if has_normalizer else 0.0,
            "production_ready": has_normalizer,
        }
    return report


def format_coverage_metrics(report: dict) -> str:
    lines = ["Label Coverage Report:"]
    for task, info in report.items():
        status = "READY" if info["production_ready"] else "INSUFFICIENT"
        lines.append(
            f"  {task}: {info['coverage_pct']:.1f}% ({info['n_labeled']}/{info['n_total']}) [{status}]"
        )
    return "\n".join(lines)
