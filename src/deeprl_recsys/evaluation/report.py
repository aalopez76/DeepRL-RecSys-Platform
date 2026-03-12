"""Evaluation reporting — generate Markdown and JSON reports from OPE results.

Produces structured reports from :class:`~evaluation.ope.diagnostics.ReliabilityVerdict`
and estimator results, suitable for storing in ``reports/`` and for CI review.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deeprl_recsys.core.logging import get_logger
from deeprl_recsys.evaluation.ope.diagnostics import ReliabilityVerdict

logger = get_logger(__name__)


def generate_report(
    estimates: dict[str, float],
    verdict: ReliabilityVerdict,
    *,
    output_dir: Path | str | None = None,
    format: str = "markdown",
) -> str:
    """Generate an evaluation report.

    Args:
        estimates: OPE estimator results (name → value).
        verdict: Reliability verdict from diagnostics.
        output_dir: If provided, save the report to this directory.
        format: ``"markdown"`` (default) or ``"json"``.

    Returns:
        Report contents as a string.
    """
    if format == "json":
        content = _generate_json(estimates, verdict)
    else:
        content = _generate_markdown(estimates, verdict)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ext = ".json" if format == "json" else ".md"
        report_path = out / f"ope_report{ext}"
        report_path.write_text(content, encoding="utf-8")
        logger.info("report_saved", path=str(report_path), format=format)

    return content


def _generate_markdown(
    estimates: dict[str, float],
    verdict: ReliabilityVerdict,
) -> str:
    """Generate a Markdown report."""
    lines: list[str] = [
        "# OPE Evaluation Report",
        "",
        "## Estimates",
        "",
        "| Estimator | Value |",
        "|-----------|-------|",
    ]
    for name, value in estimates.items():
        lines.append(f"| {name} | {value:.6f} |")

    lines.extend([
        "",
        "## Diagnostics",
        "",
        f"- **Severity**: {verdict.severity}",
        f"- **Reliable**: {verdict.reliable}",
    ])

    if verdict.warnings:
        lines.append("")
        lines.append("### Warnings")
        for w in verdict.warnings:
            lines.append(f"- {w}")

    if verdict.stats:
        lines.append("")
        lines.append("### Statistics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for k, v in verdict.stats.items():
            lines.append(f"| {k} | {v:.4f} |")

    lines.append("")
    return "\n".join(lines)


def _generate_json(
    estimates: dict[str, float],
    verdict: ReliabilityVerdict,
) -> str:
    """Generate a JSON report."""
    report: dict[str, Any] = {
        "estimates": estimates,
        "verdict": {
            "reliable": verdict.reliable,
            "severity": verdict.severity,
            "warnings": verdict.warnings,
            "stats": verdict.stats,
        },
    }
    return json.dumps(report, indent=2)
