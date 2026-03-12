"""Unit tests for evaluation/report.py — Markdown and JSON report generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deeprl_recsys.evaluation.ope.diagnostics import ReliabilityVerdict
from deeprl_recsys.evaluation.report import generate_report


@pytest.fixture()
def sample_verdict() -> ReliabilityVerdict:
    """Create a sample ReliabilityVerdict for testing."""
    return ReliabilityVerdict(
        reliable=True,
        severity="warning",
        warnings=["Low ESS detected"],
        stats={"ess": 15.5, "clipping_rate": 0.05, "n_samples": 100.0},
        policy={"clip_epsilon": 0.01},
    )


@pytest.fixture()
def sample_estimates() -> dict[str, float]:
    return {"ips": 0.456789, "dr": 0.512345}


@pytest.mark.unit
class TestReportMarkdown:
    """Tests for Markdown report generation."""

    def test_markdown_contains_header(
        self, sample_estimates: dict, sample_verdict: ReliabilityVerdict
    ) -> None:
        content = generate_report(sample_estimates, sample_verdict, format="markdown")
        assert "# OPE Evaluation Report" in content

    def test_markdown_contains_estimates(
        self, sample_estimates: dict, sample_verdict: ReliabilityVerdict
    ) -> None:
        content = generate_report(sample_estimates, sample_verdict, format="markdown")
        assert "ips" in content
        assert "0.456789" in content

    def test_markdown_contains_verdict(
        self, sample_estimates: dict, sample_verdict: ReliabilityVerdict
    ) -> None:
        content = generate_report(sample_estimates, sample_verdict, format="markdown")
        assert "warning" in content
        assert "Low ESS" in content

    def test_markdown_saves_to_file(
        self, tmp_path: Path, sample_estimates: dict, sample_verdict: ReliabilityVerdict
    ) -> None:
        generate_report(
            sample_estimates, sample_verdict, output_dir=tmp_path, format="markdown"
        )
        assert (tmp_path / "ope_report.md").exists()


@pytest.mark.unit
class TestReportJSON:
    """Tests for JSON report generation."""

    def test_json_is_valid(
        self, sample_estimates: dict, sample_verdict: ReliabilityVerdict
    ) -> None:
        content = generate_report(sample_estimates, sample_verdict, format="json")
        data = json.loads(content)
        assert "estimates" in data
        assert "verdict" in data
        assert data["estimates"]["ips"] == pytest.approx(0.456789)

    def test_json_saves_to_file(
        self, tmp_path: Path, sample_estimates: dict, sample_verdict: ReliabilityVerdict
    ) -> None:
        generate_report(
            sample_estimates, sample_verdict, output_dir=tmp_path, format="json"
        )
        assert (tmp_path / "ope_report.json").exists()
        data = json.loads((tmp_path / "ope_report.json").read_text())
        assert data["verdict"]["severity"] == "warning"
