"""
Report generation for eval results.

Produces JSON output files and markdown summary reports
from accumulated GradingResult records.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from evals.schemas import GradingResult

logger = logging.getLogger(__name__)

_EVALS_DIR = Path(__file__).parent
_RESULTS_DIR = _EVALS_DIR / "results"


def save_result_json(result: GradingResult, results_dir: Path | None = None) -> Path:
    """Save a GradingResult as a JSON file.

    Filename: ``{scenario_id}_{timestamp}.json``

    Returns:
        Path to the saved file.
    """
    results_dir = results_dir or _RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    ts = result.timestamp.strftime("%Y%m%d_%H%M%S")
    filename = f"{result.scenario_id}_{ts}.json"
    path = results_dir / filename

    with open(path, "w") as f:
        json.dump(result.to_json(), f, indent=2, default=str)

    logger.info("Saved result: %s", path.name)
    return path


def load_historical_results(results_dir: Path | None = None) -> list[GradingResult]:
    """Load all JSON results from the results directory."""
    results_dir = results_dir or _RESULTS_DIR
    results: list[GradingResult] = []

    for path in sorted(results_dir.glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            # Parse timestamp back
            if isinstance(data.get("timestamp"), str):
                data["timestamp"] = datetime.fromisoformat(data["timestamp"])
            results.append(GradingResult(**data))
        except Exception:
            logger.warning("Failed to load result: %s", path.name, exc_info=True)

    return results


def generate_summary_table(results: list[GradingResult]) -> str:
    """Generate a markdown summary table from results."""
    if not results:
        return "No results to display."

    lines = [
        "| Scenario | Ticker | Type | Score | Pass | Critical Failures |",
        "|----------|--------|------|------:|:----:|-------------------|",
    ]

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        failures = ", ".join(r.critical_failures[:2]) if r.critical_failures else "-"
        lines.append(
            f"| {r.scenario_id} | {r.ticker} | {r.scenario_type} "
            f"| {r.total_score:.1f} | {status} | {failures} |"
        )

    return "\n".join(lines)


def _likert_distribution(levels: list[int]) -> str:
    """Format a compact Likert distribution string.

    Example: ``1:0 2:1 3:2 4:3 5:1`` showing count at each level.
    """
    counts = {i: 0 for i in range(1, 6)}
    for lv in levels:
        counts[lv] = counts.get(lv, 0) + 1
    return " ".join(f"{k}:{v}" for k, v in sorted(counts.items()))


def generate_run_report(results: list[GradingResult]) -> str:
    """Generate a full markdown report from results."""
    if not results:
        return "# Eval Report\n\nNo results to report."

    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    sections = [f"# Eval Report — {ts}\n"]

    # Summary stats
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    avg_score = sum(r.total_score for r in results) / total if total else 0

    sections.append("## Summary\n")
    sections.append(f"- **Scenarios run:** {total}")
    sections.append(f"- **Pass rate:** {passed}/{total} ({passed/total*100:.0f}%)")
    sections.append(f"- **Average score:** {avg_score:.1f}/100")

    adversarial = [r for r in results if r.scenario_type == "adversarial"]
    if adversarial:
        adv_avg = sum(r.total_score for r in adversarial) / len(adversarial)
        sections.append(f"- **Adversarial avg score:** {adv_avg:.1f}/100")

    sections.append("")

    # Per-dimension averages with Likert distribution
    dim_totals: dict[str, list[float]] = {}
    dim_likert: dict[str, list[int]] = {}
    for r in results:
        for ds in r.dimension_scores:
            dim_totals.setdefault(ds.dimension_id, []).append(ds.raw_score)
            if ds.likert_level:
                dim_likert.setdefault(ds.dimension_id, []).append(ds.likert_level.level)

    if dim_totals:
        sections.append("## Dimension Averages\n")
        has_likert = bool(dim_likert)
        if has_likert:
            sections.append(
                "| Dimension | Avg Score | Min | Max | Avg Likert | Distribution |"
            )
            sections.append(
                "|-----------|----------:|----:|----:|-----------:|--------------|"
            )
        else:
            sections.append("| Dimension | Avg Score | Min | Max |")
            sections.append("|-----------|----------:|----:|----:|")
        for dim_id, dim_scores in sorted(dim_totals.items()):
            avg = sum(dim_scores) / len(dim_scores)
            if has_likert and dim_id in dim_likert:
                levels = dim_likert[dim_id]
                avg_likert = sum(levels) / len(levels)
                dist = _likert_distribution(levels)
                sections.append(
                    f"| {dim_id} | {avg:.1f} | {min(dim_scores):.1f} "
                    f"| {max(dim_scores):.1f} | {avg_likert:.1f}/5 | {dist} |"
                )
            elif has_likert:
                sections.append(
                    f"| {dim_id} | {avg:.1f} | {min(dim_scores):.1f} "
                    f"| {max(dim_scores):.1f} | - | - |"
                )
            else:
                sections.append(
                    f"| {dim_id} | {avg:.1f} | {min(dim_scores):.1f} "
                    f"| {max(dim_scores):.1f} |"
                )
        sections.append("")

    # Results table
    sections.append("## Results\n")
    sections.append(generate_summary_table(results))
    sections.append("")

    # Per-scenario detail
    sections.append("## Scenario Details\n")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        sections.append(
            f"### {r.scenario_id} ({r.ticker}) — {status} ({r.total_score:.1f})\n"
        )

        for ds in r.dimension_scores:
            likert_tag = ""
            if ds.likert_level:
                likert_tag = f" [{ds.likert_level.label}]"
            sections.append(
                f"- **{ds.dimension_name}** ({ds.weight:.0f}w): "
                f"{ds.raw_score:.0f}/100{likert_tag} — {ds.explanation}"
            )
            if ds.likert_level and ds.likert_level.anchor:
                sections.append(f"  - *Anchor: {ds.likert_level.anchor}*")

        if r.critical_failures:
            sections.append(
                f"\n**Critical failures:** {', '.join(r.critical_failures)}"
            )

        summary = r.committee_result_summary
        if summary.get("recommendation"):
            sections.append(
                f"\n**Committee output:** {summary.get('recommendation', '')} "
                f"(dir={summary.get('position_direction', '?')}, "
                f"conv={summary.get('conviction', '?')}, "
                f"T={summary.get('t_signal', '?')})"
            )
        sections.append("")

    return "\n".join(sections)
