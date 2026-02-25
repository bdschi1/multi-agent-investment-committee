#!/usr/bin/env python3
"""
Compute inter-rater reliability (Cohen's Kappa) across scoring methods.

Compares deterministic grader scores vs. LLM-as-judge scores by mapping
both to 5-point Likert scales and computing weighted Cohen's Kappa per
dimension.

Usage:
    python scripts/inter_rater_reliability.py --results evals/results/
    python scripts/inter_rater_reliability.py --results evals/results/ --output kappa_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Boundaries for mapping 0-100 scores to Likert 1-5
_LIKERT_BOUNDARIES = [0, 25, 50, 70, 80]

# Boundaries for mapping 0-1 LLM judge scores to Likert 1-5
_LLM_LIKERT_BOUNDARIES = [0.0, 0.25, 0.50, 0.70, 0.80]

# Kappa interpretation (Landis & Koch 1977)
_KAPPA_INTERPRETATION = [
    (0.0, "Poor"),
    (0.20, "Slight"),
    (0.40, "Fair"),
    (0.60, "Moderate"),
    (0.80, "Substantial"),
    (1.0, "Almost Perfect"),
]


def _score_to_likert(score: float, boundaries: list[float]) -> int:
    """Map a score to Likert 1-5 using boundaries."""
    level = 1
    for i, threshold in enumerate(boundaries):
        if score >= threshold:
            level = i + 1
    return level


def _interpret_kappa(kappa: float) -> str:
    """Return Landis & Koch interpretation for a kappa value."""
    for threshold, label in _KAPPA_INTERPRETATION:
        if kappa <= threshold:
            return label
    return "Almost Perfect"


def load_results(results_dir: Path) -> list[dict]:
    """Load all grading result JSON files from a directory."""
    results = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            if "dimension_scores" in data and "llm_judge_scores" in data:
                if data["llm_judge_scores"]:  # Skip results without LLM scores
                    results.append(data)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Skipping %s: %s", path.name, exc)
    return results


def compute_kappa(results: list[dict]) -> dict[str, dict]:
    """Compute Cohen's weighted Kappa per dimension.

    Returns dict mapping dimension_id → {kappa, interpretation, n_samples,
    deterministic_mean, llm_mean}.
    """
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        logger.error(
            "scikit-learn required: pip install scikit-learn"
        )
        return {}

    # Collect paired ratings per dimension
    paired: dict[str, list[tuple[int, int]]] = {}

    for result in results:
        llm_scores = result.get("llm_judge_scores", {})
        for dim_score in result.get("dimension_scores", []):
            dim_id = dim_score["dimension_id"]
            if dim_id not in llm_scores:
                continue

            # Deterministic: raw_score 0-100 → Likert 1-5
            det_likert = _score_to_likert(
                dim_score["raw_score"], _LIKERT_BOUNDARIES,
            )
            # LLM: 0-1 → Likert 1-5
            llm_likert = _score_to_likert(
                llm_scores[dim_id], _LLM_LIKERT_BOUNDARIES,
            )

            paired.setdefault(dim_id, []).append((det_likert, llm_likert))

    # Compute kappa per dimension
    report: dict[str, dict] = {}
    for dim_id, pairs in sorted(paired.items()):
        if len(pairs) < 2:
            report[dim_id] = {
                "kappa": None,
                "interpretation": "Insufficient data",
                "n_samples": len(pairs),
            }
            continue

        det_ratings = [p[0] for p in pairs]
        llm_ratings = [p[1] for p in pairs]

        try:
            kappa = cohen_kappa_score(
                det_ratings, llm_ratings,
                weights="linear",
                labels=[1, 2, 3, 4, 5],
            )
        except ValueError:
            kappa = 0.0

        report[dim_id] = {
            "kappa": round(kappa, 3),
            "interpretation": _interpret_kappa(kappa),
            "n_samples": len(pairs),
            "deterministic_mean": round(sum(det_ratings) / len(det_ratings), 2),
            "llm_mean": round(sum(llm_ratings) / len(llm_ratings), 2),
        }

    return report


def print_report(report: dict[str, dict]) -> None:
    """Print a formatted table of kappa results."""
    print(f"\n{'Dimension':<30} {'Kappa':>8} {'Interp.':<16} {'N':>4} {'Det μ':>6} {'LLM μ':>6}")
    print("-" * 78)
    for dim_id, data in report.items():
        kappa_str = f"{data['kappa']:.3f}" if data["kappa"] is not None else "  N/A"
        det_mean = f"{data.get('deterministic_mean', 0):.2f}" if data.get("deterministic_mean") else "  N/A"
        llm_mean = f"{data.get('llm_mean', 0):.2f}" if data.get("llm_mean") else "  N/A"
        print(
            f"{dim_id:<30} {kappa_str:>8} {data['interpretation']:<16} "
            f"{data['n_samples']:>4} {det_mean:>6} {llm_mean:>6}"
        )

    # Identify highest-variance dimensions
    scored = {k: v for k, v in report.items() if v["kappa"] is not None}
    if scored:
        worst = min(scored, key=lambda k: scored[k]["kappa"])
        best = max(scored, key=lambda k: scored[k]["kappa"])
        print(f"\nLowest agreement:  {worst} (κ={scored[worst]['kappa']:.3f}) — prioritize for human review")
        print(f"Highest agreement: {best} (κ={scored[best]['kappa']:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute inter-rater reliability (Cohen's Kappa) between "
        "deterministic grader and LLM-as-judge.",
    )
    parser.add_argument(
        "--results", type=Path, required=True,
        help="Directory containing grading result JSON files",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Optional path to save kappa report as JSON",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not args.results.is_dir():
        print(f"Error: {args.results} is not a directory", file=sys.stderr)
        sys.exit(1)

    results = load_results(args.results)
    if not results:
        print("No results with both deterministic and LLM judge scores found.")
        sys.exit(0)

    print(f"Loaded {len(results)} results with paired scores")

    report = compute_kappa(results)
    if not report:
        print("Could not compute kappa (check scikit-learn installation)")
        sys.exit(1)

    print_report(report)

    if args.output:
        args.output.write_text(json.dumps(report, indent=2))
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
