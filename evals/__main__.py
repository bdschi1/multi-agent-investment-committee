"""
CLI entry point for the eval harness.

Usage:
    python -m evals run                          # Run all scenarios
    python -m evals run --scenario nvda_2024     # Run one scenario
    python -m evals run --type adversarial       # Run only adversarial
    python -m evals run --tag semiconductors     # Filter by tag
    python -m evals list                         # List available scenarios
    python -m evals report                       # Generate report from results/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

_EVALS_DIR = Path(__file__).parent


def _cmd_list(args: argparse.Namespace) -> None:
    """List available scenarios."""
    from evals.loader import discover_scenarios

    scenarios = discover_scenarios(
        filter_type=args.type,
        filter_tags=args.tag.split(",") if args.tag else None,
    )

    if not scenarios:
        print("No scenarios found.")
        return

    print(f"\n{'ID':<40} {'Ticker':<8} {'Type':<15} {'Difficulty':<10} Tags")
    print("-" * 100)
    for s in scenarios:
        tags = ", ".join(s.tags) if s.tags else "-"
        print(f"{s.id:<40} {s.ticker:<8} {s.type:<15} {s.difficulty:<10} {tags}")
    print(f"\n{len(scenarios)} scenario(s) found.\n")


def _cmd_run(args: argparse.Namespace) -> None:
    """Run eval scenarios."""
    from evals.reporter import generate_summary_table
    from evals.runner import run_all

    results = run_all(
        filter_type=args.type,
        filter_tags=args.tag.split(",") if args.tag else None,
        filter_scenario=args.scenario,
    )

    if not results:
        print("No scenarios were run.")
        return

    # Print summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"  EVAL COMPLETE: {passed}/{total} passed")
    print(f"{'=' * 60}\n")
    print(generate_summary_table(results))
    print()


def _cmd_report(args: argparse.Namespace) -> None:
    """Generate report from existing results."""
    from evals.reporter import generate_run_report, load_historical_results

    results = load_historical_results()
    if not results:
        print("No results found in evals/results/. Run some scenarios first.")
        return

    report = generate_run_report(results)
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(report)
        print(f"Report saved to {out_path}")
    else:
        print(report)


def main() -> None:
    """Parse arguments and dispatch to subcommand."""
    parser = argparse.ArgumentParser(
        prog="python -m evals",
        description="Investment Committee Eval Harness",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- list ---
    p_list = subparsers.add_parser("list", help="List available scenarios")
    p_list.add_argument("--type", choices=["ground_truth", "adversarial"])
    p_list.add_argument("--tag", help="Comma-separated tags to filter by")

    # --- run ---
    p_run = subparsers.add_parser("run", help="Run eval scenarios")
    p_run.add_argument("--scenario", help="Run only this scenario (partial ID match)")
    p_run.add_argument("--type", choices=["ground_truth", "adversarial"])
    p_run.add_argument("--tag", help="Comma-separated tags to filter by")

    # --- report ---
    p_report = subparsers.add_parser("report", help="Generate report from results")
    p_report.add_argument("--output", "-o", help="Save report to file")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    commands = {
        "list": _cmd_list,
        "run": _cmd_run,
        "report": _cmd_report,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
