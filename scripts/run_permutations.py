#!/usr/bin/env python3
"""
Run the investment committee pipeline across multiple provider/ticker permutations.
Outputs a summary report to ~/Desktop.
"""

import sys
import os
import time
import traceback
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings, LLMProvider
from app import create_model
from tools.data_aggregator import DataAggregator
from orchestrator.committee import InvestmentCommittee


# ── Permutations ─────────────────────────────────────────────────
TICKERS = ["NVDA", "COST"]

PROVIDERS = [
    {"provider": LLMProvider.ANTHROPIC, "model": "claude-sonnet-4-20250514", "label": "Anthropic (Sonnet)"},
    {"provider": LLMProvider.GOOGLE,    "model": "gemini-2.0-flash",          "label": "Google (Gemini Flash)"},
    {"provider": LLMProvider.OLLAMA,    "model": "llama3.1:8b",              "label": "Ollama (Llama 3.1 8B)"},
]

# ── Runner ───────────────────────────────────────────────────────

def run_one(ticker: str, provider_info: dict) -> dict:
    """Run a single permutation. Returns a result dict."""
    label = provider_info["label"]
    print(f"\n{'='*60}")
    print(f"  {label} × {ticker}")
    print(f"{'='*60}")

    result_info = {
        "ticker": ticker,
        "provider": label,
        "model": provider_info["model"],
        "status": "FAILED",
        "error": None,
        "duration_s": 0,
        "tokens": 0,
        "recommendation": None,
        "conviction": None,
        "t_signal": None,
        "position_direction": None,
        "raw_confidence": None,
        "bull_conviction": None,
        "bear_conviction": None,
        "macro_favorability": None,
        "sharpe_heuristic": None,
        "sortino_heuristic": None,
        "optimizer_success": None,
        "bl_optimal_weight": None,
        "bl_sharpe": None,
        "bl_sortino": None,
    }

    t0 = time.time()
    try:
        # Override settings for this run
        settings.llm_provider = provider_info["provider"]

        # Create model
        model = create_model(
            provider=provider_info["provider"],
            model_name=provider_info["model"],
        )

        # Gather data (same for all providers)
        print(f"  Gathering data for {ticker}...")
        context = DataAggregator.gather_context(ticker=ticker, user_context="")

        # Run committee
        print(f"  Running committee pipeline...")
        committee = InvestmentCommittee(model=model)
        result = committee.run(
            ticker=ticker,
            context=context,
            on_status=lambda msg: print(f"    [{label}] {msg}"),
        )

        elapsed = time.time() - t0
        result_info["duration_s"] = round(elapsed, 1)
        result_info["tokens"] = result.total_tokens

        # Extract memo fields
        memo = result.committee_memo
        if memo:
            result_info["status"] = "OK"
            result_info["recommendation"] = memo.recommendation
            result_info["conviction"] = memo.conviction
            result_info["t_signal"] = getattr(memo, "t_signal", None)
            result_info["position_direction"] = getattr(memo, "position_direction", None)
            result_info["raw_confidence"] = getattr(memo, "raw_confidence", None)
            result_info["sharpe_heuristic"] = getattr(memo, "sharpe_ratio", None)
            result_info["sortino_heuristic"] = getattr(memo, "sortino_ratio", None)
        else:
            result_info["status"] = "NO MEMO"

        # Bull/bear
        if result.bull_case:
            result_info["bull_conviction"] = result.bull_case.conviction_score
        if result.bear_case:
            result_info["bear_conviction"] = result.bear_case.bearish_conviction

        # Macro
        if result.macro_view:
            result_info["macro_favorability"] = result.macro_view.macro_favorability

        # Optimizer
        opt = result.optimization_result
        if opt and hasattr(opt, "success"):
            result_info["optimizer_success"] = opt.success
            if opt.success:
                result_info["bl_optimal_weight"] = getattr(opt, "optimal_weight", None)
                result_info["bl_sharpe"] = getattr(opt, "computed_sharpe", None)
                result_info["bl_sortino"] = getattr(opt, "computed_sortino", None)
        elif opt and hasattr(opt, "error_message"):
            result_info["optimizer_success"] = False

        print(f"  Done in {elapsed:.1f}s — {result_info['status']}")

    except Exception as e:
        elapsed = time.time() - t0
        result_info["duration_s"] = round(elapsed, 1)
        result_info["error"] = f"{type(e).__name__}: {e}"
        print(f"  FAILED after {elapsed:.1f}s: {e}")
        traceback.print_exc()

    return result_info


def format_report(results: list[dict]) -> str:
    """Format results into a markdown report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    total_time = sum(r["duration_s"] for r in results)

    lines = [
        f"# IC Pipeline Permutation Report",
        f"",
        f"**Date:** {now}",
        f"**Permutations:** {len(results)} ({len(TICKERS)} tickers x {len(PROVIDERS)} providers)",
        f"**Total runtime:** {total_time:.0f}s ({total_time/60:.1f} min)",
        f"**Version:** v3.7.0",
        f"",
        f"---",
        f"",
        f"## Summary Table",
        f"",
        f"| Ticker | Provider | Status | Duration | Tokens | T Signal | Recommendation | Conviction |",
        f"|--------|----------|--------|----------|--------|----------|----------------|------------|",
    ]

    for r in results:
        t_sig = f"{r['t_signal']:+.2f}" if r['t_signal'] is not None else "—"
        conv = f"{r['conviction']}/10" if r['conviction'] is not None else "—"
        tok = f"{r['tokens']:,}" if r['tokens'] else "—"
        rec = r['recommendation'] or "—"
        if len(rec) > 30:
            rec = rec[:27] + "..."
        lines.append(
            f"| {r['ticker']} | {r['provider']} | {r['status']} | {r['duration_s']}s | {tok} | {t_sig} | {rec} | {conv} |"
        )

    lines += [
        f"",
        f"---",
        f"",
        f"## Detailed Results",
        f"",
    ]

    for r in results:
        lines.append(f"### {r['ticker']} — {r['provider']}")
        lines.append(f"")

        if r["status"] == "FAILED":
            lines.append(f"**Error:** `{r['error']}`")
            lines.append(f"")
            continue

        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Model | `{r['model']}` |")
        lines.append(f"| Duration | {r['duration_s']}s |")
        lines.append(f"| Tokens | {r['tokens']:,} |")
        lines.append(f"| Recommendation | {r['recommendation'] or '—'} |")
        lines.append(f"| PM Conviction | {r['conviction']}/10 |" if r['conviction'] is not None else f"| PM Conviction | — |")
        lines.append(f"| T Signal | {r['t_signal']:+.3f} |" if r['t_signal'] is not None else f"| T Signal | — |")
        lines.append(f"| Position Direction | {r['position_direction'] or '—'} |")
        lines.append(f"| Raw Confidence | {r['raw_confidence']:.2f} |" if r['raw_confidence'] is not None else f"| Raw Confidence | — |")
        lines.append(f"| Bull Conviction | {r['bull_conviction']}/10 |" if r['bull_conviction'] is not None else f"| Bull Conviction | — |")
        lines.append(f"| Bear Conviction | {r['bear_conviction']}/10 |" if r['bear_conviction'] is not None else f"| Bear Conviction | — |")
        lines.append(f"| Macro Favorability | {r['macro_favorability']} |" if r['macro_favorability'] is not None else f"| Macro Favorability | — |")

        # Heuristic metrics
        if r['sharpe_heuristic'] is not None:
            lines.append(f"| Sharpe (heuristic) | {r['sharpe_heuristic']:.2f} |")
        if r['sortino_heuristic'] is not None:
            lines.append(f"| Sortino (heuristic) | {r['sortino_heuristic']:.2f} |")

        # Optimizer
        if r['optimizer_success'] is True:
            lines.append(f"| **BL Optimizer** | **Success** |")
            if r['bl_optimal_weight'] is not None:
                lines.append(f"| BL Optimal Weight | {r['bl_optimal_weight']:.1%} |")
            if r['bl_sharpe'] is not None:
                lines.append(f"| BL Sharpe (computed) | {r['bl_sharpe']:.2f} |")
            if r['bl_sortino'] is not None:
                lines.append(f"| BL Sortino (computed) | {r['bl_sortino']:.2f} |")
        elif r['optimizer_success'] is False:
            lines.append(f"| BL Optimizer | Fallback (graceful) |")
        else:
            lines.append(f"| BL Optimizer | Not run |")

        lines.append(f"")

    # Cross-provider comparison
    lines += [
        f"---",
        f"",
        f"## Cross-Provider Comparison",
        f"",
    ]

    for ticker in TICKERS:
        ticker_results = [r for r in results if r["ticker"] == ticker and r["status"] == "OK"]
        if not ticker_results:
            continue
        lines.append(f"### {ticker}")
        lines.append(f"")
        lines.append(f"| Metric | " + " | ".join(r["provider"] for r in ticker_results) + " |")
        lines.append(f"|--------| " + " | ".join("---" for _ in ticker_results) + " |")

        for metric, key, fmt in [
            ("T Signal", "t_signal", lambda v: f"{v:+.2f}" if v is not None else "—"),
            ("Conviction", "conviction", lambda v: f"{v}/10" if v is not None else "—"),
            ("Bull Conv.", "bull_conviction", lambda v: f"{v}/10" if v is not None else "—"),
            ("Bear Conv.", "bear_conviction", lambda v: f"{v}/10" if v is not None else "—"),
            ("Duration", "duration_s", lambda v: f"{v}s"),
            ("Tokens", "tokens", lambda v: f"{v:,}" if v else "—"),
            ("BL Optimizer", "optimizer_success", lambda v: "OK" if v is True else ("Fallback" if v is False else "—")),
        ]:
            vals = " | ".join(fmt(r[key]) for r in ticker_results)
            lines.append(f"| {metric} | {vals} |")
        lines.append(f"")

    return "\n".join(lines)


def main():
    results = []

    for ticker in TICKERS:
        for prov in PROVIDERS:
            info = run_one(ticker, prov)
            results.append(info)

    # Write report
    report = format_report(results)
    out_path = os.path.expanduser("~/Desktop/ic-permutation-report.md")
    with open(out_path, "w") as f:
        f.write(report)

    print(f"\n{'='*60}")
    print(f"  Report written to {out_path}")
    print(f"  {len(results)} permutations, {sum(1 for r in results if r['status']=='OK')} succeeded")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
