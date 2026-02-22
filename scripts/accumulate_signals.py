#!/usr/bin/env python3
"""
Batch signal accumulation — runs the full IC pipeline on multiple tickers
and persists results to the backtest database.

Usage:
    python scripts/accumulate_signals.py
    python scripts/accumulate_signals.py --provider anthropic --model claude-sonnet-4-20250514
    python scripts/accumulate_signals.py --tickers AAPL,NVDA,GOOGL --max-workers 3
    python scripts/accumulate_signals.py --provider ollama --model llama3.1:8b --debate-rounds 1
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.database import SignalDatabase  # noqa: E402
from backtest.persist import persist_signal  # noqa: E402
from config.settings import LLMProvider, settings  # noqa: E402
from orchestrator.graph import run_graph  # noqa: E402
from tools.data_aggregator import DataAggregator  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_TICKERS = [
    # Tech
    ("AAPL", "Focus on iPhone cycle, services growth, and capital returns."),
    ("NVDA", "Focus on AI/datacenter demand, gaming, and valuation after run-up."),
    ("GOOGL", "Focus on search/ads moat, cloud growth, and AI integration."),
    ("META", "Focus on ad revenue recovery, Reels monetization, and Reality Labs spend."),
    # Healthcare
    ("JNJ", "Focus on MedTech growth, pharma pipeline, and Kenvue separation impact."),
    ("LLY", "Focus on GLP-1 drugs (Mounjaro/Zepbound), pipeline, and premium valuation."),
    ("UNH", "Focus on Optum growth, Medicare Advantage, and regulatory risk."),
    # Financials
    ("JPM", "Focus on NII trajectory, credit quality, and capital markets revenue."),
    ("GS", "Focus on trading revenue, asset management pivot, and expense discipline."),
    # Consumer
    ("COST", "Focus on membership growth, same-store sales, and margin expansion."),
    ("AMZN", "Focus on AWS growth reacceleration, retail margins, and ad business."),
    # Energy
    ("XOM", "Focus on Permian production, refining margins, and capital allocation."),
    # Industrials
    ("CAT", "Focus on infrastructure spending, dealer inventory, and pricing power."),
    # Semis
    ("AMD", "Focus on datacenter GPU share gains, AI inference chips, and Xilinx integration."),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch signal accumulation for the investment committee."
    )
    parser.add_argument(
        "--provider", default=None,
        help="LLM provider (anthropic, google, openai, ollama, etc.). "
             "Defaults to LLM_PROVIDER env var or settings.",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model name override (e.g. claude-sonnet-4-20250514, llama3.1:8b).",
    )
    parser.add_argument(
        "--tickers", default=None,
        help="Comma-separated ticker list (overrides defaults). "
             "E.g. --tickers AAPL,NVDA,GOOGL",
    )
    parser.add_argument(
        "--max-workers", type=int, default=2,
        help="Number of parallel ticker threads (default: 2).",
    )
    parser.add_argument(
        "--debate-rounds", type=int, default=1,
        help="Number of adversarial debate rounds (default: 1).",
    )
    parser.add_argument(
        "--db-path", default=None,
        help="SQLite database path (default: store/signals.db).",
    )
    return parser.parse_args()


def analyze_ticker(ticker, guidance, model, debate_rounds):
    """Run the full pipeline for a single ticker. Returns CommitteeResult."""
    context = DataAggregator.gather_context(ticker, guidance)
    return run_graph(
        ticker=ticker,
        context=context,
        model=model,
        max_debate_rounds=debate_rounds,
    )


def main():
    args = parse_args()

    # Resolve provider
    provider_str = args.provider or os.environ.get("LLM_PROVIDER") or settings.llm_provider.value
    provider = LLMProvider(provider_str)

    # Create model via app factory (handles rate limiting)
    from app import create_model
    model = create_model(provider, model_name=args.model)
    resolved_model = args.model or settings.get_active_model()

    # Resolve ticker list
    if args.tickers:
        tickers = [(t.strip(), "") for t in args.tickers.split(",")]
    else:
        tickers = DEFAULT_TICKERS

    # Database connection (shared across batch)
    db = SignalDatabase(db_path=args.db_path) if args.db_path else SignalDatabase()

    total = len(tickers)
    results = []
    failures = []
    t_start = time.time()

    print(f"Starting signal accumulation: {total} tickers via {provider_str} / {resolved_model}")
    print(f"Parallel workers: {args.max_workers} | Debate rounds: {args.debate_rounds}")
    print("=" * 70)

    def _process_one(ticker_guidance):
        ticker, guidance = ticker_guidance
        t0 = time.time()
        try:
            result = analyze_ticker(ticker, guidance, model, args.debate_rounds)
            elapsed = time.time() - t0

            signal_id = persist_signal(result, provider_str, resolved_model, db=db)

            memo = result.committee_memo
            return {
                "ticker": ticker,
                "status": "OK",
                "recommendation": memo.recommendation if memo else "N/A",
                "conviction": memo.conviction if memo else 0,
                "t_signal": round(memo.t_signal, 4) if memo else 0,
                "signal_id": signal_id,
                "elapsed": elapsed,
            }
        except Exception as e:
            elapsed = time.time() - t0
            logger.exception(f"Failed: {ticker}")
            return {
                "ticker": ticker,
                "status": f"ERROR: {e}",
                "recommendation": "FAILED",
                "conviction": "-",
                "t_signal": "-",
                "signal_id": None,
                "elapsed": elapsed,
            }

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(_process_one, tg): tg[0] for tg in tickers}
        for i, future in enumerate(as_completed(futures), 1):
            ticker = futures[future]
            r = future.result()
            results.append(r)
            ok = r["status"] == "OK"
            if not ok:
                failures.append((ticker, r["status"]))
            print(
                f"[{i}/{total}] {ticker}: {'OK' if ok else 'FAIL'} | "
                f"{r['recommendation']} | T={r['t_signal']} | {r['elapsed']:.0f}s"
            )

    db.close()

    # Summary
    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"Complete: {total_time:.0f}s ({total_time / 60:.1f} min)")
    print(f"Succeeded: {total - len(failures)}/{total}")
    if failures:
        print(f"Failed ({len(failures)}):")
        for t, err in failures:
            print(f"  {t}: {err}")

    print(f"\n{'Ticker':<8} {'Rec':<18} {'Conv':<8} {'T Signal':<12} {'DB ID':<8} {'Time':>6}")
    print("-" * 62)
    for r in sorted(results, key=lambda x: x["ticker"]):
        print(
            f"{r['ticker']:<8} {str(r['recommendation']):<18} {str(r['conviction']):<8} "
            f"{str(r['t_signal']):<12} {str(r['signal_id'] or '-'):<8} {r['elapsed']:>5.0f}s"
        )


if __name__ == "__main__":
    main()
