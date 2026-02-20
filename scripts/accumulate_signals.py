#!/usr/bin/env python3
"""
Batch signal accumulation — runs the full IC pipeline on multiple tickers.

Usage:
    python scripts/accumulate_signals.py

Populates store/signals.db with one signal per ticker.
"""

import os
import sys
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("OLLAMA_MODEL", "llama3.1:8b")

from app import run_committee_analysis  # noqa: E402

TICKERS = [
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


def main():
    total = len(TICKERS)
    results = []
    failures = []
    t_start = time.time()

    print(f"Starting signal accumulation: {total} tickers via Ollama llama3.1:8b")
    print("=" * 70)

    for i, (ticker, guidance) in enumerate(TICKERS, 1):
        print(f"\n[{i}/{total}] {ticker}")
        print(f"  Guidance: {guidance}")
        t0 = time.time()

        try:
            outputs = run_committee_analysis(
                ticker=ticker,
                user_context=guidance,
                provider_name="ollama",
                debate_rounds=1,
                model_choice="llama3.1:8b",
            )
            elapsed = time.time() - t0

            # Extract key info from status tab (first output)
            status_text = str(outputs[0]) if outputs else ""
            rec = "UNKNOWN"
            conviction = "?"
            t_signal = "?"
            for line in status_text.split("\n"):
                if "**Recommendation**" in line:
                    rec = line.split("**")[-1].strip().strip("|").strip()
                elif "T Signal" in line and "Trading" in line:
                    t_signal = line.split("**")[3] if line.count("**") >= 4 else "?"
                elif "**Conviction**" in line:
                    conviction = line.split("|")[-2].strip() if "|" in line else "?"

            results.append({
                "ticker": ticker,
                "recommendation": rec,
                "conviction": conviction,
                "t_signal": t_signal,
                "elapsed": elapsed,
                "status": "OK",
            })
            print(f"  Result: {rec} | Conviction: {conviction} | T: {t_signal} | {elapsed:.0f}s")

        except Exception as e:
            elapsed = time.time() - t0
            results.append({
                "ticker": ticker,
                "recommendation": "FAILED",
                "conviction": "-",
                "t_signal": "-",
                "elapsed": elapsed,
                "status": f"ERROR: {e}",
            })
            failures.append((ticker, str(e)))
            print(f"  FAILED: {e} ({elapsed:.0f}s)")

    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"Signal accumulation complete: {total_time:.0f}s total ({total_time/60:.1f} min)")
    print(f"  Succeeded: {total - len(failures)}/{total}")
    if failures:
        print(f"  Failed: {len(failures)}")
        for t, err in failures:
            print(f"    {t}: {err}")

    # Summary table
    print("\n" + "-" * 70)
    print(f"{'Ticker':<8} {'Recommendation':<18} {'Conviction':<12} {'T Signal':<16} {'Time':>6}")
    print("-" * 70)
    for r in results:
        print(f"{r['ticker']:<8} {r['recommendation']:<18} {r['conviction']:<12} {r['t_signal']:<16} {r['elapsed']:>5.0f}s")
    print("-" * 70)


if __name__ == "__main__":
    main()
