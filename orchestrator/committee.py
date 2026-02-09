"""
Investment Committee Orchestrator.

Coordinates the multi-agent workflow:
    1. Sector Analyst and Risk Manager run in parallel (ThreadPoolExecutor)
    2. Adversarial debate round (rebuttals)
    3. Portfolio Manager synthesizes and decides

This is the core orchestration engine that demonstrates multi-agent
reasoning — not just sequential prompting.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agents.base import (
    BearCase,
    BullCase,
    CommitteeMemo,
    MacroView,
    ReasoningTrace,
    Rebuttal,
)
from agents.sector_analyst import SectorAnalystAgent
from agents.risk_manager import RiskManagerAgent
from agents.portfolio_manager import PortfolioManagerAgent
from agents.macro_analyst import MacroAnalystAgent
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ConvictionSnapshot:
    """A single point in the conviction evolution timeline."""
    phase: str           # e.g. "Initial Analysis", "Post-Debate", "PM Decision"
    agent: str           # e.g. "Sector Analyst", "Risk Manager", "Portfolio Manager"
    score: float         # conviction or risk score (0-10)
    score_type: str      # "conviction" or "risk"


@dataclass
class CommitteeResult:
    """Complete result from an investment committee session."""

    ticker: str
    bull_case: Optional[BullCase] = None
    bear_case: Optional[BearCase] = None
    macro_view: Optional[MacroView] = None
    analyst_rebuttal: Optional[Rebuttal] = None
    risk_rebuttal: Optional[Rebuttal] = None
    committee_memo: Optional[CommitteeMemo] = None
    traces: dict[str, ReasoningTrace] = field(default_factory=dict)
    conviction_timeline: list[ConvictionSnapshot] = field(default_factory=list)
    total_duration_ms: float = 0.0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for UI consumption."""
        return {
            "ticker": self.ticker,
            "bull_case": self.bull_case.model_dump() if self.bull_case else None,
            "bear_case": self.bear_case.model_dump() if self.bear_case else None,
            "macro_view": self.macro_view.model_dump() if self.macro_view else None,
            "analyst_rebuttal": self.analyst_rebuttal.model_dump() if self.analyst_rebuttal else None,
            "risk_rebuttal": self.risk_rebuttal.model_dump() if self.risk_rebuttal else None,
            "committee_memo": self.committee_memo.model_dump() if self.committee_memo else None,
            "conviction_timeline": [
                {"phase": s.phase, "agent": s.agent, "score": s.score, "score_type": s.score_type}
                for s in self.conviction_timeline
            ],
            "total_duration_ms": self.total_duration_ms,
            "total_tokens": self.total_tokens,
        }


class InvestmentCommittee:
    """
    Orchestrates the multi-agent investment committee workflow.

    Architecture:
        ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
        │ Sector       │  │ Risk         │  │ Macro        │
        │ Analyst      │  │ Manager      │  │ Analyst      │
        │ (Bull Case)  │  │ (Bear Case)  │  │ (Top-Down)   │
        └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
               │                  │                  │
               │  ┌────────────┐  │        ┌────────┘
               └─►│  DEBATE   │◄─┘        │ (no debate)
                  │ (Rebuttals)│           │
                  └─────┬──────┘           │
                        │                  │
                  ┌─────▼──────────────────▼──┐
                  │  Portfolio Manager         │
                  │  (Synthesis + Macro Ctx)   │
                  └───────────────────────────┘
    """

    def __init__(self, model: Any):
        """
        Initialize the committee with a shared model.

        Args:
            model: The LLM callable (smolagents model or any callable)
        """
        self.model = model
        self.analyst = SectorAnalystAgent(model=model)
        self.risk_mgr = RiskManagerAgent(model=model)
        self.macro = MacroAnalystAgent(model=model)
        self.pm = PortfolioManagerAgent(model=model)

    def run(
        self,
        ticker: str,
        context: dict[str, Any] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> CommitteeResult:
        """
        Run the full investment committee process.

        Args:
            ticker: Stock ticker to analyze
            context: Additional context (market_data, news, user_context, etc.)
            on_status: Optional callback for status updates (for UI streaming)

        Returns:
            CommitteeResult with all analyses, debate, and final memo
        """
        context = context or {}
        result = CommitteeResult(ticker=ticker)
        t_start = time.time()

        def status(msg: str) -> None:
            logger.info(msg)
            if on_status:
                on_status(msg)

        # ── Phase 1: Parallel Analysis ──────────────────────────────
        status("Phase 1/3: Sector Analyst, Risk Manager, and Macro Analyst analyzing in parallel...")

        with ThreadPoolExecutor(max_workers=3) as executor:
            analyst_future = executor.submit(self.analyst.run, ticker, context)
            risk_future = executor.submit(self.risk_mgr.run, ticker, context)
            macro_future = executor.submit(self.macro.run, ticker, context)

            analyst_result = analyst_future.result()
            risk_result = risk_future.result()
            macro_result = macro_future.result()

        result.bull_case = analyst_result["output"]
        result.bear_case = risk_result["output"]
        result.macro_view = macro_result["output"]
        result.traces["sector_analyst"] = analyst_result["trace"]
        result.traces["risk_manager"] = risk_result["trace"]
        result.traces["macro_analyst"] = macro_result["trace"]

        # Track conviction evolution — Phase 1 snapshots
        result.conviction_timeline.append(ConvictionSnapshot(
            phase="Initial Analysis",
            agent="Sector Analyst",
            score=result.bull_case.conviction_score,
            score_type="conviction",
        ))
        result.conviction_timeline.append(ConvictionSnapshot(
            phase="Initial Analysis",
            agent="Risk Manager",
            score=result.bear_case.risk_score,
            score_type="risk",
        ))
        result.conviction_timeline.append(ConvictionSnapshot(
            phase="Initial Analysis",
            agent="Macro Analyst",
            score=result.macro_view.macro_favorability,
            score_type="favorability",
        ))

        status(
            f"  Bull case: conviction {result.bull_case.conviction_score}/10 | "
            f"Bear case: risk {result.bear_case.risk_score}/10 | "
            f"Macro favorability: {result.macro_view.macro_favorability}/10"
        )

        # ── Phase 2: Adversarial Debate ─────────────────────────────
        status("Phase 2/3: Adversarial debate — agents challenging each other...")

        for round_num in range(1, settings.max_debate_rounds + 1):
            status(f"  Debate round {round_num}/{settings.max_debate_rounds}...")

            with ThreadPoolExecutor(max_workers=2) as executor:
                analyst_reb_future = executor.submit(
                    self.analyst.rebut, ticker, result.bear_case, result.bull_case
                )
                risk_reb_future = executor.submit(
                    self.risk_mgr.rebut, ticker, result.bull_case, result.bear_case
                )

                try:
                    result.analyst_rebuttal = analyst_reb_future.result()
                except Exception as e:
                    logger.warning(f"Analyst rebuttal failed: {e}")
                try:
                    result.risk_rebuttal = risk_reb_future.result()
                except Exception as e:
                    logger.warning(f"Risk rebuttal failed: {e}")

        # Track conviction evolution — Phase 2 snapshots (post-debate)
        if result.analyst_rebuttal and result.analyst_rebuttal.revised_conviction is not None:
            result.conviction_timeline.append(ConvictionSnapshot(
                phase="Post-Debate",
                agent="Sector Analyst",
                score=result.analyst_rebuttal.revised_conviction,
                score_type="conviction",
            ))
        if result.risk_rebuttal and result.risk_rebuttal.revised_conviction is not None:
            result.conviction_timeline.append(ConvictionSnapshot(
                phase="Post-Debate",
                agent="Risk Manager",
                score=result.risk_rebuttal.revised_conviction,
                score_type="risk",
            ))

        status(
            f"  Debate complete | "
            f"Analyst revised conviction: "
            f"{result.analyst_rebuttal.revised_conviction if result.analyst_rebuttal else 'N/A'} | "
            f"Risk Mgr revised risk: "
            f"{result.risk_rebuttal.revised_conviction if result.risk_rebuttal else 'N/A'}"
        )

        # ── Phase 3: PM Synthesis ───────────────────────────────────
        status("Phase 3/3: Portfolio Manager synthesizing committee decision...")

        pm_context = {
            "bull_case": result.bull_case,
            "bear_case": result.bear_case,
            "macro_view": result.macro_view,
            "debate": {
                "analyst_rebuttal": result.analyst_rebuttal.model_dump() if result.analyst_rebuttal else {},
                "risk_rebuttal": result.risk_rebuttal.model_dump() if result.risk_rebuttal else {},
            },
            "user_context": context.get("user_context", ""),
        }

        pm_result = self.pm.run(ticker, pm_context)
        result.committee_memo = pm_result["output"]
        result.traces["portfolio_manager"] = pm_result["trace"]

        # Track conviction evolution — Phase 3 snapshot (PM decision)
        if result.committee_memo:
            result.conviction_timeline.append(ConvictionSnapshot(
                phase="PM Decision",
                agent="Portfolio Manager",
                score=result.committee_memo.conviction,
                score_type="conviction",
            ))

        # Finalize
        result.total_duration_ms = (time.time() - t_start) * 1000
        result.total_tokens = sum(t.total_tokens for t in result.traces.values())

        status(
            f"  Decision: {result.committee_memo.recommendation} | "
            f"Conviction: {result.committee_memo.conviction}/10 | "
            f"Size: {result.committee_memo.position_size}"
        )
        status(
            f"\nCommittee complete in {result.total_duration_ms/1000:.1f}s"
        )

        return result
