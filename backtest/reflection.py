"""
Post-trade reflection engine.

Generates rule-based (or optionally LLM-powered) reflections for signals
that have realized returns but haven't been reflected on yet. Each signal
produces one ReflectionRecord per agent role, capturing what worked, what
failed, and a transferable lesson for the agent's memory.
"""

from __future__ import annotations

import logging
from typing import Any

from backtest.database import SignalDatabase
from backtest.models import ReflectionRecord, SignalRecord

logger = logging.getLogger(__name__)

# Agent roles and how they map to signal fields
_AGENT_CONFIGS = {
    "sector_analyst": {
        "conviction_field": "bull_conviction",
        "direction": 1,  # bull → expects positive returns
    },
    "short_analyst": {
        "conviction_field": "bear_conviction",
        "direction": -1,  # short → expects negative returns
    },
    "risk_manager": {
        "conviction_field": "bear_conviction",
        "direction": -1,  # bearish → expects negative returns
    },
    "macro_analyst": {
        "conviction_field": "macro_favorability",
        "direction": 1,  # favorable macro → expects positive returns
    },
    "portfolio_manager": {
        "conviction_field": "conviction",
        "direction": None,  # uses position_direction from signal
    },
}


class ReflectionEngine:
    """Generates post-trade reflections for unreflected signals."""

    def __init__(self, db: SignalDatabase, model: Any = None):
        self.db = db
        self.model = model  # Optional LLM for richer reflections

    def run_reflections(self, horizon: str = "return_20d") -> int:
        """Generate reflections for all unreflected signals with realized returns.

        Returns:
            Number of reflection records generated.
        """
        unreflected = self.db.get_unreflected_signals(horizon=horizon)
        if not unreflected:
            logger.info("No unreflected signals found for horizon=%s", horizon)
            return 0

        count = 0
        for signal in unreflected:
            actual_return = getattr(signal, horizon, None)
            if actual_return is None:
                continue

            for role, cfg in _AGENT_CONFIGS.items():
                reflection = self._generate_reflection(signal, role, cfg, actual_return, horizon)
                if reflection:
                    self.db.store_reflection(reflection)
                    count += 1

        logger.info("Generated %d reflections for %d signals", count, len(unreflected))
        return count

    def _generate_reflection(
        self,
        signal: SignalRecord,
        role: str,
        cfg: dict,
        actual_return: float,
        horizon: str,
    ) -> ReflectionRecord | None:
        """Generate a single reflection for one agent role on one signal."""
        conviction = getattr(signal, cfg["conviction_field"], 5.0) or 5.0

        # Determine predicted direction
        if cfg["direction"] is not None:
            predicted_direction = cfg["direction"]
        else:
            predicted_direction = signal.position_direction or 0

        # Determine correctness
        if predicted_direction == 0:
            was_correct = 0
        elif predicted_direction > 0:
            was_correct = 1 if actual_return > 0 else 0
        else:
            was_correct = 1 if actual_return < 0 else 0

        if self.model:
            return self._llm_reflection(signal, role, conviction, predicted_direction, actual_return, was_correct, horizon)

        return self._rule_based_reflection(signal, role, conviction, predicted_direction, actual_return, was_correct, horizon)

    def _rule_based_reflection(
        self,
        signal: SignalRecord,
        role: str,
        conviction: float,
        predicted_direction: int,
        actual_return: float,
        was_correct: int,
        horizon: str,
    ) -> ReflectionRecord:
        """Generate a rule-based reflection without LLM."""
        dir_label = "bullish" if predicted_direction > 0 else "bearish" if predicted_direction < 0 else "neutral"
        ret_pct = f"{actual_return:+.1%}"

        if was_correct:
            lesson = (
                f"{role} was correct on {signal.ticker}: predicted {dir_label}, "
                f"actual return {ret_pct} over {horizon}."
            )
            what_worked = f"Conviction {conviction:.1f}/10 aligned with realized {ret_pct} return."
            what_failed = ""
        else:
            lesson = (
                f"{role} was wrong on {signal.ticker}: predicted {dir_label}, "
                f"actual return {ret_pct} over {horizon}."
            )
            what_worked = ""
            what_failed = f"Conviction {conviction:.1f}/10 predicted {dir_label} but actual was {ret_pct}."

        # Calibration assessment
        if conviction >= 8.0:
            if was_correct:
                calibration = "High conviction correctly placed."
            else:
                calibration = "High conviction was misplaced — consider lowering conviction for similar setups."
        elif conviction <= 3.0:
            if was_correct:
                calibration = "Low conviction underestimated the opportunity."
            else:
                calibration = "Low conviction was appropriately cautious."
        else:
            calibration = "Moderate conviction — calibration appears reasonable."

        return ReflectionRecord(
            signal_id=signal.id,
            agent_role=role,
            ticker=signal.ticker,
            horizon=horizon,
            predicted_direction=predicted_direction,
            actual_return=actual_return,
            was_correct=was_correct,
            lesson=lesson,
            what_worked=what_worked,
            what_failed=what_failed,
            confidence_calibration=calibration,
        )

    def _llm_reflection(
        self,
        signal: SignalRecord,
        role: str,
        conviction: float,
        predicted_direction: int,
        actual_return: float,
        was_correct: int,
        horizon: str,
    ) -> ReflectionRecord:
        """Generate an LLM-powered reflection for richer lessons."""
        dir_label = "bullish" if predicted_direction > 0 else "bearish" if predicted_direction < 0 else "neutral"
        prompt = f"""You are a {role} on an investment committee reflecting on a past trade.

Signal: {signal.ticker} on {signal.signal_date.strftime('%Y-%m-%d')}
Your predicted direction: {dir_label} (conviction: {conviction:.1f}/10)
Actual return over {horizon}: {actual_return:+.1%}
Outcome: {"CORRECT" if was_correct else "INCORRECT"}

In 2-3 sentences each, provide:
1. LESSON: What transferable insight should you carry to future analyses?
2. WHAT_WORKED: What part of your analysis was well-calibrated?
3. WHAT_FAILED: What did you get wrong or miss?
4. CALIBRATION: Was your conviction level appropriate for the outcome?

Be specific and actionable. Avoid generic platitudes."""

        try:
            response = self.model(prompt)
            # Parse sections from free-form response
            lesson = self._extract_section(response, "LESSON") or f"{role} {'correct' if was_correct else 'incorrect'} on {signal.ticker}"
            what_worked = self._extract_section(response, "WHAT_WORKED") or ""
            what_failed = self._extract_section(response, "WHAT_FAILED") or ""
            calibration = self._extract_section(response, "CALIBRATION") or ""
        except Exception:
            logger.warning("LLM reflection failed for %s/%s, falling back to rule-based", role, signal.ticker)
            return self._rule_based_reflection(signal, role, conviction, predicted_direction, actual_return, was_correct, horizon)

        return ReflectionRecord(
            signal_id=signal.id,
            agent_role=role,
            ticker=signal.ticker,
            horizon=horizon,
            predicted_direction=predicted_direction,
            actual_return=actual_return,
            was_correct=was_correct,
            lesson=lesson,
            what_worked=what_worked,
            what_failed=what_failed,
            confidence_calibration=calibration,
        )

    @staticmethod
    def _extract_section(text: str, section: str) -> str:
        """Extract a named section from LLM response text."""
        import re
        pattern = rf"(?:{section})\s*[:]\s*(.*?)(?=\n\s*(?:LESSON|WHAT_WORKED|WHAT_FAILED|CALIBRATION)\s*:|$)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""
