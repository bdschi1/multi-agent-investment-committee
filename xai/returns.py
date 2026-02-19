"""Steps 3-4: Distress screening and expected return estimation.

Step 3: Screen companies with PFD > threshold as "distressed".
Step 4: Compute expected return = (1 - PFD) * earnings_yield_proxy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DistressScreenResult:
    """Result of distress screening (Step 3)."""

    is_distressed: bool
    flag: str
    pfd: float
    threshold: float


@dataclass
class ReturnEstimate:
    """Expected return estimate (Step 4)."""

    earnings_yield_proxy: float
    expected_return: float
    expected_return_pct: str
    yield_source: str


def screen_distress(pfd: float, threshold: float = 0.5) -> DistressScreenResult:
    """Screen for financial distress based on PFD threshold.

    Args:
        pfd: Probability of Financial Distress [0, 1]
        threshold: PFD cutoff for distress flag (default 0.5)

    Returns:
        DistressScreenResult with screening outcome.
    """
    is_distressed = pfd >= threshold

    if is_distressed:
        flag = f"DISTRESSED: PFD={pfd:.1%} exceeds threshold ({threshold:.0%})"
    elif pfd >= threshold * 0.6:
        flag = f"WATCH: PFD={pfd:.1%} approaching threshold ({threshold:.0%})"
    else:
        flag = f"CLEAR: PFD={pfd:.1%} well below threshold ({threshold:.0%})"

    return DistressScreenResult(
        is_distressed=is_distressed,
        flag=flag,
        pfd=pfd,
        threshold=threshold,
    )


def _get_earnings_yield(fundamentals: dict[str, Any]) -> tuple[float, str]:
    """Extract earnings yield proxy from fundamentals.

    Priority: pe_forward → pe_trailing → profit_margin.
    Returns (yield_value, source_description).
    """
    # Try forward PE first
    pe_fwd = fundamentals.get("pe_forward")
    if pe_fwd is not None and isinstance(pe_fwd, (int, float)) and pe_fwd > 0:
        return 1.0 / float(pe_fwd), "1/pe_forward"

    # Trailing PE
    pe_trail = fundamentals.get("pe_trailing")
    if pe_trail is not None and isinstance(pe_trail, (int, float)) and pe_trail > 0:
        return 1.0 / float(pe_trail), "1/pe_trailing"

    # Profit margin fallback
    pm = fundamentals.get("profit_margin")
    if pm is not None:
        if isinstance(pm, str):
            try:
                pm_val = float(pm.replace("%", "").strip()) / 100.0
                if pm_val > 0:
                    return pm_val, "profit_margin"
            except (ValueError, TypeError):
                pass
        elif isinstance(pm, (int, float)) and pm > 0:
            return float(pm), "profit_margin"

    # Last resort: assume 5% yield
    return 0.05, "default_5pct"


def compute_expected_return(
    pfd: float,
    fundamentals: dict[str, Any],
) -> ReturnEstimate:
    """Compute risk-adjusted expected return.

    ER = (1 - PFD) * earnings_yield_proxy

    Args:
        pfd: Probability of Financial Distress [0, 1]
        fundamentals: Dict from context["financial_metrics"]

    Returns:
        ReturnEstimate with expected return calculation.
    """
    earnings_yield, source = _get_earnings_yield(fundamentals)

    er = (1.0 - pfd) * earnings_yield
    er_pct = f"{er * 100:.1f}%"

    return ReturnEstimate(
        earnings_yield_proxy=earnings_yield,
        expected_return=er,
        expected_return_pct=er_pct,
        yield_source=source,
    )
