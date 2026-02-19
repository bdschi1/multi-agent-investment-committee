"""
Investor risk tolerance profiles for committee evaluation.

Defines conservative, moderate, and aggressive risk profiles with
allocation constraints, loss tolerance thresholds, and position sizing
limits.  These profiles enable evaluating whether committee
recommendations respect different investor mandates.

Inspired by:
    - FinRobot (2024): Risk-based portfolio allocation constraints
    - Regulatory guidelines (MiFID II suitability, SEC Reg BI)

Each profile specifies:
    - Max single-position weight
    - Max sector concentration
    - Max portfolio volatility target
    - Loss tolerance (max drawdown before forced de-risk)
    - Allowed recommendation buckets
    - Conviction thresholds for position sizing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class RiskTolerance(str, Enum):
    """Investor risk tolerance levels."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass(frozen=True)
class RiskProfile:
    """Allocation constraints for an investor risk profile."""

    tolerance: RiskTolerance
    label: str

    # Position limits
    max_single_position_pct: float  # Max weight for one name
    max_sector_concentration_pct: float  # Max weight for one sector
    max_short_exposure_pct: float  # Max gross short exposure

    # Volatility and drawdown
    target_annual_vol_pct: float  # Target portfolio volatility
    max_drawdown_pct: float  # Max acceptable drawdown

    # Recommendation filters
    allowed_buckets: tuple[str, ...]  # Allowed recommendation buckets
    min_conviction_to_size: float  # Min conviction to take a position
    max_conviction_override: float  # Max conviction (cap)

    # Leverage
    max_gross_leverage: float  # 1.0 = no leverage

    def allows_recommendation(self, bucket: str) -> bool:
        """Check if a recommendation bucket is permitted."""
        return bucket.upper() in (b.upper() for b in self.allowed_buckets)

    def position_size(self, conviction: float) -> float:
        """Compute position size as pct of portfolio given conviction.

        Returns 0 if conviction below threshold. Linear scale from
        min_conviction_to_size up to max_single_position_pct.
        """
        if conviction < self.min_conviction_to_size:
            return 0.0
        capped = min(conviction, self.max_conviction_override)
        scale = (capped - self.min_conviction_to_size) / (
            10.0 - self.min_conviction_to_size
        )
        return round(scale * self.max_single_position_pct, 2)

    def validate_allocation(self, allocation: Dict[str, float]) -> List[str]:
        """Check an allocation dict for constraint violations.

        Args:
            allocation: Mapping of ticker → portfolio weight (0-100).

        Returns:
            List of violation descriptions (empty = compliant).
        """
        violations: List[str] = []

        for ticker, weight in allocation.items():
            if abs(weight) > self.max_single_position_pct:
                violations.append(
                    f"{ticker} weight {weight:.1f}% exceeds "
                    f"max {self.max_single_position_pct:.1f}%"
                )

        short_total = sum(abs(w) for w in allocation.values() if w < 0)
        if short_total > self.max_short_exposure_pct:
            violations.append(
                f"Short exposure {short_total:.1f}% exceeds "
                f"max {self.max_short_exposure_pct:.1f}%"
            )

        gross = sum(abs(w) for w in allocation.values())
        if gross > self.max_gross_leverage * 100:
            violations.append(
                f"Gross leverage {gross:.1f}% exceeds "
                f"max {self.max_gross_leverage * 100:.1f}%"
            )

        return violations

    def to_dict(self) -> dict:
        return {
            "tolerance": self.tolerance.value,
            "label": self.label,
            "max_single_position_pct": self.max_single_position_pct,
            "max_sector_concentration_pct": self.max_sector_concentration_pct,
            "max_short_exposure_pct": self.max_short_exposure_pct,
            "target_annual_vol_pct": self.target_annual_vol_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "allowed_buckets": list(self.allowed_buckets),
            "min_conviction_to_size": self.min_conviction_to_size,
            "max_conviction_override": self.max_conviction_override,
            "max_gross_leverage": self.max_gross_leverage,
        }


# ---------------------------------------------------------------------------
# Default profiles
# ---------------------------------------------------------------------------

CONSERVATIVE = RiskProfile(
    tolerance=RiskTolerance.CONSERVATIVE,
    label="Conservative — Capital Preservation",
    max_single_position_pct=5.0,
    max_sector_concentration_pct=20.0,
    max_short_exposure_pct=0.0,
    target_annual_vol_pct=8.0,
    max_drawdown_pct=10.0,
    allowed_buckets=("STRONG BUY", "BUY", "HOLD"),
    min_conviction_to_size=7.0,
    max_conviction_override=9.0,
    max_gross_leverage=1.0,
)

MODERATE = RiskProfile(
    tolerance=RiskTolerance.MODERATE,
    label="Moderate — Balanced Growth",
    max_single_position_pct=10.0,
    max_sector_concentration_pct=30.0,
    max_short_exposure_pct=10.0,
    target_annual_vol_pct=15.0,
    max_drawdown_pct=20.0,
    allowed_buckets=("STRONG BUY", "BUY", "HOLD", "UNDERWEIGHT", "SELL"),
    min_conviction_to_size=5.0,
    max_conviction_override=10.0,
    max_gross_leverage=1.3,
)

AGGRESSIVE = RiskProfile(
    tolerance=RiskTolerance.AGGRESSIVE,
    label="Aggressive — Alpha Maximization",
    max_single_position_pct=20.0,
    max_sector_concentration_pct=40.0,
    max_short_exposure_pct=30.0,
    target_annual_vol_pct=25.0,
    max_drawdown_pct=35.0,
    allowed_buckets=("STRONG BUY", "BUY", "HOLD", "UNDERWEIGHT", "SELL", "ACTIVE SHORT", "AVOID"),
    min_conviction_to_size=3.0,
    max_conviction_override=10.0,
    max_gross_leverage=2.0,
)

PROFILES: Dict[str, RiskProfile] = {
    RiskTolerance.CONSERVATIVE.value: CONSERVATIVE,
    RiskTolerance.MODERATE.value: MODERATE,
    RiskTolerance.AGGRESSIVE.value: AGGRESSIVE,
}


def get_profile(tolerance: str | RiskTolerance) -> RiskProfile:
    """Look up a risk profile by tolerance level."""
    key = tolerance.value if isinstance(tolerance, RiskTolerance) else tolerance
    if key not in PROFILES:
        raise ValueError(
            f"Unknown risk tolerance '{key}'. "
            f"Choose from: {list(PROFILES.keys())}"
        )
    return PROFILES[key]


def check_recommendation_suitability(
    recommendation: str,
    conviction: float,
    tolerance: str | RiskTolerance,
) -> Dict[str, Any]:
    """Check if a recommendation is suitable for a given risk profile.

    Returns:
        Dict with 'suitable' bool, 'violations' list, and 'position_size'.
    """
    profile = get_profile(tolerance)
    violations: List[str] = []

    if not profile.allows_recommendation(recommendation):
        violations.append(
            f"'{recommendation}' not allowed for {profile.tolerance.value} profile"
        )

    position = profile.position_size(conviction)

    return {
        "suitable": len(violations) == 0,
        "violations": violations,
        "position_size_pct": position,
        "profile": profile.tolerance.value,
        "recommendation": recommendation,
        "conviction": conviction,
    }
