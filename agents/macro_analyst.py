"""
Macro Analyst Agent — provides top-down economic context + portfolio strategy.

This agent acts as a global macro strategist AND portfolio strategist. It
assesses the economic cycle, rate environment, sector rotation, geopolitical
landscape, and cross-asset signals. Additionally, it provides portfolio-level
guidance: annualized vol boundaries, net exposure direction (L/S),
sector/style agnostic positioning, and correlation regime awareness.

It does NOT act as a PM — it provides the strategic overlay that constrains
and informs PM decisions.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agents.base import (
    AgentRole,
    BaseInvestmentAgent,
    MacroView,
    Rebuttal,
    extract_json,
    retry_extract_json,
    clean_json_artifacts,
)

logger = logging.getLogger(__name__)


class MacroAnalystAgent(BaseInvestmentAgent):
    """Top-down macro + portfolio strategist: cycle, rates, rotation, vol regime, net exposure."""

    def __init__(self, model: Any, tool_registry: Any = None):
        super().__init__(model=model, role=AgentRole.MACRO_ANALYST, tool_registry=tool_registry)

    def think(self, ticker: str, context: dict[str, Any]) -> str:
        """Assess the macro landscape and form initial hypotheses."""
        market_data = context.get("market_data", {})
        news = context.get("news", [])

        user_context = context.get('user_context', '').strip()
        expert_section = ""
        if user_context:
            expert_section = f"""

EXPERT GUIDANCE FROM COMMITTEE CHAIR:
The following expert context has been provided. You MUST incorporate these
considerations into your macro analysis — they represent domain expertise
that may highlight macro factors you would otherwise miss:
{user_context}
"""

        user_kb = context.get('user_kb', '').strip()
        kb_section = ""
        if user_kb:
            kb_section = f"""

{user_kb}
"""

        prompt = f"""You are a senior global macro strategist AND portfolio strategist on an investment committee.
Your job is to provide the TOP-DOWN MACRO CONTEXT AND PORTFOLIO-LEVEL STRATEGY GUIDANCE for analyzing {ticker}.

You are NOT building a bull or bear case. You are NOT a PM. You are the strategist who:
(a) Provides the economic backdrop that informs the investment thesis
(b) Sets the portfolio-level guardrails: vol budgets, net exposure, sector/style positioning

Available market data: {json.dumps(market_data, indent=2, default=str)}
Recent news headlines: {json.dumps(news[:5], default=str) if news else 'None available'}
{expert_section}{kb_section}
THINK about the current macro environment AND portfolio strategy. Consider:

MACRO CONTEXT:
1. Where are we in the ECONOMIC CYCLE? (early expansion, mid-cycle, late cycle, recession)
   What evidence supports your assessment?
2. What is the RATE ENVIRONMENT? (tightening, pausing, easing)
   What is the central bank likely to do next and why?
3. SECTOR ROTATION — where does this stock's sector sit?
   Is money flowing into or out of this type of name?
4. GEOPOLITICAL RISKS — what global events could impact this stock?
   Trade wars, sanctions, elections, conflicts, regulatory shifts?
5. CROSS-ASSET SIGNALS — what are bonds, commodities, credit, and FX telling us?
   Do they confirm or contradict equity positioning?
6. What is the CONSENSUS MACRO VIEW? Where does the street think we are in the cycle?
   Where might they be WRONG? The best macro calls come from variant cycle timing.

PORTFOLIO STRATEGY (sector/style AGNOSTIC — you are not biased toward any sector or style):
7. VOLATILITY REGIME: What is the current annualized vol regime?
   - Low vol (<12 VIX): Can take larger positions, but beware complacency
   - Normal (12-18): Standard sizing applies
   - Elevated (18-25): Reduce gross, tighten stops
   - Crisis (>25): Defensive mode, reduce net exposure, increase hedges
   How does the vol regime affect appropriate position sizing for THIS name?
8. NET EXPOSURE GUIDANCE: Given the macro environment, should the book be:
   - Net long (bullish cycle, risk-on)
   - Market neutral (late cycle, uncertain)
   - Net short (recession risk, risk-off)
   What is the appropriate L/S directionality?
9. SECTOR & STYLE ASSESSMENT (be agnostic — don't favor any sector):
   - Growth vs Value rotation: which is working and where are we in the rotation?
   - Large vs Small cap: is the spread widening or narrowing?
   - Defensive vs Cyclical: what does the cycle favor?
   - How does this stock fit within the current style regime?
10. CORRELATION REGIME: Are correlations high (macro-driven, hard to diversify) or
    low (stock-picking environment)? What does this mean for position sizing?
11. POSITION SIZING METHOD (heuristic reasoning — not optimization):
    Given the vol regime and correlation environment, which sizing approach is most appropriate?
    - Proportional to alpha (simple, works in stable regimes)
    - Risk parity (normalize by vol — good when vol dispersion is wide)
    - Mean-variance (penalize vol² — good when vol-conscious)
    - Shrunk mean-variance (blend stock + sector vol — good when estimates are noisy)
    Vol targeting > GMV targeting: vol is persistent and predictable.
    What portfolio annualized vol target makes sense in this regime?

ANTI-OBVIOUS REQUIREMENT:
- "The Fed might raise/cut rates" is not an insight — everyone knows that.
- Dig deeper: what SECOND-ORDER macro effects is the market underpricing?
- Seek non-consensus macro views backed by cross-asset confirmation.

Think deeply. The macro backdrop often matters more than company fundamentals,
especially at inflection points. And the portfolio strategy guardrails prevent
the PM from taking positions that are inappropriate for the regime."""

        response = self.model(prompt)
        return response if isinstance(response, str) else str(response)

    def plan(self, ticker: str, context: dict[str, Any], thinking: str) -> str:
        """Plan the macro analysis approach."""
        tool_catalog = self.get_tool_catalog()
        tool_section = ""
        if tool_catalog:
            tool_section = f"""

AVAILABLE TOOLS (optional — call any that would add macro context):
{tool_catalog}

Suggested tools for macro analysis: get_price_data_extended

To request tool data, add a TOOL_CALLS block at the END of your plan:
TOOL_CALLS:
- tool_name(ticker="{ticker}")
"""

        prompt = f"""Based on your initial macro thinking about {ticker}:

{thinking}

Now PLAN your macro analysis AND portfolio strategy assessment. Specifically:

MACRO ANALYSIS:
1. Which macro dimensions are MOST relevant to this stock and why?
2. How will you assess the economic cycle — what indicators matter?
3. What cross-asset signals will you examine for confirmation/divergence?
4. How will you determine if the macro is a tailwind or headwind for this name?
5. What sector rotation dynamics are relevant, and how do you measure them?
6. What geopolitical risks need quantifying and over what timeframes?

PORTFOLIO STRATEGY:
7. How will you assess the current vol regime and its implications for sizing?
8. What is your framework for determining net exposure direction (L/S)?
9. How will you evaluate sector/style rotation — what signals matter most?
10. What correlation dynamics are relevant and how do they affect diversification?
{tool_section}
Prioritize — not all macro factors are equally relevant for every stock.
A pharma stock cares less about oil prices than an airline stock.
But ALL stocks are affected by the vol regime and correlation environment."""

        response = self.model(prompt)
        return response if isinstance(response, str) else str(response)

    def act(self, ticker: str, context: dict[str, Any], plan: str, tool_results: dict[str, Any] | None = None) -> MacroView:
        """Execute macro analysis and produce the MacroView."""
        market_data = context.get("market_data", {})
        news = context.get("news", [])
        metrics = context.get("financial_metrics", {})

        user_context = context.get('user_context', '').strip()
        expert_section = ""
        if user_context:
            expert_section = f"""
EXPERT GUIDANCE (incorporate into your macro analysis):
{user_context}
"""

        user_kb = context.get('user_kb', '').strip()
        kb_section = ""
        if user_kb:
            kb_section = f"""
{user_kb}
"""

        # Inject dynamic tool results if available
        tool_data_section = ""
        if tool_results:
            tool_data_section = f"""
ADDITIONAL DATA FROM TOOL CALLS (use this for macro context):
{json.dumps(tool_results, indent=2, default=str)}
"""

        prompt = f"""You are executing your macro analysis AND portfolio strategy assessment for {ticker}.
Provide the TOP-DOWN CONTEXT + PORTFOLIO GUARDRAILS.

Your analysis plan:
{plan}

Market data: {json.dumps(market_data, indent=2, default=str)}
Financial metrics: {json.dumps(metrics, indent=2, default=str)}
Recent news: {json.dumps(news[:10], default=str) if news else 'None available'}
{expert_section}{kb_section}{tool_data_section}
Produce a STRUCTURED macro view + portfolio strategy. This is context and guardrails — not a buy/sell recommendation.

Respond in valid JSON matching this exact schema:
{{
    "ticker": "{ticker}",
    "economic_cycle_phase": "early expansion / mid-cycle / late cycle / recession",
    "cycle_evidence": [
        "GDP growth at X% — consistent with mid-cycle",
        "Unemployment at Y% — labor market tight",
        "ISM manufacturing at Z — expansion territory",
        ...
    ],
    "rate_environment": "tightening / pausing / easing / QE",
    "central_bank_outlook": "Fed expected to ... over next 6-12 months because ...",
    "sector_positioning": "This sector (e.g. tech/healthcare/energy) is currently ...",
    "rotation_implications": "Money is flowing toward/away from this sector because ...",
    "geopolitical_risks": [
        "Risk 1: specific geopolitical risk and impact mechanism",
        "Risk 2: ...",
        ...
    ],
    "cross_asset_signals": {{
        "bonds": "10Y yield at X%, signal for ...",
        "credit": "IG/HY spreads at Xbps, indicating ...",
        "commodities": "Oil at $X, gold at $X — signaling ...",
        "fx": "DXY at X — dollar strength/weakness implies ...",
        "volatility": "VIX at X — market pricing in ..."
    }},
    "macro_impact_on_stock": "Net narrative: how the macro backdrop specifically affects {ticker}",
    "macro_favorability": 6.0,
    "tailwinds": ["macro factor helping this stock", ...],
    "headwinds": ["macro factor hurting this stock", ...],
    "annualized_vol_regime": "low (<12) / normal (12-18) / elevated (18-25) / crisis (>25) — current VIX level and regime classification",
    "vol_budget_guidance": "Given the vol regime, appropriate position sizing is: [guidance]. Max single-name weight should be X%. If vol is elevated, reduce gross exposure by Y%.",
    "portfolio_directionality": "net long / market neutral / net short — based on cycle phase, cross-asset signals, and vol regime. Rationale: ...",
    "sector_style_assessment": "Growth vs Value: [which is working]. Large vs Small: [spread dynamics]. Defensive vs Cyclical: [cycle implication]. This stock ({ticker}) fits as a [classification] name in the current rotation.",
    "correlation_regime": "Correlations are [high/moderate/low]. This is a [macro-driven/stock-picking] environment. Implication for {ticker}: [diversification benefit or lack thereof].",
    "sector_avg_volatility": "Estimated annualized vol for this stock's sector is ~X%. This stock's vol is [above/below/in-line] at ~Y%. The spread matters for relative sizing.",
    "recommended_sizing_method": "proportional / risk_parity / mean_variance / shrunk_mean_variance — [rationale]. See sizing framework below.",
    "portfolio_vol_target": "Recommended portfolio annualized vol target: X%. Rationale: [why this level given the regime]. Vol targeting > GMV targeting because vol is persistent and predictable."
}}

macro_favorability: 0-10 scale (0 = extremely hostile macro environment, 10 = perfect macro tailwind)

POSITION SIZING FRAMEWORK (heuristic guidance — not precise optimization):
The PM will use your sizing recommendation as a guardrail. Recommend ONE method:
- PROPORTIONAL (NMV = κ × α): Simple — size proportional to forecasted alpha.
  Best when: you trust alpha estimates and vol is stable.
- RISK PARITY (NMV = κ × α / σ): Normalize by vol so high-vol names get smaller sizes.
  Best when: vol dispersion across names is wide; prevents one name dominating risk.
- MEAN-VARIANCE (NMV = κ × α / σ²): Penalize vol more heavily.
  Best when: you're vol-conscious and want to tilt toward lower-risk alpha.
- SHRUNK MEAN-VARIANCE (NMV = κ × α / [p×σ² + (1-p)×σ²_sector]):
  Blend stock vol with sector vol to reduce estimation error. Shrinkage factor p ∈ [0,1].
  Best when: stock vol estimates are noisy or the stock has limited history.
Vol targeting > GMV targeting: vol is more persistent and predictable than returns.
Setting a portfolio vol target (e.g., 12% annualized) and sizing to stay within it
produces better risk-adjusted outcomes than simply capping gross market value.

PORTFOLIO STRATEGY FIELDS ARE REQUIRED:
- annualized_vol_regime: Classify the current regime and reference actual VIX level
- vol_budget_guidance: Be specific about position sizing constraints
- portfolio_directionality: State the recommended net exposure (L/S) with rationale
- sector_style_assessment: Be sector and style AGNOSTIC — objectively assess where rotations are
- correlation_regime: Assess whether this is a macro-driven or idiosyncratic market

QUANTITATIVE SIZING FIELDS ARE REQUIRED (heuristic estimates, not optimizer outputs):
- sector_avg_volatility: Estimate the sector's annualized vol and compare to this stock's vol
- recommended_sizing_method: Choose ONE method (proportional/risk_parity/mean_variance/shrunk_mean_variance) with rationale
- portfolio_vol_target: Recommend an annualized vol target for the portfolio given the current regime

VARIANT VIEW REQUIREMENT:
- For each dimension, state the CONSENSUS macro view first, then where you differ (if you do).
- Avoid restating what every macro newsletter says. Seek non-consensus signals.
- Cross-asset divergences are often the most powerful variant signals — highlight them.
- If your view IS consensus, say so honestly and explain why consensus might still be actionable.

Be specific. Use real data where available. Think about what a portfolio strategist at
a $50B+ AUM fund would focus on for this name. You are not the PM — you set the
guardrails within which the PM operates.
Respond ONLY with the JSON object, no other text."""

        response = self.model(prompt)
        response_text = response if isinstance(response, str) else str(response)

        try:
            parsed, retried = retry_extract_json(self.model, prompt, response_text, max_retries=1)
            if retried:
                logger.info(f"MacroView JSON extraction required retry for {ticker}")
            return MacroView(**parsed)
        except Exception as e:
            logger.warning(f"Failed to parse macro view JSON: {e}. Using fallback.")
            return MacroView(
                ticker=ticker,
                economic_cycle_phase="Analysis generated but structured parsing failed",
                cycle_evidence=[],
                rate_environment="unknown",
                central_bank_outlook=clean_json_artifacts(response_text),
                macro_favorability=5.0,
            )

    def reflect(self, ticker: str, result: Any) -> str:
        """Evaluate the quality and balance of the macro analysis."""
        macro_view = result if isinstance(result, MacroView) else result

        prompt = f"""Review the macro analysis AND portfolio strategy guidance you just produced for {ticker}:

Cycle Phase: {macro_view.economic_cycle_phase}
Rate Environment: {macro_view.rate_environment}
Sector Positioning: {macro_view.sector_positioning}
Macro Favorability: {macro_view.macro_favorability}/10
Tailwinds: {macro_view.tailwinds}
Headwinds: {macro_view.headwinds}
Vol Regime: {macro_view.annualized_vol_regime if hasattr(macro_view, 'annualized_vol_regime') else 'N/A'}
Portfolio Direction: {macro_view.portfolio_directionality if hasattr(macro_view, 'portfolio_directionality') else 'N/A'}
Sector/Style Assessment: {macro_view.sector_style_assessment if hasattr(macro_view, 'sector_style_assessment') else 'N/A'}
Correlation Regime: {macro_view.correlation_regime if hasattr(macro_view, 'correlation_regime') else 'N/A'}

REFLECT on your analysis:
1. Is your cycle assessment well-calibrated? Are you using the right indicators?
2. Are you too focused on US macro, or have you adequately considered global factors?
3. Are your cross-asset signals internally consistent?
4. Is your macro_favorability score appropriate for this specific stock?
5. CONVICTION SENSITIVITY — what specific macro developments would:
   a) INCREASE favorability by 2+ points? (Be specific: "If 10Y yield drops below 3.5%...")
   b) DECREASE favorability by 2+ points? (Be specific: "If ISM drops below 48...")
   c) Completely FLIP the macro backdrop from tailwind to headwind (or vice versa)?
6. What would a dissenting macro strategist argue? Where is your biggest uncertainty?
7. VARIANT CHECK — is your macro view genuinely different from the Bloomberg consensus,
   or are you restating what every macro commentator says?

PORTFOLIO STRATEGY SELF-CHECK:
8. Is your vol regime classification accurate? Would you trust this reading for actual sizing?
9. Is your net exposure recommendation consistent with the cycle phase and vol regime?
   (e.g., "net long" during "crisis vol" would be inconsistent unless you have a strong reason)
10. Is your sector/style assessment truly AGNOSTIC, or did you drift toward favoring
    the stock's own sector? As a strategist, you must be objective.
11. Does your vol budget guidance give the PM enough specificity to act on?
    "Be careful" is useless. "Max 3% of AUM per name, reduce gross by 20%" is actionable.
12. Is your correlation regime assessment informing the right diversification message?
13. SIZING METHOD CHECK (heuristic, not optimization):
    a) Is your recommended sizing method appropriate for the current vol regime?
       (e.g., shrunk MV is better in high-uncertainty regimes; proportional is fine in stable ones)
    b) Is your sector vol estimate reasonable? Does it align with recent realized vol?
    c) Is your portfolio vol target achievable given the current VIX level and correlation regime?
    d) Would the PM have enough specificity to actually apply your sizing guidance?

Good macro analysis is humble — the economy is complex and surprises are frequent.
Good portfolio strategy is disciplined — the guardrails must hold even when conviction is high.
Are you appropriately uncertain where the data is ambiguous?"""

        response = self.model(prompt)
        return response if isinstance(response, str) else str(response)

    def rebut(self, ticker: str, opposing_view: Any, own_result: Any) -> Rebuttal:
        """Macro Analyst doesn't debate — provides context only. No-op for protocol compliance."""
        return Rebuttal(
            agent_role=AgentRole.MACRO_ANALYST,
            responding_to=AgentRole.SECTOR_ANALYST,
            points=["Macro Analyst provides context, not thesis — does not participate in debate"],
            concessions=[],
        )

    # _extract_json removed — using shared extract_json() from agents.base
