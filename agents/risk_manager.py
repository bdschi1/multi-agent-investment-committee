"""
Risk Manager Agent — risk assessment with sizing/structuring focus.

v4.0: The Risk Manager no longer generates short theses (that's now the
Short Analyst's job). Instead, the Risk Manager focuses on:
- Position structure recommendation (outright, hedged, paired, options_overlay)
- Stop-loss levels
- Max risk allocation in risk units
- Stress scenario analysis
- Correlation/crowding flags
- 2nd and 3rd order risk effects
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agents.base import (
    AgentRole,
    BaseInvestmentAgent,
    BearCase,
    Rebuttal,
    clean_json_artifacts,
    extract_json,
    retry_extract_json,
)

logger = logging.getLogger(__name__)


class RiskManagerAgent(BaseInvestmentAgent):
    """Risk assessment agent: sizing, structuring, and stress testing."""

    def __init__(self, model: Any, tool_registry: Any = None):
        super().__init__(model=model, role=AgentRole.RISK_MANAGER, tool_registry=tool_registry)

    def think(self, ticker: str, context: dict[str, Any]) -> str:
        """Identify initial risk vectors and areas of concern."""
        market_data = context.get("market_data", {})
        news = context.get("news", [])

        user_context = context.get('user_context', '').strip()
        expert_section = ""
        if user_context:
            expert_section = f"""

EXPERT GUIDANCE FROM COMMITTEE CHAIR:
The following expert context has been provided. You MUST incorporate these
considerations into your risk analysis — they represent domain expertise
that may highlight risks you would otherwise miss:
{user_context}
"""

        user_kb = context.get('user_kb', '').strip()
        kb_section = ""
        if user_kb:
            kb_section = f"""

{user_kb}
"""

        prompt = f"""You are a senior risk manager on an investment committee.
Your job is to BUILD THE BEAR CASE for {ticker}. You are the devil's advocate.

First, THINK about what could go wrong. Consider:
- Macro risks (interest rates, recession, geopolitical)
- Industry/competitive risks
- Company-specific risks (execution, governance, balance sheet)
- Regulatory and legal risks
- Technological disruption risks
- Technical/momentum risks (price action relative to sector and market, moving average trends)
- Valuation risks (is the stock priced for perfection? what if growth disappoints?)

Available market data: {json.dumps(market_data, indent=2, default=str)}
Recent news headlines: {json.dumps(news[:5], default=str) if news else 'None available'}
{expert_section}{kb_section}
Respond with your initial risk assessment:
1. What are the most obvious risks? (Acknowledge them, but these are usually PRICED IN.)
2. What are the HIDDEN risks that most analysts miss? THIS IS YOUR ALPHA.
   Find the non-consensus risk the market is underpricing or ignoring.
3. What second-order effects could cascade from primary risks?
4. What would a worst-case scenario look like?
5. What is the CONSENSUS BEAR VIEW? Where does your risk assessment DIFFER?

RETURN DECOMPOSITION CHALLENGE (heuristic reasoning — not precise computation):
6. ALPHA SKEPTICISM: If a bull case claims X% idiosyncratic return, challenge it:
   - Is the "alpha" actually disguised factor exposure? (e.g., momentum or sector beta)
   - What portion of the expected return depends on the SECTOR moving, not the company?
   - If you strip out industry return, does the stock-specific thesis still hold?
7. DOWNSIDE VOL & SORTINO STRESS TEST: The bull case may understate downside risk:
   - What is the realistic DOWNSIDE volatility? (Not symmetric vol — focus on loss distribution)
   - Stress-test the Sortino: if downside vol is 1.5x what bulls assume, does the
     risk-adjusted return still justify the position?
   - For potential shorts: what is the upside risk (short squeeze, mean reversion)?
8. FACTOR-AS-ALPHA FLAG: Is this stock's return driven by systematic factors
   (momentum, value, quality, size) rather than genuine idiosyncratic alpha?
   If the return decomposes mostly into factor returns, it can be replicated cheaper
   with an ETF — the active position doesn't add value.

ANTI-OBVIOUS REQUIREMENT:
- "Recession risk" or "competition" alone are not actionable insights.
- Dig deeper: WHAT SPECIFIC mechanism creates the risk? WHICH competitor and WHY?
- Seek variant risk views: risks that are underpriced or misunderstood by the market.
- The best short pitches come from seeing what the crowd doesn't.

Think deeply. Your job is to protect capital, not be optimistic."""

        response = self.model(prompt)
        return response if isinstance(response, str) else str(response)

    def plan(self, ticker: str, context: dict[str, Any], thinking: str) -> str:
        """Plan the risk analysis approach."""
        tool_catalog = self.get_tool_catalog()
        tool_section = ""
        if tool_catalog:
            tool_section = f"""

AVAILABLE TOOLS (optional — call any that would strengthen your risk analysis):
{tool_catalog}

Suggested tools for risk analysis: compare_peers, get_insider_activity, get_price_data_extended

To request tool data, add a TOOL_CALLS block at the END of your plan:
TOOL_CALLS:
- tool_name(ticker="{ticker}")
"""

        prompt = f"""Based on your initial risk thinking about {ticker}:

{thinking}

Now PLAN your risk analysis. Specifically:

1. Which risks will you investigate first and why (prioritize by severity)?
2. For each major risk, what 2nd-order effects will you trace?
3. For the most severe 2nd-order effects, what 3rd-order cascades do you see?
4. What data would help you quantify these risks?
5. How will you score overall risk (your framework)?
{tool_section}
Think in CAUSAL CHAINS:
  Primary risk → 2nd order consequence → 3rd order cascade

This chain-of-consequence reasoning is what distinguishes good risk management
from simple risk listing."""

        response = self.model(prompt)
        return response if isinstance(response, str) else str(response)

    def act(self, ticker: str, context: dict[str, Any], plan: str, tool_results: dict[str, Any] | None = None) -> BearCase:
        """Execute risk analysis and produce the bear case."""
        market_data = context.get("market_data", {})
        news = context.get("news", [])
        metrics = context.get("financial_metrics", {})

        user_context = context.get('user_context', '').strip()
        expert_section = ""
        if user_context:
            expert_section = f"""
EXPERT GUIDANCE (incorporate into your risk analysis):
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
ADDITIONAL DATA FROM TOOL CALLS (use this to strengthen your risk analysis):
{json.dumps(tool_results, indent=2, default=str)}
"""

        # XAI pre-screen context (quantitative)
        xai_section = ""
        xai_data = context.get("xai_analysis", {})
        if xai_data:
            narrative = xai_data.get("narrative", "") if isinstance(xai_data, dict) else getattr(xai_data, "narrative", "")
            distress = xai_data.get("distress", {}) if isinstance(xai_data, dict) else {}
            pfd = distress.get("pfd", None) if isinstance(distress, dict) else None
            zone = distress.get("distress_zone", "") if isinstance(distress, dict) else ""
            if narrative:
                pfd_warning = ""
                if pfd is not None and pfd > 0.3:
                    pfd_warning = f"\n⚠️ ELEVATED DISTRESS PROBABILITY: PFD={pfd:.1%}, zone={zone}. Investigate financial health risks."
                xai_section = f"""
XAI QUANTITATIVE PRE-SCREEN (Shapley value analysis — use for risk identification):
{narrative}{pfd_warning}
"""

        prompt = f"""You are executing your risk analysis for {ticker}. BUILD THE RISK ASSESSMENT.

Your analysis plan:
{plan}

Market data: {json.dumps(market_data, indent=2, default=str)}
Financial metrics: {json.dumps(metrics, indent=2, default=str)}
Recent news: {json.dumps(news[:10], default=str) if news else 'None available'}
{expert_section}{kb_section}{tool_data_section}{xai_section}
Produce a STRUCTURED risk assessment with SIZING AND STRUCTURING recommendations.
Think in CAUSAL CHAINS for 2nd/3rd order effects.

NOTE: You are NOT generating a short thesis — that is the Short Analyst's job.
Your role is to ASSESS RISKS, RECOMMEND POSITION STRUCTURE, SET STOP-LOSS LEVELS,
and RUN STRESS SCENARIOS. You are the risk guardian, not the short seller.

VARIANT VIEW REQUIREMENT:
- State what the CONSENSUS BEAR CASE is (what most bears worry about), then go DEEPER.
- Your best risks should be NON-OBVIOUS — things the market is underpricing.
- Avoid lazy risk points ("macro headwinds", "competition"). Be specific: which macro factor,
  which competitor, what mechanism, what timeline.
- The best risk assessments find asymmetry: risks with high impact but low market pricing.

SIZING & STRUCTURING (your PRIMARY v4 deliverable):
- POSITION STRUCTURE: Should this be outright, hedged, paired, or options_overlay?
  Why does the risk profile favor one structure over another?
- STOP-LOSS LEVEL: At what specific price level should the position be cut?
  Reference technical levels, fundamental thresholds, or time-based stops.
- MAX RISK ALLOCATION: How many risk units (1-10 scale) should this consume?
  What is the maximum allocation given the risk profile?
- STRESS SCENARIOS: Run 2-3 stress tests:
  - Rates +100bps scenario: P&L impact
  - Sector -15% scenario: P&L impact
  - Company-specific stress (earnings miss, guidance cut): P&L impact
- CORRELATION FLAGS: Is this position crowded? Are correlations elevated with
  other positions in the book? Flag any crowding or correlation warnings.

RETURN DECOMPOSITION CHALLENGE (heuristic reasoning — not precise computation):
- If a bull thesis claims alpha, challenge the decomposition: how much of the
  expected return is actually sector/industry return vs. true idiosyncratic alpha?
- Stress-test the Sortino ratio: what if downside vol is 1.5-2x the bull's assumption?
  Many blow-ups come from underestimating tail risk in the loss distribution.
- FLAG factor-masquerading-as-alpha: if the stock's returns decompose mostly into
  momentum, value, or quality factors, the "alpha" can be replicated with cheaper
  factor ETFs. The active position doesn't add value — it adds risk.

Example of causal chain thinking:
  Risk: "Rising interest rates"
  → 2nd order: "Higher borrowing costs compress margins by 200bps"
  → 3rd order: "Forced to cut R&D spend, losing competitive position in 18-24 months"

Respond in valid JSON matching this exact schema:
{{
    "ticker": "{ticker}",
    "risks": ["primary risk 1", "primary risk 2", ...],
    "second_order_effects": [
        "If [risk] then [consequence] because [mechanism]",
        ...
    ],
    "third_order_effects": [
        "If [2nd order effect] then [cascade] leading to [ultimate impact]",
        ...
    ],
    "worst_case_scenario": "Narrative description of how things could compound",
    "bearish_conviction": 6.5,
    "key_vulnerabilities": {{"area": "description"}},
    "position_structure": "outright / hedged / paired / options_overlay — rationale",
    "stop_loss_level": "$XXX — rationale (e.g. '3-day close below 200 DMA' or '15% from entry')",
    "max_risk_allocation": "X risk units — rationale",
    "stress_scenarios": [
        {{"scenario": "Rates +100bps", "pnl_impact": "-X% on position"}},
        {{"scenario": "Sector -15%", "pnl_impact": "-X% on position"}},
        {{"scenario": "Earnings miss by 10%", "pnl_impact": "-X% on position"}}
    ],
    "correlation_flags": ["Crowding warning: X% of float shorted", "High correlation with Y position"]
}}

bearish_conviction: 0-10 scale (0 = minimal concern, 10 = maximum bearish conviction)

Be specific. Use numbers. Trace the causal chains.
Respond ONLY with the JSON object, no other text."""

        response = self.model(prompt)
        response_text = response if isinstance(response, str) else str(response)

        try:
            parsed, retried = retry_extract_json(self.model, prompt, response_text, max_retries=1)
            if retried:
                logger.info(f"BearCase JSON extraction required retry for {ticker}")
            return BearCase(**parsed)
        except Exception as e:
            logger.warning(f"Failed to parse bear case JSON: {e}. Using fallback.")
            return BearCase(
                ticker=ticker,
                risks=["Analysis generated but structured parsing failed"],
                second_order_effects=[],
                third_order_effects=[],
                worst_case_scenario=clean_json_artifacts(response_text),
                bearish_conviction=5.0,
                key_vulnerabilities={},
            )

    def reflect(self, ticker: str, result: Any) -> str:
        """Evaluate the quality and depth of the risk analysis."""
        bear_case = result if isinstance(result, BearCase) else result

        prompt = f"""Review the bear case you just produced for {ticker}:

Risks: {bear_case.risks}
2nd Order Effects: {bear_case.second_order_effects}
3rd Order Effects: {bear_case.third_order_effects}
Worst Case: {bear_case.worst_case_scenario}
Bearish Conviction: {bear_case.bearish_conviction}/10

REFLECT on your analysis:
1. Are your causal chains logically sound or are you stretching?
2. Did you miss any major risk categories?
3. Are you being appropriately bearish or catastrophizing?
4. CONVICTION SENSITIVITY — what specific, observable events would:
   a) INCREASE your bearish conviction by 2+ points? (Be specific: "If debt/EBITDA exceeds 4x...")
   b) DECREASE your bearish conviction by 2+ points? (Be specific: "If free cash flow grows 20% YoY...")
   c) Make you flip to BULLISH? What would it take?
5. What data would help validate or invalidate your risk assessment?
6. VARIANT CHECK — are your risks genuinely non-consensus, or are you restating obvious bears?
   The market already knows about "competition" and "macro risk." What are you seeing that others don't?
7. RETURN DECOMPOSITION CHALLENGE CHECK (heuristic, not precise):
   a) Did you adequately challenge whether the bull's "alpha" is real idiosyncratic return
      or disguised factor/sector exposure?
   b) Did you stress-test the downside vol assumption? Sortino is only as good as the
      downside estimate — if you didn't push back on it, the bull gets a free pass.
   c) Did you flag any factor-as-alpha risk? If not, reconsider whether the stock's
      returns can be replicated cheaper with systematic factor exposure.

Be calibrated. Good risk management isn't about being maximally pessimistic —
it's about being accurately pessimistic."""

        response = self.model(prompt)
        return response if isinstance(response, str) else str(response)

    def rebut(self, ticker: str, opposing_view: Any, own_result: Any) -> Rebuttal:
        """Provide sizing/structuring commentary on the debate between Long and Short analysts."""
        bear_case = own_result if isinstance(own_result, BearCase) else own_result

        # opposing_view may be bull_case or short_case depending on context
        prompt = f"""You are the Risk Manager for {ticker}. The Long and Short analysts have debated.
You provide SIZING AND STRUCTURING COMMENTARY — not a thesis debate.

YOUR RISK ASSESSMENT:
Bearish Conviction: {bear_case.bearish_conviction if hasattr(bear_case, 'bearish_conviction') else bear_case.get('bearish_conviction', 'N/A')}/10
Key Risks: {bear_case.risks if hasattr(bear_case, 'risks') else bear_case.get('risks', [])}
Position Structure: {bear_case.position_structure if hasattr(bear_case, 'position_structure') else bear_case.get('position_structure', 'N/A')}

Based on the debate so far, provide your risk sizing feedback:

Respond in valid JSON:
{{
    "points": ["sizing/structuring commentary point 1", "risk sizing feedback 2", ...],
    "concessions": ["where the analysts' debate revealed useful risk info", ...],
    "revised_conviction": 6.5
}}

revised_conviction is YOUR bearish conviction (0-10). Focus on RISK MANAGEMENT, not thesis debate.
Respond ONLY with the JSON object."""

        response = self.model(prompt)
        response_text = response if isinstance(response, str) else str(response)

        try:
            parsed, _ = extract_json(response_text)
            return Rebuttal(
                agent_role=AgentRole.RISK_MANAGER,
                responding_to=AgentRole.SECTOR_ANALYST,
                **parsed,
            )
        except Exception as e:
            logger.warning(f"Failed to parse rebuttal JSON: {e}")
            return Rebuttal(
                agent_role=AgentRole.RISK_MANAGER,
                responding_to=AgentRole.SECTOR_ANALYST,
                points=["Rebuttal generated but structured parsing failed"],
                concessions=[],
            )
