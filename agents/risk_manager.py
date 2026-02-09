"""
Risk Manager Agent — builds the bear case with 2nd and 3rd order effects.

This agent acts as a senior risk manager / devil's advocate, identifying
what could go wrong and tracing cascading consequences that most
analysts miss.
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
    StepType,
)

logger = logging.getLogger(__name__)


class RiskManagerAgent(BaseInvestmentAgent):
    """Bear-case agent: adversarial risk analysis with cascading effects."""

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
{expert_section}
Respond with your initial risk assessment:
1. What are the most obvious risks? (Acknowledge them, but these are usually PRICED IN.)
2. What are the HIDDEN risks that most analysts miss? THIS IS YOUR ALPHA.
   Find the non-consensus risk the market is underpricing or ignoring.
3. What second-order effects could cascade from primary risks?
4. What would a worst-case scenario look like?
5. What is the CONSENSUS BEAR VIEW? Where does your risk assessment DIFFER?

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

        # Inject dynamic tool results if available
        tool_data_section = ""
        if tool_results:
            tool_data_section = f"""
ADDITIONAL DATA FROM TOOL CALLS (use this to strengthen your risk analysis):
{json.dumps(tool_results, indent=2, default=str)}
"""

        prompt = f"""You are executing your risk analysis for {ticker}. BUILD THE BEAR CASE.

Your analysis plan:
{plan}

Market data: {json.dumps(market_data, indent=2, default=str)}
Financial metrics: {json.dumps(metrics, indent=2, default=str)}
Recent news: {json.dumps(news[:10], default=str) if news else 'None available'}
{expert_section}{tool_data_section}
Produce a STRUCTURED bear case. Think in CAUSAL CHAINS for 2nd/3rd order effects.

IMPORTANT: Your output should NOT default to "avoid/no position." If the data supports it,
produce an ACTIVE SHORT PITCH — a real alpha-generating short thesis that a fund would trade.
Consider: Is this stock overvalued? Is the narrative ahead of fundamentals? Are there
accounting risks, competitive threats, or secular declines that make this a short candidate?

VARIANT VIEW REQUIREMENT:
- State what the CONSENSUS BEAR CASE is (what most bears worry about), then go DEEPER.
- Your best risks should be NON-OBVIOUS — things the market is underpricing.
- Avoid lazy risk points ("macro headwinds", "competition"). Be specific: which macro factor,
  which competitor, what mechanism, what timeline.
- The best bear cases find asymmetry: risks with high impact but low market pricing.

Also include TECHNICAL ANALYSIS:
- Recent price action vs. 50-day and 200-day moving averages
- Key support/resistance levels and potential breakdown triggers
- Relative performance vs. sector and market over 1m/3m/6m

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
    "risk_score": 6.5,
    "key_vulnerabilities": {{"area": "description"}},
    "short_thesis": "If warranted: 1-2 sentence active short pitch. If not warranted, leave empty.",
    "actionable_recommendation": "AVOID or UNDERWEIGHT or ACTIVE SHORT or HEDGE",
    "technical_levels": {{
        "current_vs_50dma": "above/below by X%",
        "current_vs_200dma": "above/below by X%",
        "key_support": "$XXX",
        "key_resistance": "$XXX",
        "breakdown_trigger": "$XXX — if breached, indicates..."
    }}
}}

risk_score: 0-10 scale (0 = no risk, 10 = extreme risk / uninvestable)
actionable_recommendation: Choose the most appropriate — don't default to AVOID if ACTIVE SHORT is warranted.

Be specific. Use numbers. Trace the causal chains.
Respond ONLY with the JSON object, no other text."""

        response = self.model(prompt)
        response_text = response if isinstance(response, str) else str(response)

        try:
            parsed = self._extract_json(response_text)
            return BearCase(**parsed)
        except Exception as e:
            logger.warning(f"Failed to parse bear case JSON: {e}. Using fallback.")
            return BearCase(
                ticker=ticker,
                risks=["Analysis generated but structured parsing failed"],
                second_order_effects=[],
                third_order_effects=[],
                worst_case_scenario=response_text[:500],
                risk_score=5.0,
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
Risk Score: {bear_case.risk_score}/10

REFLECT on your analysis:
1. Are your causal chains logically sound or are you stretching?
2. Did you miss any major risk categories?
3. Are you being appropriately bearish or catastrophizing?
4. CONVICTION SENSITIVITY — what specific, observable events would:
   a) INCREASE your risk score by 2+ points? (Be specific: "If debt/EBITDA exceeds 4x...")
   b) DECREASE your risk score by 2+ points? (Be specific: "If free cash flow grows 20% YoY...")
   c) Make you flip to BULLISH? What would it take?
5. What data would help validate or invalidate your risk assessment?
6. VARIANT CHECK — are your risks genuinely non-consensus, or are you restating obvious bears?
   The market already knows about "competition" and "macro risk." What are you seeing that others don't?

Be calibrated. Good risk management isn't about being maximally pessimistic —
it's about being accurately pessimistic."""

        response = self.model(prompt)
        return response if isinstance(response, str) else str(response)

    def rebut(self, ticker: str, opposing_view: Any, own_result: Any) -> Rebuttal:
        """Respond to the Sector Analyst's bull case."""
        bear_case = own_result if isinstance(own_result, BearCase) else own_result
        bull_case = opposing_view

        prompt = f"""You are the Risk Manager for {ticker}. The Sector Analyst has presented
their bull case. You must CHALLENGE their analysis.

SECTOR ANALYST'S BULL CASE:
Thesis: {bull_case.thesis if hasattr(bull_case, 'thesis') else bull_case.get('thesis', '')}
Evidence: {bull_case.supporting_evidence if hasattr(bull_case, 'supporting_evidence') else bull_case.get('supporting_evidence', [])}
Catalysts: {bull_case.catalysts if hasattr(bull_case, 'catalysts') else bull_case.get('catalysts', [])}
Conviction: {bull_case.conviction_score if hasattr(bull_case, 'conviction_score') else bull_case.get('conviction_score', 'N/A')}

YOUR BEAR CASE:
Risk Score: {bear_case.risk_score}/10
Key Risks: {bear_case.risks}

Respond in valid JSON:
{{
    "points": ["challenge point 1", "challenge point 2", ...],
    "concessions": ["point where analyst makes a fair argument", ...],
    "revised_conviction": 6.5
}}

revised_conviction is YOUR risk score after considering the bull case (0-10, where 10 = highest risk).

Be rigorous but fair — acknowledge strong bull points, but press on the weaknesses.
Respond ONLY with the JSON object."""

        response = self.model(prompt)
        response_text = response if isinstance(response, str) else str(response)

        try:
            parsed = self._extract_json(response_text)
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

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract JSON from a response that might contain markdown or extra text."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            return json.loads(text[start:end].strip())

        if "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            return json.loads(text[start:end].strip())

        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
