"""
Sector Analyst Agent — builds the bull case.

This agent acts as a senior equity analyst constructing a buy thesis.
It gathers fundamental data, identifies catalysts, and builds a
conviction-scored investment case.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agents.base import (
    AgentRole,
    BaseInvestmentAgent,
    BullCase,
    Rebuttal,
    StepType,
)

logger = logging.getLogger(__name__)


class SectorAnalystAgent(BaseInvestmentAgent):
    """Bull-case agent: builds the affirmative investment thesis."""

    def __init__(self, model: Any):
        super().__init__(model=model, role=AgentRole.SECTOR_ANALYST)

    def think(self, ticker: str, context: dict[str, Any]) -> str:
        """Assess what we know and form initial hypotheses about the opportunity."""
        market_data = context.get("market_data", {})
        news = context.get("news", [])

        user_context = context.get('user_context', '').strip()
        expert_section = ""
        if user_context:
            expert_section = f"""

EXPERT GUIDANCE FROM COMMITTEE CHAIR:
The following expert context has been provided. You MUST incorporate these
considerations into your analysis — they represent domain expertise that
should materially influence your thinking:
{user_context}
"""

        prompt = f"""You are a senior sector analyst on an investment committee.
Your job is to BUILD THE BULL CASE for {ticker}.

First, THINK about what you know and what hypotheses you want to test.

Available market data: {json.dumps(market_data, indent=2, default=str)}
Recent news headlines: {json.dumps(news[:5], default=str) if news else 'None available'}
{expert_section}
Respond with your initial thinking:
1. What is this company/sector and why might it be interesting?
2. What are your initial hypotheses for a bull case?
3. What data points would strengthen or weaken your thesis?
4. What catalysts might you look for?
5. What is the CONSENSUS VIEW on this stock? What does "the street" think?
6. Where might you find a VARIANT PERCEPTION — something the market is missing or
   mispricing? The best alpha comes from non-consensus, well-researched views.

IMPORTANT — avoid the obvious:
- Do NOT build a thesis around widely-known narratives (e.g. "AI is growing" for big tech).
  These are already priced in.
- Instead, seek VARIANT VIEWS: what would a top-decile analyst see that the market doesn't?
- Triangulate: can you find 2-3 independent data points that converge on a non-obvious conclusion?

Be specific and analytical. Think like an analyst, not a chatbot."""

        response = self.model(prompt)
        return response if isinstance(response, str) else str(response)

    def plan(self, ticker: str, context: dict[str, Any], thinking: str) -> str:
        """Plan what analysis to perform based on initial thinking."""
        prompt = f"""Based on your initial thinking about {ticker}:

{thinking}

Now PLAN your analysis. What specific steps will you take to build the bull case?

Outline:
1. Which financial metrics will you examine and why?
2. What competitive advantages will you assess?
3. What growth drivers will you analyze?
4. What catalysts will you identify?
5. How will you score conviction?

Be structured and specific. This plan will guide your analysis."""

        response = self.model(prompt)
        return response if isinstance(response, str) else str(response)

    def act(self, ticker: str, context: dict[str, Any], plan: str) -> BullCase:
        """Execute the analysis plan and produce the bull case."""
        market_data = context.get("market_data", {})
        news = context.get("news", [])
        metrics = context.get("financial_metrics", {})

        user_context = context.get('user_context', '').strip()
        expert_section = ""
        if user_context:
            expert_section = f"""
EXPERT GUIDANCE (incorporate into your analysis):
{user_context}
"""

        prompt = f"""You are executing your analysis plan for {ticker}. BUILD THE BULL CASE.

Your plan:
{plan}

Market data: {json.dumps(market_data, indent=2, default=str)}
Financial metrics: {json.dumps(metrics, indent=2, default=str)}
Recent news: {json.dumps(news[:10], default=str) if news else 'None available'}
{expert_section}

Produce a STRUCTURED bull case. Include:
1. Technical analysis: recent price action, trend, momentum, key support/resistance levels
2. Relative performance: how has this stock performed vs. its sector and the broader market recently?
3. A 12-month catalyst calendar with specific upcoming events

CRITICAL — VARIANT VIEW REQUIREMENT:
- State the CONSENSUS VIEW first (what most analysts think), then explain WHERE YOU DIFFER and why.
- Your thesis should contain at least one NON-OBVIOUS insight — something not in the headlines.
- Avoid generic bull points like "strong brand" or "growing TAM" unless you can quantify WHY
  they're underappreciated by the market RIGHT NOW.
- Triangulate: support your thesis with 2-3 independent, converging data points.
- If you cannot find a genuinely differentiated view, lower your conviction score accordingly.
  Honesty > forced bullishness.

Respond in valid JSON matching this exact schema:
{{
    "ticker": "{ticker}",
    "thesis": "Clear, 2-3 sentence investment thesis",
    "supporting_evidence": ["evidence point 1", "evidence point 2", ...],
    "catalysts": ["catalyst 1", "catalyst 2", ...],
    "catalyst_calendar": [
        {{"timeframe": "Q1 2025", "event": "Earnings report", "impact": "Expected beat on margins"}},
        {{"timeframe": "Q2 2025", "event": "Product launch", "impact": "TAM expansion catalyst"}},
        ...
    ],
    "conviction_score": 7.5,
    "time_horizon": "6-12 months",
    "key_metrics": {{"metric_name": "value"}},
    "technical_outlook": "Stock is trading above 50/200 DMA with bullish momentum..."
}}

conviction_score: 0-10 scale (0 = no conviction, 10 = highest conviction)

Ground your analysis in the data provided. Be specific with numbers.
For the catalyst_calendar, include at least 4-6 events over the next 12 months.
For technical_outlook, reference actual price levels, moving averages, and recent performance.
Respond ONLY with the JSON object, no other text."""

        response = self.model(prompt)
        response_text = response if isinstance(response, str) else str(response)

        try:
            # Try to parse the JSON from the response
            parsed = self._extract_json(response_text)
            return BullCase(**parsed)
        except Exception as e:
            logger.warning(f"Failed to parse bull case JSON: {e}. Using fallback.")
            return BullCase(
                ticker=ticker,
                thesis=response_text[:500],
                supporting_evidence=["Analysis generated but structured parsing failed"],
                catalysts=[],
                conviction_score=5.0,
                time_horizon="unknown",
                key_metrics={},
            )

    def reflect(self, ticker: str, result: Any) -> str:
        """Evaluate the quality and completeness of the bull case."""
        bull_case = result if isinstance(result, BullCase) else result

        prompt = f"""Review the bull case you just produced for {ticker}:

Thesis: {bull_case.thesis}
Evidence: {bull_case.supporting_evidence}
Catalysts: {bull_case.catalysts}
Conviction: {bull_case.conviction_score}/10

REFLECT on your analysis:
1. What are the strongest parts of this thesis?
2. What gaps or weaknesses exist?
3. Are you being appropriately skeptical or overly optimistic?
4. CONVICTION SENSITIVITY — what specific, observable events or data points would:
   a) INCREASE your conviction by 2+ points? (Be specific: "If Q2 margins exceed 28%...")
   b) DECREASE your conviction by 2+ points? (Be specific: "If customer churn rises above 5%...")
   c) KILL the thesis entirely? (What would make you walk away?)
5. Rate your confidence in this analysis (high/medium/low) and why.
6. VARIANT CHECK — is your thesis genuinely non-consensus, or are you restating the obvious?
   If it's consensus, acknowledge that and explain why consensus might be right this time.

Be honest and self-critical."""

        response = self.model(prompt)
        return response if isinstance(response, str) else str(response)

    def rebut(self, ticker: str, opposing_view: Any, own_result: Any) -> Rebuttal:
        """Respond to the Risk Manager's bear case."""
        bull_case = own_result if isinstance(own_result, BullCase) else own_result
        bear_case = opposing_view

        prompt = f"""You are the Sector Analyst for {ticker}. The Risk Manager has presented
their bear case. You must REBUT their analysis.

YOUR BULL CASE:
Thesis: {bull_case.thesis}
Conviction: {bull_case.conviction_score}/10

RISK MANAGER'S BEAR CASE:
Risks: {bear_case.risks if hasattr(bear_case, 'risks') else bear_case.get('risks', [])}
2nd Order Effects: {bear_case.second_order_effects if hasattr(bear_case, 'second_order_effects') else bear_case.get('second_order_effects', [])}
3rd Order Effects: {bear_case.third_order_effects if hasattr(bear_case, 'third_order_effects') else bear_case.get('third_order_effects', [])}
Risk Score: {bear_case.risk_score if hasattr(bear_case, 'risk_score') else bear_case.get('risk_score', 'N/A')}

Respond in valid JSON:
{{
    "points": ["rebuttal point 1", "rebuttal point 2", ...],
    "concessions": ["point where you agree with risk manager", ...],
    "revised_conviction": 7.0
}}

Be intellectually honest — concede valid points, but defend your thesis where evidence supports it.
Respond ONLY with the JSON object."""

        response = self.model(prompt)
        response_text = response if isinstance(response, str) else str(response)

        try:
            parsed = self._extract_json(response_text)
            return Rebuttal(
                agent_role=AgentRole.SECTOR_ANALYST,
                responding_to=AgentRole.RISK_MANAGER,
                **parsed,
            )
        except Exception as e:
            logger.warning(f"Failed to parse rebuttal JSON: {e}")
            return Rebuttal(
                agent_role=AgentRole.SECTOR_ANALYST,
                responding_to=AgentRole.RISK_MANAGER,
                points=["Rebuttal generated but structured parsing failed"],
                concessions=[],
            )

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract JSON from a response that might contain markdown or extra text."""
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block in markdown
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            return json.loads(text[start:end].strip())

        if "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            return json.loads(text[start:end].strip())

        # Try to find JSON object boundaries
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
