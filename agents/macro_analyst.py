"""
Macro Analyst Agent — provides top-down economic context.

This agent acts as a global macro strategist, assessing the economic cycle,
rate environment, sector rotation, geopolitical landscape, and cross-asset
signals. Its output is a contextual overlay — it doesn't debate, but feeds
the Portfolio Manager additional macro intelligence for synthesis.
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
    StepType,
)

logger = logging.getLogger(__name__)


class MacroAnalystAgent(BaseInvestmentAgent):
    """Top-down macro agent: economic cycle, rates, sector rotation, geopolitical."""

    def __init__(self, model: Any):
        super().__init__(model=model, role=AgentRole.MACRO_ANALYST)

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

        prompt = f"""You are a senior global macro strategist on an investment committee.
Your job is to provide the TOP-DOWN MACRO CONTEXT for analyzing {ticker}.

You are NOT building a bull or bear case. You are providing the economic
backdrop that both the bull and bear analysts need to consider.

Available market data: {json.dumps(market_data, indent=2, default=str)}
Recent news headlines: {json.dumps(news[:5], default=str) if news else 'None available'}
{expert_section}
THINK about the current macro environment. Consider:
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

ANTI-OBVIOUS REQUIREMENT:
- "The Fed might raise/cut rates" is not an insight — everyone knows that.
- Dig deeper: what SECOND-ORDER macro effects is the market underpricing?
- Seek non-consensus macro views backed by cross-asset confirmation.

Think deeply. The macro backdrop often matters more than company fundamentals,
especially at inflection points."""

        response = self.model(prompt)
        return response if isinstance(response, str) else str(response)

    def plan(self, ticker: str, context: dict[str, Any], thinking: str) -> str:
        """Plan the macro analysis approach."""
        prompt = f"""Based on your initial macro thinking about {ticker}:

{thinking}

Now PLAN your macro analysis. Specifically:

1. Which macro dimensions are MOST relevant to this stock and why?
2. How will you assess the economic cycle — what indicators matter?
3. What cross-asset signals will you examine for confirmation/divergence?
4. How will you determine if the macro is a tailwind or headwind for this name?
5. What sector rotation dynamics are relevant, and how do you measure them?
6. What geopolitical risks need quantifying and over what timeframes?

Prioritize — not all macro factors are equally relevant for every stock.
A pharma stock cares less about oil prices than an airline stock."""

        response = self.model(prompt)
        return response if isinstance(response, str) else str(response)

    def act(self, ticker: str, context: dict[str, Any], plan: str) -> MacroView:
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

        prompt = f"""You are executing your macro analysis for {ticker}. Provide the TOP-DOWN CONTEXT.

Your analysis plan:
{plan}

Market data: {json.dumps(market_data, indent=2, default=str)}
Financial metrics: {json.dumps(metrics, indent=2, default=str)}
Recent news: {json.dumps(news[:10], default=str) if news else 'None available'}
{expert_section}
Produce a STRUCTURED macro view. This is context — not a buy/sell recommendation.

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
    "headwinds": ["macro factor hurting this stock", ...]
}}

macro_favorability: 0-10 scale (0 = extremely hostile macro environment, 10 = perfect macro tailwind)

VARIANT VIEW REQUIREMENT:
- For each dimension, state the CONSENSUS macro view first, then where you differ (if you do).
- Avoid restating what every macro newsletter says. Seek non-consensus signals.
- Cross-asset divergences are often the most powerful variant signals — highlight them.
- If your view IS consensus, say so honestly and explain why consensus might still be actionable.

Be specific. Use real data where available. Think about what macro analysts at
a $50B+ AUM fund would focus on for this name.
Respond ONLY with the JSON object, no other text."""

        response = self.model(prompt)
        response_text = response if isinstance(response, str) else str(response)

        try:
            parsed = self._extract_json(response_text)
            return MacroView(**parsed)
        except Exception as e:
            logger.warning(f"Failed to parse macro view JSON: {e}. Using fallback.")
            return MacroView(
                ticker=ticker,
                economic_cycle_phase="Analysis generated but structured parsing failed",
                cycle_evidence=[],
                rate_environment="unknown",
                central_bank_outlook=response_text[:500],
                macro_favorability=5.0,
            )

    def reflect(self, ticker: str, result: Any) -> str:
        """Evaluate the quality and balance of the macro analysis."""
        macro_view = result if isinstance(result, MacroView) else result

        prompt = f"""Review the macro analysis you just produced for {ticker}:

Cycle Phase: {macro_view.economic_cycle_phase}
Rate Environment: {macro_view.rate_environment}
Sector Positioning: {macro_view.sector_positioning}
Macro Favorability: {macro_view.macro_favorability}/10
Tailwinds: {macro_view.tailwinds}
Headwinds: {macro_view.headwinds}

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

Good macro analysis is humble — the economy is complex and surprises are frequent.
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
