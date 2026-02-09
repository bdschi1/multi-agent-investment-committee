"""
Portfolio Manager Agent — synthesizes bull and bear cases into a decision.

This agent acts as the PM who chairs the investment committee. They weigh
both sides, identify where conviction is strongest, and produce the
final recommendation with explicit reasoning.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agents.base import (
    AgentRole,
    BaseInvestmentAgent,
    BearCase,
    BullCase,
    CommitteeMemo,
    MacroView,
    Rebuttal,
    StepType,
)

logger = logging.getLogger(__name__)


class PortfolioManagerAgent(BaseInvestmentAgent):
    """Synthesizer agent: weighs bull vs bear and makes the final call."""

    def __init__(self, model: Any):
        super().__init__(model=model, role=AgentRole.PORTFOLIO_MANAGER)

    def think(self, ticker: str, context: dict[str, Any]) -> str:
        """Assess both cases, macro context, and the debate before forming a view."""
        bull_case = context.get("bull_case", {})
        bear_case = context.get("bear_case", {})
        macro_view = context.get("macro_view", {})
        debate = context.get("debate", {})

        # Serialize Pydantic models if needed
        bull_data = bull_case.model_dump() if hasattr(bull_case, "model_dump") else bull_case
        bear_data = bear_case.model_dump() if hasattr(bear_case, "model_dump") else bear_case
        macro_data = macro_view.model_dump() if hasattr(macro_view, "model_dump") else macro_view

        user_context = context.get('user_context', '').strip()
        expert_section = ""
        if user_context:
            expert_section = f"""

EXPERT GUIDANCE FROM COMMITTEE CHAIR:
The following expert context was provided by the user. Factor this domain
expertise into your synthesis — it may highlight considerations that
neither analyst fully addressed:
{user_context}
"""

        prompt = f"""You are the Portfolio Manager chairing the investment committee for {ticker}.

You have received:
1. A BULL CASE from the Sector Analyst
2. A BEAR CASE from the Risk Manager
3. A MACRO ENVIRONMENT assessment from the Macro Analyst
4. Their DEBATE (rebuttals to each other)

BULL CASE:
{json.dumps(bull_data, indent=2, default=str)}

BEAR CASE:
{json.dumps(bear_data, indent=2, default=str)}

MACRO ENVIRONMENT:
{json.dumps(macro_data, indent=2, default=str)}

DEBATE SUMMARY:
{json.dumps(debate, indent=2, default=str)}
{expert_section}
Now THINK about what you've heard. Consider:
1. Where do the analyst and risk manager AGREE? (These are likely reliable conclusions)
2. Where do they DISAGREE? Who has stronger evidence?
3. What is the quality of reasoning on each side?
4. How does the MACRO BACKDROP affect the thesis? Does the cycle, rate environment,
   or sector rotation favor the bull or bear case?
5. What would your initial instinct be and why?
6. What additional factors should you weigh (portfolio fit, timing, sizing)?
7. VARIANT VIEW ASSESSMENT — did either agent present a genuinely NON-CONSENSUS view?
   Or are both restating widely-known narratives? You should weight differentiated,
   evidence-backed variant views more highly than consensus-restatement.
8. What is the CONSENSUS STREET VIEW on this name? Where does this committee's
   combined analysis differ from what the market already believes?

Think like a PM who has to put capital at risk. The macro context often tips the
balance when bull and bear are closely matched. This isn't academic — it's a real decision.
The best trades come from non-consensus views backed by rigorous analysis."""

        response = self.model(prompt)
        return response if isinstance(response, str) else str(response)

    def plan(self, ticker: str, context: dict[str, Any], thinking: str) -> str:
        """Plan the synthesis and decision framework."""
        prompt = f"""Based on your initial thinking about {ticker}:

{thinking}

PLAN your decision framework:
1. What weight will you give to the bull vs bear case and why?
2. What are the key swing factors that determine the recommendation?
3. How will you determine position sizing?
4. What risk mitigants would you require?
5. What would make you revisit this decision?

Be explicit about your decision framework. Good PMs are transparent about
how they weigh evidence."""

        response = self.model(prompt)
        return response if isinstance(response, str) else str(response)

    def act(self, ticker: str, context: dict[str, Any], plan: str) -> CommitteeMemo:
        """Synthesize everything into the final committee memo."""
        bull_case = context.get("bull_case", {})
        bear_case = context.get("bear_case", {})
        macro_view = context.get("macro_view", {})
        debate = context.get("debate", {})

        bull_data = bull_case.model_dump() if hasattr(bull_case, "model_dump") else bull_case
        bear_data = bear_case.model_dump() if hasattr(bear_case, "model_dump") else bear_case
        macro_data = macro_view.model_dump() if hasattr(macro_view, "model_dump") else macro_view

        user_context = context.get('user_context', '').strip()
        expert_section = ""
        if user_context:
            expert_section = f"""
EXPERT GUIDANCE (factor into your final decision):
{user_context}
"""

        prompt = f"""You are the Portfolio Manager making the FINAL DECISION on {ticker}.

Your decision framework:
{plan}

BULL CASE: {json.dumps(bull_data, indent=2, default=str)}
BEAR CASE: {json.dumps(bear_data, indent=2, default=str)}
MACRO ENVIRONMENT: {json.dumps(macro_data, indent=2, default=str)}
DEBATE: {json.dumps(debate, indent=2, default=str)}
{expert_section}
IMPORTANT: Factor in the macro environment. Consider:
- Does the economic cycle favor or hinder this stock?
- Does the rate trajectory support or pressure the thesis?
- Does sector rotation align with this trade?
- Do cross-asset signals confirm or diverge from equity positioning?

VARIANT VIEW & ALPHA STANDARD:
- Your thesis_summary MUST state whether this is a CONSENSUS or NON-CONSENSUS view.
- If consensus: explain why timing or sizing creates alpha even without a variant view.
- If non-consensus: explain clearly what the market is missing and why you're right.
- Avoid recommending based on widely-known, already-priced-in narratives.
- If neither analyst presented a compelling variant view, lower conviction accordingly.
  It's better to HOLD with honesty than BUY with a consensus thesis.

Produce the INVESTMENT COMMITTEE MEMO. Respond in valid JSON:
{{
    "ticker": "{ticker}",
    "recommendation": "STRONG BUY / BUY / HOLD / UNDERWEIGHT / SELL / ACTIVE SHORT",
    "position_size": "Full position / Half position / Small starter / No position",
    "conviction": 7.0,
    "thesis_summary": "2-3 sentence summary of the final thesis",
    "key_factors": ["factor that drove the decision", ...],
    "bull_points_accepted": ["bull arguments you found compelling", ...],
    "bear_points_accepted": ["bear arguments you found compelling", ...],
    "dissenting_points": ["points where you overruled one side and why", ...],
    "risk_mitigants": ["what you'd require to manage downside", ...],
    "time_horizon": "e.g. 6-12 months"
}}

conviction: 0-10 (your overall confidence in this recommendation)

IMPORTANT: Explain WHY you weighted one side over the other.
Every recommendation must explicitly state what you'd need to see to change your mind.
Respond ONLY with the JSON object, no other text."""

        response = self.model(prompt)
        response_text = response if isinstance(response, str) else str(response)

        try:
            parsed = self._extract_json(response_text)
            return CommitteeMemo(**parsed)
        except Exception as e:
            logger.warning(f"Failed to parse committee memo JSON: {e}. Using fallback.")
            return CommitteeMemo(
                ticker=ticker,
                recommendation="HOLD",
                position_size="No position — parsing error",
                conviction=0.0,
                thesis_summary=response_text[:500],
                key_factors=["Memo generated but structured parsing failed"],
            )

    def reflect(self, ticker: str, result: Any) -> str:
        """Evaluate the quality of the final decision."""
        memo = result if isinstance(result, CommitteeMemo) else result

        prompt = f"""Review the investment committee memo you just produced for {ticker}:

Recommendation: {memo.recommendation}
Position Size: {memo.position_size}
Conviction: {memo.conviction}/10
Summary: {memo.thesis_summary}

REFLECT as the PM:
1. Is this recommendation internally consistent?
2. Did you give fair weight to both the bull and bear arguments?
3. Are the risk mitigants sufficient?
4. Would you actually put your own capital behind this?
5. CONVICTION SENSITIVITY — what specific, observable events would:
   a) INCREASE your conviction by 2+ points? (Be precise)
   b) DECREASE your conviction by 2+ points? (Be precise)
   c) Cause you to REVERSE your recommendation entirely?
6. ALPHA CHECK — is this recommendation actionable and differentiated, or is it just
   restating what any analyst with a Bloomberg terminal would say? If it's consensus,
   say so honestly. The worst thing a PM can do is dress up a consensus view as insight.
7. What would a critic say about this decision?

The best PMs are brutally honest about their own decision quality."""

        response = self.model(prompt)
        return response if isinstance(response, str) else str(response)

    def rebut(self, ticker: str, opposing_view: Any, own_result: Any) -> Rebuttal:
        """PM doesn't rebut — they synthesize. This is a no-op for protocol compliance."""
        return Rebuttal(
            agent_role=AgentRole.PORTFOLIO_MANAGER,
            responding_to=AgentRole.SECTOR_ANALYST,
            points=["PM synthesizes rather than rebuts"],
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
