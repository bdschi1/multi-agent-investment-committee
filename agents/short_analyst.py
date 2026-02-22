"""
Short Analyst Agent — builds the dedicated short thesis.

This agent acts as a senior short-selling specialist constructing a
short thesis. It identifies alpha shorts vs. beta hedges, assesses
borrow conditions, maps event paths for the short to work, and
decomposes return into idiosyncratic vs. systematic components.

New in v4.0: Dedicated 5th agent replacing the Risk Manager's
former dual role of risk assessment AND short thesis generation.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agents.base import (
    AgentRole,
    BaseInvestmentAgent,
    Rebuttal,
    ShortCase,
    clean_json_artifacts,
    extract_json,
    retry_extract_json,
)

logger = logging.getLogger(__name__)


class ShortAnalystAgent(BaseInvestmentAgent):
    """Short-case agent: builds the dedicated short thesis with alpha/beta classification."""

    def __init__(self, model: Any, tool_registry: Any = None):
        super().__init__(model=model, role=AgentRole.SHORT_ANALYST, tool_registry=tool_registry)

    def think(self, ticker: str, context: dict[str, Any]) -> str:
        """Identify short opportunities and form initial short hypotheses."""
        market_data = context.get("market_data", {})
        news = context.get("news", [])

        user_context = context.get('user_context', '').strip()
        expert_section = ""
        if user_context:
            expert_section = f"""

EXPERT GUIDANCE FROM COMMITTEE CHAIR:
The following expert context has been provided. You MUST incorporate these
considerations into your short analysis — they represent domain expertise
that may highlight short opportunities you would otherwise miss:
{user_context}
"""

        user_kb = context.get('user_kb', '').strip()
        kb_section = ""
        if user_kb:
            kb_section = f"""

{user_kb}
"""

        memory_section = ""
        agent_memory = context.get("agent_memory", [])
        if agent_memory:
            lessons = "\n".join(
                f"- [{m['ticker']}] {'CORRECT' if m['was_correct'] else 'WRONG'}: {m['lesson']}"
                for m in agent_memory
            )
            memory_section = f"""

LESSONS FROM SIMILAR PAST ANALYSES:
{lessons}
Use these lessons to calibrate your short thesis conviction and avoid repeating past mistakes.
"""

        prompt = f"""You are a senior short-selling specialist on an investment committee.
Your job is to evaluate whether {ticker} is a SHORT CANDIDATE.

You are NOT the Risk Manager — the Risk Manager handles sizing, structuring, and
stop-loss levels. You are the SHORT THESIS GENERATOR. Your job is to identify
whether this stock deserves an active short position and why.

First, THINK about what could make this stock decline. Consider:

SHORT THESIS CLASSIFICATION:
1. ALPHA SHORT — genuine idiosyncratic decline driver (earnings deterioration,
   accounting issues, competitive disruption, product failure, secular decline).
   This is the gold standard. The stock declines REGARDLESS of what the market does.
2. HEDGE — you're shorting to offset long exposure. The thesis is portfolio-level,
   not stock-specific. Lower conviction, but useful for risk management.
3. PAIR LEG — one half of a relative value trade. You're short this name against
   a long in a related name. The thesis is about RELATIVE performance.
4. NO POSITION — you don't see a compelling short case. Be honest about this.

Available market data: {json.dumps(market_data, indent=2, default=str)}
Recent news headlines: {json.dumps(news[:5], default=str) if news else 'None available'}
{expert_section}{kb_section}{memory_section}
Respond with your initial short thesis thinking:
1. What is the BULL NARRATIVE the market believes? (You must understand the long
   case before you can break it.)
2. Where is the bull narrative VULNERABLE? What are the specific mechanisms
   that could cause this stock to decline?
3. Is this a potential ALPHA SHORT (stock-specific decline) or just a BETA SHORT
   (market/sector decline in disguise)?
4. BORROW CONDITIONS — what is the short interest, days to cover, and likely
   borrow cost? High short interest with expensive borrow changes the risk/reward.
5. EVENT PATH — what is the ordered sequence of events that would cause the
   short to work? (e.g., "earnings miss → guidance cut → multiple compression")
6. PERIPHERAL VIEW — how does your short thesis interact with the long case?
   Where does it overlap with the bull analyst's thesis, and where does it diverge?

ALPHA-SHORT STANDARD:
- The best shorts are NOT "this company is bad." They're "the market's expectations
  are specifically wrong about THIS variable, and when it's revealed, the stock
  reprices by X%."
- Avoid generic shorts ("overvalued," "too expensive"). Those are beta shorts in disguise.
- Seek structural shorts: secular declines, accounting red flags, competitive disruption,
  or exhausted growth narratives where expectations haven't adjusted.

Think deeply. The best short sellers see what the crowd doesn't."""

        response = self.model(prompt)
        return response if isinstance(response, str) else str(response)

    def plan(self, ticker: str, context: dict[str, Any], thinking: str) -> str:
        """Plan the short analysis approach."""
        tool_catalog = self.get_tool_catalog()
        tool_section = ""
        if tool_catalog:
            tool_section = f"""

AVAILABLE TOOLS (optional — call any that would strengthen your short analysis):
{tool_catalog}

Suggested tools for short analysis: compare_peers, get_insider_activity, get_earnings_history

To request tool data, add a TOOL_CALLS block at the END of your plan:
TOOL_CALLS:
- tool_name(ticker="{ticker}")
"""

        prompt = f"""Based on your initial short thesis thinking about {ticker}:

{thinking}

Now PLAN your short analysis. Specifically:

1. What is the SPECIFIC MECHANISM for stock decline? Not just "overvalued" —
   what changes, when, and by how much?
2. How will you classify the short? (alpha_short / hedge / pair_leg / no_position)
3. What EVENT PATH will you map? List the ordered events.
4. How will you assess borrow conditions?
5. How will you decompose the return into idiosyncratic vs. systematic?
6. What data would validate or invalidate your short thesis?
7. How does your short view interact with what the long analyst might see?
{tool_section}
Be structured and specific. This plan will guide your short analysis."""

        response = self.model(prompt)
        return response if isinstance(response, str) else str(response)

    def act(self, ticker: str, context: dict[str, Any], plan: str, tool_results: dict[str, Any] | None = None) -> ShortCase:
        """Execute the short analysis plan and produce the ShortCase."""
        market_data = context.get("market_data", {})
        news = context.get("news", [])
        metrics = context.get("financial_metrics", {})

        user_context = context.get('user_context', '').strip()
        expert_section = ""
        if user_context:
            expert_section = f"""
EXPERT GUIDANCE (incorporate into your short analysis):
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
ADDITIONAL DATA FROM TOOL CALLS (use this to strengthen your short analysis):
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
                pfd_note = ""
                if pfd is not None and pfd > 0.3:
                    pfd_note = f"\nELEVATED DISTRESS PROBABILITY: PFD={pfd:.1%}, zone={zone}. This supports a short thesis."
                xai_section = f"""
XAI QUANTITATIVE PRE-SCREEN (Shapley value analysis — use for short thesis support):
{narrative}{pfd_note}
"""

        prompt = f"""You are executing your short analysis for {ticker}. BUILD THE SHORT CASE.

Your analysis plan:
{plan}

Market data: {json.dumps(market_data, indent=2, default=str)}
Financial metrics: {json.dumps(metrics, indent=2, default=str)}
Recent news: {json.dumps(news[:10], default=str) if news else 'None available'}
{expert_section}{kb_section}{tool_data_section}{xai_section}
Produce a STRUCTURED short case. You MUST classify the short thesis type honestly.

IMPORTANT CLASSIFICATION RULES:
- "alpha_short": You have a SPECIFIC, IDIOSYNCRATIC reason this stock declines.
  The decline is stock-specific, not just the sector going down.
- "hedge": This is a portfolio hedge — you're offsetting long exposure, not making
  an alpha call on the stock itself.
- "pair_leg": One side of a relative value trade. The thesis is about THIS stock
  vs. ANOTHER stock, not about the absolute direction.
- "no_position": You do NOT see a compelling short case. Be honest. Not every stock
  is a short. Forced shorts lose money.

Respond in valid JSON matching this exact schema:
{{
    "ticker": "{ticker}",
    "short_thesis": "1-2 sentence specific mechanism for stock decline (or empty if no_position)",
    "thesis_type": "alpha_short / hedge / pair_leg / no_position",
    "event_path": [
        "Event 1: specific trigger and timeline",
        "Event 2: consequence of Event 1",
        "Event 3: how this causes the stock to decline by X%"
    ],
    "supporting_evidence": ["evidence point 1", "evidence point 2"],
    "alpha_vs_beta_assessment": "Is this idiosyncratic (alpha) or systematic (beta)? How much of the expected decline is stock-specific vs. sector/market?",
    "borrow_assessment": "Short interest: X%, days to cover: Y, estimated borrow cost: Z bps. If hard to borrow, explain impact on risk/reward.",
    "conviction_score": 6.0,
    "key_vulnerabilities": {{"area": "description of vulnerability"}},
    "estimated_short_return": "X% — methodology and assumptions (e.g., 'If earnings miss by 10%, P/E compresses from 30x to 22x → 27% decline')",
    "idiosyncratic_return": "X% — stock-specific return vs sector. If most of the decline is sector-wide, this is a beta short, not alpha.",
    "estimated_sharpe": "X.XX — using (-1 * expected_return) / vol. A short with Sharpe > 0.5 is actionable."
}}

conviction_score: 0-10 scale (0 = no short conviction, 10 = maximum short conviction).
If thesis_type is "no_position", conviction_score should be 0-2.

RETURN DECOMPOSITION IS REQUIRED (heuristic estimates):
- estimated_short_return: Total expected decline + how you got there
- idiosyncratic_return: Stock-specific decline vs. sector. If near zero, this is a beta short.
- estimated_sharpe: Using (-1 * expected_return) / vol. For shorts, the return is negative,
  so Sharpe = (positive expected profit) / vol.

Respond ONLY with the JSON object, no other text."""

        response = self.model(prompt)
        response_text = response if isinstance(response, str) else str(response)

        try:
            parsed, retried = retry_extract_json(self.model, prompt, response_text, max_retries=1)
            if retried:
                logger.info(f"ShortCase JSON extraction required retry for {ticker}")
            return ShortCase(**parsed)
        except Exception as e:
            logger.warning(f"Failed to parse short case JSON: {e}. Using fallback.")
            return ShortCase(
                ticker=ticker,
                short_thesis=clean_json_artifacts(response_text),
                thesis_type="no_position",
                supporting_evidence=["Analysis generated but structured parsing failed"],
                conviction_score=3.0,
                key_vulnerabilities={},
            )

    def reflect(self, ticker: str, result: Any) -> str:
        """Evaluate the quality and honesty of the short thesis."""
        short_case = result if isinstance(result, ShortCase) else result

        prompt = f"""Review the short case you just produced for {ticker}:

Short Thesis: {short_case.short_thesis}
Thesis Type: {short_case.thesis_type}
Event Path: {short_case.event_path}
Alpha vs Beta: {short_case.alpha_vs_beta_assessment}
Borrow Assessment: {short_case.borrow_assessment}
Conviction: {short_case.conviction_score}/10

REFLECT on your analysis:
1. Is your SHORT THESIS TYPE classification honest?
   - If you classified as "alpha_short," is the decline truly stock-specific?
   - If the whole sector would decline similarly, this is a beta short — reclassify.
2. Is your EVENT PATH specific enough to be actionable?
   - Each event should have a timeline and a probability assessment.
3. BORROW REALITY CHECK:
   - Did you consider short interest and borrow cost?
   - If borrow is expensive (>100bps), does the expected return justify the carry?
4. CONVICTION SENSITIVITY — what specific events would:
   a) INCREASE your short conviction by 2+ points?
   b) DECREASE your short conviction by 2+ points?
   c) KILL the short thesis entirely?
5. RETURN DECOMPOSITION CHECK:
   a) Is your estimated_short_return defensible?
   b) Is the idiosyncratic_return truly stock-specific, or are you claiming
      alpha on what is really a sector bet?
   c) Is the Sharpe above 0.5? If not, the risk/reward may not justify the short.
6. HONESTY CHECK — if you don't have a compelling short case, it's better to
   classify as "no_position" with low conviction than to force a weak thesis.
   Forced shorts lose money.

Be brutally honest. The best short sellers kill their own bad ideas before the market does."""

        response = self.model(prompt)
        return response if isinstance(response, str) else str(response)

    def rebut(self, ticker: str, opposing_view: Any, own_result: Any) -> Rebuttal:
        """Respond to the Long Analyst's bull case."""
        short_case = own_result if isinstance(own_result, ShortCase) else own_result
        bull_case = opposing_view

        prompt = f"""You are the Short Analyst for {ticker}. The Long Analyst has presented
their bull case. You must CHALLENGE their analysis from the short side.

LONG ANALYST'S BULL CASE:
Thesis: {bull_case.thesis if hasattr(bull_case, 'thesis') else bull_case.get('thesis', '')}
Evidence: {bull_case.supporting_evidence if hasattr(bull_case, 'supporting_evidence') else bull_case.get('supporting_evidence', [])}
Catalysts: {bull_case.catalysts if hasattr(bull_case, 'catalysts') else bull_case.get('catalysts', [])}
Conviction: {bull_case.conviction_score if hasattr(bull_case, 'conviction_score') else bull_case.get('conviction_score', 'N/A')}

YOUR SHORT CASE:
Short Thesis: {short_case.short_thesis if hasattr(short_case, 'short_thesis') else short_case.get('short_thesis', '')}
Thesis Type: {short_case.thesis_type if hasattr(short_case, 'thesis_type') else short_case.get('thesis_type', '')}
Short Conviction: {short_case.conviction_score if hasattr(short_case, 'conviction_score') else short_case.get('conviction_score', 'N/A')}

Respond in valid JSON:
{{
    "points": ["challenge point 1", "challenge point 2", ...],
    "concessions": ["point where the long analyst makes a fair argument", ...],
    "revised_conviction": 6.5
}}

revised_conviction is YOUR short conviction after considering the bull case (0-10).

Be rigorous — challenge the bull's return decomposition, question whether catalysts
are truly non-consensus, and press on valuation assumptions.
Respond ONLY with the JSON object."""

        response = self.model(prompt)
        response_text = response if isinstance(response, str) else str(response)

        try:
            parsed, _ = extract_json(response_text)
            return Rebuttal(
                agent_role=AgentRole.SHORT_ANALYST,
                responding_to=AgentRole.SECTOR_ANALYST,
                **parsed,
            )
        except Exception as e:
            logger.warning(f"Failed to parse short rebuttal JSON: {e}")
            return Rebuttal(
                agent_role=AgentRole.SHORT_ANALYST,
                responding_to=AgentRole.SECTOR_ANALYST,
                points=["Rebuttal generated but structured parsing failed"],
                concessions=[],
            )
