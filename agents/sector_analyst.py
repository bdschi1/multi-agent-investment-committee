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
    clean_json_artifacts,
    extract_json,
    retry_extract_json,
)

logger = logging.getLogger(__name__)


class SectorAnalystAgent(BaseInvestmentAgent):
    """Bull-case agent: builds the affirmative investment thesis."""

    def __init__(self, model: Any, tool_registry: Any = None):
        super().__init__(model=model, role=AgentRole.SECTOR_ANALYST, tool_registry=tool_registry)

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

        user_kb = context.get('user_kb', '').strip()
        kb_section = ""
        if user_kb:
            kb_section = f"""

{user_kb}
"""

        prompt = f"""You are a senior sector analyst on an investment committee.
Your job is to BUILD THE BULL CASE for {ticker}.

First, THINK about what you know and what hypotheses you want to test.

Available market data: {json.dumps(market_data, indent=2, default=str)}
Recent news headlines: {json.dumps(news[:5], default=str) if news else 'None available'}
{expert_section}{kb_section}
Respond with your initial thinking:
1. What is this company/sector and why might it be interesting?
2. What are your initial hypotheses for a bull case?
3. What data points would strengthen or weaken your thesis?
4. What catalysts might you look for?
5. What is the CONSENSUS VIEW on this stock? What does "the street" think?
6. Where might you find a VARIANT PERCEPTION — something the market is missing or
   mispricing? The best alpha comes from non-consensus, well-researched views.

NEWS SENTIMENT ANALYSIS:
7. For each news headline, extract the SENTIMENT SIGNAL:
   - Is the headline bullish, bearish, or neutral for the stock?
   - What is the signal strength (strong/moderate/weak)?
   - What type of catalyst does it represent (earnings, product, regulatory, management, macro)?
   - Does the news sentiment DIVERGE from price action? (Positive news + falling price = red flag;
     Negative news + rising price = potential strength)
8. What is the AGGREGATE sentiment across all news? Is it shifting over time?
9. Are there sentiment signals the market hasn't fully processed yet?

RETURN DECOMPOSITION (heuristic reasoning — not precise computation):
10. PRICE TARGET & TOTAL RETURN: Based on your thesis, what is a reasonable 12-month
    price target? What total return does that imply from the current price?
11. INDUSTRY vs IDIOSYNCRATIC RETURN: Of the total return you expect, how much is
    the sector/industry likely to contribute vs. company-specific alpha?
    - Idiosyncratic return = total return - estimated industry return
    - THIS is the alpha signal. If most of the return comes from the sector moving,
      there's no stock-specific edge — you're just making a sector bet.
12. HEURISTIC SHARPE / SORTINO: Given the stock's approximate volatility:
    - Sharpe ≈ idiosyncratic_return / estimated_annual_vol
    - Sortino ≈ idiosyncratic_return / downside_vol (focus on loss volatility)
    - For SHORT positions: flip the return sign (use -1 × expected return)
    - These are rough mental-model estimates, not precise calculations.
    - A Sharpe below 0.3 means the risk-adjusted return may not justify the position.

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
        tool_catalog = self.get_tool_catalog()
        tool_section = ""
        if tool_catalog:
            tool_section = f"""

AVAILABLE TOOLS (optional — call any that would strengthen your bull case):
{tool_catalog}

Suggested tools for bull-case analysis: get_earnings_history, compare_peers, get_insider_activity

To request tool data, add a TOOL_CALLS block at the END of your plan:
TOOL_CALLS:
- tool_name(ticker="{ticker}")
"""

        prompt = f"""Based on your initial thinking about {ticker}:

{thinking}

Now PLAN your analysis. What specific steps will you take to build the bull case?

Outline:
1. Which financial metrics will you examine and why?
2. What competitive advantages will you assess?
3. What growth drivers will you analyze?
4. What catalysts will you identify?
5. How will you score conviction?
{tool_section}
Be structured and specific. This plan will guide your analysis."""

        response = self.model(prompt)
        return response if isinstance(response, str) else str(response)

    def act(self, ticker: str, context: dict[str, Any], plan: str, tool_results: dict[str, Any] | None = None) -> BullCase:
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
ADDITIONAL DATA FROM TOOL CALLS (use this to strengthen your analysis):
{json.dumps(tool_results, indent=2, default=str)}
"""

        prompt = f"""You are executing your analysis plan for {ticker}. BUILD THE BULL CASE.

Your plan:
{plan}

Market data: {json.dumps(market_data, indent=2, default=str)}
Financial metrics: {json.dumps(metrics, indent=2, default=str)}
Recent news: {json.dumps(news[:10], default=str) if news else 'None available'}
{kb_section}
{expert_section}{tool_data_section}
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
    "technical_outlook": "Stock is trading above 50/200 DMA with bullish momentum...",
    "sentiment_factors": [
        {{"headline": "headline text", "sentiment": "bullish/bearish/neutral", "signal_strength": "strong/moderate/weak", "catalyst_type": "earnings/product/regulatory/management/macro"}},
        ...
    ],
    "aggregate_news_sentiment": "strongly_bullish / bullish / neutral / bearish / strongly_bearish",
    "sentiment_divergence": "Where news sentiment diverges from price action (e.g. positive news + declining price = institutional selling)",
    "price_target": "$XXX in [horizon] — [methodology and key assumptions, e.g. 'DCF with 11% WACC, 22x terminal multiple on $8.50 normalized FCF']",
    "forecasted_total_return": "X% — [how derived, e.g. '(185/152)-1 = 22%, includes 1.2% dividend yield over 12m horizon']",
    "estimated_industry_return": "X% — [reasoning, e.g. 'semis historically track GDP+4%, mid-cycle expansion supports 8-10% sector return']",
    "idiosyncratic_return": "X% — [reasoning, e.g. '14% alpha = 22% total - 8% sector; driven by share gains in data center + 200bp margin expansion']",
    "estimated_sharpe": "X.XX — [reasoning, e.g. '0.47 = 14% idio / 30% ann vol; moderate, reflects binary regulatory outcome']",
    "estimated_sortino": "X.XX — [reasoning, e.g. '0.64 = 14% idio / 22% downside vol; asymmetric — asset floor limits downside']"
}}

conviction_score: 0-10 scale (0 = no conviction, 10 = highest conviction)

RETURN DECOMPOSITION IS REQUIRED (heuristic estimates, not precise calculations):
Each field MUST include both the estimate AND a 1-2 sentence explanation of how you arrived at it:
- price_target: Target + methodology and key assumptions (DCF, multiple, sum-of-parts)
- forecasted_total_return: Pct return + how derived from price target vs current price
- estimated_industry_return: Sector return + why (cycle position, historical comps, macro backdrop)
- idiosyncratic_return: Alpha + what drives it (company-specific catalysts above sector drift).
  If near zero, your thesis is a sector bet, not a stock pick — say so.
- estimated_sharpe / estimated_sortino: Risk-adjusted number + what vol assumption you used and why.
  For SHORT theses, flip the return sign: Sharpe = (-1 * expected_return) / vol.
  These are heuristic mental-model numbers, not optimizer outputs.

NEWS SENTIMENT EXTRACTION IS REQUIRED:
- For EVERY news headline provided, extract a sentiment factor with classification.
- Aggregate into an overall sentiment reading.
- Identify any divergence between sentiment and price action — this is where alpha lives.

Ground your analysis in the data provided. Be specific with numbers.
For the catalyst_calendar, include at least 4-6 events over the next 12 months.
For technical_outlook, reference actual price levels, moving averages, and recent performance.
Respond ONLY with the JSON object, no other text."""

        response = self.model(prompt)
        response_text = response if isinstance(response, str) else str(response)

        try:
            parsed, retried = retry_extract_json(self.model, prompt, response_text, max_retries=1)
            if retried:
                logger.info(f"Bull case JSON recovered via retry for {ticker}")
            return BullCase(**parsed)
        except Exception as e:
            logger.warning(f"Failed to parse bull case JSON: {e}. Using fallback.")
            return BullCase(
                ticker=ticker,
                thesis=clean_json_artifacts(response_text),
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
News Sentiment: {bull_case.aggregate_news_sentiment if hasattr(bull_case, 'aggregate_news_sentiment') else 'N/A'}
Sentiment Divergence: {bull_case.sentiment_divergence if hasattr(bull_case, 'sentiment_divergence') else 'N/A'}

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
7. SENTIMENT QUALITY CHECK:
   a) Does your sentiment extraction accurately capture the tone of each headline?
   b) Is the aggregate sentiment consistent with your conviction score? If not, explain the divergence.
   c) Are there any sentiment signals you may have over- or under-weighted?
   d) Does the sentiment-price divergence (if any) strengthen or weaken your thesis?
8. RETURN DECOMPOSITION CHECK (these are heuristic estimates, not precise calculations):
   a) Is your price target defensible? What methodology did you use and is it appropriate?
   b) How much of your forecasted return is idiosyncratic vs. just riding the sector?
      If idio_return is near zero, you're making a sector bet, not a stock pick — acknowledge this.
   c) Is your heuristic Sharpe reasonable? Below 0.3 is weak. Above 1.0 should be scrutinized.
   d) Does your Sortino tell a different story than Sharpe? (It should, if downside is asymmetric.)
   e) If this were a SHORT thesis, did you properly flip the return sign?

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
Bearish Conviction: {bear_case.bearish_conviction if hasattr(bear_case, 'bearish_conviction') else bear_case.get('bearish_conviction', 'N/A')}

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
            parsed, _ = extract_json(response_text)
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
