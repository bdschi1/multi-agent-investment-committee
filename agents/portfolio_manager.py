"""
Portfolio Manager Agent — synthesizes bull and bear cases into a decision.

This agent acts as the PM who chairs the investment committee. The PM is
deeply trading-fluent and works closely with his head trader — he thinks
in vol surfaces, factor tilts, and event paths. He weighs both sides,
assesses implied/historical vol, factor exposures, and produces the final
recommendation with explicit reasoning and a T signal for downstream RL
consumption.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agents.base import (
    AgentRole,
    BaseInvestmentAgent,
    CommitteeMemo,
    Rebuttal,
    extract_json,
    retry_extract_json,
    clean_json_artifacts,
)

logger = logging.getLogger(__name__)


class PortfolioManagerAgent(BaseInvestmentAgent):
    """Trading-fluent PM: weighs bull vs bear, consults his trader on vol/factors, and makes the final call."""

    def __init__(self, model: Any, tool_registry: Any = None):
        super().__init__(model=model, role=AgentRole.PORTFOLIO_MANAGER, tool_registry=tool_registry)

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

        user_kb = context.get('user_kb', '').strip()
        kb_section = ""
        if user_kb:
            kb_section = f"""

{user_kb}
"""

        # Phase C: PM guidance from HITL review step
        pm_guidance = context.get('pm_guidance', '').strip()
        guidance_section = ""
        if pm_guidance:
            guidance_section = f"""

COMMITTEE CHAIR GUIDANCE (post-review):
After reviewing the analyst outputs and debate, the committee chair has
provided the following direction. You MUST factor this into your final decision:
{pm_guidance}
"""

        # Phase C: Prior analyses from session memory
        prior_analyses = context.get('prior_analyses', [])
        memory_section = ""
        if prior_analyses:
            memory_section = f"""

PRIOR ANALYSES OF {ticker} (from this session):
{json.dumps(prior_analyses, indent=2, default=str)}
Consider: how has the investment case evolved? What changed since the last analysis?
If conviction or recommendation shifted, explain why.
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
{expert_section}{kb_section}{guidance_section}{memory_section}
Now THINK about what you've heard — as a PM who speaks fluently with his trader. Consider:
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

TRADING LENS — you speak to your head trader daily and think fluently in market microstructure:
9. IMPLIED vs HISTORICAL VOLATILITY: Is the options market pricing in more or less
   risk than realized? If IV >> HV, the market is scared — is it right? If IV << HV,
   complacency or opportunity? What does the vol surface (skew, term structure) tell you?
10. FACTOR EXPOSURES: What systematic factor tilts does this position carry?
    (momentum, value, quality, size, volatility, growth). How does adding this name
    affect your overall factor exposure?
11. EVENT PATH: Map the ordered sequence of near-term events (earnings, FOMC, product
    launch, regulatory decision) and how each event changes the conviction envelope.
    What is the event-by-event path dependency?
12. CONVICTION CHANGE TRIGGERS: Be precise — what specific, observable, measurable
    data points would make you:
    a) Size UP the position? (e.g. "IV crush post-earnings below 25th percentile")
    b) Cut the position? (e.g. "3-day close below 200 DMA on 2x volume")
    c) REVERSE the thesis entirely?

QUANTITATIVE SIZING HEURISTICS (heuristic reasoning — not precise optimization):
13. RETURN DECOMPOSITION VALIDATION: The Sector Analyst estimated idiosyncratic return.
    Do you agree with their decomposition? Is the "alpha" genuine or is it sector/factor beta?
    After hearing the Risk Manager's challenge, what is YOUR validated idio return estimate?
14. SHARPE / SORTINO SYNTHESIS: Given both sides, what is your heuristic risk-adjusted return?
    - Sharpe ≈ idio_return / vol (for shorts: use -1 × return)
    - Sortino ≈ idio_return / downside_vol (focuses on loss distribution)
    - If Sharpe < 0.3, the position may not justify the risk budget.
    - If Sortino diverges significantly from Sharpe, the tail risk matters — size accordingly.
15. SIZING METHOD SELECTION: The Macro Analyst recommended a sizing method.
    Do you agree? Apply it heuristically:
    - Proportional: NMV ∝ α (simple, if alpha is trusted)
    - Risk Parity: NMV ∝ α/σ (normalize by vol)
    - Mean-Variance: NMV ∝ α/σ² (penalize vol more)
    - Shrunk MV: NMV ∝ α/[p×σ² + (1-p)×σ²_sector] (blend when estimates are noisy)
    What sizing rationale fits this specific name?
16. VOL TARGETING: The Macro Analyst set a portfolio vol target.
    Does this position's vol contribution stay within that budget?
    Vol targeting > GMV targeting because vol is persistent and predictable.

Think like a PM who puts capital at risk and manages the book's P&L daily.
You are a fundamental expert who also uses quantitative metrics and tools to size
and manage the portfolio. Most names are fundamental thesis-driven, but you also
run statistical positions to manage portfolio risk metrics. You speak to your
head trader frequently — he keeps you honest on levels, flow, and execution.
The macro context often tips the balance when bull and bear are closely matched.
This isn't academic — it's a real decision with real mark-to-market.
The best trades come from non-consensus views backed by rigorous analysis."""

        response = self.model(prompt)
        return response if isinstance(response, str) else str(response)

    def plan(self, ticker: str, context: dict[str, Any], thinking: str) -> str:
        """Plan the synthesis and decision framework."""
        tool_catalog = self.get_tool_catalog()
        tool_section = ""
        if tool_catalog:
            tool_section = f"""

AVAILABLE TOOLS (optional — call any that would help your synthesis):
{tool_catalog}

Suggested tools for PM synthesis: get_earnings_history, compare_peers

To request tool data, add a TOOL_CALLS block at the END of your plan:
TOOL_CALLS:
- tool_name(ticker="{ticker}")
"""

        prompt = f"""Based on your initial thinking about {ticker}:

{thinking}

PLAN your decision framework (PM with trading fluency):
1. What weight will you give to the bull vs bear case and why?
2. What are the key swing factors that determine the recommendation?
3. How will you determine position sizing given the vol regime?
4. What risk mitigants would you require?
5. What would make you revisit this decision?
6. VOLATILITY FRAMEWORK: How does implied vs historical vol inform your sizing and timing?
7. FACTOR AWARENESS: What factor tilts does this position create? Is the timing right given factor rotation?
8. EVENT PATH PLANNING: What is the critical path of events over the next 30/60/90 days?
   How does each event change the position's expected value?
9. CONVICTION LADDER: Define the specific triggers at each conviction level — what moves you
   from "small starter" to "full size" or from "hold" to "cut"?

QUANTITATIVE SIZING PLAN (heuristic reasoning — not optimization):
10. How will you validate the analyst's idiosyncratic return estimate after hearing the bear challenge?
11. What is your framework for computing heuristic Sharpe and Sortino?
    (Remember: for shorts, flip the return sign)
12. Which sizing method will you use (proportional / risk_parity / mean_variance / shrunk_MV)?
    Does the Macro Analyst's recommendation fit this name?
13. How does this position fit within the portfolio vol target? What is the max NMV?
{tool_section}
Be explicit about your decision framework. Good PMs are transparent about
how they weigh evidence. Good traders are precise about triggers and levels — and you speak their language."""

        response = self.model(prompt)
        return response if isinstance(response, str) else str(response)

    def act(self, ticker: str, context: dict[str, Any], plan: str, tool_results: dict[str, Any] | None = None) -> CommitteeMemo:
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
ADDITIONAL DATA FROM TOOL CALLS (factor into your synthesis):
{json.dumps(tool_results, indent=2, default=str)}
"""

        # Phase C: PM guidance from HITL review step
        pm_guidance = context.get('pm_guidance', '').strip()
        guidance_section = ""
        if pm_guidance:
            guidance_section = f"""
COMMITTEE CHAIR GUIDANCE (post-review):
After reviewing the analyst outputs and debate, the committee chair has
provided the following direction. You MUST factor this into your final decision:
{pm_guidance}
"""

        # Phase C: Prior analyses from session memory
        prior_analyses = context.get('prior_analyses', [])
        memory_section = ""
        if prior_analyses:
            memory_section = f"""
PRIOR ANALYSES OF {ticker} (from this session):
{json.dumps(prior_analyses, indent=2, default=str)}
Consider: how has the investment case evolved? What changed since the last analysis?
"""

        prompt = f"""You are the Portfolio Manager making the FINAL DECISION on {ticker}.

Your decision framework:
{plan}

BULL CASE: {json.dumps(bull_data, indent=2, default=str)}
BEAR CASE: {json.dumps(bear_data, indent=2, default=str)}
MACRO ENVIRONMENT: {json.dumps(macro_data, indent=2, default=str)}
DEBATE: {json.dumps(debate, indent=2, default=str)}
{expert_section}{kb_section}{tool_data_section}{guidance_section}{memory_section}
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
    "time_horizon": "e.g. 6-12 months",
    "implied_vol_assessment": "IV is X% vs HV of Y%. The vol surface shows [skew/term structure assessment]. This implies the market is pricing [more/less] risk than realized. For sizing: [implication].",
    "event_path": [
        "Event 1 (date/timeframe): expected impact and how it changes conviction",
        "Event 2 (date/timeframe): expected impact and how it changes conviction",
        ...
    ],
    "conviction_change_triggers": {{
        "size_up": "Specific trigger to increase position (e.g. 'Post-earnings IV crush below 25th pctile')",
        "cut_position": "Specific trigger to reduce/exit (e.g. '3-day close below 200 DMA on 2x avg volume')",
        "reverse_thesis": "What would make you flip from long to short or vice versa"
    }},
    "factor_exposures": {{
        "momentum": "positive/negative/neutral — because...",
        "value": "positive/negative/neutral — because...",
        "quality": "positive/negative/neutral — because...",
        "size": "large/mid/small cap tilt",
        "volatility": "low/high vol name relative to market"
    }},
    "idio_return_estimate": "X% — PM's validated idio return after weighing bull vs bear decomposition. This is the alpha the position captures.",
    "sharpe_estimate": "X.XX — heuristic Sharpe: idio_return / vol. For shorts, uses -1*expected_return. Below 0.3 = weak.",
    "sortino_estimate": "X.XX — heuristic Sortino: idio_return / downside_vol. For shorts, uses -1*expected_return. Divergence from Sharpe signals tail risk.",
    "sizing_method_used": "proportional / risk_parity / mean_variance / shrunk_mean_variance — chosen because [rationale].",
    "target_nmv_rationale": "Target NMV is X% of AUM because: alpha estimate is Y%, vol is Z%, using [method] → NMV ∝ α/σ^n. Constrained by vol budget.",
    "vol_target_rationale": "Portfolio vol target is X%. This position contributes ~Y% marginal vol. Vol targeting > GMV targeting because vol is persistent.",
    "pm_synthesis_rationale": "2-4 sentences: What tipped the balance in this decision? Why did you weight one side over the other? What would you tell the IC in the meeting — the real reason for this call, not just restating the thesis. Be candid about what gave you conviction or what held you back.",
    "position_direction": 1,
    "raw_confidence": 0.75
}}

conviction: 0-10 (your overall confidence in this recommendation)

POSITION DIRECTION AND CONFIDENCE (for T signal computation):
- position_direction: +1 if your recommendation is STRONG BUY/BUY, -1 if SELL/ACTIVE SHORT, 0 if HOLD/UNDERWEIGHT with no position
- raw_confidence: your conviction scaled to [0, 1] — this is conviction/10 multiplied by how certain you are in the analysis quality.
  If you are confident in your analysis AND the recommendation, raw_confidence should be high (0.7-1.0).
  If the data is ambiguous or you're unsure, raw_confidence should be lower (0.3-0.6).
  This feeds the T signal: T = direction * entropy-adjusted confidence, which an RL agent uses downstream.

TRADING REQUIREMENTS (your trader will hold you to these):
- implied_vol_assessment is REQUIRED — assess IV vs HV and what it means for sizing/timing
- event_path is REQUIRED — map at least 3 near-term events in chronological order
- conviction_change_triggers is REQUIRED — be precise about what moves you up/down the conviction ladder
- factor_exposures is REQUIRED — identify the dominant factor tilts this position creates

QUANTITATIVE SIZING HEURISTICS ARE REQUIRED (heuristic estimates, not optimizer outputs):
- idio_return_estimate: Your validated alpha estimate after weighing bull/bear decompositions
- sharpe_estimate: Heuristic Sharpe = idio_return / vol. For shorts, flip return sign.
- sortino_estimate: Heuristic Sortino = idio_return / downside_vol. For shorts, flip return sign.
- sizing_method_used: Which method you chose and WHY it fits this name
- target_nmv_rationale: How the alpha, vol, and sizing method combine to set the NMV
- vol_target_rationale: How this position fits the portfolio vol target.
  Vol targeting > GMV targeting because volatility is persistent and predictable.

IMPORTANT: Explain WHY you weighted one side over the other.
Every recommendation must explicitly state what you'd need to see to change your mind.
Respond ONLY with the JSON object, no other text."""

        response = self.model(prompt)
        response_text = response if isinstance(response, str) else str(response)

        try:
            parsed, retried = retry_extract_json(self.model, prompt, response_text, max_retries=1)
            if retried:
                logger.info(f"CommitteeMemo JSON extraction required retry for {ticker}")
            return CommitteeMemo(**parsed)
        except Exception as e:
            logger.warning(f"Failed to parse committee memo JSON: {e}. Using fallback.")
            return CommitteeMemo(
                ticker=ticker,
                recommendation="HOLD",
                position_size="Not determined — parsing degraded",
                conviction=5.0,
                thesis_summary=clean_json_artifacts(response_text),
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
IV Assessment: {memo.implied_vol_assessment if hasattr(memo, 'implied_vol_assessment') else 'N/A'}
Event Path: {memo.event_path if hasattr(memo, 'event_path') else 'N/A'}
Factor Exposures: {memo.factor_exposures if hasattr(memo, 'factor_exposures') else 'N/A'}
T Signal Components: direction={memo.position_direction if hasattr(memo, 'position_direction') else 'N/A'}, confidence={memo.raw_confidence if hasattr(memo, 'raw_confidence') else 'N/A'}

REFLECT as a trading-fluent PM (you speak to your trader daily):
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

TRADING SELF-CHECK (what would your head trader push back on?):
8. Is your IV assessment consistent with your position sizing? (High IV = smaller size or wait for crush)
9. Does your event path properly sequence the risk? Are you sized correctly for the next binary event?
10. Are your conviction change triggers specific enough to be actionable intraday?
    (A good trigger is "3-day close below $150 on >2x avg volume", not "if things get worse")
11. Do your factor exposures create any unintended portfolio-level tilts?
12. Is your raw_confidence score honest? If you're uncertain about the data quality,
    this should be reflected in a lower confidence, not just a lower conviction.

QUANTITATIVE SIZING SELF-CHECK (heuristic, not optimization):
13. Is your idio_return_estimate defensible? Did you properly strip out sector/factor return,
    or are you claiming alpha that's really beta?
14. Is your heuristic Sharpe realistic? Below 0.3 is weak risk-adjusted return.
    Above 1.0 deserves scrutiny. Does it justify the position size?
15. Does your Sortino tell a different story than Sharpe? If downside vol is much higher
    than symmetric vol, you should be sized smaller than Sharpe alone would suggest.
16. Is the sizing method appropriate for this name's characteristics?
    (e.g., shrunk MV for noisy estimates, risk parity when vol dispersion is wide)
17. Does the position's vol contribution stay within the portfolio vol target?
    If not, you need to reduce NMV or acknowledge the budget breach.
18. Is your NMV rationale specific enough to be actionable? "Small position" is vague.
    "2.5% of AUM using risk-parity sizing: 8% alpha / 32% vol = 0.25 normalized" is actionable.

The best PMs are brutally honest about their own decision quality.
Your trader knows his levels cold — make sure yours are just as precise."""

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

    # _extract_json removed — using shared extract_json() from agents.base
