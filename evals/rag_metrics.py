"""
RAG evaluation metrics for multi-agent investment committee.

Implements retrieval-augmented generation quality metrics adapted for
financial analysis.  Measures how faithfully agents use retrieved data
and whether their reasoning is grounded in the available context.

Metrics:
    1. Faithfulness — claims in output are supported by retrieved context
    2. Context Relevance — retrieved chunks are relevant to the query
    3. Answer Relevance — output actually addresses the question asked
    4. Groundedness — fraction of key claims traceable to source data

Inspired by:
    - RAGAS (2024): Retrieval Augmented Generation Assessment
    - FinTextQA (2024): Financial text question answering evaluation
    - Fin-RATE (2026): Financial reasoning evaluation with RAG pathways
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Claim:
    """A single factual claim extracted from agent output."""

    text: str
    supported: bool = False
    source_snippet: str = ""


@dataclass
class RAGScore:
    """Score for a single RAG metric."""

    metric: str
    score: float  # 0.0 to 1.0
    details: str = ""
    claims_checked: int = 0
    claims_supported: int = 0

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "score": round(self.score, 4),
            "details": self.details,
            "claims_checked": self.claims_checked,
            "claims_supported": self.claims_supported,
        }


@dataclass
class RAGEvalResult:
    """Complete RAG evaluation across all metrics."""

    faithfulness: RAGScore
    context_relevance: RAGScore
    answer_relevance: RAGScore
    groundedness: RAGScore

    @property
    def composite_score(self) -> float:
        """Weighted average: faithfulness 40%, groundedness 30%,
        context_relevance 15%, answer_relevance 15%."""
        return (
            0.40 * self.faithfulness.score
            + 0.30 * self.groundedness.score
            + 0.15 * self.context_relevance.score
            + 0.15 * self.answer_relevance.score
        )

    def to_dict(self) -> dict:
        return {
            "composite_score": round(self.composite_score, 4),
            "faithfulness": self.faithfulness.to_dict(),
            "context_relevance": self.context_relevance.to_dict(),
            "answer_relevance": self.answer_relevance.to_dict(),
            "groundedness": self.groundedness.to_dict(),
        }


# ---------------------------------------------------------------------------
# Claim extraction (heuristic)
# ---------------------------------------------------------------------------

# Patterns that indicate a factual claim in financial text
_CLAIM_PATTERNS = [
    r"revenue (?:of|was|is|grew|declined|reached)\s+[\$\d]",
    r"(?:EPS|eps|earnings per share)\s+(?:of|was|is)\s+[\$\d]",
    r"(?:margin|margins?)\s+(?:of|at|was|is|were)\s+[\d]+",
    r"(?:growth|decline)\s+(?:of|at|was|is)\s+[\d]+",
    r"(?:P/E|PE|price.to.earnings)\s+(?:of|at|ratio|was|is)\s+[\d]+",
    r"(?:market cap|valuation)\s+(?:of|at|was|is)\s+[\$\d]",
    r"(?:debt|leverage)\s+(?:of|at|was|is|ratio)\s+[\d]+",
    r"\d+%\s+(?:revenue|growth|margin|return|decline)",
    r"(?:FY|fiscal|quarter|Q[1-4])\s*\d{2,4}",
]

_CLAIM_RE = re.compile("|".join(_CLAIM_PATTERNS), re.IGNORECASE)


def extract_claims(text: str) -> List[str]:
    """Extract factual claim sentences from text using heuristics.

    Splits text into sentences and returns those containing
    financial claim patterns (numbers, metrics, percentages).
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    claims: List[str] = []
    for sent in sentences:
        sent = sent.strip()
        if not sent or len(sent) < 10:
            continue
        if _CLAIM_RE.search(sent):
            claims.append(sent)
    return claims


def _normalize(text: str) -> str:
    """Lowercase and strip non-alphanumeric for fuzzy matching."""
    return re.sub(r"[^a-z0-9\s.]", "", text.lower()).strip()


def _sentence_overlap(claim: str, context: str) -> float:
    """Compute word overlap between a claim and context.

    Returns fraction of claim words found in context.
    """
    claim_words = set(_normalize(claim).split())
    context_words = set(_normalize(context).split())
    if not claim_words:
        return 0.0
    return len(claim_words & context_words) / len(claim_words)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

_SUPPORT_THRESHOLD = 0.50  # Min word overlap to count as "supported"


def compute_faithfulness(
    output_text: str,
    context_chunks: Sequence[str],
) -> RAGScore:
    """Measure fraction of output claims supported by context.

    For each claim-like sentence in the output, check if any context
    chunk contains enough overlapping words to constitute support.
    """
    claims = extract_claims(output_text)
    if not claims:
        return RAGScore(
            metric="faithfulness",
            score=1.0,
            details="No factual claims detected in output",
        )

    context_blob = " ".join(context_chunks)
    supported = 0
    for claim in claims:
        if _sentence_overlap(claim, context_blob) >= _SUPPORT_THRESHOLD:
            supported += 1

    score = supported / len(claims)
    return RAGScore(
        metric="faithfulness",
        score=score,
        details=f"{supported}/{len(claims)} claims supported by context",
        claims_checked=len(claims),
        claims_supported=supported,
    )


def compute_context_relevance(
    query: str,
    context_chunks: Sequence[str],
) -> RAGScore:
    """Measure how relevant retrieved context chunks are to the query.

    Scores each chunk's word overlap with the query, returns average.
    """
    if not context_chunks:
        return RAGScore(
            metric="context_relevance",
            score=0.0,
            details="No context chunks provided",
        )

    relevance_scores: List[float] = []
    for chunk in context_chunks:
        overlap = _sentence_overlap(query, chunk)
        relevance_scores.append(min(overlap * 2.0, 1.0))  # Scale up, cap at 1.0

    avg = sum(relevance_scores) / len(relevance_scores)
    return RAGScore(
        metric="context_relevance",
        score=avg,
        details=f"Average relevance across {len(context_chunks)} chunks: {avg:.2f}",
    )


def compute_answer_relevance(
    query: str,
    output_text: str,
) -> RAGScore:
    """Measure whether the output addresses the query.

    Uses word overlap between query and output as a proxy.
    """
    if not output_text.strip():
        return RAGScore(
            metric="answer_relevance",
            score=0.0,
            details="Empty output",
        )

    overlap = _sentence_overlap(query, output_text)
    score = min(overlap * 2.5, 1.0)  # Scale and cap
    return RAGScore(
        metric="answer_relevance",
        score=score,
        details=f"Query-output word overlap: {overlap:.2f}",
    )


def compute_groundedness(
    output_text: str,
    source_data: Dict[str, Any],
) -> RAGScore:
    """Measure how grounded the output is in structured source data.

    Flattens source_data values to strings and checks what fraction
    of output claims reference specific data points from the source.
    """
    # Flatten source data to searchable text
    source_parts: List[str] = []

    def _flatten(obj: Any) -> None:
        if isinstance(obj, str):
            source_parts.append(obj)
        elif isinstance(obj, (int, float)):
            source_parts.append(str(obj))
        elif isinstance(obj, dict):
            for v in obj.values():
                _flatten(v)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _flatten(item)

    _flatten(source_data)
    source_blob = " ".join(source_parts)

    claims = extract_claims(output_text)
    if not claims:
        return RAGScore(
            metric="groundedness",
            score=1.0,
            details="No factual claims to verify",
        )

    grounded = 0
    for claim in claims:
        if _sentence_overlap(claim, source_blob) >= _SUPPORT_THRESHOLD:
            grounded += 1

    score = grounded / len(claims)
    return RAGScore(
        metric="groundedness",
        score=score,
        details=f"{grounded}/{len(claims)} claims grounded in source data",
        claims_checked=len(claims),
        claims_supported=grounded,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate_rag(
    query: str,
    output_text: str,
    context_chunks: Sequence[str],
    source_data: Dict[str, Any] | None = None,
) -> RAGEvalResult:
    """Run all RAG evaluation metrics.

    Args:
        query: The original investment question or ticker.
        output_text: The committee's textual output.
        context_chunks: Retrieved text chunks used for analysis.
        source_data: Optional structured data (fundamentals, etc.).

    Returns:
        RAGEvalResult with all four metric scores.
    """
    return RAGEvalResult(
        faithfulness=compute_faithfulness(output_text, context_chunks),
        context_relevance=compute_context_relevance(query, context_chunks),
        answer_relevance=compute_answer_relevance(query, output_text),
        groundedness=compute_groundedness(
            output_text, source_data or {},
        ),
    )
