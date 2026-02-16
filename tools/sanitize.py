"""
Input sanitization for uploaded documents.

Defends against prompt injection from PDF/DOCX/TXT content by
detecting and redacting 8 common injection patterns. Ported from
llm-long-short-arena (src/sanitize.py).
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patterns that look like prompt-injection attempts
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an)\s+", re.IGNORECASE),
    re.compile(r"system\s*:\s*", re.IGNORECASE),
    re.compile(r"<\s*/?\s*system\s*>", re.IGNORECASE),
    re.compile(r"assistant\s*:\s*", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all|your)\s+", re.IGNORECASE),
    re.compile(r"new\s+instructions?\s*:", re.IGNORECASE),
    re.compile(r"override\s+(your\s+)?(instructions|rules|prompt)", re.IGNORECASE),
]


def sanitize_document_text(text: str) -> str:
    """Strip potential prompt-injection content from extracted document text.

    Scans for 8 known injection patterns and replaces matches with
    ``[REDACTED]``. Safe to call on any string — returns the input
    unchanged when no patterns are found.
    """
    cleaned = text
    found_any = False

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(cleaned):
            found_any = True
            cleaned = pattern.sub("[REDACTED]", cleaned)

    if found_any:
        logger.warning(
            "Prompt injection patterns detected and redacted from document text"
        )

    return cleaned
