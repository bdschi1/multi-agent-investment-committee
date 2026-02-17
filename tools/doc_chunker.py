"""
Document chunker for user-uploaded knowledge base files.

Adapted from the local chunker project (bds_repos). Provides:
- File ingestion: PDF, DOCX, TXT
- Chunking decision: small files injected whole, large files chunked
- Token-aware chunking: respects LLM context limits with tiktoken
- Boilerplate removal: strips disclosure sections from equity research
- Smart page scoring: importance-ranked page selection for large PDFs
- Prompt injection defense: 8-pattern sanitization on all extracted text

The processed KB is injected into agent context as supplementary research.
Agents treat it as one resource among many — they rely on their own judgment.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from tools.sanitize import sanitize_document_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token counting — tiktoken with graceful fallback
# ---------------------------------------------------------------------------

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    logger.warning("tiktoken not installed — falling back to word-based token estimate")

    def _count_tokens(text: str) -> int:
        return len(text.split()) * 4 // 3  # rough 1.33 tokens/word estimate


# ---------------------------------------------------------------------------
# Smart page scoring (ported from llm-long-short-arena)
# ---------------------------------------------------------------------------

_HIGH_VALUE_HEADERS = re.compile(
    r"(executive\s+summary|investment\s+(thesis|summary|conclusion)"
    r"|key\s+(findings|takeaways|drivers|risks)"
    r"|financial\s+(summary|highlights|overview)"
    r"|valuation|price\s+target|recommendation"
    r"|conclusion|risk\s+factors|catalysts"
    r"|earnings|revenue|guidance|outlook)",
    re.IGNORECASE,
)

# Large-PDF char threshold: if total extracted chars exceed this, apply
# importance-scored page selection instead of using every page.
_SMART_TRUNCATE_CHAR_LIMIT = 60_000


def _score_page(text: str, has_tables: bool = False) -> float:
    """Heuristic importance score for a PDF page (higher = more important).

    Scoring rules (ported from llm-long-short-arena ``_score_page``):
    - +3.0 for pages containing high-value section headers
    - +2.0 for pages with tables (likely financials)
    - +1.5 for pages with >5% digit density (quantitative content)
    - -2.0 for very short pages (<200 chars — cover pages, disclaimers)
    """
    score = 1.0

    if _HIGH_VALUE_HEADERS.search(text):
        score += 3.0

    if has_tables:
        score += 2.0

    digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
    if digit_ratio > 0.05:
        score += 1.5

    if len(text) < 200:
        score -= 2.0

    return score


def _smart_select_pages(
    pages: list[dict],
    char_limit: int = _SMART_TRUNCATE_CHAR_LIMIT,
) -> str:
    """Build the best possible text within *char_limit* using importance scoring.

    Strategy (ported from llm-long-short-arena ``smart_truncate``):
    - Always include the first 2 pages (context/intro) and last page
    - Fill remaining budget with highest-scored pages in document order
    - Sanitize the combined result against prompt injection
    """
    if not pages:
        return ""

    total_chars = sum(p["chars"] for p in pages)

    # Everything fits — concatenate, sanitize, return
    if total_chars <= char_limit:
        combined = "\n\n".join(p["text"] for p in pages)
        return sanitize_document_text(combined)

    logger.info(
        "PDF exceeds char limit (%d > %d), applying smart page selection",
        total_chars,
        char_limit,
    )

    # Reserve slots for first 2 and last page
    must_include: set[int] = set()
    for i in range(min(2, len(pages))):
        must_include.add(i)
    must_include.add(len(pages) - 1)

    # Score remaining pages
    scored = []
    for i, page in enumerate(pages):
        if i not in must_include:
            scored.append((i, _score_page(page["text"], page.get("has_tables", False))))
    scored.sort(key=lambda x: x[1], reverse=True)

    # Build selection within budget
    selected_indices = set(must_include)
    budget = char_limit - sum(pages[i]["chars"] for i in must_include)

    for idx, _sc in scored:
        if budget <= 0:
            break
        if pages[idx]["chars"] <= budget:
            selected_indices.add(idx)
            budget -= pages[idx]["chars"]

    # Reassemble in document order
    selected = sorted(selected_indices)
    parts = [pages[i]["text"] for i in selected]

    skipped = len(pages) - len(selected)
    if skipped > 0:
        logger.info(
            "Smart page selection: kept %d/%d pages (skipped %d low-value pages)",
            len(selected),
            len(pages),
            skipped,
        )

    combined = "\n\n".join(parts)
    return sanitize_document_text(combined)


# ---------------------------------------------------------------------------
# File readers — PDF, DOCX, TXT
# ---------------------------------------------------------------------------

def _parse_page_range(page_range: str | None, total_pages: int) -> list[int]:
    """
    Parse a page range string into a sorted list of 0-based page indices.

    Accepts human-friendly 1-based page numbers in these formats:
        "5"         → single page
        "1-10"      → inclusive range
        "1-5,8,12-15" → mixed ranges and singles

    Returns sorted, deduplicated list of 0-based indices clamped to [0, total_pages).
    Returns all pages if page_range is None or empty.
    """
    if not page_range:
        return list(range(total_pages))

    indices: set[int] = set()
    for part in page_range.split(","):
        part = part.strip()
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = max(int(start_s.strip()) - 1, 0)
            end = min(int(end_s.strip()), total_pages)  # end is exclusive after -1+1
            indices.update(range(start, end))
        else:
            idx = int(part) - 1
            if 0 <= idx < total_pages:
                indices.add(idx)

    return sorted(indices)


def _read_pdf(path: Path, page_range: str | None = None) -> str:
    """
    Extract text from a PDF file.

    Args:
        path: Path to the PDF.
        page_range: Optional page selection string (1-based, human-friendly).
                    Examples: "5", "1-10", "1-5,8,12-15".
                    None or "" reads all pages.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.warning("pypdf not installed — cannot read PDF files")
        return ""

    try:
        reader = PdfReader(str(path))
        total = len(reader.pages)
        indices = _parse_page_range(page_range, total)

        if not indices:
            logger.warning(
                f"Page range '{page_range}' produced no valid pages "
                f"(PDF has {total} pages)"
            )
            return ""

        # Extract per-page text with metadata for smart scoring
        extracted: list[dict] = []
        for i in indices:
            text = reader.pages[i].extract_text() or ""
            extracted.append({"page": i + 1, "text": text, "chars": len(text)})

        logger.info(
            f"PDF {path.name}: extracted {len(indices)}/{total} pages "
            f"(pages {indices[0]+1}-{indices[-1]+1})"
        )

        total_chars = sum(p["chars"] for p in extracted)

        # Large PDF → smart page selection with importance scoring
        if total_chars > _SMART_TRUNCATE_CHAR_LIMIT:
            return _smart_select_pages(extracted)

        # Normal PDF → concatenate and sanitize
        return sanitize_document_text("\n\n".join(p["text"] for p in extracted))
    except Exception as e:
        logger.warning(f"Failed to read PDF {path.name}: {e}")
        return ""


def _read_docx(path: Path) -> str:
    """Extract text from a DOCX file."""
    try:
        from docx import Document
    except ImportError:
        logger.warning("python-docx not installed — cannot read DOCX files")
        return ""

    try:
        doc = Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return sanitize_document_text("\n\n".join(paragraphs))
    except Exception as e:
        logger.warning(f"Failed to read DOCX {path.name}: {e}")
        return ""


def _read_txt(path: Path) -> str:
    """Read a plain text file."""
    try:
        return sanitize_document_text(path.read_text(errors="ignore"))
    except Exception as e:
        logger.warning(f"Failed to read TXT {path.name}: {e}")
        return ""


_READERS = {
    ".pdf": _read_pdf,
    ".docx": _read_docx,
    ".doc": _read_docx,  # best-effort — python-docx handles some .doc files
    ".txt": _read_txt,
}


# ---------------------------------------------------------------------------
# Boilerplate removal (equity research disclosures)
# ---------------------------------------------------------------------------

_DISCARD_PATTERNS = [
    r"disclosure appendix",
    r"important disclosures",
    r"regulatory disclosures",
    r"analyst certification",
    r"not an offer to sell",
    r"past performance",
]

_SECTION_BREAK = re.compile(r"^(exhibit|figure|table)\s+\d+", re.IGNORECASE)


def _strip_boilerplate(text: str) -> str:
    """Remove trailing disclosure / boilerplate sections."""
    for pattern in _DISCARD_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            text = text[:match.start()]
    return text.strip()


# ---------------------------------------------------------------------------
# Chunking logic
# ---------------------------------------------------------------------------

# Files under this token count are injected whole (no chunking needed)
SMALL_FILE_THRESHOLD = 3000  # ~3K tokens ≈ 2-3 pages

# Max tokens per chunk when chunking is needed
MAX_CHUNK_TOKENS = 800


def _chunk_text(text: str) -> list[str]:
    """
    Split text into token-bounded chunks, respecting paragraph and section breaks.

    Adapted from EquityResearchChunker in the local chunker project.
    """
    text = _strip_boilerplate(text)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current: list[str] = []
    token_count = 0

    for para in paragraphs:
        para_tokens = _count_tokens(para)

        # Section break → flush current chunk
        if _SECTION_BREAK.match(para):
            if current:
                chunks.append("\n\n".join(current))
                current = []
                token_count = 0

        # Would exceed limit → flush and start new chunk
        if token_count + para_tokens > MAX_CHUNK_TOKENS and current:
            chunks.append("\n\n".join(current))
            current = [para]
            token_count = para_tokens
        else:
            current.append(para)
            token_count += para_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

MAX_FILES = 5


def process_uploads(
    file_paths: list[str | Path],
    page_range: str | None = None,
) -> list[dict[str, Any]]:
    """
    Process uploaded files into a knowledge base for agent consumption.

    Never raises — individual file failures are captured and reported
    so the pipeline continues with whatever documents succeeded.

    Args:
        file_paths: List of file paths (up to MAX_FILES). Gradio gives
                    temporary paths after upload.
        page_range: Optional page selection for PDFs (1-based, human-friendly).
                    Examples: "5", "1-10", "1-5,8,12-15".
                    None reads all pages. Ignored for non-PDF files.

    Returns:
        List of dicts, each with:
            - filename: original filename
            - chunks: list of text strings (1 if small file, N if chunked)
            - chunked: bool — whether the file was chunked
            - token_count: approximate total tokens
            - chunk_count: number of chunks
            - status: "ok" | "skipped" | "error"
            - error: optional error description (only when status != "ok")
    """
    if not file_paths:
        return []

    # Enforce file limit
    paths = [Path(p) for p in file_paths[:MAX_FILES]]
    kb_docs: list[dict[str, Any]] = []

    for path in paths:
        try:
            ext = path.suffix.lower()
            reader = _READERS.get(ext)

            if reader is None:
                logger.warning(f"Unsupported file type: {ext} ({path.name}) — skipping")
                kb_docs.append({
                    "filename": path.name,
                    "chunks": [],
                    "chunked": False,
                    "token_count": 0,
                    "chunk_count": 0,
                    "status": "skipped",
                    "error": f"Unsupported file type: {ext}",
                })
                continue

            # Pass page_range to PDF reader; other readers ignore it
            if ext == ".pdf":
                text = reader(path, page_range=page_range)
            else:
                text = reader(path)
            if not text.strip():
                logger.warning(f"Empty content from {path.name} — skipping")
                kb_docs.append({
                    "filename": path.name,
                    "chunks": [],
                    "chunked": False,
                    "token_count": 0,
                    "chunk_count": 0,
                    "status": "skipped",
                    "error": "File was empty or could not be read",
                })
                continue

            total_tokens = _count_tokens(text)

            if total_tokens <= SMALL_FILE_THRESHOLD:
                # Small file → inject whole
                kb_docs.append({
                    "filename": path.name,
                    "chunks": [text],
                    "chunked": False,
                    "token_count": total_tokens,
                    "chunk_count": 1,
                    "status": "ok",
                })
                logger.info(
                    f"KB: {path.name} — {total_tokens} tokens, injecting whole"
                )
            else:
                # Large file → chunk
                chunks = _chunk_text(text)
                if not chunks:
                    kb_docs.append({
                        "filename": path.name,
                        "chunks": [],
                        "chunked": True,
                        "token_count": total_tokens,
                        "chunk_count": 0,
                        "status": "error",
                        "error": "Chunking produced no output",
                    })
                    logger.warning(
                        f"KB: {path.name} — {total_tokens} tokens, "
                        f"chunking produced 0 chunks"
                    )
                else:
                    kb_docs.append({
                        "filename": path.name,
                        "chunks": chunks,
                        "chunked": True,
                        "token_count": total_tokens,
                        "chunk_count": len(chunks),
                        "status": "ok",
                    })
                    logger.info(
                        f"KB: {path.name} — {total_tokens} tokens, "
                        f"chunked into {len(chunks)} pieces"
                    )

        except Exception as e:
            logger.warning(f"KB: Failed to process {path.name}: {e}")
            kb_docs.append({
                "filename": path.name,
                "chunks": [],
                "chunked": False,
                "token_count": 0,
                "chunk_count": 0,
                "status": "error",
                "error": str(e),
            })

    return kb_docs


def format_kb_for_prompt(kb_docs: list[dict[str, Any]], max_tokens: int = 12000) -> str:
    """
    Format processed KB documents into a prompt-ready string.

    Concatenates all chunks across all documents up to max_tokens,
    with clear document boundaries. Agents receive this as a
    SUPPLEMENTARY RESEARCH section.

    Args:
        kb_docs: Output from process_uploads()
        max_tokens: Token budget for KB content in prompt

    Returns:
        Formatted string ready for prompt injection, or empty string if no KB.
    """
    if not kb_docs:
        return ""

    # Separate usable docs from failed ones
    usable_docs = [d for d in kb_docs if d.get("status") == "ok"]
    failed_docs = [d for d in kb_docs if d.get("status") in ("skipped", "error")]

    sections: list[str] = []
    tokens_used = 0

    for doc in usable_docs:
        header = f"--- Document: {doc['filename']} ---"
        header_tokens = _count_tokens(header)

        for chunk in doc["chunks"]:
            chunk_tokens = _count_tokens(chunk)

            if tokens_used + header_tokens + chunk_tokens > max_tokens:
                # Budget exhausted — stop adding content
                if not sections:
                    # At least include a truncated version of the first chunk
                    sections.append(header)
                    words = chunk.split()
                    truncated = " ".join(words[:max_tokens])
                    sections.append(truncated + "\n[... truncated ...]")
                break

            if not sections or sections[-1] != header:
                sections.append(header)
                tokens_used += header_tokens

            sections.append(chunk)
            tokens_used += chunk_tokens

        # Check if we've hit the budget
        if tokens_used >= max_tokens:
            break

    # If no usable content but there were failed docs, still return a note
    if not sections and not failed_docs:
        return ""

    total_usable = len(usable_docs)
    total_chunks = sum(d["chunk_count"] for d in usable_docs)

    preamble = (
        f"SUPPLEMENTARY RESEARCH ({total_usable} document(s), "
        f"{total_chunks} section(s)):\n"
        "The following documents were uploaded by the user as additional context.\n"
        "Treat this as one resource among many — rely on your own analytical "
        "judgment. The user's documents may contain valuable insights, data, "
        "or perspectives, but they may also contain biases or errors.\n"
    )

    # Add note about failed documents so agents are aware
    if failed_docs:
        failed_names = [
            f"{d['filename']} ({d.get('error', 'unknown error')})"
            for d in failed_docs
        ]
        preamble += (
            f"\nNote: {len(failed_docs)} document(s) could not be processed: "
            + ", ".join(failed_names) + "\n"
        )

    body = "\n".join(sections) if sections else "[No document content available]"
    return preamble + body


def get_upload_summary(kb_docs: list[dict[str, Any]]) -> str:
    """
    Return a human-readable summary of upload results including failures.

    Used by the UI to show which documents were processed and which failed.
    """
    if not kb_docs:
        return ""

    ok_docs = [d for d in kb_docs if d.get("status") == "ok"]
    failed_docs = [d for d in kb_docs if d.get("status") in ("skipped", "error")]

    parts: list[str] = []

    if ok_docs:
        total_chunks = sum(d["chunk_count"] for d in ok_docs)
        parts.append(
            f"{len(ok_docs)} document(s) processed ({total_chunks} section(s))"
        )

    if failed_docs:
        failed_names = [
            f"{d['filename']}: {d.get('error', 'unknown error')}"
            for d in failed_docs
        ]
        parts.append(
            f"⚠️ {len(failed_docs)} document(s) could not be used: "
            + "; ".join(failed_names)
        )

    return " | ".join(parts)
