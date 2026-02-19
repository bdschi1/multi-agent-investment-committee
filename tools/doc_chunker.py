"""
Document chunker for user-uploaded knowledge base files.

Provides:
- File ingestion: PDF, DOCX, TXT, Excel (XLSX/XLS/CSV)
- Section-aware chunking: detects headers/sections before splitting
- Chunk overlap: configurable token overlap between consecutive chunks
- Token-aware chunking: respects LLM context limits with tiktoken
- Boilerplate removal: strips disclosure sections from equity research
- Smart page scoring: importance-ranked page selection for large PDFs
- Prompt injection defense: 8-pattern sanitization on all extracted text

Section-aware chunking ported from knowledge-base/chunking/chunker.py.
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
        return len(_enc.encode(text, disallowed_special=()))
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
# Section / header detection (ported from knowledge-base chunker)
# ---------------------------------------------------------------------------

# Numbered section headers: "1. Introduction", "3.2 Methodology", "II. Results"
_NUMBERED_HEADER_RE = re.compile(
    r"^(?:"
    r"(?:\d{1,3}\.)+\s+"                    # "1. ", "3.2. ", "1.2.3 "
    r"|[IVXLC]+\.\s+"                       # "I. ", "IV. "
    r"|[A-Z]\.\s+"                           # "A. ", "B. "
    r"|(?:Chapter|Section|Part|Appendix)\s+\d+"  # "Chapter 1", "Section 3"
    r")"
    r"[A-Z].*$",                             # Must start with uppercase
    re.MULTILINE,
)

# ALL-CAPS headers or Title Case headers (standalone short lines)
_STANDALONE_HEADER_RE = re.compile(
    r"^(?:"
    r"[A-Z][A-Z\s&,/()-]{4,80}$"            # ALL CAPS line (5-80 chars)
    r"|(?:[A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|and|or|of|the|in|for|to|&|vs\.?)){1,8})$"  # Title Case
    r")",
    re.MULTILINE,
)

# Markdown-style headers
_MARKDOWN_HEADER_RE = re.compile(r"^#{1,4}\s+.+$", re.MULTILINE)

# Common section keywords that signal topic boundaries
_SECTION_KEYWORDS = re.compile(
    r"^(?:"
    r"Abstract|Introduction|Background|Overview|"
    r"Methodology|Methods?|Data(?:\s+and\s+Methods?)?|"
    r"Results?|Findings|Discussion|Analysis|"
    r"Conclusion|Summary|Recommendations?|"
    r"Risk Factors?|Valuation|Catalysts?|"
    r"Key Takeaways?|Executive Summary|"
    r"Appendix|References?|Bibliography|"
    r"Exhibit|Figure|Table"
    r")\s*[:.]?\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def _detect_sections(text: str) -> list[dict]:
    """Split text into sections based on structural header detection.

    Returns a list of dicts: [{"header": str|None, "body": str}, ...]
    Each section is a coherent topical unit bounded by detected headers.

    Ported from knowledge-base/chunking/chunker.py::_detect_sections().
    """
    # Collect all header positions
    header_spans = []
    for pattern in [_NUMBERED_HEADER_RE, _MARKDOWN_HEADER_RE, _SECTION_KEYWORDS]:
        for m in pattern.finditer(text):
            header_spans.append((m.start(), m.end(), m.group().strip()))

    # Also check standalone headers, but filter out false positives
    for m in _STANDALONE_HEADER_RE.finditer(text):
        candidate = m.group().strip()
        if len(candidate) < 5:
            continue
        if re.match(r"^[A-Z]{1,5}$", candidate):
            continue
        header_spans.append((m.start(), m.end(), candidate))

    if not header_spans:
        return [{"header": None, "body": text}]

    # Sort by position and deduplicate overlapping spans
    header_spans.sort(key=lambda x: x[0])
    deduped = []
    for start, end, header_text in header_spans:
        if deduped and start < deduped[-1][1]:
            if (end - start) > (deduped[-1][1] - deduped[-1][0]):
                deduped[-1] = (start, end, header_text)
            continue
        deduped.append((start, end, header_text))

    # Build sections
    sections = []

    # Text before first header (preamble)
    if deduped[0][0] > 0:
        preamble = text[: deduped[0][0]].strip()
        if preamble:
            sections.append({"header": None, "body": preamble})

    for i, (start, end, header_text) in enumerate(deduped):
        next_start = deduped[i + 1][0] if i + 1 < len(deduped) else len(text)
        body = text[end:next_start].strip()
        if body or header_text:
            sections.append({"header": header_text, "body": body})

    return sections


# ---------------------------------------------------------------------------
# File readers — PDF, DOCX, TXT, Excel
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
    """Extract text from a PDF file."""
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

        extracted: list[dict] = []
        for i in indices:
            text = reader.pages[i].extract_text() or ""
            extracted.append({"page": i + 1, "text": text, "chars": len(text)})

        logger.info(
            f"PDF {path.name}: extracted {len(indices)}/{total} pages "
            f"(pages {indices[0]+1}-{indices[-1]+1})"
        )

        total_chars = sum(p["chars"] for p in extracted)

        if total_chars > _SMART_TRUNCATE_CHAR_LIMIT:
            return _smart_select_pages(extracted)

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


def _read_excel(path: Path) -> str:
    """Read an Excel file (.xlsx/.xls/.csv) and convert to markdown tables.

    Each sheet is rendered as a separate markdown table with a header.
    For CSV files, the single table is returned directly.
    """
    ext = path.suffix.lower()

    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not installed — cannot read Excel/CSV files")
        return ""

    try:
        if ext == ".csv":
            dfs = {"Sheet1": pd.read_csv(str(path))}
        else:
            # openpyxl for .xlsx, xlrd for .xls (pandas auto-selects engine)
            dfs = pd.read_excel(str(path), sheet_name=None)

        parts: list[str] = []
        for sheet_name, df in dfs.items():
            if df.empty:
                continue
            # Cap at 500 rows to avoid blowing up context
            if len(df) > 500:
                df = df.head(500)
                truncated = True
            else:
                truncated = False

            # Clean: drop fully-empty columns, fill NaN
            df = df.dropna(axis=1, how="all")
            df = df.fillna("")

            header = f"## Sheet: {sheet_name}" if len(dfs) > 1 else f"## {path.stem}"
            md_table = df.to_markdown(index=False)
            if truncated:
                md_table += f"\n\n[... truncated at 500 rows, {len(dfs[sheet_name])} total ...]"

            parts.append(f"{header}\n\n{md_table}")

        combined = "\n\n---\n\n".join(parts)
        if not combined.strip():
            return ""

        logger.info(
            f"Excel {path.name}: {len(dfs)} sheet(s), "
            f"{sum(len(df) for df in dfs.values())} total rows"
        )
        return sanitize_document_text(combined)

    except Exception as e:
        logger.warning(f"Failed to read Excel {path.name}: {e}")
        return ""


_READERS = {
    ".pdf": _read_pdf,
    ".docx": _read_docx,
    ".doc": _read_docx,  # best-effort — python-docx handles some .doc files
    ".txt": _read_txt,
    ".xlsx": _read_excel,
    ".xls": _read_excel,
    ".csv": _read_excel,
}


# ---------------------------------------------------------------------------
# Boilerplate removal (equity research disclosures)
# ---------------------------------------------------------------------------

# Phrases that start a boilerplate block — ported from KB chunker
_BOILERPLATE_STARTS = [
    r"this (?:document|report|material|presentation|publication) is (?:provided|published|distributed|intended) (?:for|solely|by)",
    r"this (?:report|material) (?:has been|was) (?:prepared|produced|published) by",
    r"(?:important|required) disclosures?",
    r"(?:important|general) (?:legal )?information",
    r"analyst certification",
    r"the analyst(?:s)? (?:responsible|named|hereby certif)",
    r"past performance (?:is|does) not (?:necessarily )?(?:indicative|a guarantee|guarantee)",
    r"(?:this|the) (?:document|report|research|material) (?:is|should) not (?:be )?(?:considered|construed|relied|used)",
    r"not (?:fdic )?insured",
    r"all rights reserved",
    r"for (?:important|additional|further) disclosures?",
    r"see (?:important|additional|required) disclosures?",
    r"regulatory disclosures?",
    r"disclosure appendix",
    r"(?:copyright|©)\s*\d{4}",
    r"registered (?:broker[- ]?dealer|investment advis|with the (?:sec|fca|finra))",
    r"member (?:finra|sipc|fdic|nyse|nfa)",
    r"conflicts? of interest",
    r"for the exclusive use of",
    r"not an offer to sell",
]

_BOILERPLATE_PATTERNS = [
    re.compile(p, re.IGNORECASE | re.MULTILINE) for p in _BOILERPLATE_STARTS
]

# Full-block patterns — if a paragraph matches these entirely, remove it
_FULL_BLOCK_REMOVALS = [
    re.compile(r"^\s*(?:source|sources?):\s*.{0,100}\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d{1,3}\s*$"),  # standalone page numbers
    re.compile(r"^\s*[0-9a-f]{32}\s*$"),  # document tracking hex codes
]


def _strip_boilerplate(text: str) -> str:
    """Remove boilerplate legal, disclosure, and publication paragraphs.

    Upgraded from simple trailing-match to per-paragraph scanning (KB approach).
    """
    if not text:
        return text

    paragraphs = re.split(r"\n\s*\n", text)
    clean = []

    for para in paragraphs:
        stripped = para.strip()
        if not stripped:
            continue

        # Skip standalone page numbers / tiny artifacts
        if len(stripped) < 10:
            is_artifact = any(pat.match(stripped) for pat in _FULL_BLOCK_REMOVALS)
            if is_artifact:
                continue

        # Check if paragraph starts with a boilerplate phrase
        is_boilerplate = any(
            pat.search(stripped[:300]) for pat in _BOILERPLATE_PATTERNS
        )
        if not is_boilerplate:
            clean.append(stripped)

    return "\n\n".join(clean)


# ---------------------------------------------------------------------------
# Chunking logic — section-aware with overlap (ported from KB)
# ---------------------------------------------------------------------------

# Files under this token count are injected whole (no chunking needed)
SMALL_FILE_THRESHOLD = 3000  # ~3K tokens ~ 2-3 pages

# Max tokens per chunk when chunking is needed
MAX_CHUNK_TOKENS = 800

# Overlap between consecutive chunks (tokens)
CHUNK_OVERLAP = 100

# Minimum chunk size — discard smaller fragments
MIN_CHUNK_TOKENS = 50

# Separator priority for recursive splitting
_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _recursive_split(text: str, separators: list[str]) -> list[str]:
    """Recursively split text using priority-ordered separators.

    Tries the first separator; if any piece is still too large,
    recurses with the next separator.
    """
    if not separators:
        return [text] if text.strip() else []

    sep = separators[0]
    remaining = separators[1:]

    parts = text.split(sep)
    result = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if _count_tokens(part) <= MAX_CHUNK_TOKENS:
            result.append(part)
        else:
            result.extend(_recursive_split(part, remaining))

    return result


def _chunk_text(text: str) -> list[str]:
    """
    Section-aware chunking with overlap.

    Strategy (ported from knowledge-base chunker):
      1. Strip boilerplate
      2. Detect structural sections (headers, numbered sections, topic shifts)
      3. Chunk each section independently (prevents cross-section straddling)
      4. Within sections, use recursive separator splitting with overlap
    """
    text = _strip_boilerplate(text)
    if not text.strip():
        return []

    sections = _detect_sections(text)

    all_chunks: list[str] = []

    for section in sections:
        section_text = ""
        if section["header"]:
            section_text = section["header"] + "\n\n"
        section_text += section["body"]
        section_text = section_text.strip()

        if not section_text:
            continue

        # If the section fits in one chunk, take it whole
        section_tokens = _count_tokens(section_text)
        if section_tokens <= MAX_CHUNK_TOKENS:
            if section_tokens >= MIN_CHUNK_TOKENS:
                all_chunks.append(section_text)
            continue

        # Split section into pieces using recursive separators
        splits = _recursive_split(section_text, _SEPARATORS)

        # Merge splits into chunks with overlap
        current_parts: list[str] = []
        current_tokens = 0

        for split_text in splits:
            split_tokens = _count_tokens(split_text)

            if current_tokens + split_tokens <= MAX_CHUNK_TOKENS:
                current_parts.append(split_text)
                current_tokens += split_tokens
            else:
                # Flush current chunk
                if current_parts:
                    chunk_text = " ".join(current_parts).strip()
                    ct = _count_tokens(chunk_text)
                    if ct >= MIN_CHUNK_TOKENS:
                        all_chunks.append(chunk_text)

                    # Overlap: keep last N tokens worth of parts
                    overlap_parts: list[str] = []
                    overlap_tokens = 0
                    for part in reversed(current_parts):
                        pt = _count_tokens(part)
                        if overlap_tokens + pt <= CHUNK_OVERLAP:
                            overlap_parts.insert(0, part)
                            overlap_tokens += pt
                        else:
                            break

                    current_parts = overlap_parts + [split_text]
                    current_tokens = overlap_tokens + split_tokens
                else:
                    current_parts = [split_text]
                    current_tokens = split_tokens

        # Flush final chunk for this section
        if current_parts:
            chunk_text = " ".join(current_parts).strip()
            ct = _count_tokens(chunk_text)
            if ct >= MIN_CHUNK_TOKENS:
                all_chunks.append(chunk_text)

    return all_chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

MAX_FILES = 10


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
                # Large file → section-aware chunk with overlap
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
