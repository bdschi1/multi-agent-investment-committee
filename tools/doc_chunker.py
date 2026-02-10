"""
Document chunker for user-uploaded knowledge base files.

Adapted from the local chunker project (bds_repos). Provides:
- File ingestion: PDF, DOCX, TXT
- Chunking decision: small files injected whole, large files chunked
- Token-aware chunking: respects LLM context limits with tiktoken
- Boilerplate removal: strips disclosure sections from equity research

The processed KB is injected into agent context as supplementary research.
Agents treat it as one resource among many — they rely on their own judgment.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

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
# File readers — PDF, DOCX, TXT
# ---------------------------------------------------------------------------

def _read_pdf(path: Path) -> str:
    """Extract text from a PDF file."""
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.warning("pypdf not installed — cannot read PDF files")
        return ""

    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
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
        return "\n\n".join(paragraphs)
    except Exception as e:
        logger.warning(f"Failed to read DOCX {path.name}: {e}")
        return ""


def _read_txt(path: Path) -> str:
    """Read a plain text file."""
    try:
        return path.read_text(errors="ignore")
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


def process_uploads(file_paths: list[str | Path]) -> list[dict[str, Any]]:
    """
    Process uploaded files into a knowledge base for agent consumption.

    Never raises — individual file failures are captured and reported
    so the pipeline continues with whatever documents succeeded.

    Args:
        file_paths: List of file paths (up to MAX_FILES). Gradio gives
                    temporary paths after upload.

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
