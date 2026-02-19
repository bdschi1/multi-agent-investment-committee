"""
Knowledge Base Tool — search the curated KB from ~/Projects/knowledge-base.

Provides access to RAG-powered search over financial research, quant finance,
biotech, AI/ML, and investment research documents.

Falls back gracefully if the knowledge base is not available (returns error dict).

Usage:
    from tools.knowledge_base import KnowledgeBaseTool
    results = KnowledgeBaseTool.search_kb("momentum factor")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Path to knowledge-base project
_KB_PATH = Path.home() / "Projects" / "knowledge-base"
_kb_available = False

# Try to make KB importable
if _KB_PATH.exists():
    _kb_str = str(_KB_PATH)
    if _kb_str not in sys.path:
        sys.path.insert(0, _kb_str)
    try:
        # Load .env for API keys
        _env_path = _KB_PATH / ".env"
        if _env_path.exists():
            import os
            for line in _env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = value
        _kb_available = True
    except Exception as e:
        logger.warning(f"Knowledge base path found but env setup failed: {e}")


def _not_available(func_name: str) -> dict:
    return {
        "error": f"Knowledge base not available (searched {_KB_PATH})",
        "tool": func_name,
    }


class KnowledgeBaseTool:
    """Static methods for querying the knowledge base."""

    @staticmethod
    def search_kb(query: str, top_k: int = 5, category: str = "") -> dict:
        """Search KB for relevant document chunks.

        Args:
            query: Natural language search query
            top_k: Number of results (default 5)
            category: Optional category filter (e.g. "quantitative finance")

        Returns:
            dict with results list, or error dict if KB unavailable.
        """
        if not _kb_available:
            return _not_available("search_kb")

        try:
            from query import search
            results = search(
                query,
                top_k=top_k,
                category=category or None,
            )
            return {
                "results": [
                    {
                        "text": r["text"][:500],
                        "source": r.get("metadata", {}).get("filename", "unknown"),
                        "category": r.get("metadata", {}).get("category", ""),
                        "score": r.get("score", 0),
                    }
                    for r in results
                ],
                "count": len(results),
            }
        except Exception as e:
            logger.error(f"KB search failed: {e}")
            return {"error": str(e), "tool": "search_kb"}

    @staticmethod
    def ask_kb(query: str, top_k: int = 5, category: str = "") -> dict:
        """Get formatted KB context string for LLM injection.

        Args:
            query: Natural language query
            top_k: Number of chunks to retrieve
            category: Optional category filter

        Returns:
            dict with context string, or error dict.
        """
        if not _kb_available:
            return _not_available("ask_kb")

        try:
            from query import ask
            context = ask(
                query,
                top_k=top_k,
                category=category or None,
            )
            return {"context": context}
        except Exception as e:
            logger.error(f"KB ask failed: {e}")
            return {"error": str(e), "tool": "ask_kb"}

    @staticmethod
    def answer_kb(query: str, category: str = "") -> dict:
        """End-to-end KB answer with LLM synthesis and fallback.

        Args:
            query: Natural language question
            category: Optional category filter

        Returns:
            dict with answer, source, top_score, or error dict.
        """
        if not _kb_available:
            return _not_available("answer_kb")

        try:
            from query import answer_with_kb
            result = answer_with_kb(
                query,
                category=category or None,
            )
            return {
                "answer": result["answer"],
                "source": result["source"],
                "top_score": result["top_score"],
                "num_sources": len(result.get("sources", [])),
            }
        except Exception as e:
            logger.error(f"KB answer failed: {e}")
            return {"error": str(e), "tool": "answer_kb"}
