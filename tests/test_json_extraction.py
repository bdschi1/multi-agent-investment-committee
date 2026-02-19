"""
Tests for robust JSON extraction and artifact cleaning utilities.

Tests the shared extract_json() and clean_json_artifacts() functions
from agents/base.py — the foundation of small-LLM resilience (v3.4).
"""

import json

import pytest

from agents.base import clean_json_artifacts, extract_json

# ---------------------------------------------------------------------------
# extract_json() tests
# ---------------------------------------------------------------------------

class TestExtractJson:
    """Test the progressive JSON extraction strategies."""

    def test_valid_json_direct(self):
        """Strategy 1: Direct JSON parse — no repair needed."""
        text = '{"ticker": "NVDA", "conviction": 7.5}'
        result, repaired = extract_json(text)
        assert result == {"ticker": "NVDA", "conviction": 7.5}
        assert repaired is False

    def test_json_in_markdown_block(self):
        """Strategy 2: JSON wrapped in ```json code block."""
        text = 'Here is the analysis:\n```json\n{"ticker": "AAPL", "score": 8.0}\n```\nDone.'
        result, repaired = extract_json(text)
        assert result["ticker"] == "AAPL"
        assert result["score"] == 8.0
        assert repaired is True

    def test_json_in_generic_code_block(self):
        """Strategy 3: JSON wrapped in ``` code block (no json tag)."""
        text = 'Result:\n```\n{"ticker": "MSFT", "risks": ["competition"]}\n```'
        result, repaired = extract_json(text)
        assert result["ticker"] == "MSFT"
        assert repaired is True

    def test_json_with_surrounding_prose(self):
        """Strategy 4: JSON embedded in prose, extracted by brace boundary."""
        text = 'Based on my analysis, {"ticker": "GOOG", "thesis": "Strong AI moat"} is the result.'
        result, repaired = extract_json(text)
        assert result["ticker"] == "GOOG"
        assert repaired is True

    def test_trailing_commas(self):
        """Strategy 5: JSON with trailing commas (common LLM error)."""
        text = '{"ticker": "TSLA", "risks": ["valuation", "competition",], "score": 6.0,}'
        result, repaired = extract_json(text)
        assert result["ticker"] == "TSLA"
        assert result["score"] == 6.0
        assert repaired is True

    def test_single_quotes(self):
        """Strategy 6: JSON with single quotes instead of double quotes."""
        text = "{'ticker': 'META', 'conviction': 7.0}"
        result, repaired = extract_json(text)
        assert result["ticker"] == "META"
        assert repaired is True

    def test_unbalanced_braces(self):
        """Strategy 7: JSON with missing closing braces."""
        text = '{"ticker": "AMZN", "thesis": "Cloud dominance"'
        result, repaired = extract_json(text)
        assert result["ticker"] == "AMZN"
        assert repaired is True

    def test_completely_unparseable_raises(self):
        """All strategies fail — should raise ValueError."""
        text = "This is just plain text with no JSON at all."
        with pytest.raises(ValueError, match="All JSON extraction strategies failed"):
            extract_json(text)

    def test_empty_string_raises(self):
        """Empty input should raise ValueError."""
        with pytest.raises((ValueError, KeyError)):
            extract_json("")

    def test_nested_json(self):
        """Nested JSON objects should parse correctly."""
        data = {"ticker": "SPY", "metrics": {"pe": 25.5, "volume": 1000000}}
        text = json.dumps(data)
        result, repaired = extract_json(text)
        assert result == data
        assert repaired is False

    def test_json_with_newlines_in_values(self):
        """JSON with newlines inside string values."""
        text = '{"ticker": "NVDA", "thesis": "Strong AI\\nGrowth momentum"}'
        result, repaired = extract_json(text)
        assert result["ticker"] == "NVDA"

    def test_json_with_code_block_language_tag(self):
        """JSON in code block with language tag on same line as backticks."""
        text = '```python\n{"ticker": "AAPL", "score": 9.0}\n```'
        result, repaired = extract_json(text)
        assert result["ticker"] == "AAPL"
        assert repaired is True


# ---------------------------------------------------------------------------
# clean_json_artifacts() tests
# ---------------------------------------------------------------------------

class TestCleanJsonArtifacts:
    """Test the JSON artifact cleaning utility."""

    def test_normal_text_unchanged(self):
        """Plain text without JSON artifacts passes through unchanged."""
        text = "Strong bull case driven by AI momentum and data center TAM."
        result = clean_json_artifacts(text)
        assert result == text

    def test_json_string_extracts_longest_value(self):
        """Full JSON string — should extract the longest string value."""
        data = {
            "ticker": "NVDA",
            "thesis": "Strong AI momentum driven by data center TAM expansion and CUDA moat.",
            "score": 8.0,
        }
        text = json.dumps(data)
        result = clean_json_artifacts(text)
        assert "Strong AI momentum" in result
        # Should NOT contain JSON syntax
        assert "{" not in result
        assert '"ticker"' not in result

    def test_partial_json_cleaned(self):
        """Partial/broken JSON gets cleaned via regex."""
        text = '{"ticker": "SPY", "thesis": "Bull case for SPY based on'
        result = clean_json_artifacts(text)
        # Should not contain raw JSON keys
        assert '"ticker"' not in result or "{" not in result
        # Should contain some readable text
        assert len(result) > 0

    def test_empty_string(self):
        """Empty input returns empty string."""
        assert clean_json_artifacts("") == ""

    def test_truncation(self):
        """Long text gets truncated to max_length."""
        text = "A" * 1000
        result = clean_json_artifacts(text, max_length=100)
        assert len(result) <= 100

    def test_json_array_not_dict(self):
        """JSON array at top level — should still clean."""
        text = '["risk1", "risk2", "risk3"]'
        result = clean_json_artifacts(text)
        assert len(result) > 0

    def test_mixed_json_and_prose(self):
        """Text that starts with prose but contains JSON-like patterns."""
        text = 'The analysis shows {"key": "value"} embedded in text.'
        result = clean_json_artifacts(text)
        # Should return the text, possibly cleaned
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Integration: parsing failure detection
# ---------------------------------------------------------------------------

class TestParsingFailureDetection:
    """Test that sentinel values correctly trigger parsing failure tracking."""

    def test_bull_case_sentinel(self):
        """Bull case fallback with sentinel is detected as degraded."""
        from agents.base import BullCase
        from orchestrator.nodes import _is_parsing_degraded_bull

        # Fallback bull case (as constructed in sector_analyst.py)
        bc = BullCase(
            ticker="TEST",
            thesis="some text",
            supporting_evidence=["Analysis generated but structured parsing failed"],
            catalysts=[],
            conviction_score=5.0,
            time_horizon="unknown",
            key_metrics={},
        )
        assert _is_parsing_degraded_bull(bc) is True

    def test_bull_case_normal(self):
        """Normal bull case is NOT detected as degraded."""
        from agents.base import BullCase
        from orchestrator.nodes import _is_parsing_degraded_bull

        bc = BullCase(
            ticker="TEST",
            thesis="Strong thesis",
            supporting_evidence=["Revenue growth at 25%", "Market share expanding"],
            catalysts=["Earnings beat"],
            conviction_score=7.5,
            time_horizon="12 months",
            key_metrics={"PE": "25x"},
        )
        assert _is_parsing_degraded_bull(bc) is False

    def test_bear_case_sentinel(self):
        """Bear case fallback with sentinel is detected as degraded."""
        from agents.base import BearCase
        from orchestrator.nodes import _is_parsing_degraded_bear

        bc = BearCase(
            ticker="TEST",
            risks=["Analysis generated but structured parsing failed"],
            second_order_effects=[],
            third_order_effects=[],
            worst_case_scenario="some text",
            bearish_conviction=5.0,
            key_vulnerabilities={},
        )
        assert _is_parsing_degraded_bear(bc) is True

    def test_macro_sentinel(self):
        """Macro view fallback with sentinel is detected as degraded."""
        from agents.base import MacroView
        from orchestrator.nodes import _is_parsing_degraded_macro

        mv = MacroView(
            ticker="TEST",
            economic_cycle_phase="Analysis generated but structured parsing failed",
            cycle_evidence=[],
            rate_environment="unknown",
            central_bank_outlook="some text",
            macro_favorability=5.0,
        )
        assert _is_parsing_degraded_macro(mv) is True

    def test_memo_sentinel(self):
        """Committee memo fallback with sentinel is detected as degraded."""
        from agents.base import CommitteeMemo
        from orchestrator.nodes import _is_parsing_degraded_memo

        memo = CommitteeMemo(
            ticker="TEST",
            recommendation="HOLD",
            position_size="Not determined",
            conviction=5.0,
            thesis_summary="some text",
            key_factors=["Memo generated but structured parsing failed"],
        )
        assert _is_parsing_degraded_memo(memo) is True
