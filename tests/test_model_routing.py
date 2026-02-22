"""Tests for orchestrator.model_routing — per-node model selection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from orchestrator.model_routing import (
    build_model_cache,
    get_node_model_spec,
    parse_model_spec,
)


# ---------------------------------------------------------------------------
# parse_model_spec
# ---------------------------------------------------------------------------

class TestParseModelSpec:
    def test_provider_and_model(self):
        assert parse_model_spec("anthropic:claude-sonnet-4-20250514") == (
            "anthropic", "claude-sonnet-4-20250514",
        )

    def test_model_only(self):
        assert parse_model_spec("gemini-2.0-flash") == (None, "gemini-2.0-flash")

    def test_strips_whitespace(self):
        assert parse_model_spec("  google : gemini-2.0-flash  ") == (
            "google", "gemini-2.0-flash",
        )

    def test_empty_string(self):
        assert parse_model_spec("") == (None, "")

    def test_multiple_colons(self):
        """Only splits on the first colon."""
        provider, model = parse_model_spec("ollama:llama3:8b")
        assert provider == "ollama"
        assert model == "llama3:8b"


# ---------------------------------------------------------------------------
# get_node_model_spec
# ---------------------------------------------------------------------------

class TestGetNodeModelSpec:
    def test_returns_none_when_no_override(self):
        with patch("orchestrator.model_routing.settings") as mock_settings:
            mock_settings.task_models = {}
            assert get_node_model_spec("run_sector_analyst") is None

    def test_returns_spec_when_override_set(self):
        with patch("orchestrator.model_routing.settings") as mock_settings:
            mock_settings.task_models = {
                "run_portfolio_manager": "anthropic:claude-sonnet-4-20250514",
            }
            assert get_node_model_spec("run_portfolio_manager") == "anthropic:claude-sonnet-4-20250514"
            assert get_node_model_spec("run_sector_analyst") is None


# ---------------------------------------------------------------------------
# build_model_cache
# ---------------------------------------------------------------------------

class TestBuildModelCache:
    def test_empty_when_no_overrides(self):
        with patch("orchestrator.model_routing.settings") as mock_settings:
            mock_settings.task_models = {}
            cache = build_model_cache("default_model", lambda *a, **kw: "model")
            assert cache == {}

    def test_creates_models_for_overrides(self):
        mock_factory = MagicMock(return_value="custom_model")
        with patch("orchestrator.model_routing.settings") as mock_settings:
            mock_settings.task_models = {
                "run_portfolio_manager": "anthropic:claude-sonnet-4-20250514",
            }
            mock_settings.llm_provider = "ollama"
            cache = build_model_cache("default", mock_factory)
            assert "run_portfolio_manager" in cache
            assert cache["run_portfolio_manager"] == "custom_model"

    def test_shares_model_for_same_spec(self):
        call_count = 0

        def counting_factory(provider=None, model_name=None):
            nonlocal call_count
            call_count += 1
            return f"model_{call_count}"

        with patch("orchestrator.model_routing.settings") as mock_settings:
            mock_settings.task_models = {
                "run_sector_analyst": "google:gemini-2.0-flash",
                "run_short_analyst": "google:gemini-2.0-flash",
                "run_risk_manager": "google:gemini-2.0-flash",
            }
            mock_settings.llm_provider = "ollama"
            cache = build_model_cache("default", counting_factory)

            # All three should share the same model instance
            assert cache["run_sector_analyst"] is cache["run_short_analyst"]
            assert cache["run_short_analyst"] is cache["run_risk_manager"]
            assert call_count == 1  # Factory called only once

    def test_graceful_fallback_on_creation_error(self):
        def failing_factory(provider=None, model_name=None):
            raise RuntimeError("No API key")

        with patch("orchestrator.model_routing.settings") as mock_settings:
            mock_settings.task_models = {
                "run_portfolio_manager": "anthropic:claude-sonnet-4-20250514",
            }
            mock_settings.llm_provider = "ollama"
            cache = build_model_cache("default", failing_factory)
            assert cache == {}  # Failed, not in cache


# ---------------------------------------------------------------------------
# _get_model integration with node_name
# ---------------------------------------------------------------------------

class TestGetModelWithNodeName:
    def test_uses_cache_when_available(self):
        from orchestrator.nodes import _get_model

        config = {"configurable": {
            "model": "default_model",
            "model_cache": {"run_portfolio_manager": "premium_model"},
        }}
        result = _get_model({}, config, node_name="run_portfolio_manager")
        assert result == "premium_model"

    def test_falls_back_to_default_model(self):
        from orchestrator.nodes import _get_model

        config = {"configurable": {
            "model": "default_model",
            "model_cache": {"run_portfolio_manager": "premium_model"},
        }}
        # Node not in cache → use default
        result = _get_model({}, config, node_name="run_sector_analyst")
        assert result == "default_model"

    def test_backward_compatible_without_node_name(self):
        from orchestrator.nodes import _get_model

        config = {"configurable": {"model": "shared_model"}}
        # No node_name → old behavior
        result = _get_model({}, config)
        assert result == "shared_model"

    def test_empty_cache_uses_default(self):
        from orchestrator.nodes import _get_model

        config = {"configurable": {
            "model": "default_model",
            "model_cache": {},
        }}
        result = _get_model({}, config, node_name="run_sector_analyst")
        assert result == "default_model"

    def test_no_config_uses_state(self):
        from orchestrator.nodes import _get_model

        state = {"model": "state_model"}
        result = _get_model(state, None, node_name="run_sector_analyst")
        assert result == "state_model"


# ---------------------------------------------------------------------------
# Settings integration
# ---------------------------------------------------------------------------

class TestSettingsTaskModels:
    def test_default_is_empty_dict(self):
        from config.settings import Settings
        s = Settings()
        assert s.task_models == {}

    def test_accepts_dict(self):
        from config.settings import Settings
        s = Settings(task_models={"run_portfolio_manager": "anthropic:claude-sonnet-4-20250514"})
        assert s.task_models["run_portfolio_manager"] == "anthropic:claude-sonnet-4-20250514"
