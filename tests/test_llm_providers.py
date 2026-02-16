"""
Tests for LLM provider resolution and fallback logic.

Verifies that resolve_provider() correctly selects the active provider
based on configured API keys, falling back to Ollama when no keys are set.
"""

from __future__ import annotations

from config.settings import LLMProvider, Settings


class TestResolveProvider:
    """Test Settings.resolve_provider() fallback logic."""

    def _make_settings(self, **overrides) -> Settings:
        """Create a Settings instance with explicit values, bypassing .env."""
        defaults = {
            "llm_provider": LLMProvider.ANTHROPIC,
            "anthropic_api_key": None,
            "google_api_key": None,
            "openai_api_key": None,
            "hf_token": None,
            "deepseek_api_key": None,
        }
        defaults.update(overrides)
        return Settings(**defaults)

    def test_no_keys_defaults_to_ollama(self):
        """No API keys set at all → falls back to Ollama."""
        s = self._make_settings()
        assert s.resolve_provider() == LLMProvider.OLLAMA

    def test_anthropic_key_returns_anthropic(self):
        """Anthropic selected with valid key → returns Anthropic."""
        s = self._make_settings(anthropic_api_key="sk-ant-test")
        assert s.resolve_provider() == LLMProvider.ANTHROPIC

    def test_google_key_returns_google(self):
        """Google selected with valid key → returns Google."""
        s = self._make_settings(
            llm_provider=LLMProvider.GOOGLE,
            google_api_key="test-google-key",
        )
        assert s.resolve_provider() == LLMProvider.GOOGLE

    def test_openai_key_returns_openai(self):
        """OpenAI selected with valid key → returns OpenAI."""
        s = self._make_settings(
            llm_provider=LLMProvider.OPENAI,
            openai_api_key="sk-test",
        )
        assert s.resolve_provider() == LLMProvider.OPENAI

    def test_deepseek_key_returns_deepseek(self):
        """DeepSeek selected with valid key → returns DeepSeek."""
        s = self._make_settings(
            llm_provider=LLMProvider.DEEPSEEK,
            deepseek_api_key="test-deepseek-key",
        )
        assert s.resolve_provider() == LLMProvider.DEEPSEEK

    def test_huggingface_key_returns_huggingface(self):
        """HuggingFace selected with valid token → returns HuggingFace."""
        s = self._make_settings(
            llm_provider=LLMProvider.HUGGINGFACE,
            hf_token="hf_test_token",
        )
        assert s.resolve_provider() == LLMProvider.HUGGINGFACE

    def test_ollama_selected_returns_ollama_regardless(self):
        """Ollama explicitly selected → returns Ollama even with no keys."""
        s = self._make_settings(llm_provider=LLMProvider.OLLAMA)
        assert s.resolve_provider() == LLMProvider.OLLAMA

    def test_ollama_selected_with_other_keys_still_ollama(self):
        """Ollama explicitly selected → returns Ollama even if other keys exist."""
        s = self._make_settings(
            llm_provider=LLMProvider.OLLAMA,
            anthropic_api_key="sk-ant-test",
            google_api_key="test-google-key",
        )
        assert s.resolve_provider() == LLMProvider.OLLAMA

    def test_selected_provider_no_key_falls_to_other(self):
        """Anthropic selected but no Anthropic key, Google key exists → returns Google."""
        s = self._make_settings(
            llm_provider=LLMProvider.ANTHROPIC,
            google_api_key="test-google-key",
        )
        assert s.resolve_provider() == LLMProvider.GOOGLE

    def test_selected_provider_no_key_falls_to_deepseek(self):
        """OpenAI selected but no OpenAI key, DeepSeek key exists → returns DeepSeek."""
        s = self._make_settings(
            llm_provider=LLMProvider.OPENAI,
            deepseek_api_key="test-deepseek-key",
        )
        assert s.resolve_provider() == LLMProvider.DEEPSEEK

    def test_selected_provider_no_key_no_others_falls_to_ollama(self):
        """Google selected but no keys at all → falls back to Ollama."""
        s = self._make_settings(llm_provider=LLMProvider.GOOGLE)
        assert s.resolve_provider() == LLMProvider.OLLAMA


class TestGetActiveModel:
    """Test Settings.get_active_model() for all providers."""

    def test_anthropic_model(self):
        s = Settings(llm_provider=LLMProvider.ANTHROPIC)
        assert s.get_active_model() == s.anthropic_model

    def test_google_model(self):
        s = Settings(llm_provider=LLMProvider.GOOGLE)
        assert s.get_active_model() == s.google_model

    def test_openai_model(self):
        s = Settings(llm_provider=LLMProvider.OPENAI)
        assert s.get_active_model() == s.openai_model

    def test_huggingface_model(self):
        s = Settings(llm_provider=LLMProvider.HUGGINGFACE)
        assert s.get_active_model() == s.hf_model

    def test_ollama_model(self):
        s = Settings(llm_provider=LLMProvider.OLLAMA)
        assert s.get_active_model() == s.ollama_model

    def test_deepseek_model(self):
        s = Settings(llm_provider=LLMProvider.DEEPSEEK)
        assert s.get_active_model() == s.deepseek_model

    def test_custom_deepseek_model(self):
        s = Settings(llm_provider=LLMProvider.DEEPSEEK, deepseek_model="deepseek-reasoner")
        assert s.get_active_model() == "deepseek-reasoner"


class TestLLMProviderEnum:
    """Test LLMProvider enum values."""

    def test_all_providers_present(self):
        providers = {p.value for p in LLMProvider}
        expected = {"anthropic", "google", "openai", "huggingface", "ollama", "deepseek"}
        assert providers == expected

    def test_string_enum(self):
        assert LLMProvider.DEEPSEEK == "deepseek"
        assert LLMProvider.OLLAMA == "ollama"
