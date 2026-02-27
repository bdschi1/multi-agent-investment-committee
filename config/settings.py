"""
Application settings with Pydantic validation.
Supports .env file and environment variable overrides.

Supported LLM providers:
    - anthropic  (Claude)
    - google     (Gemini)
    - openai     (GPT)
    - huggingface (HF Inference API)
    - ollama     (local open-source models)
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    """Global application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM Provider ---
    llm_provider: LLMProvider = Field(
        default=LLMProvider.ANTHROPIC,
        description="Which LLM backend to use: anthropic, google, openai, huggingface, ollama",
    )

    # --- Anthropic (Claude) ---
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key")
    anthropic_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Anthropic model name",
    )

    # --- Google (Gemini) ---
    google_api_key: str | None = Field(default=None, description="Google AI API key")
    google_model: str = Field(
        default="gemini-2.0-flash",
        description="Google Gemini model name",
    )

    # --- OpenAI ---
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model name",
    )

    # --- Hugging Face ---
    hf_token: str | None = Field(default=None, description="Hugging Face API token")
    hf_model: str = Field(
        default="Qwen/Qwen2.5-72B-Instruct",
        description="HuggingFace model ID for inference",
    )

    # --- Ollama (local open-source) ---
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL",
    )
    ollama_model: str = Field(
        default="llama3.1:8b",
        description="Ollama model name (must be pulled first)",
    )

    # --- Reasoning Parameters ---
    max_debate_rounds: int = Field(default=2, ge=1, le=20, description="Max debate rounds")
    max_tokens_per_agent: int = Field(default=4096, ge=256, description="Token budget per agent")
    max_tool_calls_per_agent: int = Field(
        default=5, ge=0, le=20,
        description="Max dynamic tool calls per agent per run (0 disables tool calling)",
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature (global default)")
    task_temperatures: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-node temperature overrides. Keys are node names "
            "(e.g. 'run_sector_analyst', 'run_portfolio_manager'). "
            "Unset nodes use built-in defaults from orchestrator/temperature.py."
        ),
    )
    task_models: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Per-node model overrides. Keys are node names "
            "(e.g. 'run_sector_analyst', 'run_portfolio_manager'). "
            "Values are 'provider:model_name' strings "
            "(e.g. 'google:gemini-2.0-flash', 'anthropic:claude-sonnet-4-20250514'). "
            "Unset nodes use the shared model from the primary provider."
        ),
    )

    # --- LLM Call Timeout ---
    model_timeout_seconds: float = Field(
        default=120.0, ge=0.0,
        description="Timeout in seconds for individual LLM calls (0 = no timeout)",
    )

    # --- Rate Limiting ---
    # Per-provider defaults (85% of official Tier 1 limits for safety margin).
    # Set any value to 0 to disable that limit.  Providers not listed run
    # unrestricted.  Override via env vars: RATE_LIMIT_RPM, etc.
    rate_limit_rpm: int = Field(
        default=45, ge=0,
        description="Anthropic RPM (legacy, still used as fallback if provider_rate_limits is empty)",
    )
    rate_limit_input_tpm: int = Field(
        default=25000, ge=0,
        description="Anthropic input TPM (legacy fallback)",
    )
    rate_limit_output_tpm: int = Field(
        default=7000, ge=0,
        description="Anthropic output TPM (legacy fallback)",
    )

    # --- Application ---
    log_level: str = Field(default="INFO", description="Logging level")
    enable_reasoning_trace: bool = Field(
        default=True, description="Capture detailed reasoning traces"
    )
    enable_hitl: bool = Field(
        default=True,
        description="Enable two-phase human-in-the-loop mode in UI (review before PM synthesis)",
    )

    # --- Optimizer ---
    optimizer_method: str = Field(
        default="black_litterman",
        description=(
            "Portfolio optimization strategy: black_litterman, hrp, mean_variance, "
            "min_variance, risk_parity, equal_weight, ensemble"
        ),
    )

    # --- XAI Module ---
    enable_xai: bool = Field(default=True, description="Enable XAI pre-screen analysis in pipeline")
    xai_distress_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="PFD threshold for distress classification",
    )
    xai_model_path: str = Field(
        default="xai/artifacts/distress_model.joblib",
        description="Path to trained XGBoost distress model (optional)",
    )

    def get_active_model(self) -> str:
        """Return the model identifier for the active provider."""
        model_map = {
            LLMProvider.ANTHROPIC: self.anthropic_model,
            LLMProvider.GOOGLE: self.google_model,
            LLMProvider.OPENAI: self.openai_model,
            LLMProvider.HUGGINGFACE: self.hf_model,
            LLMProvider.OLLAMA: self.ollama_model,
        }
        return model_map[self.llm_provider]

    def resolve_provider(self) -> LLMProvider:
        """Return the effective provider, falling back to Ollama if no API keys configured.

        Logic:
            1. If the selected provider has a valid API key → use it.
            2. If the selected provider is Ollama → use it (no key needed).
            3. If the selected provider has no key → scan for any provider with a key.
            4. If no API keys at all → fall back to Ollama (local).
        """
        key_map = {
            LLMProvider.ANTHROPIC: self.anthropic_api_key,
            LLMProvider.GOOGLE: self.google_api_key,
            LLMProvider.OPENAI: self.openai_api_key,
            LLMProvider.HUGGINGFACE: self.hf_token,
        }

        # If the chosen provider has a valid key, use it
        if self.llm_provider in key_map and key_map[self.llm_provider]:
            return self.llm_provider

        # Ollama doesn't need a key
        if self.llm_provider == LLMProvider.OLLAMA:
            return LLMProvider.OLLAMA

        # No key for the selected provider — try to find any provider with a key
        for provider, key in key_map.items():
            if key:
                return provider

        # No API keys at all — fall back to Ollama
        return LLMProvider.OLLAMA


# Singleton instance
settings = Settings()
