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
from typing import Optional

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
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    anthropic_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Anthropic model name",
    )

    # --- Google (Gemini) ---
    google_api_key: Optional[str] = Field(default=None, description="Google AI API key")
    google_model: str = Field(
        default="gemini-2.0-flash",
        description="Google Gemini model name",
    )

    # --- OpenAI ---
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model name",
    )

    # --- Hugging Face ---
    hf_token: Optional[str] = Field(default=None, description="Hugging Face API token")
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
        default="llama3.2:3b",
        description="Ollama model name (must be pulled first)",
    )

    # --- Reasoning Parameters ---
    max_debate_rounds: int = Field(default=2, ge=1, le=20, description="Max debate rounds")
    max_tokens_per_agent: int = Field(default=4096, ge=256, description="Token budget per agent")
    max_tool_calls_per_agent: int = Field(
        default=5, ge=0, le=20,
        description="Max dynamic tool calls per agent per run (0 disables tool calling)",
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")

    # --- Rate Limiting (Anthropic Tier 1 — 85% of actual limits for safety margin) ---
    rate_limit_rpm: int = Field(
        default=45, ge=0,
        description="Max requests per minute for rate-limited providers (0 = no limit)",
    )
    rate_limit_input_tpm: int = Field(
        default=25000, ge=0,
        description="Max input tokens per minute for rate-limited providers (0 = no limit)",
    )
    rate_limit_output_tpm: int = Field(
        default=7000, ge=0,
        description="Max output tokens per minute for rate-limited providers (0 = no limit)",
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


# Singleton instance
settings = Settings()
