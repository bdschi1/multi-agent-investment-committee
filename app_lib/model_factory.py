"""
Model factory functions and rate limiting for LLM providers.

Extracted from app.py — contains all provider-specific model factories,
the unified create_model() entry point, rate limiting, and timeout logic.
"""

from __future__ import annotations

import logging
import os
import threading
import time


from config.settings import LLMProvider, settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider-specific model factories
# ---------------------------------------------------------------------------

def _create_anthropic_model(model_name: str | None = None) -> callable:
    """Create a Claude (Anthropic) callable."""
    from anthropic import Anthropic

    client = Anthropic(api_key=settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY"))
    model = model_name or settings.anthropic_model

    def call(prompt: str, *, temperature: float | None = None) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=settings.max_tokens_per_agent,
            temperature=temperature if temperature is not None else settings.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    return call


def _create_google_model(model_name: str | None = None) -> callable:
    """Create a Gemini (Google) callable."""
    from google import genai

    client = genai.Client(api_key=settings.google_api_key or os.environ.get("GOOGLE_API_KEY"))
    model = model_name or settings.google_model

    def call(prompt: str, *, temperature: float | None = None) -> str:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "temperature": temperature if temperature is not None else settings.temperature,
                "max_output_tokens": settings.max_tokens_per_agent,
            },
        )
        return response.text

    return call


def _create_openai_model(model_name: str | None = None) -> callable:
    """Create an OpenAI callable."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI package not installed. Run: pip install -e '.[openai]'"
        ) from exc

    client = OpenAI(api_key=settings.openai_api_key or os.environ.get("OPENAI_API_KEY"))
    model = model_name or settings.openai_model

    def call(prompt: str, *, temperature: float | None = None) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature if temperature is not None else settings.temperature,
            max_tokens=settings.max_tokens_per_agent,
        )
        return response.choices[0].message.content

    return call


def _create_huggingface_model(model_name: str | None = None) -> callable:
    """Create a HuggingFace Inference API callable."""
    from huggingface_hub import InferenceClient

    model = model_name or settings.hf_model
    client = InferenceClient(
        model=model,
        token=settings.hf_token or os.environ.get("HF_TOKEN"),
    )

    def call(prompt: str, *, temperature: float | None = None) -> str:
        temp = temperature if temperature is not None else settings.temperature
        response = client.text_generation(
            prompt,
            max_new_tokens=settings.max_tokens_per_agent,
            temperature=max(temp, 0.01),  # HF needs >0
            do_sample=True,
        )
        return response

    return call


def _create_ollama_model(model_name: str | None = None) -> callable:
    """Create an Ollama (local) callable for open-source models."""
    try:
        import ollama as ollama_lib
    except ImportError as exc:
        raise RuntimeError(
            "Ollama package not installed. Run: pip install -e '.[ollama]'\n"
            "Also ensure Ollama is running: ollama serve"
        ) from exc

    model = model_name or settings.ollama_model
    client = ollama_lib.Client(host=settings.ollama_base_url)

    # Verify the model is available before returning the callable
    try:
        client.show(model)
    except Exception as exc:
        raise RuntimeError(
            f"Ollama model '{model}' not found. Pull it first:\n\n"
            f"    ollama pull {model}\n\n"
            f"Make sure Ollama is running (ollama serve) and the model name is correct.\n"
            f"Available models can be listed with: ollama list"
        ) from exc

    def call(prompt: str, *, temperature: float | None = None, json_mode: bool = False) -> str:
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "options": {
                "temperature": temperature if temperature is not None else settings.temperature,
                "num_predict": settings.max_tokens_per_agent,
            },
        }
        # JSON mode: instruct Ollama to constrain output to valid JSON
        if json_mode or _prompt_requests_json(prompt):
            kwargs["format"] = "json"
        response = client.chat(**kwargs)
        return response["message"]["content"]

    return call


def _prompt_requests_json(prompt: str) -> bool:
    """Heuristic: detect if a prompt is asking for JSON output."""
    indicators = [
        "Respond ONLY with the JSON",
        "Respond in valid JSON",
        "respond with a JSON",
        "output as JSON",
        "JSON object",
        "JSON matching this",
    ]
    return any(ind.lower() in prompt.lower() for ind in indicators)


# ---------------------------------------------------------------------------
# Unified model factory
# ---------------------------------------------------------------------------

PROVIDER_FACTORIES = {
    LLMProvider.ANTHROPIC: _create_anthropic_model,
    LLMProvider.GOOGLE: _create_google_model,
    LLMProvider.OPENAI: _create_openai_model,
    LLMProvider.HUGGINGFACE: _create_huggingface_model,
    LLMProvider.OLLAMA: _create_ollama_model,
}

# Display names for the UI dropdown
PROVIDER_DISPLAY = {
    "Claude (Anthropic)": LLMProvider.ANTHROPIC,
    "Gemini (Google)": LLMProvider.GOOGLE,
    "GPT (OpenAI)": LLMProvider.OPENAI,
    "HuggingFace API": LLMProvider.HUGGINGFACE,
    "Ollama (Local)": LLMProvider.OLLAMA,
}

# Reverse map for default selection
PROVIDER_TO_DISPLAY = {v: k for k, v in PROVIDER_DISPLAY.items()}

# Model choices per provider — first entry is the default
PROVIDER_MODEL_CHOICES: dict[str, list[str]] = {
    "Claude (Anthropic)": [
        "claude-sonnet-4-20250514",
        "claude-haiku-4-20250414",
    ],
    "Gemini (Google)": [
        "gemini-2.0-flash",
        "gemini-2.5-pro-preview-06-05",
        "gemini-2.5-flash-preview-04-17",
    ],
    "GPT (OpenAI)": [
        "gpt-4o-mini",
        "gpt-4o",
    ],
    "HuggingFace API": [
        "Qwen/Qwen2.5-72B-Instruct",
    ],
    "Ollama (Local)": [
        "llama3.1:8b",
        "llama3.2:3b",
        "llama3:70b",
    ],
}


def _get_default_model_for_provider(provider_display_name: str) -> str:
    """Return the default (first) model for a provider display name."""
    choices = PROVIDER_MODEL_CHOICES.get(provider_display_name, [])
    return choices[0] if choices else ""


# ---------------------------------------------------------------------------
# Timeout wrapper — prevents indefinite hangs on LLM calls
# ---------------------------------------------------------------------------

class _ModelTimeout(Exception):
    """Raised when an LLM call exceeds the configured timeout."""


def with_timeout(model_fn: callable, timeout_seconds: float) -> callable:
    """Wrap a model callable with a timeout guard.

    Uses a daemon thread so the main thread isn't blocked forever.
    If the call exceeds *timeout_seconds*, raises _ModelTimeout.
    If timeout_seconds <= 0, returns the model unchanged (no timeout).
    """
    if timeout_seconds <= 0:
        return model_fn

    import concurrent.futures

    def timed_call(prompt: str, **kwargs) -> str:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(model_fn, prompt, **kwargs)
            try:
                return future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError:
                raise _ModelTimeout(
                    f"LLM call timed out after {timeout_seconds:.0f}s"
                )

    # Preserve attributes from wrapped model (e.g. RateLimitedModel internals)
    timed_call._original_model = model_fn
    timed_call._timeout_seconds = timeout_seconds
    return timed_call


# ---------------------------------------------------------------------------
# Rate limiter — wraps model callable to respect per-minute token budgets
# ---------------------------------------------------------------------------

# Provider-specific rate limits (configurable via settings / .env)
# Only providers listed here get wrapped. Others run unrestricted.
PROVIDER_RATE_LIMITS: dict[LLMProvider, dict[str, int]] = {
    LLMProvider.ANTHROPIC: {
        "max_rpm": settings.rate_limit_rpm,
        "max_input_tpm": settings.rate_limit_input_tpm,
        "max_output_tpm": settings.rate_limit_output_tpm,
    },
    LLMProvider.OPENAI: {
        "max_rpm": 400,
        "max_input_tpm": 80_000,
        "max_output_tpm": 25_000,
    },
    LLMProvider.GOOGLE: {
        "max_rpm": 300,
        "max_input_tpm": 80_000,
        "max_output_tpm": 25_000,
    },
    # HuggingFace, Ollama: no wrapping needed (generous limits or local)
}


class RateLimitedModel:
    """
    Wraps a model callable with per-minute token budget enforcement.

    Uses a 60-second sliding window to track requests and token usage.
    Thread-safe — parallel agents (LangGraph Send / ThreadPoolExecutor)
    coordinate through a shared lock.

    Token estimation uses the same word-based heuristic as doc_chunker:
    ~1.33 tokens per word (no tiktoken dependency required).
    """

    def __init__(
        self,
        model_fn: callable,
        max_rpm: int = 50,
        max_input_tpm: int = 30_000,
        max_output_tpm: int = 8_000,
    ):
        self._model = model_fn
        self._max_rpm = max_rpm
        self._max_input_tpm = max_input_tpm
        self._max_output_tpm = max_output_tpm
        self._call_log: list[tuple[float, int, int]] = []  # (timestamp, in_tokens, out_tokens)
        self._lock = threading.Lock()

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Word-based token estimate: ~1.33 tokens per whitespace-separated word."""
        return int(len(text.split()) * 4 / 3)

    def _prune_old(self) -> None:
        """Remove entries older than 60 seconds from the sliding window."""
        cutoff = time.time() - 60
        self._call_log = [(t, i, o) for t, i, o in self._call_log if t > cutoff]

    def _wait_for_budget(self, input_tokens: int) -> None:
        """Block until the sliding window has enough budget for this call."""
        while True:
            with self._lock:
                self._prune_old()
                used_rpm = len(self._call_log)
                used_input = sum(i for _, i, _ in self._call_log)

                if (used_rpm < self._max_rpm
                        and used_input + input_tokens <= self._max_input_tpm):
                    return  # budget available

                # Calculate how long until enough budget frees up
                if self._call_log:
                    oldest = self._call_log[0][0]
                    wait_secs = max(0, 60 - (time.time() - oldest)) + 0.5
                else:
                    wait_secs = 2.0

            logger.debug(
                f"Rate limiter: {used_rpm} reqs, {used_input}+{input_tokens} input tokens "
                f"in window — waiting {wait_secs:.1f}s for budget"
            )
            time.sleep(min(wait_secs, 5.0))  # cap individual sleep at 5s for responsiveness

    def __call__(self, prompt: str, **kwargs) -> str:
        input_tokens = self._estimate_tokens(prompt)
        self._wait_for_budget(input_tokens)

        result = self._model(prompt, **kwargs)
        output_tokens = self._estimate_tokens(result)

        with self._lock:
            self._call_log.append((time.time(), input_tokens, output_tokens))

        return result


def create_model(
    provider: LLMProvider | None = None,
    model_name: str | None = None,
) -> callable:
    """
    Create the LLM callable for the given provider.

    Architecture note: Every provider returns a simple `callable(str) -> str`.
    This uniform interface means agents don't know or care which LLM they're
    talking to. Swapping providers is a one-line change.

    For providers with tight rate limits (Anthropic Tier 1), the callable is
    wrapped with a RateLimitedModel that enforces per-minute token budgets.

    Args:
        provider: Which LLM backend to use
        model_name: Specific model to use (overrides settings default)
    """
    provider = provider or settings.resolve_provider()
    factory = PROVIDER_FACTORIES.get(provider)
    if not factory:
        raise ValueError(f"Unknown provider: {provider}")

    model_fn = factory(model_name=model_name)

    # Wrap with rate limiter if provider has tight limits
    limits = PROVIDER_RATE_LIMITS.get(provider)
    if limits:
        logger.info(
            f"Rate limiter active for {provider.value}: "
            f"{limits['max_rpm']} RPM, {limits['max_input_tpm']} input TPM"
        )
        model_fn = RateLimitedModel(model_fn, **limits)

    # Wrap with timeout guard (applies to all providers)
    timeout = settings.model_timeout_seconds
    if timeout > 0:
        model_fn = with_timeout(model_fn, timeout)

    return model_fn
