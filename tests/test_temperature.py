"""
Tests for per-node temperature routing.

Verifies that:
1. with_temperature() wraps models correctly
2. Temperature kwarg is passed through to models that support it
3. Graceful fallback for models that don't support temperature kwarg
4. get_node_temperature() returns correct defaults
5. User overrides via settings.task_temperatures work
"""

from __future__ import annotations


class TestWithTemperature:
    """Test the with_temperature() wrapper."""

    def test_passes_temperature_kwarg(self):
        """Model receives temperature when it accepts the kwarg."""
        from orchestrator.temperature import with_temperature

        calls = []

        def model(prompt: str, *, temperature: float | None = None) -> str:
            calls.append({"prompt": prompt, "temperature": temperature})
            return "response"

        wrapped = with_temperature(model, 0.3)
        result = wrapped("hello")

        assert result == "response"
        assert len(calls) == 1
        assert calls[0]["temperature"] == 0.3

    def test_fallback_for_old_model(self):
        """Models without temperature kwarg still work (TypeError caught)."""
        from orchestrator.temperature import with_temperature

        def old_model(prompt: str) -> str:
            return f"echo: {prompt}"

        wrapped = with_temperature(old_model, 0.5)
        result = wrapped("test")

        assert result == "echo: test"

    def test_preserves_original_model_ref(self):
        """Wrapper stores reference to original model."""
        from orchestrator.temperature import with_temperature

        def model(prompt: str) -> str:
            return prompt

        wrapped = with_temperature(model, 0.7)
        assert wrapped._original_model is model
        assert wrapped._temperature == 0.7

    def test_different_temperatures(self):
        """Different wrappers use different temperatures."""
        from orchestrator.temperature import with_temperature

        temps = []

        def model(prompt: str, *, temperature: float | None = None) -> str:
            temps.append(temperature)
            return "ok"

        low = with_temperature(model, 0.1)
        high = with_temperature(model, 0.9)

        low("a")
        high("b")

        assert temps == [0.1, 0.9]

    def test_zero_temperature(self):
        """T=0.0 is passed correctly (not treated as falsy)."""
        from orchestrator.temperature import with_temperature

        received = []

        def model(prompt: str, *, temperature: float | None = None) -> str:
            received.append(temperature)
            return "ok"

        wrapped = with_temperature(model, 0.0)
        wrapped("test")

        assert received == [0.0]


class TestGetNodeTemperature:
    """Test the get_node_temperature() lookup."""

    def test_known_nodes(self):
        """Built-in defaults exist for all pipeline nodes."""
        from orchestrator.temperature import get_node_temperature

        assert get_node_temperature("run_sector_analyst") == 0.5
        assert get_node_temperature("run_risk_manager") == 0.5
        assert get_node_temperature("run_macro_analyst") == 0.5
        assert get_node_temperature("run_debate_round") == 0.5
        assert get_node_temperature("run_portfolio_manager") == 0.7
        assert get_node_temperature("run_optimizer") == 0.0
        assert get_node_temperature("gather_data") == 0.1

    def test_unknown_node_falls_back(self):
        """Unknown nodes fall back to settings.temperature."""
        from config.settings import settings
        from orchestrator.temperature import get_node_temperature

        result = get_node_temperature("nonexistent_node")
        assert result == settings.temperature

    def test_user_override(self):
        """settings.task_temperatures overrides built-in defaults."""
        from config.settings import settings
        from orchestrator.temperature import get_node_temperature

        original = settings.task_temperatures
        try:
            settings.task_temperatures = {"run_sector_analyst": 0.9}
            assert get_node_temperature("run_sector_analyst") == 0.9
            # Other nodes unaffected
            assert get_node_temperature("run_risk_manager") == 0.5
        finally:
            settings.task_temperatures = original


class TestTemperatureTable:
    """Verify the temperature assignments match the recommended ranges."""

    def test_data_extraction_low(self):
        """Data extraction should have low temperature (0.0-0.2)."""
        from orchestrator.temperature import get_node_temperature
        assert get_node_temperature("gather_data") <= 0.2

    def test_risk_assessment_moderate(self):
        """Risk assessment should be moderate (0.4-0.6)."""
        from orchestrator.temperature import get_node_temperature
        temp = get_node_temperature("run_risk_manager")
        assert 0.4 <= temp <= 0.6

    def test_synthesis_higher(self):
        """PM synthesis should be higher (0.6-0.8)."""
        from orchestrator.temperature import get_node_temperature
        temp = get_node_temperature("run_portfolio_manager")
        assert 0.6 <= temp <= 0.8

    def test_math_zero(self):
        """Math/optimizer should be T=0."""
        from orchestrator.temperature import get_node_temperature
        assert get_node_temperature("run_optimizer") == 0.0


class TestRateLimitedModelPassthrough:
    """Verify RateLimitedModel passes kwargs to wrapped model."""

    def test_kwargs_passthrough(self):
        """RateLimitedModel should forward **kwargs to the inner model."""
        import sys
        sys.path.insert(0, ".")

        received_kwargs = {}

        def mock_model(prompt: str, **kwargs) -> str:
            received_kwargs.update(kwargs)
            return "response"

        from app import RateLimitedModel

        wrapped = RateLimitedModel(mock_model, max_rpm=100, max_input_tpm=100000, max_output_tpm=50000)
        wrapped("test prompt", temperature=0.3)

        assert received_kwargs.get("temperature") == 0.3
