"""
Tests for the implied volatility surface tool.

Tests cover:
- Black-Scholes pricing and implied vol round-trip
- COS method convergence against BS closed-form
- Heston model pricing and smile generation
- CEV model pricing and skew behavior
- VolatilitySurfaceTool interface and error handling
- Tool registry integration

All tests are pure numerical — no API keys or network access needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.volatility_surface import (
    OptionType,
    VolatilitySurfaceTool,
    bs_call_put_price,
    cev_price,
    cos_price,
    generate_vol_surface,
    heston_cf,
    heston_price,
    implied_volatility,
)


# ---------------------------------------------------------------------------
# Black-Scholes tests
# ---------------------------------------------------------------------------

class TestBlackScholes:
    """Test BS pricing and implied vol extraction."""

    def test_call_price_atm(self):
        price = bs_call_put_price(OptionType.CALL, 100.0, 100.0, 0.2, 1.0, 0.05)
        # ATM call with 20% vol, 1Y, 5% rate ~= 10.45
        assert 10.0 < float(price[0]) < 11.0

    def test_put_price_atm(self):
        price = bs_call_put_price(OptionType.PUT, 100.0, 100.0, 0.2, 1.0, 0.05)
        assert 5.0 < float(price[0]) < 7.0

    def test_put_call_parity(self):
        S0, K, sigma, T, r = 100.0, 105.0, 0.25, 0.5, 0.03
        call = bs_call_put_price(OptionType.CALL, S0, K, sigma, T, r)[0]
        put = bs_call_put_price(OptionType.PUT, S0, K, sigma, T, r)[0]
        # C - P = S - K*exp(-rT)
        lhs = call - put
        rhs = S0 - K * np.exp(-r * T)
        assert abs(lhs - rhs) < 1e-10

    def test_vector_strikes(self):
        strikes = [90.0, 100.0, 110.0]
        prices = bs_call_put_price(OptionType.CALL, 100.0, strikes, 0.2, 1.0, 0.05)
        assert len(prices) == 3
        # Higher strike => lower call price
        assert prices[0] > prices[1] > prices[2]

    def test_implied_vol_round_trip(self):
        sigma = 0.30
        price = bs_call_put_price(OptionType.CALL, 100.0, 100.0, sigma, 1.0, 0.05)[0]
        recovered = implied_volatility(OptionType.CALL, float(price), 100.0, 1.0, 100.0, 0.05)
        assert abs(recovered - sigma) < 1e-6

    def test_implied_vol_otm_put(self):
        sigma = 0.25
        price = bs_call_put_price(OptionType.PUT, 100.0, 85.0, sigma, 0.5, 0.03)[0]
        recovered = implied_volatility(OptionType.PUT, float(price), 85.0, 0.5, 100.0, 0.03)
        assert abs(recovered - sigma) < 1e-4


# ---------------------------------------------------------------------------
# COS method tests
# ---------------------------------------------------------------------------

class TestCOSMethod:
    """Test COS method against BS closed-form."""

    def test_gbm_call_matches_bs(self):
        """COS method with GBM characteristic function should match BS."""
        S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

        # GBM characteristic function for log(S_T/S_0)
        def cf_gbm(u):
            i = 1j
            return np.exp(i * u * (r - 0.5 * sigma**2) * T - 0.5 * sigma**2 * u**2 * T)

        cos_val = cos_price(OptionType.CALL, cf_gbm, S0, K, T, r)[0]
        bs_val = bs_call_put_price(OptionType.CALL, S0, K, sigma, T, r)[0]
        assert abs(cos_val - bs_val) < 0.05  # Within 5 cents

    def test_gbm_put_matches_bs(self):
        S0, K, T, r, sigma = 100.0, 110.0, 0.5, 0.03, 0.3

        def cf_gbm(u):
            i = 1j
            return np.exp(i * u * (r - 0.5 * sigma**2) * T - 0.5 * sigma**2 * u**2 * T)

        cos_val = cos_price(OptionType.PUT, cf_gbm, S0, K, T, r)[0]
        bs_val = bs_call_put_price(OptionType.PUT, S0, K, sigma, T, r)[0]
        assert abs(cos_val - bs_val) < 0.10

    def test_cos_convergence(self):
        """More Fourier terms should improve accuracy."""
        S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        bs_ref = bs_call_put_price(OptionType.CALL, S0, K, sigma, T, r)[0]

        def cf_gbm(u):
            i = 1j
            return np.exp(i * u * (r - 0.5 * sigma**2) * T - 0.5 * sigma**2 * u**2 * T)

        errors = []
        for N in [64, 256, 1024, 4096]:
            cos_val = cos_price(OptionType.CALL, cf_gbm, S0, K, T, r, N=N)[0]
            errors.append(abs(cos_val - bs_ref))

        # Errors should decrease monotonically (approximately)
        assert errors[-1] < errors[0]
        assert errors[-1] < 0.01  # High-N should be very accurate

    def test_vector_strikes_cos(self):
        S0, T, r, sigma = 100.0, 1.0, 0.05, 0.2

        def cf_gbm(u):
            i = 1j
            return np.exp(i * u * (r - 0.5 * sigma**2) * T - 0.5 * sigma**2 * u**2 * T)

        strikes = np.array([90.0, 100.0, 110.0])
        prices = cos_price(OptionType.CALL, cf_gbm, S0, strikes, T, r)
        assert len(prices) == 3
        assert prices[0] > prices[1] > prices[2]


# ---------------------------------------------------------------------------
# Heston model tests
# ---------------------------------------------------------------------------

class TestHestonModel:
    """Test Heston stochastic volatility model."""

    # Standard test parameters
    PARAMS = dict(kappa=1.5, gamma=0.5, vbar=0.06, v0=0.04, rho=-0.7)

    def test_cf_at_zero(self):
        """Characteristic function at u=0 should be 1."""
        cf = heston_cf(0.0, T=1.0, r=0.05, **self.PARAMS)
        assert abs(float(np.abs(cf[0])) - 1.0) < 1e-10

    def test_call_price_positive(self):
        price = heston_price(OptionType.CALL, 100.0, 100.0, 1.0, 0.05, **self.PARAMS)
        assert float(price[0]) > 0

    def test_put_call_parity_heston(self):
        S0, K, T, r = 100.0, 105.0, 1.0, 0.05
        call = heston_price(OptionType.CALL, S0, K, T, r, **self.PARAMS)[0]
        put = heston_price(OptionType.PUT, S0, K, T, r, **self.PARAMS)[0]
        parity = abs((call - put) - (S0 - K * np.exp(-r * T)))
        assert parity < 0.5  # COS method has finite truncation error

    def test_smile_shape(self):
        """Heston with negative rho should produce downward-sloping skew."""
        S0, T, r = 100.0, 0.5, 0.05
        strikes = np.linspace(80, 120, 11)
        prices = heston_price(OptionType.CALL, S0, strikes, T, r, **self.PARAMS)

        ivs = []
        for k_idx, Ki in enumerate(strikes):
            try:
                iv = implied_volatility(OptionType.CALL, float(prices[k_idx]), Ki, T, S0, r)
                ivs.append(iv)
            except Exception:
                ivs.append(np.nan)

        valid = [v for v in ivs if not np.isnan(v)]
        assert len(valid) >= 5
        # With rho=-0.7, OTM puts (low strikes) should have higher IV
        assert valid[0] > valid[-1]

    def test_heston_reduces_to_bs_when_flat(self):
        """When gamma=0 (no vol-of-vol), Heston should approach BS."""
        S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        flat_params = dict(kappa=1.5, gamma=1e-6, vbar=sigma**2, v0=sigma**2, rho=0.0)

        heston_val = heston_price(OptionType.CALL, S0, K, T, r, **flat_params)[0]
        bs_val = bs_call_put_price(OptionType.CALL, S0, K, sigma, T, r)[0]
        assert abs(heston_val - bs_val) < 0.5


# ---------------------------------------------------------------------------
# CEV model tests
# ---------------------------------------------------------------------------

class TestCEVModel:
    """Test CEV local volatility model."""

    def test_call_price_positive(self):
        price = cev_price(OptionType.CALL, 100.0, 100.0, 1.0, 0.05, sigma=0.3, beta=0.5)
        assert float(price[0]) > 0

    def test_put_price_positive(self):
        price = cev_price(OptionType.PUT, 100.0, 110.0, 1.0, 0.05, sigma=0.3, beta=0.5)
        assert float(price[0]) > 0

    def test_beta_less_than_one_produces_skew(self):
        """beta < 1 should produce downward-sloping vol skew (leverage effect)."""
        S0, T, r, sigma, beta = 100.0, 1.0, 0.05, 0.3, 0.5
        strikes = np.linspace(80, 120, 11)
        prices = cev_price(OptionType.CALL, S0, strikes, T, r, sigma, beta)

        ivs = []
        for k_idx, Ki in enumerate(strikes):
            try:
                iv = implied_volatility(OptionType.CALL, float(prices[k_idx]), Ki, T, S0, r)
                ivs.append(iv)
            except Exception:
                ivs.append(np.nan)

        valid = [v for v in ivs if not np.isnan(v)]
        assert len(valid) >= 5
        # beta < 1 => skew: low strikes have higher IV
        assert valid[0] > valid[-1]

    def test_vector_strikes(self):
        strikes = [90.0, 100.0, 110.0]
        prices = cev_price(OptionType.CALL, 100.0, strikes, 1.0, 0.05, sigma=0.3, beta=0.5)
        assert len(prices) == 3
        assert prices[0] > prices[1] > prices[2]


# ---------------------------------------------------------------------------
# Surface generation tests
# ---------------------------------------------------------------------------

class TestGenerateVolSurface:
    """Test the high-level surface generator."""

    def test_heston_surface_structure(self):
        result = generate_vol_surface(
            model="heston", S0=100.0, r=0.05,
            kappa=1.5, gamma=0.5, vbar=0.06, v0=0.04, rho=-0.7,
        )
        assert result["model"] == "heston"
        assert result["spot"] == 100.0
        assert len(result["strikes"]) == 15
        assert len(result["maturities"]) == 5
        assert len(result["iv_surface_pct"]) == 5
        assert len(result["iv_surface_pct"][0]) == 15
        assert "summary" in result
        assert "atm_term_structure" in result
        assert "skew_by_maturity" in result

    def test_cev_surface_structure(self):
        result = generate_vol_surface(
            model="cev", S0=100.0, r=0.05,
            sigma=0.3, beta=0.5,
        )
        assert result["model"] == "cev"
        assert len(result["iv_surface_pct"]) == 5

    def test_custom_strikes_maturities(self):
        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
        mats = [0.25, 1.0]
        result = generate_vol_surface(
            model="heston", S0=100.0, r=0.05,
            strikes=strikes, maturities=mats,
            kappa=1.5, gamma=0.5, vbar=0.06, v0=0.04, rho=-0.7,
        )
        assert len(result["strikes"]) == 5
        assert len(result["maturities"]) == 2
        assert len(result["iv_surface_pct"]) == 2

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            generate_vol_surface(model="sabr", S0=100.0, r=0.05)

    def test_surface_has_valid_ivs(self):
        result = generate_vol_surface(
            model="heston", S0=100.0, r=0.05,
            kappa=1.5, gamma=0.5, vbar=0.06, v0=0.04, rho=-0.7,
        )
        # At least half the surface should have valid IVs
        flat = [v for row in result["iv_surface_pct"] for v in row if v is not None]
        total = len(result["strikes"]) * len(result["maturities"])
        assert len(flat) > total * 0.5

    def test_summary_interpretation(self):
        result = generate_vol_surface(
            model="heston", S0=100.0, r=0.05,
            kappa=1.5, gamma=0.5, vbar=0.06, v0=0.04, rho=-0.7,
        )
        summary = result["summary"]
        assert summary["model"] == "heston"
        assert summary["term_structure"] in ("contango", "backwardation")
        assert isinstance(summary["interpretation"], str)
        assert len(summary["interpretation"]) > 10


# ---------------------------------------------------------------------------
# Tool interface tests
# ---------------------------------------------------------------------------

class TestVolatilitySurfaceTool:
    """Test the ToolRegistry-facing interface."""

    def test_get_vol_surface_default_heston(self):
        result = VolatilitySurfaceTool.get_vol_surface(spot=150.0)
        assert "error" not in result
        assert result["model"] == "heston"
        assert result["spot"] == 150.0

    def test_get_vol_surface_cev(self):
        result = VolatilitySurfaceTool.get_vol_surface(spot=100.0, model="cev")
        assert "error" not in result
        assert result["model"] == "cev"

    def test_get_vol_surface_unknown_model(self):
        result = VolatilitySurfaceTool.get_vol_surface(spot=100.0, model="sabr")
        assert "error" in result

    def test_get_vol_smile_default(self):
        result = VolatilitySurfaceTool.get_vol_smile(spot=100.0)
        assert "error" not in result
        assert result["model"] == "heston"
        assert result["maturity"] == 0.25
        assert len(result["strikes"]) == 21
        assert len(result["implied_vols_pct"]) == 21
        assert result["atm_vol_pct"] is not None

    def test_get_vol_smile_custom_maturity(self):
        result = VolatilitySurfaceTool.get_vol_smile(spot=200.0, maturity=1.0, model="cev")
        assert "error" not in result
        assert result["maturity"] == 1.0

    def test_get_vol_smile_has_moneyness(self):
        result = VolatilitySurfaceTool.get_vol_smile(spot=100.0)
        assert "moneyness" in result
        assert len(result["moneyness"]) == 21
        # ATM moneyness should be ~1.0
        mid = result["moneyness"][10]
        assert 0.99 < mid < 1.01


# ---------------------------------------------------------------------------
# Registry integration test
# ---------------------------------------------------------------------------

class TestRegistryIntegration:
    """Test that vol surface tools integrate with the tool registry."""

    def test_registry_includes_vol_tools(self):
        from tools.registry import build_default_registry

        registry = build_default_registry(max_calls=10)
        tools = registry.list_tools()
        assert "get_vol_surface" in tools
        assert "get_vol_smile" in tools

    def test_registry_execute_vol_surface(self):
        from tools.registry import build_default_registry

        registry = build_default_registry(max_calls=10)
        result = registry.execute(
            "test_agent", "get_vol_surface",
            {"spot": 100.0, "r": 0.05},
        )
        assert "error" not in result
        assert result["model"] == "heston"

    def test_registry_execute_vol_smile(self):
        from tools.registry import build_default_registry

        registry = build_default_registry(max_calls=10)
        result = registry.execute(
            "test_agent", "get_vol_smile",
            {"spot": 100.0},
        )
        assert "error" not in result
        assert "atm_vol_pct" in result

    def test_catalog_includes_vol_tools(self):
        from tools.registry import build_default_registry

        registry = build_default_registry()
        catalog = registry.get_catalog()
        assert "get_vol_surface" in catalog
        assert "get_vol_smile" in catalog
        assert "implied volatility" in catalog.lower()
