"""
Implied volatility surface generation for analyst agents.

Provides Heston stochastic volatility and CEV local volatility models
with COS (Fourier-cosine expansion) method pricing and implied volatility
surface construction.

Agents can call these tools to get structured vol surface data for
options-aware risk assessment, position structuring (options_overlay),
and implied vs. realized vol analysis.

Models implemented:
    - **Heston** (stochastic volatility): captures smile/skew via
      stochastic variance with mean reversion and vol-of-vol.
    - **CEV** (constant elasticity of variance): captures skew via
      local volatility with leverage effect parameter beta.
    - **COS method**: Fast Fourier-cosine expansion pricer that works
      with any model providing a characteristic function.

Numerical methods adapted from:
    C.W. Oosterlee and L.A. Grzelak, "Mathematical Modeling and Computation
    in Finance: With Exercises and Python and MATLAB Computer Codes,"
    World Scientific, 2019. Code: https://github.com/LechGrzelak/QuantFinanceBook
    Licensed under BSD 3-Clause. See THIRD_PARTY_NOTICES.md for full license.
"""

from __future__ import annotations

import enum
import logging
from typing import Any

import numpy as np
import scipy.optimize as optimize
import scipy.stats as st

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Option type enum
# ---------------------------------------------------------------------------

class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0


# ---------------------------------------------------------------------------
# Black-Scholes reference pricer (for implied vol inversion)
# ---------------------------------------------------------------------------

def bs_call_put_price(
    cp: OptionType,
    S0: float,
    K: np.ndarray | float,
    sigma: float | np.ndarray,
    T: float,
    r: float,
) -> np.ndarray:
    """Black-Scholes European option price.

    Adapted from Oosterlee & Grzelak (2019), BSD 3-Clause.
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if cp == OptionType.CALL:
        return st.norm.cdf(d1) * S0 - st.norm.cdf(d2) * K * np.exp(-r * T)
    return st.norm.cdf(-d2) * K * np.exp(-r * T) - st.norm.cdf(-d1) * S0


def implied_volatility(
    cp: OptionType,
    market_price: float,
    K: float,
    T: float,
    S0: float,
    r: float = 0.0,
) -> float:
    """Extract BS implied volatility from a model price via Newton's method.

    Uses a grid search for initial guess to ensure convergence.

    Adapted from Oosterlee & Grzelak (2019), BSD 3-Clause.
    """
    sigma_grid = np.linspace(0.01, 3.0, 5000)
    price_grid = bs_call_put_price(cp, S0, K, sigma_grid, T, r).ravel()
    sigma_init = float(np.interp(market_price, price_grid, sigma_grid))

    func = lambda sigma: float(bs_call_put_price(cp, S0, K, sigma, T, r).ravel()[0]) - market_price
    try:
        return float(optimize.newton(func, sigma_init, tol=1e-10, maxiter=200))
    except RuntimeError:
        return float(sigma_init)


# ---------------------------------------------------------------------------
# COS method — Fourier-cosine expansion option pricer
# ---------------------------------------------------------------------------

def _chi(k: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Chi coefficients for the COS method.

    Adapted from Oosterlee & Grzelak (2019), BSD 3-Clause.
    """
    bma = b - a
    kpi = k * np.pi / bma
    denom = 1.0 + kpi**2
    val = (
        1.0 / denom
        * (
            np.cos(kpi * (d - a)) * np.exp(d)
            - np.cos(kpi * (c - a)) * np.exp(c)
            + kpi * np.sin(kpi * (d - a)) * np.exp(d)
            - kpi * np.sin(kpi * (c - a)) * np.exp(c)
        )
    )
    return val


def _psi(k: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Psi coefficients for the COS method.

    Adapted from Oosterlee & Grzelak (2019), BSD 3-Clause.
    """
    bma = b - a
    kpi = k * np.pi / bma
    # k=0 case: psi = d - c; k>0: use sin formula
    # Suppress divide-by-zero for k=0 (handled by np.where)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            np.abs(k) < 1e-14,
            d - c,
            (np.sin(kpi * (d - a)) - np.sin(kpi * (c - a))) / kpi,
        )
    return result


def cos_price(
    cp: OptionType,
    cf_func,
    S0: float,
    K: np.ndarray | float,
    T: float,
    r: float,
    N: int = 2**12,
) -> np.ndarray:
    """Price European options using the COS method.

    Prices puts directly via cosine expansion, then uses put-call parity
    for calls (better convergence on left tail).

    Args:
        cp: CALL or PUT
        cf_func: Characteristic function cf(u) -> complex array
        S0: Spot price
        K: Strike(s)
        T: Time to expiry
        r: Risk-free rate
        N: Number of Fourier terms (higher = more accurate, default 4096)

    Returns:
        Array of option prices for each strike.

    Adapted from Oosterlee & Grzelak (2019), BSD 3-Clause.
    """
    K_arr = np.atleast_1d(np.asarray(K, dtype=float)).reshape(-1, 1)

    # Truncation domain for log-moneyness
    L = 10.0
    a = 0.0 - L * np.sqrt(T)
    b = 0.0 + L * np.sqrt(T)

    # Fourier frequencies — column vector (N, 1)
    k = np.linspace(0, N - 1, N).reshape(N, 1)
    u = k * np.pi / (b - a)

    # Put payoff coefficients H_k — column vector (N, 1)
    chi_k = _chi(k.ravel(), a, b, a, 0.0).reshape(N, 1)
    psi_k = _psi(k.ravel(), a, b, a, 0.0).reshape(N, 1)
    H_k = 2.0 / (b - a) * (-chi_k + psi_k)

    # Log-moneyness per strike — column vector (nK, 1)
    x0 = np.log(S0 / K_arr)

    # Exponential matrix: (nK, N) via outer product of (x0 - a) and u
    # The (x0 - a) shift incorporates the exp(-iu*a) factor from Fang-Oosterlee (2009)
    mat = np.exp(1j * np.outer((x0 - a).ravel(), u.ravel()))

    # CF values * payoff coefficients — column vector (N, 1)
    cf_vals = cf_func(u.ravel())
    if np.isscalar(cf_vals):
        cf_vals = np.array([cf_vals])
    cf_vals = cf_vals.reshape(N, 1)
    temp = cf_vals * H_k
    temp[0] = 0.5 * temp[0]  # halve k=0 term

    # Put prices: (nK, N) @ (N, 1) -> (nK, 1)
    put_prices = np.exp(-r * T) * K_arr * np.real(mat.dot(temp))
    put_prices = put_prices.ravel()

    if cp == OptionType.PUT:
        return np.maximum(put_prices, 0.0)

    # Call via put-call parity: C = P + S - K*exp(-rT)
    call_prices = put_prices + S0 - K_arr.ravel() * np.exp(-r * T)
    return np.maximum(call_prices, 0.0)


# ---------------------------------------------------------------------------
# Heston stochastic volatility model
# ---------------------------------------------------------------------------

def heston_cf(
    u: np.ndarray | float,
    T: float,
    r: float,
    kappa: float,
    gamma: float,
    vbar: float,
    v0: float,
    rho: float,
) -> np.ndarray:
    """Characteristic function of log(S_T/S_0) under the Heston model.

    Parameters:
        u: Fourier frequency
        T: Time to maturity
        r: Risk-free rate
        kappa: Mean reversion speed of variance
        gamma: Vol-of-vol (sigma_v)
        vbar: Long-run variance (theta)
        v0: Initial variance
        rho: Correlation between stock and variance Brownian motions

    Returns:
        Complex characteristic function values.

    Adapted from Oosterlee & Grzelak (2019), Ch. 10, BSD 3-Clause.
    """
    u = np.atleast_1d(np.asarray(u, dtype=float))
    i = 1j

    D = np.sqrt(
        (kappa - i * rho * gamma * u) ** 2
        + (u**2 + i * u) * gamma**2
    )
    g = (kappa - i * rho * gamma * u - D) / (kappa - i * rho * gamma * u + D)

    exp_DT = np.exp(-D * T)

    C = (1.0 / (gamma**2)) * (
        (kappa - i * rho * gamma * u - D) * T
        - 2.0 * np.log((1.0 - g * exp_DT) / (1.0 - g))
    )
    E = (1.0 / (gamma**2)) * (
        (kappa - i * rho * gamma * u - D)
        * (1.0 - exp_DT)
        / (1.0 - g * exp_DT)
    )

    cf = np.exp(i * u * r * T + C * vbar * kappa + E * v0)
    return cf


def heston_price(
    cp: OptionType,
    S0: float,
    K: np.ndarray | float,
    T: float,
    r: float,
    kappa: float,
    gamma: float,
    vbar: float,
    v0: float,
    rho: float,
    N: int = 2**12,
) -> np.ndarray:
    """Price European options under the Heston model using COS method."""
    cf = lambda u: heston_cf(u, T, r, kappa, gamma, vbar, v0, rho)
    return cos_price(cp, cf, S0, K, T, r, N)


# ---------------------------------------------------------------------------
# CEV (Constant Elasticity of Variance) model
# ---------------------------------------------------------------------------

def cev_price(
    cp: OptionType,
    S0: float,
    K: np.ndarray | float,
    T: float,
    r: float,
    sigma: float,
    beta: float,
) -> np.ndarray:
    """Price European options under the CEV model.

    Uses the non-central chi-squared distribution.

    Parameters:
        cp: CALL or PUT
        S0: Spot price
        K: Strike(s)
        T: Time to maturity
        r: Risk-free rate
        sigma: Volatility parameter
        beta: Elasticity parameter (0 < beta < 1 gives skew)

    Adapted from Oosterlee & Grzelak (2019), Ch. 14, Fig14_03, BSD 3-Clause.
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    prices = np.zeros(len(K))

    for i, Ki in enumerate(K):
        if Ki <= 0:
            if cp == OptionType.CALL:
                prices[i] = S0 * np.exp(-r * T)
            else:
                prices[i] = 0.0
            continue

        nu = 1.0 / (1.0 - beta)
        omega = sigma**2 * (1.0 - beta) / r
        z_K = (Ki ** (2.0 * (1.0 - beta))) / omega / (1.0 - np.exp(-r * (1.0 - beta) * 2.0 * T))
        z_S = (S0 ** (2.0 * (1.0 - beta))) / omega / (1.0 - np.exp(-r * (1.0 - beta) * 2.0 * T))
        z_S_adj = z_S * np.exp(-r * (1.0 - beta) * 2.0 * T)

        if beta < 1.0:
            # beta < 1: standard CEV
            if cp == OptionType.CALL:
                prices[i] = np.exp(-r * T) * (
                    S0 * np.exp(r * T) * (1.0 - st.ncx2.cdf(z_K, 2.0 * nu + 2.0, z_S_adj))
                    - Ki * st.ncx2.cdf(z_S_adj, 2.0 * nu, z_K)
                )
            else:
                prices[i] = np.exp(-r * T) * (
                    Ki * (1.0 - st.ncx2.cdf(z_S_adj, 2.0 * nu, z_K))
                    - S0 * np.exp(r * T) * st.ncx2.cdf(z_K, 2.0 * nu + 2.0, z_S_adj)
                )
        else:
            # beta > 1: reversed roles
            if cp == OptionType.CALL:
                prices[i] = np.exp(-r * T) * (
                    S0 * np.exp(r * T) * st.ncx2.cdf(z_S_adj, -2.0 * nu + 2.0, z_K)
                    - Ki * (1.0 - st.ncx2.cdf(z_K, -2.0 * nu, z_S_adj))
                )
            else:
                prices[i] = np.exp(-r * T) * (
                    Ki * st.ncx2.cdf(z_K, -2.0 * nu, z_S_adj)
                    - S0 * np.exp(r * T) * (1.0 - st.ncx2.cdf(z_S_adj, -2.0 * nu + 2.0, z_K))
                )

    return np.maximum(prices, 0.0)


# ---------------------------------------------------------------------------
# Implied volatility surface generation
# ---------------------------------------------------------------------------

def generate_vol_surface(
    model: str,
    S0: float,
    r: float,
    strikes: np.ndarray | list[float] | None = None,
    maturities: np.ndarray | list[float] | None = None,
    **model_params,
) -> dict[str, Any]:
    """Generate an implied volatility surface.

    Args:
        model: "heston" or "cev"
        S0: Current spot price
        r: Risk-free rate
        strikes: Array of strikes (default: 0.7*S0 to 1.3*S0 in 15 steps)
        maturities: Array of maturities in years (default: [0.083, 0.25, 0.5, 1.0, 2.0])
        **model_params: Model-specific parameters:
            Heston: kappa, gamma, vbar, v0, rho
            CEV: sigma, beta

    Returns:
        Dict with keys: model, spot, strikes, maturities, iv_surface (2D list),
        skew_by_maturity, term_structure_atm, summary.
    """
    if strikes is None:
        strikes = np.linspace(0.7 * S0, 1.3 * S0, 15)
    else:
        strikes = np.asarray(strikes, dtype=float)

    if maturities is None:
        maturities = np.array([1 / 12, 0.25, 0.5, 1.0, 2.0])
    else:
        maturities = np.asarray(maturities, dtype=float)

    cp = OptionType.CALL
    iv_surface = np.full((len(maturities), len(strikes)), np.nan)

    for t_idx, T in enumerate(maturities):
        if T <= 0:
            continue
        # Price under chosen model
        if model.lower() == "heston":
            prices = heston_price(cp, S0, strikes, T, r, **model_params)
        elif model.lower() == "cev":
            prices = cev_price(cp, S0, strikes, T, r, **model_params)
        else:
            raise ValueError(f"Unknown model: {model}. Use 'heston' or 'cev'.")

        # Extract implied vols
        for k_idx, Ki in enumerate(strikes):
            price = prices[k_idx]
            intrinsic = max(S0 - Ki * np.exp(-r * T), 0.0)
            if price <= intrinsic + 1e-10:
                continue
            try:
                iv_surface[t_idx, k_idx] = implied_volatility(cp, price, Ki, T, S0, r)
            except Exception:
                pass

    # Derived analytics
    atm_idx = int(np.argmin(np.abs(strikes - S0)))
    atm_term_structure = {}
    for t_idx, T in enumerate(maturities):
        iv = iv_surface[t_idx, atm_idx]
        if not np.isnan(iv):
            atm_term_structure[f"{T:.3f}y"] = round(float(iv) * 100, 2)

    skew_by_maturity = {}
    for t_idx, T in enumerate(maturities):
        row = iv_surface[t_idx, :]
        valid = ~np.isnan(row)
        if valid.sum() >= 3:
            otm_put = row[valid][0]
            atm = row[valid][len(row[valid]) // 2]
            otm_call = row[valid][-1]
            skew_by_maturity[f"{T:.3f}y"] = {
                "put_wing": round(float(otm_put) * 100, 2),
                "atm": round(float(atm) * 100, 2),
                "call_wing": round(float(otm_call) * 100, 2),
                "skew_25d": round(float(otm_put - otm_call) * 100, 2),
            }

    # Summary for agent consumption
    atm_vols = [v for v in iv_surface[:, atm_idx] if not np.isnan(v)]
    short_term_vol = atm_vols[0] * 100 if atm_vols else None
    long_term_vol = atm_vols[-1] * 100 if atm_vols else None
    term_slope = "contango" if long_term_vol and short_term_vol and long_term_vol > short_term_vol else "backwardation"

    # Mean skew across maturities
    skew_values = [v["skew_25d"] for v in skew_by_maturity.values() if "skew_25d" in v]
    avg_skew = round(np.mean(skew_values), 2) if skew_values else None

    summary = {
        "model": model,
        "short_term_atm_vol": round(short_term_vol, 2) if short_term_vol else None,
        "long_term_atm_vol": round(long_term_vol, 2) if long_term_vol else None,
        "term_structure": term_slope,
        "avg_skew_25d": avg_skew,
        "interpretation": _interpret_surface(term_slope, avg_skew, short_term_vol, long_term_vol),
    }

    return {
        "model": model,
        "spot": float(S0),
        "risk_free_rate": float(r),
        "model_params": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in model_params.items()},
        "strikes": [round(float(k), 4) for k in strikes],
        "maturities": [round(float(t), 4) for t in maturities],
        "iv_surface_pct": [[round(float(v) * 100, 2) if not np.isnan(v) else None for v in row] for row in iv_surface],
        "atm_term_structure": atm_term_structure,
        "skew_by_maturity": skew_by_maturity,
        "summary": summary,
    }


def _interpret_surface(
    term_slope: str,
    avg_skew: float | None,
    short_vol: float | None,
    long_vol: float | None,
) -> str:
    """Generate plain-English interpretation of vol surface for agents."""
    parts = []

    if term_slope == "backwardation" and short_vol and long_vol:
        parts.append(
            f"Term structure in backwardation (short-term {short_vol:.1f}% > "
            f"long-term {long_vol:.1f}%): market pricing near-term event risk "
            f"or elevated uncertainty."
        )
    elif short_vol and long_vol:
        parts.append(
            f"Term structure in contango (short-term {short_vol:.1f}% < "
            f"long-term {long_vol:.1f}%): normal conditions, no near-term stress priced."
        )

    if avg_skew is not None:
        if avg_skew > 5:
            parts.append(
                f"Pronounced negative skew ({avg_skew:.1f}pp): heavy demand for "
                f"downside protection. Market pricing tail risk to the left."
            )
        elif avg_skew > 2:
            parts.append(
                f"Moderate skew ({avg_skew:.1f}pp): typical demand for downside hedging."
            )
        elif avg_skew < -1:
            parts.append(
                f"Positive skew ({avg_skew:.1f}pp): unusual — market pricing upside "
                f"risk higher than downside. Often seen in short-squeeze or "
                f"momentum names."
            )
        else:
            parts.append(f"Symmetric smile ({avg_skew:.1f}pp): balanced risk pricing.")

    return " ".join(parts) if parts else "Insufficient data for interpretation."


# ---------------------------------------------------------------------------
# Tool interface for ToolRegistry
# ---------------------------------------------------------------------------

class VolatilitySurfaceTool:
    """Agent-facing tool for implied volatility surface generation.

    Provides two entry points:
        - get_vol_surface: Full strike x maturity grid
        - get_vol_smile: Single-maturity smile for quick assessment
    """

    @staticmethod
    def get_vol_surface(
        spot: float,
        r: float = 0.04,
        model: str = "heston",
        kappa: float = 1.5,
        gamma: float = 0.5,
        vbar: float = 0.06,
        v0: float = 0.04,
        rho: float = -0.7,
        sigma: float = 0.3,
        beta: float = 0.5,
    ) -> dict[str, Any]:
        """Generate implied volatility surface for agent consumption.

        Args:
            spot: Current stock price
            r: Risk-free rate (default 0.04)
            model: "heston" (default) or "cev"
            kappa: Heston mean reversion speed (default 1.5)
            gamma: Heston vol-of-vol (default 0.5)
            vbar: Heston long-run variance (default 0.06)
            v0: Heston initial variance (default 0.04)
            rho: Heston correlation (default -0.7)
            sigma: CEV volatility parameter (default 0.3)
            beta: CEV elasticity (default 0.5)

        Returns:
            Structured dict with iv_surface_pct, atm_term_structure,
            skew_by_maturity, and plain-English summary.
        """
        try:
            if model.lower() == "heston":
                params = dict(kappa=kappa, gamma=gamma, vbar=vbar, v0=v0, rho=rho)
            elif model.lower() == "cev":
                params = dict(sigma=sigma, beta=beta)
            else:
                return {"error": f"Unknown model '{model}'. Use 'heston' or 'cev'."}

            return generate_vol_surface(model=model, S0=spot, r=r, **params)
        except Exception as e:
            logger.error(f"Vol surface generation failed: {e}")
            return {"error": f"Vol surface generation failed: {e}"}

    @staticmethod
    def get_vol_smile(
        spot: float,
        maturity: float = 0.25,
        r: float = 0.04,
        model: str = "heston",
        kappa: float = 1.5,
        gamma: float = 0.5,
        vbar: float = 0.06,
        v0: float = 0.04,
        rho: float = -0.7,
        sigma: float = 0.3,
        beta: float = 0.5,
    ) -> dict[str, Any]:
        """Generate a single-maturity implied volatility smile.

        Lighter-weight alternative to full surface. Good for quick
        skew assessment at a specific tenor.

        Args:
            spot: Current stock price
            maturity: Time to expiry in years (default 0.25 = 3 months)
            r: Risk-free rate (default 0.04)
            model: "heston" (default) or "cev"
            (remaining args: model parameters, see get_vol_surface)

        Returns:
            Structured dict with strikes, implied_vols_pct, moneyness,
            atm_vol, skew_25d, and interpretation.
        """
        try:
            strikes = np.linspace(0.8 * spot, 1.2 * spot, 21)

            if model.lower() == "heston":
                params = dict(kappa=kappa, gamma=gamma, vbar=vbar, v0=v0, rho=rho)
            elif model.lower() == "cev":
                params = dict(sigma=sigma, beta=beta)
            else:
                return {"error": f"Unknown model '{model}'. Use 'heston' or 'cev'."}

            result = generate_vol_surface(
                model=model, S0=spot, r=r,
                strikes=strikes, maturities=[maturity],
                **params,
            )

            # Flatten to single-maturity output
            ivs = result["iv_surface_pct"][0]
            atm_idx = len(strikes) // 2
            moneyness = [round(float(k / spot), 4) for k in strikes]

            return {
                "model": model,
                "spot": float(spot),
                "maturity": float(maturity),
                "strikes": result["strikes"],
                "moneyness": moneyness,
                "implied_vols_pct": ivs,
                "atm_vol_pct": ivs[atm_idx] if ivs[atm_idx] is not None else None,
                "skew": result["skew_by_maturity"].get(f"{maturity:.3f}y", {}),
                "summary": result["summary"],
            }
        except Exception as e:
            logger.error(f"Vol smile generation failed: {e}")
            return {"error": f"Vol smile generation failed: {e}"}
