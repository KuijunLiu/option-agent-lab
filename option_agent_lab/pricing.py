"""Vectorized European call reference prices, including boundary cases."""

import numpy as np
from scipy.special import ndtr


def bs_call(spot, strike, maturity_years, volatility, rate, dividend_yield=0.0):
    s, k, t, v, r, q = np.broadcast_arrays(*[
        np.asarray(a, dtype=np.float64)
        for a in (spot, strike, maturity_years, volatility, rate, dividend_yield)
    ])
    if not all(np.isfinite(a).all() for a in (s, k, t, v, r, q)):
        raise ValueError("All parameters must be finite")
    if np.any(s <= 0) or np.any(k <= 0) or np.any(t < 0) or np.any(v < 0):
        raise ValueError("Require spot,strike>0 and maturity,volatility>=0")
    sd, kd = s * np.exp(-q * t), k * np.exp(-r * t)
    std = v * np.sqrt(t)
    safe_std = np.where(std > 0, std, 1.0)
    d1 = (np.log(s / k) + (r - q + v * v / 2) * t) / safe_std
    formula = sd * ndtr(d1) - kd * ndtr(d1 - std)
    return np.where(std > 0, formula, np.maximum(sd - kd, 0.0))
