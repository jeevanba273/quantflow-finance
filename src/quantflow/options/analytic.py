"""
Vectorised Black-Scholes-Merton formulas for European options.

The functions here accept scalars or NumPy arrays and broadcast them against
each other, so a whole option chain can be priced in one call. They are the
single implementation of the closed-form price and Greeks:
:class:`~quantflow.BlackScholes` calls the same code for one option.

Conventions
-----------
* ``T`` is the time to expiry as a year fraction. The library does not apply a
  day count; ``days / 365`` (ACT/365) is the usual choice.
* ``r`` and ``q`` are continuously compounded annual rates; ``sigma`` is the
  annualised volatility of log returns.
* ``vega`` and ``rho`` are per one percentage point (the derivative divided
  by 100); ``theta`` is the calendar-time derivative per year,
  ``theta = -dV/dT``.

References
----------
Black, F. and Scholes, M. (1973). The Pricing of Options and Corporate
Liabilities. Journal of Political Economy, 81(3), 637-654.

Merton, R. C. (1973). Theory of Rational Option Pricing. Bell Journal of
Economics and Management Science, 4(1), 141-183.
"""

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr

__all__ = ["black_scholes_price", "black_scholes_greeks"]

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _norm_pdf(x):
    return _INV_SQRT_2PI * np.exp(-0.5 * x * x)


def _prepare(S, K, T, r, sigma, q, option_type):
    """Validate the inputs, broadcast them, and return float arrays."""
    arrays = [np.asarray(v, dtype=float) for v in (S, K, T, r, sigma, q)]
    names = ("S", "K", "T", "r", "sigma", "q")
    for name, arr in zip(names, arrays, strict=True):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be a finite number")
    S, K, T, r, sigma, q = arrays
    if np.any(S <= 0):
        raise ValueError("Stock price (S) must be positive")
    if np.any(K <= 0):
        raise ValueError("Strike price (K) must be positive")
    if np.any(T <= 0):
        raise ValueError("Time to expiration (T) must be positive")
    if np.any(sigma <= 0):
        raise ValueError("Volatility (sigma) must be positive")

    kinds = np.asarray(option_type)
    if kinds.dtype.kind not in "UO":
        raise ValueError("option_type must be 'call' or 'put'")
    kinds = np.char.lower(kinds.astype(str))
    if not np.all(np.isin(kinds, ("call", "put"))):
        raise ValueError("option_type must be 'call' or 'put'")
    is_call = kinds == "call"

    return np.broadcast_arrays(S, K, T, r, sigma, q, is_call)


def _d1_d2(S, K, T, r, sigma, q):
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def _price_core(S, K, T, r, sigma, q, is_call):
    """Closed-form price on validated, broadcast arrays."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    fwd_s = S * np.exp(-q * T)
    pv_k = K * np.exp(-r * T)
    call = fwd_s * ndtr(d1) - pv_k * ndtr(d2)
    put = pv_k * ndtr(-d2) - fwd_s * ndtr(-d1)
    return np.where(is_call, call, put)


def _greeks_core(S, K, T, r, sigma, q, is_call):
    """Closed-form price and Greeks on validated, broadcast arrays."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    sqrt_t = np.sqrt(T)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    pdf_d1 = _norm_pdf(d1)
    n_d1, n_d2 = ndtr(d1), ndtr(d2)
    n_md1, n_md2 = ndtr(-d1), ndtr(-d2)

    price = np.where(
        is_call,
        S * disc_q * n_d1 - K * disc_r * n_d2,
        K * disc_r * n_md2 - S * disc_q * n_md1,
    )
    delta = np.where(is_call, disc_q * n_d1, -disc_q * n_md1)
    gamma = disc_q * pdf_d1 / (S * sigma * sqrt_t)
    decay = -S * disc_q * pdf_d1 * sigma / (2.0 * sqrt_t)
    theta = np.where(
        is_call,
        decay - r * K * disc_r * n_d2 + q * S * disc_q * n_d1,
        decay + r * K * disc_r * n_md2 - q * S * disc_q * n_md1,
    )
    vega = S * disc_q * pdf_d1 * sqrt_t / 100.0
    rho = np.where(is_call, K * T * disc_r * n_d2, -K * T * disc_r * n_md2) / 100.0
    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho,
    }


def _unwrap(x):
    """Return a Python float for 0-d results and the array otherwise."""
    x = np.asarray(x)
    return float(x) if x.ndim == 0 else x


def black_scholes_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    option_type: ArrayLike = "call",
    q: ArrayLike = 0.0,
) -> float | np.ndarray:
    r"""
    Black-Scholes-Merton price of European calls and puts.

    All arguments broadcast against each other, so any of them may be an
    array (for example a strip of strikes, or a matching array of
    ``'call'``/``'put'`` labels).

    Parameters
    ----------
    S : array_like
        Spot price of the underlying, > 0.
    K : array_like
        Strike, > 0.
    T : array_like
        Time to expiry in years, > 0.
    r : array_like
        Continuously compounded risk-free rate (annual).
    sigma : array_like
        Annualised volatility, > 0.
    option_type : str or array_like of str, default 'call'
        ``'call'`` or ``'put'`` (case-insensitive).
    q : array_like, default 0.0
        Continuous dividend yield (annual). For an FX option use the foreign
        rate (Garman-Kohlhagen); for an option on a future use ``q = r``
        (Black 1976).

    Returns
    -------
    float or numpy.ndarray
        A float when every input is a scalar, otherwise an array with the
        broadcast shape.

    Raises
    ------
    ValueError
        If any input is not finite, ``S``, ``K``, ``T`` or ``sigma`` is not
        positive, or ``option_type`` is not ``'call'``/``'put'``.

    Notes
    -----
    .. math::

        C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2), \qquad
        P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1),

        d_1 = \frac{\ln(S/K) + (r - q + \sigma^2/2) T}{\sigma \sqrt{T}},
        \qquad d_2 = d_1 - \sigma\sqrt{T}.

    Examples
    --------
    >>> import numpy as np
    >>> from quantflow import black_scholes_price
    >>> round(black_scholes_price(42, 40, 0.5, 0.10, 0.20, "call"), 4)
    4.7594
    >>> strikes = np.array([90.0, 100.0, 110.0])
    >>> black_scholes_price(100, strikes, 1.0, 0.05, 0.2).round(4)
    array([16.6994, 10.4506,  6.0401])
    >>> black_scholes_price(100, 100, 1.0, 0.05, 0.2, ["call", "put"]).round(4)
    array([10.4506,  5.5735])
    """
    return _unwrap(_price_core(*_prepare(S, K, T, r, sigma, q, option_type)))


def black_scholes_greeks(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    option_type: ArrayLike = "call",
    q: ArrayLike = 0.0,
) -> dict[str, float | np.ndarray]:
    r"""
    Black-Scholes-Merton price and Greeks of European calls and puts.

    Takes the same arguments as :func:`black_scholes_price` and broadcasts
    them the same way.

    Returns
    -------
    dict
        Keys ``price``, ``delta``, ``gamma``, ``theta``, ``vega``, ``rho``.
        ``theta`` is per year (``-dV/dT``); ``vega`` is per 1 percentage
        point of volatility and ``rho`` per 1 percentage point of ``r``.

    Notes
    -----
    With :math:`\varphi` the standard normal density:

    .. math::

        \Delta_C = e^{-qT} N(d_1), \qquad \Delta_P = -e^{-qT} N(-d_1),

        \Gamma = \frac{e^{-qT} \varphi(d_1)}{S \sigma \sqrt{T}}, \qquad
        \nu = \frac{S e^{-qT} \varphi(d_1) \sqrt{T}}{100},

        \Theta_C = -\frac{S e^{-qT} \varphi(d_1) \sigma}{2\sqrt{T}}
                   - r K e^{-rT} N(d_2) + q S e^{-qT} N(d_1),

        \Theta_P = -\frac{S e^{-qT} \varphi(d_1) \sigma}{2\sqrt{T}}
                   + r K e^{-rT} N(-d_2) - q S e^{-qT} N(-d_1),

        \rho_C = \frac{K T e^{-rT} N(d_2)}{100}, \qquad
        \rho_P = -\frac{K T e^{-rT} N(-d_2)}{100}.

    Hull, J. C., Options, Futures, and Other Derivatives, chapter on the
    Greek letters, gives the same expressions.
    """
    greeks = _greeks_core(*_prepare(S, K, T, r, sigma, q, option_type))
    return {name: _unwrap(value) for name, value in greeks.items()}


def _ncdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _price_scalar(S, K, T, r, sigma, q, is_call):
    """Scalar price with the math module, for the implied volatility solver.

    Same formula as :func:`_price_core`; about ten times cheaper per call for
    one option because it avoids NumPy array overhead.
    """
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    fwd_s = S * math.exp(-q * T)
    pv_k = K * math.exp(-r * T)
    if is_call:
        return fwd_s * _ncdf(d1) - pv_k * _ncdf(d2)
    return pv_k * _ncdf(-d2) - fwd_s * _ncdf(-d1)
