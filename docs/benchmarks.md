# Benchmarks

Accuracy and speed of quantflow against QuantLib, vollib and a plain
NumPy/SciPy Black-Scholes, all on the same machine and the same inputs. To
reproduce:

```bash
pip install -e ".[validation]"
python benchmarks/run_benchmarks.py
```

## Method

- Timings are wall-clock (`time.perf_counter`) for a whole batch, repeated;
  each row gives the median over the repetitions and the interquartile range
  (IQR). Rates are computed from the batch time.
- Vectorised rows price one array of 1,000,000 options per call. Scalar rows
  loop over 5,000 options in Python, one call per option, which is the only
  interface QuantLib and vollib offer.
- Inputs are drawn uniformly: spot 80-120, strike 70-130, expiry 7-730 whole
  days (T = days / 365), rate 0-6%, dividend yield 0-2%, volatility 10-80%,
  half calls and half puts.
- Accuracy uses a fixed grid of 6,048 options (listed on the
  {doc}`validation` page).

The comparisons are not all like for like, and the notes column says so:
QuantLib is C++ called through SWIG and does more work per call (term
structures, observers, caching); vollib is pure Python; the NumPy baseline
skips input validation. Timings from a laptop vary with power settings and
background load, so compare rows within one run.

## Results

### Environment

| Item | Value |
|---|---|
| CPU | Intel(R) Core(TM) Ultra 9 185H |
| Logical cores | 22 |
| RAM | 31.5 GB |
| OS | Windows-11-10.0.26300-SP0 |
| Python | 3.13.5 |
| quantflow | 0.3.0 |
| numpy | 2.5.4 |
| scipy | 1.18.1 |
| QuantLib | 1.44 |
| vollib | 1.0.11 |
| lets_be_rational | 1.1.2 |

### Accuracy

Grid: 6048 European options (S in 80-120, K in 70-130, expiry 7-730 days, r 0-6%, q 0-2%, sigma 10-100%, calls and puts). T = days / 365 for every library. Relative errors use points where the reference is above 1e-6 in absolute value: below that, far out-of-the-money prices are the difference of two nearly equal small numbers in every library and agree to fewer digits.

| Quantity | Reference | Max abs error | Max rel error |
|---|---|---|---|
| price | QuantLib AnalyticEuropeanEngine | 1.5e-13 | 2.9e-09 |
| delta | QuantLib AnalyticEuropeanEngine | 8.8e-15 | 4.4e-11 |
| gamma | QuantLib AnalyticEuropeanEngine | 6.1e-16 | 4.1e-13 |
| theta | QuantLib AnalyticEuropeanEngine | 1.3e-12 | 2.8e-09 |
| vega | QuantLib AnalyticEuropeanEngine | 1.2e-15 | 3.3e-13 |
| rho | QuantLib AnalyticEuropeanEngine | 2.1e-15 | 1.9e-11 |
| price | vollib black_scholes_merton | 4.3e-14 | 3.6e-13 |

#### Implied volatility

5186 of the 6048 grid options have vega >= 1e-3 (per unit of volatility); the rest have too little time value for any solver to recover sigma. Each solver starts from the exact model price and its error is |sigma_solved - sigma_true|.

| Solver | Max abs vol error | Median abs vol error | Failures |
|---|---|---|---|
| quantflow (Brent, tol 1e-10) | 3.4e-11 | 7.2e-14 | 0 |
| vollib (Let's Be Rational) | 1.7e-11 | 3.3e-16 | 0 |
| QuantLib impliedVolatility (accuracy 1e-12) | 1.9e-11 | 1.6e-15 | 0 |

### Speed

Inputs drawn uniformly (S 80-120, K 70-130, T 7-730 days, r 0-6%, q 0-2%, sigma 10-80%, half calls). Vectorised rows price 1,000,000 options per call; scalar rows loop over 5,000 options in Python. 7 repetitions; median and IQR.

| Task | Implementation | Options per second (median) | IQR | Note |
|---|---|---|---|---|
| Price, vectorised | quantflow black_scholes_price | 5,691,834 | 5,637,157 - 5,721,629 | includes input validation |
| Price, vectorised | plain NumPy/SciPy (ndtr) | 9,059,514 | 8,946,849 - 9,140,442 | same formula, no validation |
| Price + 5 Greeks, vectorised | quantflow black_scholes_greeks | 2,813,832 | 2,799,138 - 2,930,661 |  |
| Price, scalar loop | quantflow BlackScholes(...).price() | 90,568 | 89,604 - 91,769 | object per option |
| Price + 5 Greeks, scalar loop | quantflow BlackScholes(...).greeks() | 57,468 | 56,993 - 58,254 |  |
| Price, scalar loop | vollib black_scholes_merton | 460,384 | 448,209 - 463,280 | pure Python |
| Price + 5 Greeks, scalar loop | vollib analytical greeks | 83,218 | 82,183 - 88,233 | pure Python, six calls per option |
| Price, scalar loop | QuantLib AnalyticEuropeanEngine | 272,509 | 248,268 - 288,080 | C++ via SWIG; objects built once, quote bumped |
| Price, scalar loop | QuantLib, objects built per option | 26,515 | 25,934 - 26,832 | C++ via SWIG; includes curve and process set-up |

Cost breakdown (median per call):

| Step | Time |
|---|---|
| validate and broadcast 1,000,000 inputs | 36.4 ms |
| price 1,000,000 validated inputs | 138.1 ms |
| plain NumPy price of 1,000,000 inputs | 112.3 ms |
| construct one BlackScholes | 8.0 us |
| price() on a built BlackScholes | 2.7 us |

Implied volatility: 2,000 solves per repetition from exact model prices (options with vega >= 1e-3 from the inputs above).

| Solver | Solves per second (median) | IQR | Note |
|---|---|---|---|
| quantflow implied_volatility | 36,366 | 35,916 - 36,391 | Brent's method, Python, tol 1e-10 |
| vollib implied_volatility | 56,190 | 55,205 - 56,821 | Let's Be Rational, pure Python |
| QuantLib impliedVolatility | 71,124 | 68,791 - 72,099 | C++ via SWIG, accuracy 1e-12 |

### Monte Carlo

At-the-money call, S=K=100, T=1, r=5%, sigma=20%; Black-Scholes price 10.450584. Seed 42 for every run. Error is MC price minus the closed form.

| Paths | Price | Std error | Error | Error / std error | Time (s) |
|---|---|---|---|---|---|
| 1,000 | 9.9120 | 0.44835 | -0.53860 | -1.20 | 0.000 |
| 10,000 | 10.3452 | 0.14766 | -0.10540 | -0.71 | 0.001 |
| 100,000 | 10.4205 | 0.04677 | -0.03004 | -0.64 | 0.002 |
| 1,000,000 | 10.4532 | 0.01473 | +0.00261 | +0.18 | 0.042 |
| 10,000,000 | 10.4482 | 0.00465 | -0.00239 | -0.51 | 0.437 |

Variance reduction at 1,000,000 paths. Efficiency is 1 / (std error^2 x time), relative to plain Monte Carlo (higher is better).

| Method | Std error | Time (s) | Relative efficiency |
|---|---|---|---|
| plain | 0.01473 | 0.035 | 1.0 |
| antithetic | 0.01042 | 0.028 | 2.5 |
| control variate | 0.00562 | 0.042 | 5.7 |
| antithetic + control variate | 0.00276 | 0.033 | 29.6 |

### American put

S=100, K=105, T=1 year, r=4%, sigma=25%, no dividends. Reference: QuantLib FdBlackScholesVanillaEngine on a 4000 x 4000 grid, 11.020230. Time per option, median of 5.

| Method | Steps | Price | Difference from reference | Time (ms) |
|---|---|---|---|---|
| quantflow BinomialTree (CRR) | 100 | 11.009607 | -0.010623 | 1.00 |
| QuantLib BinomialVanillaEngine crr | 100 | 11.009788 | -0.010442 | 0.09 |
| quantflow BinomialTree (CRR) | 500 | 11.022244 | +0.002014 | 7.73 |
| QuantLib BinomialVanillaEngine crr | 500 | 11.022280 | +0.002050 | 1.09 |
| quantflow BinomialTree (CRR) | 2000 | 11.021378 | +0.001148 | 63.04 |
| QuantLib BinomialVanillaEngine crr | 2000 | 11.021387 | +0.001157 | 14.91 |

## Reading the results

- The vectorised functions run at a few million options per second, slower
  than the plain NumPy baseline. The cost breakdown table shows why: input
  validation is part of the gap, and the rest is that quantflow evaluates
  the put formula directly (two more normal CDF evaluations) instead of
  deriving puts from parity, which keeps far out-of-the-money put prices
  accurate.
- One-option-at-a-time pricing through the `BlackScholes` class is slower
  than vollib's pure-Python function and than QuantLib with prebuilt
  objects. Most of the time goes into building the object and validating
  its inputs (see the cost breakdown). For many options use the vectorised
  functions.
- quantflow's implied volatility solver is slower than vollib and QuantLib
  and slightly less accurate than both, though all three are within
  $4 \times 10^{-11}$ of the true volatility on the test grid.
- quantflow's CRR tree gives nearly the same prices as QuantLib's CRR engine
  (the probability parameterisation differs slightly) and takes longer per
  option; the ratio shrinks as the number of steps grows (see the table).
