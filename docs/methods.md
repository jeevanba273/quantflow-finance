# Methods

The models behind each part of the library, the conventions they use, and
where they should not be trusted. $N$ is the standard normal distribution
function and $\varphi$ its density.

## Conventions used everywhere

- Rates $r$ and $q$ are continuously compounded per year. There are no
  curves: each is a single number.
- $T$ is a year fraction supplied by the user. No day count, calendar or
  settlement lag is applied.
- Volatility $\sigma$ is the annualised standard deviation of log returns.
- Option Greeks: $\Delta$ and $\Gamma$ per unit of the underlying, $\Theta$
  per year ($-\partial V/\partial T$), vega and rho per percentage point.
- Return statistics: simple returns per period, annualised with
  $m$ = `periods_per_year` (default 252); sample standard deviations with
  `ddof=1`; VaR and expected shortfall as signed returns at tail
  probability $\alpha$.

## Black-Scholes-Merton

Used by {class}`~quantflow.BlackScholes`, {func}`~quantflow.black_scholes_price`
and {func}`~quantflow.black_scholes_greeks`.

**Model.** Under the risk-neutral measure the underlying follows
$dS_t = (r - q) S_t\,dt + \sigma S_t\,dW_t$ with constant $r$, $q$, $\sigma$.
A European payoff is worth its discounted expectation, which gives

$$
C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2), \qquad
P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1),
$$

$$
d_1 = \frac{\ln(S/K) + (r - q + \tfrac12\sigma^2) T}{\sigma\sqrt T}, \qquad
d_2 = d_1 - \sigma\sqrt T .
$$

The Greeks are the analytic derivatives of these expressions; their formulas
are in the {doc}`API reference <api/options>`. $N$ is evaluated with
`scipy.special.ndtr` (vectorised) or `math.erfc` (the scalar path used by
the implied volatility solver); the two agree to about $10^{-12}$ relative.

**Variants through $q$.** Dividend-paying stock or index (Merton 1973);
FX option with $q$ the foreign rate (Garman and Kohlhagen 1983); option on a
future with $S$ the futures price and $q = r$ (Black 1976).

**Assumptions and limits.** Constant volatility (no smile or term
structure), constant rates and yield, lognormal prices without jumps,
continuous trading without costs, European exercise only. Discrete cash
dividends are not modelled. For far out-of-the-money options with prices
below about $10^{-6}$ of the spot, the call and put formulas subtract two
nearly equal numbers and only a few significant digits are reliable (this is
true of the other libraries compared on the {doc}`validation` page as well).

## Implied volatility

{meth}`~quantflow.BlackScholes.implied_volatility` checks the no-arbitrage
bounds and then solves $V(\sigma) = V_{\text{market}}$ with Brent's method
(Brent 1973) on $[10^{-6}, 10]$, widened up to 1000 if needed, with an
absolute tolerance of $10^{-10}$ on $\sigma$. The solution is unique because
vega is positive. Prices without time value above the lower bound raise an
error, since no volatility is determined. The achievable accuracy is about
the price rounding error divided by vega, so it degrades for very short
expiries and far from the money. Jaeckel's "Let's Be Rational" (2015), used
by vollib, is designed to converge in two iterations; on the benchmark grid
its median volatility error is smaller than quantflow's (see
{doc}`validation`).

## Cox-Ross-Rubinstein tree

Used by {class}`~quantflow.BinomialTree` (Cox, Ross and Rubinstein 1979).
With $\Delta t = T/N$, $u = e^{\sigma\sqrt{\Delta t}}$, $d = 1/u$ and
$p = (e^{(r-q)\Delta t} - d)/(u - d)$, node values are rolled back with
$V = e^{-r\Delta t}\,[p V_u + (1-p) V_d]$. American nodes take
$\max(V, \text{intrinsic})$. Delta, gamma and theta use Hull's lattice
formulas on the first two layers; vega and rho are central differences with
bumps of $10^{-4}$.

**Limits.** European prices converge to Black-Scholes-Merton with an error of
order $1/N$ that oscillates in sign; no smoothing or Richardson
extrapolation is applied. Cost grows as $N^2$. $p$ leaves $[0,1]$, with a
warning, when roughly $N < T (r-q)^2 / \sigma^2$. Only a continuous dividend
yield is supported.

## Monte Carlo

Used by {class}`~quantflow.MonteCarlo`. Terminal prices are sampled exactly,
$S_T = S\exp((r - q - \tfrac12\sigma^2)T + \sigma\sqrt T Z)$, and the price is
the mean discounted payoff. The standard error is the sample standard
deviation of the independent estimates over $\sqrt n$; the confidence
interval uses the normal approximation. Antithetic variates average the
payoffs of $Z$ and $-Z$; the control variate is $e^{-rT} S_T$ with known mean
$S e^{-qT}$ and an estimated regression coefficient (Glasserman 2003,
sections 4.1 and 4.2). Random numbers come from NumPy's PCG64 generator.

**Limits.** European payoffs on one GBM asset only; no paths, so no
path-dependent or early-exercise payoffs; no Greeks. The estimated
control-variate coefficient introduces a bias of order $1/n$.

## Risk metrics

Used by {class}`~quantflow.RiskMetrics`, on returns $r_1, \dots, r_n$ with
sample mean $\hat\mu$ and standard deviation $\hat\sigma$.

| Measure | Definition |
| --- | --- |
| Historical VaR | empirical $\alpha$-quantile, linear interpolation (`numpy.percentile`) |
| Normal VaR | $\hat\mu + z_\alpha \hat\sigma$, $z_\alpha = N^{-1}(\alpha)$ |
| Cornish-Fisher VaR | $\hat\mu + z_{cf}\hat\sigma$, $z_{cf} = z + \frac{(z^2-1)S}{6} + \frac{(z^3-3z)K}{24} - \frac{(2z^3-5z)S^2}{36}$ |
| Historical ES | mean of returns $\le$ historical VaR |
| Normal ES | $\hat\mu - \hat\sigma\varphi(z_\alpha)/\alpha$ |
| Volatility | $\hat\sigma\sqrt m$ |
| Sharpe | $(\hat\mu - r_f/m)/\hat\sigma \cdot \sqrt m$ (Sharpe 1994) |
| Sortino | $(\hat\mu - r_f/m)/\sigma_d \cdot \sqrt m$, $\sigma_d^2 = \frac1n\sum\min(r_i-\tau,0)^2$ (Sortino and Price 1994) |
| Maximum drawdown | $\min_t W_t/\max_{s\le t}W_s - 1$, $W_0 = 1$, $W_t = \prod_{i\le t}(1+r_i)$ |
| Calmar | $(\text{CAGR} - r_f)/|\text{MDD}|$, $\text{CAGR} = \prod(1+r_i)^{m/n} - 1$ |
| Omega | $\sum\max(r_i-\tau,0)/\sum\max(\tau-r_i,0)$ (Keating and Shadwick 2002) |
| Beta | $\widehat{\text{Cov}}(r,b)/\widehat{\text{Var}}(b)$ |
| Jensen's alpha | $m[(\bar r - r_f/m) - \beta(\bar b - r_f/m)]$ (Jensen 1968) |
| Information ratio | $\bar a/s_a\cdot\sqrt m$ with $a_i = r_i - b_i$ |

$S$ and $K$ in the Cornish-Fisher row are the sample skewness and excess
kurtosis (Cornish and Fisher 1938).

**Limits.** Every measure treats the returns as one stationary sample: no
volatility clustering, no overlapping-period or autocorrelation
corrections (Lo 2002 shows how much these matter for Sharpe ratios), no
multi-day VaR scaling and no VaR backtests. Normal VaR understates fat
tails; Cornish-Fisher can overshoot them at small $\alpha$. Sample ratios
have large standard errors on a few years of data.

## Mean-variance optimisation

Used by {class}`~quantflow.Portfolio` (Markowitz 1952). With sample mean
vector $\mu$ and covariance $\Sigma$ (both annualised) and weights summing to
one:

- minimum variance: $\min w^\top\Sigma w$;
- target return: $\min w^\top\Sigma w$ subject to $w^\top\mu = \mu^\star$;
- maximum Sharpe: $\max (w^\top\mu - r_f)/\sqrt{w^\top\Sigma w}$.

Long-only versions ($0 \le w_i \le 1$) are solved with SciPy's SLSQP from
equal weights, with the variance objective scaled to order one. With short
sales, the tangency portfolio is $\Sigma^{-1}(\mu - r_f\mathbf 1)$ normalised,
which exists only when the minimum-variance return exceeds $r_f$ (Merton
1972 gives the closed forms used in the tests).

**Limits.** Sample estimates are used as if they were the true parameters;
optimised weights are very sensitive to errors in $\mu$ (Michaud 1989). No
shrinkage, factor models, transaction costs, turnover or position limits
other than the long-only bounds.

## References

- Black, F. (1976). The Pricing of Commodity Contracts. *Journal of Financial Economics*, 3(1-2), 167-179.
- Black, F. and Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*, 81(3), 637-654.
- Brent, R. P. (1973). *Algorithms for Minimization Without Derivatives*. Prentice-Hall.
- Cornish, E. A. and Fisher, R. A. (1938). Moments and Cumulants in the Specification of Distributions. *Revue de l'Institut International de Statistique*, 5(4), 307-320.
- Cox, J. C., Ross, S. A. and Rubinstein, M. (1979). Option Pricing: A Simplified Approach. *Journal of Financial Economics*, 7(3), 229-263.
- Garman, M. B. and Kohlhagen, S. W. (1983). Foreign Currency Option Values. *Journal of International Money and Finance*, 2(3), 231-237.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.
- Haug, E. G. (2007). *The Complete Guide to Option Pricing Formulas* (2nd ed.). McGraw-Hill.
- Hull, J. C. *Options, Futures, and Other Derivatives* (10th ed.). Pearson.
- Jaeckel, P. (2015). Let's Be Rational. *Wilmott*, 2015(75), 40-53.
- Jensen, M. C. (1968). The Performance of Mutual Funds in the Period 1945-1964. *Journal of Finance*, 23(2), 389-416.
- Keating, C. and Shadwick, W. F. (2002). A Universal Performance Measure. *Journal of Performance Measurement*, 6(3), 59-84.
- Lo, A. W. (2002). The Statistics of Sharpe Ratios. *Financial Analysts Journal*, 58(4), 36-52.
- Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*, 7(1), 77-91.
- Merton, R. C. (1972). An Analytic Derivation of the Efficient Portfolio Frontier. *Journal of Financial and Quantitative Analysis*, 7(4), 1851-1872.
- Merton, R. C. (1973). Theory of Rational Option Pricing. *Bell Journal of Economics and Management Science*, 4(1), 141-183.
- Michaud, R. O. (1989). The Markowitz Optimization Enigma: Is 'Optimized' Optimal? *Financial Analysts Journal*, 45(1), 31-42.
- Sharpe, W. F. (1994). The Sharpe Ratio. *Journal of Portfolio Management*, 21(1), 49-58.
- Sortino, F. A. and Price, L. N. (1994). Performance Measurement in a Downside Risk Framework. *Journal of Investing*, 3(3), 59-64.
