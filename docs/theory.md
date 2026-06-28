# Mathematical Background

This page is the theory reference for `quantflow-finance`. Every formula below is the one the package actually implements; where a convention matters (per-1% Greeks, per-year theta, excess kurtosis, annualization factors), it is stated explicitly and cross-linked to the corresponding API object.

Throughout, $N(\cdot)$ denotes the standard normal cumulative distribution function (CDF) and

$$
\varphi(x) = \frac{1}{\sqrt{2\pi}}\,e^{-x^{2}/2}
$$

its probability density function (PDF).

## 1. The Black-Scholes-Merton model

Implemented by {class}`~quantflow.BlackScholes`.

### 1.1 The model assumption

The underlying asset price $S_t$ is assumed to follow a geometric Brownian motion under the physical measure,

$$
dS_t = \mu\,S_t\,dt + \sigma\,S_t\,dW_t,
$$

with constant drift $\mu$, constant volatility $\sigma$, and a standard Brownian motion $W_t$. The asset pays a **continuous dividend yield** $q$, the risk-free rate $r$ is constant, and markets are frictionless and arbitrage-free with continuous trading. Under these conditions a self-financing replicating portfolio exists, and by the Feynman-Kac / risk-neutral argument the price of a European derivative with payoff $V(S_T,T)$ is its discounted risk-neutral expectation,

$$
V(S_0,0) = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}\!\left[V(S_T,T)\right],
$$

where under the risk-neutral measure $\mathbb{Q}$ the drift is replaced by the cost of carry $r-q$:

$$
dS_t = (r-q)\,S_t\,dt + \sigma\,S_t\,dW_t^{\mathbb{Q}}.
$$

Equivalently, $V$ solves the Black-Scholes-Merton partial differential equation

$$
\frac{\partial V}{\partial t}
+ (r-q)\,S\frac{\partial V}{\partial S}
+ \tfrac{1}{2}\sigma^{2}S^{2}\frac{\partial^{2} V}{\partial S^{2}}
- rV = 0 .
$$

### 1.2 The terms $d_1$ and $d_2$

$$
d_1 = \frac{\ln(S/K) + \left(r - q + \tfrac{1}{2}\sigma^{2}\right)T}{\sigma\sqrt{T}},
\qquad
d_2 = d_1 - \sigma\sqrt{T}.
$$

Here $S$ is the spot price, $K$ the strike, $T$ the time to expiry in years, $r$ the risk-free rate, $q$ the dividend yield, and $\sigma$ the volatility. Setting $q=0$ recovers the classic 1973 Black-Scholes formulas.

### 1.3 Call and put prices

$$
C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2),
$$

$$
P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1).
$$

These satisfy **put-call parity**,

$$
C - P = S e^{-qT} - K e^{-rT},
$$

which the package reproduces exactly (see the verification example in {ref}`Section 2.4 <crr-convergence>`).

### 1.4 Closed-form Greeks

The Greeks are the partial derivatives of the option value with respect to its inputs. The table records the exact quantities returned by {meth}`~quantflow.BlackScholes.greeks`; note in particular the package's **scaling conventions**, which differ from the raw mathematical derivatives:

- **Vega** ({meth}`~quantflow.BlackScholes.vega`) and **rho** ({meth}`~quantflow.BlackScholes.rho`) are reported **per one percentage point** (i.e. the raw $\partial V/\partial\sigma$ and $\partial V/\partial r$ divided by 100), so the value approximates the price change for a $1\%$ move.
- **Theta** ({meth}`~quantflow.BlackScholes.theta`) is reported **per year**; divide by 365 for a daily figure.
- **Delta** and **vega** carry the dividend discount factor $e^{-qT}$.

**Delta** $\;\Delta = \dfrac{\partial V}{\partial S}$:

$$
\Delta_{\text{call}} = e^{-qT} N(d_1),
\qquad
\Delta_{\text{put}} = e^{-qT}\bigl(N(d_1) - 1\bigr).
$$

**Gamma** $\;\Gamma = \dfrac{\partial^{2} V}{\partial S^{2}}$ (identical for calls and puts):

$$
\Gamma = \frac{e^{-qT}\,\varphi(d_1)}{S\,\sigma\sqrt{T}}.
$$

**Vega** $\;\nu = \dfrac{\partial V}{\partial \sigma}$ (identical for calls and puts), per $1\%$:

$$
\nu = \frac{S e^{-qT}\,\varphi(d_1)\,\sqrt{T}}{100}.
$$

**Theta** $\;\Theta = \dfrac{\partial V}{\partial t} = -\dfrac{\partial V}{\partial T}$, per year:

$$
\Theta_{\text{call}}
= -\frac{S e^{-qT}\varphi(d_1)\sigma}{2\sqrt{T}}
- rK e^{-rT}N(d_2)
+ qS e^{-qT}N(d_1),
$$

$$
\Theta_{\text{put}}
= -\frac{S e^{-qT}\varphi(d_1)\sigma}{2\sqrt{T}}
+ rK e^{-rT}N(-d_2)
- qS e^{-qT}N(-d_1).
$$

**Rho** $\;\rho = \dfrac{\partial V}{\partial r}$, per $1\%$:

$$
\rho_{\text{call}} = \frac{K T e^{-rT} N(d_2)}{100},
\qquad
\rho_{\text{put}} = -\frac{K T e^{-rT} N(-d_2)}{100}.
$$

### 1.5 Implied volatility

Given an observed market price, {meth}`~quantflow.BlackScholes.implied_volatility` inverts the price-versus-volatility map. Because $\partial C/\partial\sigma = \nu \ge 0$, the price is a strictly increasing function of $\sigma$ on $(0,\infty)$, so the inverse is unique wherever it exists. The solver first checks the no-arbitrage bounds — for a call, $C \in \bigl[\max(Se^{-qT}-Ke^{-rT},\,0),\; Se^{-qT}\bigr]$ — then applies Brent's bracketing method, which is globally convergent and avoids the low-vega failure modes of an unbracketed Newton-Raphson iteration.

## 2. The Cox-Ross-Rubinstein binomial tree

Implemented by {class}`~quantflow.BinomialTree`.

### 2.1 Lattice construction

The interval $[0,T]$ is divided into $N$ steps of length $\Delta t = T/N$. Over each step the underlying moves up by a factor $u$ or down by a factor $d$, with the Cox-Ross-Rubinstein (CRR) choice

$$
u = e^{\sigma\sqrt{\Delta t}},
\qquad
d = \frac{1}{u} = e^{-\sigma\sqrt{\Delta t}}.
$$

This $u\,d = 1$ symmetry makes the lattice **recombining**: an up-then-down move returns to the starting price, so the tree has only $N+1$ terminal nodes rather than $2^N$.

### 2.2 Risk-neutral probability

The up-move probability is fixed by requiring the discounted underlying to be a martingale under $\mathbb{Q}$, i.e. $\mathbb{E}^{\mathbb{Q}}[S_{t+\Delta t}] = S_t\,e^{(r-q)\Delta t}$:

$$
p = \frac{e^{(r-q)\Delta t} - d}{u - d}.
$$

For sufficiently large $N$ one has $p \in [0,1]$; the package emits a warning if the inputs and step count push $p$ outside that range.

### 2.3 Backward induction and early exercise

At expiry, each terminal node carries the option's intrinsic payoff. With $j$ down-moves out of $N$ steps the terminal price is $S_T = S\,u^{\,N-j} d^{\,j}$ and the payoff is

$$
V_{N,j} = \max(S_T - K,\,0)\ \text{(call)}, \qquad \max(K - S_T,\,0)\ \text{(put)}.
$$

Values are then rolled back one layer at a time, discounting the risk-neutral expectation by $e^{-r\Delta t}$:

$$
V_{i,j} = e^{-r\Delta t}\bigl[\,p\,V_{i+1,j} + (1-p)\,V_{i+1,j+1}\,\bigr].
$$

For a **European** option this is the value at node $(i,j)$. For an **American** option the holder may exercise at any node, so the continuation value is compared with the immediate intrinsic value and the larger is kept:

$$
V_{i,j} = \max\!\Bigl(\underbrace{e^{-r\Delta t}\bigl[p\,V_{i+1,j} + (1-p)\,V_{i+1,j+1}\bigr]}_{\text{continuation}},\ \underbrace{g(S_{i,j})}_{\text{intrinsic}}\Bigr),
$$

where $g$ is the payoff function. The **early-exercise premium** returned by {meth}`~quantflow.BinomialTree.early_exercise_premium` is the American price minus the otherwise-identical European price; it is always non-negative and is strictly positive mainly for American puts (and for American calls on dividend-paying assets).

The lattice Greeks are read directly from the early layers of the tree as finite differences: delta from $\{V_u, V_d\}$, gamma from $\{V_{uu}, V_{ud}, V_{dd}\}$, and theta from $(V_{ud}-V_0)/(2\Delta t)$; vega and rho use a central-difference bump-and-reprice, scaled per $1\%$ to match the {class}`~quantflow.BlackScholes` convention.

(crr-convergence)=
### 2.4 Convergence to Black-Scholes

As $N \to \infty$, the binomial distribution of $\ln S_T$ converges to the normal distribution of the continuous model, so the European CRR price converges to the Black-Scholes-Merton price. The error decays as $O(1/N)$ but does **not** decay monotonically: it oscillates in sign as $N$ increases, because the strike's position relative to the discrete terminal nodes shifts with $N$. The following deterministic run shows both the $O(1/N)$ rate (the error magnitude roughly halves each time $N$ doubles) and the oscillation, and confirms put-call parity:

```python
import numpy as np
from quantflow import BlackScholes, BinomialTree

bs = BlackScholes(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call", q=0.0)
print(f"Black-Scholes call price: {bs.price():.6f}")
for n in (25, 50, 100, 200, 400):
    crr = BinomialTree(100, 100, 1.0, 0.05, 0.2, "call", "european", steps=n)
    print(f"  CRR N={n:>3}:  {crr.price():.6f}   error = {crr.price() - bs.price():+.6f}")

# Put-call parity with a dividend yield
call = BlackScholes(100, 100, 1.0, 0.05, 0.2, "call", q=0.03)
put  = BlackScholes(100, 100, 1.0, 0.05, 0.2, "put",  q=0.03)
print(f"C - P            = {call.price() - put.price():.6f}")
print(f"Se^-qT - Ke^-rT  = {100*np.exp(-0.03) - 100*np.exp(-0.05):.6f}")
```

Output:

```text
Black-Scholes call price: 10.450584
  CRR N= 25:  10.520966   error = +0.070382
  CRR N= 50:  10.410692   error = -0.039892
  CRR N=100:  10.430612   error = -0.019972
  CRR N=200:  10.440591   error = -0.009992
  CRR N=400:  10.445586   error = -0.004998
C - P            = 1.921611
Se^-qT - Ke^-rT  = 1.921611
```

## 3. Risk measures

Implemented by {class}`~quantflow.RiskMetrics`. Let $r_1,\dots,r_n$ be the per-period return observations with sample mean $\hat\mu$ and sample standard deviation $\hat\sigma$. Annualization uses `periods_per_year` $=m$ (default $252$). By convention, returns are signed, so a loss is a negative number and VaR/ES are reported as negative quantities.

### 3.1 Value at Risk

**Historical VaR** ({meth}`~quantflow.RiskMetrics.var_historical`) is the empirical quantile of the return distribution at confidence level $\alpha$ (default $\alpha = 0.05$, i.e. the $95\%$ VaR):

$$
\mathrm{VaR}^{\text{hist}}_\alpha = \widehat{Q}_{\alpha}(r),
$$

the $\alpha$-quantile of the sample.

**Parametric (Gaussian) VaR** ({meth}`~quantflow.RiskMetrics.var_parametric`) assumes normally distributed returns:

$$
\mathrm{VaR}^{\text{param}}_\alpha = \hat\mu + z_\alpha\,\hat\sigma,
\qquad z_\alpha = N^{-1}(\alpha),
$$

where $z_\alpha$ is the $\alpha$-quantile of the standard normal (e.g. $z_{0.05} = -1.645$), making the result negative for typical inputs.

### 3.2 Cornish-Fisher (modified) VaR

To account for non-normal skewness and kurtosis, {meth}`~quantflow.RiskMetrics.var_cornish_fisher` replaces the Gaussian quantile $z_\alpha$ with a Cornish-Fisher-adjusted quantile $z_{\text{cf}}$:

$$
z_{\text{cf}} = z
+ \frac{(z^{2}-1)\,S}{6}
+ \frac{(z^{3}-3z)\,K}{24}
- \frac{(2z^{3}-5z)\,S^{2}}{36},
$$

$$
\mathrm{VaR}^{\text{cf}}_\alpha = \hat\mu + z_{\text{cf}}\,\hat\sigma,
$$

where $z = z_\alpha = N^{-1}(\alpha)$, $S$ is the sample skewness, and $K$ is the **excess** kurtosis (so $K=0$ for a normal distribution; the package uses pandas' excess-kurtosis convention). When $S = K = 0$ the expansion collapses to the parametric VaR.

### 3.3 Expected Shortfall

Expected Shortfall (also called Conditional VaR) is the mean loss conditional on being in the worst $\alpha$ tail. **Historical ES** ({meth}`~quantflow.RiskMetrics.expected_shortfall`) averages the observations at or below the historical VaR:

$$
\mathrm{ES}^{\text{hist}}_\alpha = \mathbb{E}\!\left[\,r \;\middle|\; r \le \mathrm{VaR}^{\text{hist}}_\alpha\,\right].
$$

**Parametric (Gaussian) ES** ({meth}`~quantflow.RiskMetrics.expected_shortfall_parametric`) has the closed form

$$
\mathrm{ES}^{\text{param}}_\alpha = \hat\mu - \hat\sigma\,\frac{\varphi\!\bigl(z_\alpha\bigr)}{\alpha},
\qquad z_\alpha = N^{-1}(\alpha),
$$

where $\varphi$ is the standard normal PDF. Because the tail expectation is always at least as severe as the quantile, $\mathrm{ES}_\alpha \le \mathrm{VaR}_\alpha$ (both negative).

### 3.4 Performance ratios

**Sharpe ratio** ({meth}`~quantflow.RiskMetrics.sharpe_ratio`), annualized, with annual risk-free rate $r_f$:

$$
\text{Sharpe} = \frac{\hat\mu - r_f/m}{\hat\sigma}\,\sqrt{m}.
$$

**Sortino ratio** ({meth}`~quantflow.RiskMetrics.sortino_ratio`) replaces total volatility with **downside deviation** below a target $\tau$ (default $0$):

$$
\sigma_{\text{down}} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\bigl[\min(r_i - \tau,\,0)\bigr]^{2}},
\qquad
\text{Sortino} = \frac{\hat\mu - r_f/m}{\sigma_{\text{down}}}\,\sqrt{m}.
$$

The downside deviation is a population (not sample) root-mean-square; it returns $+\infty$ when there is no downside risk.

**Calmar ratio** ({meth}`~quantflow.RiskMetrics.calmar_ratio`) is the annualized compound growth rate divided by the magnitude of the maximum drawdown. With the compound annual growth rate

$$
\text{CAGR} = \left(\prod_{i=1}^{n}(1+r_i)\right)^{\!m/n} - 1,
$$

$$
\text{Calmar} = \frac{\text{CAGR} - r_f}{\lvert\text{MDD}\rvert}.
$$

**Omega ratio** ({meth}`~quantflow.RiskMetrics.omega_ratio`) at threshold $\tau$ is the ratio of probability-weighted gains to losses relative to $\tau$:

$$
\Omega(\tau) = \frac{\sum_i \max(r_i - \tau,\,0)}{\sum_i \max(\tau - r_i,\,0)}.
$$

Unlike the ratios above, Omega uses the entire return distribution and is not annualized.

### 3.5 Maximum drawdown

{meth}`~quantflow.RiskMetrics.max_drawdown` builds the cumulative wealth index $C_t = \prod_{i\le t}(1+r_i)$, tracks its running peak $P_t = \max_{s\le t} C_s$, and returns the most negative trough-to-peak decline:

$$
\text{MDD} = \min_t \frac{C_t - P_t}{P_t}.
$$

### 3.6 CAPM beta and alpha

Given benchmark returns $b_i$, {meth}`~quantflow.RiskMetrics.beta` is the slope of the portfolio's returns regressed on the benchmark, using the sample covariance and variance (with $\mathrm{ddof}=1$):

$$
\beta = \frac{\widehat{\mathrm{Cov}}(r,\,b)}{\widehat{\mathrm{Var}}(b)}.
$$

{meth}`~quantflow.RiskMetrics.alpha` is the annualized Jensen's alpha — the average return not explained by the benchmark's risk premium:

$$
\alpha = m \Bigl[\bigl(\bar r - r_f/m\bigr) - \beta\bigl(\bar b - r_f/m\bigr)\Bigr].
$$

{meth}`~quantflow.RiskMetrics.capm` additionally reports $R^{2} = \mathrm{corr}(r,b)^{2}$, the fraction of the portfolio's variance explained by the benchmark. The closely related **information ratio** ({meth}`~quantflow.RiskMetrics.information_ratio`) is the annualized mean active return divided by the tracking error (the sample standard deviation of $r_i - b_i$).

## 4. Mean-variance optimization

Implemented by {class}`~quantflow.Portfolio`. Let there be $A$ assets with expected per-period return vector $\boldsymbol{\mu}\in\mathbb{R}^{A}$, covariance matrix $\Sigma\in\mathbb{R}^{A\times A}$, and weight vector $\mathbf{w}$ with $\mathbf{1}^{\top}\mathbf{w}=1$. The portfolio's expected return and variance are

$$
\mu_p = \mathbf{w}^{\top}\boldsymbol{\mu},
\qquad
\sigma_p^{2} = \mathbf{w}^{\top}\Sigma\,\mathbf{w}.
$$

Annualized figures multiply $\mu_p$ by $m$ and $\sigma_p$ by $\sqrt{m}$. The covariance and correlation matrices are available via {meth}`~quantflow.Portfolio.covariance_matrix` and {meth}`~quantflow.Portfolio.correlation_matrix`.

### 4.1 The Markowitz problem

The mean-variance investor (Markowitz, 1952) seeks the smallest variance for a required expected return $\mu^{\star}$:

$$
\min_{\mathbf{w}}\ \tfrac{1}{2}\,\mathbf{w}^{\top}\Sigma\,\mathbf{w}
\quad\text{subject to}\quad
\mathbf{w}^{\top}\boldsymbol{\mu} = \mu^{\star},\quad
\mathbf{1}^{\top}\mathbf{w} = 1.
$$

The package solves this and its variants ({meth}`~quantflow.Portfolio.min_variance`, {meth}`~quantflow.Portfolio.max_sharpe`, {meth}`~quantflow.Portfolio.optimize` with `objective='target_return'`) numerically with sequential least-squares programming (SLSQP), so it can also honour a long-only constraint $\mathbf{w}\ge 0$ when `allow_short=False`.

### 4.2 Closed-form minimum-variance portfolio

Dropping the return constraint and keeping only the budget constraint $\mathbf{1}^{\top}\mathbf{w}=1$, the Lagrangian conditions give the **global minimum-variance** weights in closed form:

$$
\mathbf{w}_{\text{mv}} = \frac{\Sigma^{-1}\mathbf{1}}{\mathbf{1}^{\top}\Sigma^{-1}\mathbf{1}}.
$$

This holds when short positions are allowed (no inequality constraints bind). With `allow_short=True`, the package's numerical solution matches this formula; in a representative three-asset test the SLSQP weights agreed with $\mathbf{w}_{\text{mv}}$ to six decimal places.

### 4.3 Maximum-Sharpe (tangency) portfolio

The tangency portfolio maximizes the Sharpe ratio of excess returns over the annual risk-free rate $r_f$:

$$
\max_{\mathbf{w}}\ \frac{\mathbf{w}^{\top}\boldsymbol{\mu}_{\text{ann}} - r_f}{\sqrt{\mathbf{w}^{\top}\Sigma_{\text{ann}}\,\mathbf{w}}}
\quad\text{subject to}\quad
\mathbf{1}^{\top}\mathbf{w} = 1.
$$

When unconstrained, its solution is proportional to $\Sigma^{-1}(\boldsymbol{\mu} - r_f\mathbf{1})$, normalized so the weights sum to one. The tangency portfolio is the point where a ray from the risk-free asset is tangent to the efficient frontier, giving the highest reward-to-risk slope (the capital market line). It is returned by {meth}`~quantflow.Portfolio.max_sharpe`.

### 4.4 The efficient frontier

The **efficient frontier** is the set of portfolios that achieve the minimum variance for each attainable expected return; it is the upper boundary of the feasible region in mean-standard-deviation space. {meth}`~quantflow.Portfolio.efficient_frontier` traces it by sweeping target returns from the global minimum-variance return up to the best single-asset return and solving the constrained problem of {ref}`Section 4.1 <mean-variance-problem>` at each. Without short-selling or borrowing constraints the frontier is a hyperbola in $(\sigma_p,\mu_p)$ space; adding a risk-free asset turns the efficient set into the straight capital market line tangent to that hyperbola at the maximum-Sharpe portfolio.

(mean-variance-problem)=

## References

- Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*, 81(3), 637-654.
- Merton, R. C. (1973). Theory of Rational Option Pricing. *Bell Journal of Economics and Management Science*, 4(1), 141-183. (Continuous-dividend extension.)
- Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). Option Pricing: A Simplified Approach. *Journal of Financial Economics*, 7(3), 229-263.
- Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*, 7(1), 77-91.
- Cornish, E. A., & Fisher, R. A. (1938). Moments and Cumulants in the Specification of Distributions. *Revue de l'Institut International de Statistique*, 5(4), 307-320.
- Hull, J. C. *Options, Futures, and Other Derivatives* (10th ed.). Pearson. (Comprehensive reference for the option-pricing and Greeks formulas above.)

:::{seealso}
For worked, runnable examples of each formula, see the API reference for {class}`~quantflow.BlackScholes`, {class}`~quantflow.BinomialTree`, {class}`~quantflow.RiskMetrics`, and {class}`~quantflow.Portfolio`.
:::
