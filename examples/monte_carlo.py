"""Monte Carlo price of a European call: standard error and variance reduction.

Run with:  python examples/monte_carlo.py
"""

from quantflow import BlackScholes, MonteCarlo

args = dict(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")
exact = BlackScholes(**args).price()
print(f"Black-Scholes {exact:.4f}")

print("paths      price   std err   in 95% CI")
for n in (1_000, 10_000, 100_000, 1_000_000):
    res = MonteCarlo(**args, n_paths=n, seed=42).simulate()
    inside = res.ci_low <= exact <= res.ci_high
    print(f"{n:>9,}  {res.price:7.4f}  {res.std_error:8.5f}   {inside}")

print("method (100,000 paths)       std err")
for label, anti, cv in [
    ("plain", False, False),
    ("antithetic", True, False),
    ("control variate", False, True),
    ("antithetic + control", True, True),
]:
    res = MonteCarlo(
        **args, n_paths=100_000, antithetic=anti, control_variate=cv, seed=42
    ).simulate()
    print(f"{label:<26} {res.std_error:8.5f}")
