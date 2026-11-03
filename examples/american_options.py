"""American and European options on a Cox-Ross-Rubinstein tree.

Run with:  python examples/american_options.py
"""

from quantflow import BinomialTree, BlackScholes

# Hull's American put: S=K=50, r=10%, sigma=40%, five months.
for steps in (5, 30, 100, 500):
    tree = BinomialTree(50, 50, 5 / 12, 0.10, 0.40, "put", "american", steps=steps)
    print(f"steps {steps:>4}  American put {tree.price():.4f}")

european = BlackScholes(50, 50, 5 / 12, 0.10, 0.40, "put").price()
print(f"Black-Scholes European put {european:.4f}")

amer = BinomialTree(50, 50, 5 / 12, 0.10, 0.40, "put", "american", steps=500)
print(f"early-exercise premium {amer.early_exercise_premium():.4f}")

# Convergence of the European tree to the closed form.
exact = BlackScholes(100, 105, 0.5, 0.05, 0.25, "call").price()
for steps in (50, 100, 200, 400, 800):
    tree = BinomialTree(100, 105, 0.5, 0.05, 0.25, "call", "european", steps)
    print(f"steps {steps:>4}  error {tree.price() - exact:+.6f}")
