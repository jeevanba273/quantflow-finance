# Benchmarks

`run_benchmarks.py` measures the accuracy of QuantFlow Finance's prices, Greeks and
implied volatilities against QuantLib and vollib, its speed against those
libraries and a plain NumPy/SciPy Black-Scholes, Monte Carlo standard errors
against the number of paths, and American-option trees against a QuantLib
finite-difference reference.

```sh
pip install -e ".[validation]"
python benchmarks/run_benchmarks.py           # full run, a few minutes
python benchmarks/run_benchmarks.py --quick   # smaller sizes, fewer repeats
```

The script prints Markdown, starting with the CPU, RAM, OS and library
versions. Results from one machine are on the documentation's Benchmarks
page. Timings depend on the machine, the Python build and background load,
so compare rows from the same run rather than numbers across machines.
