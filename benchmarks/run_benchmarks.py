"""
Accuracy and speed of quantflow against named baselines.

Run from the repository root:

    pip install -e ".[validation]"
    python benchmarks/run_benchmarks.py            # full run
    python benchmarks/run_benchmarks.py --quick    # smaller sizes, fewer repeats

The script prints Markdown tables: the machine and library versions, accuracy
against QuantLib and vollib, timings, Monte Carlo standard errors and an
American-option comparison. Timings are wall-clock (time.perf_counter); each
row is the median over the repetitions, with the interquartile range (IQR)
as the spread. Baselines that are not installed are reported as missing.
"""

import argparse
import contextlib
import ctypes
import importlib.metadata as md
import itertools
import os
import platform
import statistics
import subprocess
import sys
import time
import warnings

import numpy as np
from scipy.special import ndtr

import quantflow as qf
from quantflow import (
    BinomialTree,
    BlackScholes,
    MonteCarlo,
    black_scholes_greeks,
    black_scholes_price,
)

try:
    import QuantLib as ql
except ImportError:  # pragma: no cover - optional baseline
    ql = None

try:
    from vollib.black_scholes_merton import black_scholes_merton as vl_price
    from vollib.black_scholes_merton.greeks import analytical as vl_greeks
    from vollib.black_scholes_merton.implied_volatility import (
        implied_volatility as vl_iv,
    )
except ImportError:  # pragma: no cover - optional baseline
    vl_price = vl_greeks = vl_iv = None


# --------------------------------------------------------------- environment
def _cpu_name():
    system = platform.system()
    try:
        if system == "Windows":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if system == "Linux":
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if system == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _ram_gb():
    system = platform.system()
    try:
        if system == "Windows":

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.ullTotalPhys / 2**30
        if system == "Linux":
            with open("/proc/meminfo") as fh:
                return int(fh.readline().split()[1]) / 2**20
        if system == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(out) / 2**30
    except Exception:
        pass
    return float("nan")


def _version(dist):
    with contextlib.suppress(md.PackageNotFoundError):
        return md.version(dist)
    return "not installed"


def environment():
    rows = [
        ("CPU", _cpu_name()),
        ("Logical cores", str(os.cpu_count())),
        ("RAM", f"{_ram_gb():.1f} GB"),
        ("OS", platform.platform()),
        ("Python", sys.version.split()[0]),
        ("quantflow", qf.__version__),
        ("numpy", np.__version__),
        ("scipy", _version("scipy")),
        ("QuantLib", _version("QuantLib")),
        ("vollib", _version("vollib")),
        ("lets_be_rational", _version("lets_be_rational")),
    ]
    print("## Environment\n")
    print("| Item | Value |\n|---|---|")
    for name, value in rows:
        print(f"| {name} | {value} |")
    print()


# ------------------------------------------------------------------- helpers
def timed(fn, repeat):
    """Run fn() `repeat` times; return (median, q1, q3) of the run times."""
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    q = statistics.quantiles(times, n=4) if repeat >= 2 else [times[0]] * 3
    return statistics.median(times), q[0], q[2]


def fmt_rate(n_items, stats_):
    """Items per second from the median, with the IQR of the rate."""
    med, q1, q3 = stats_
    return f"{n_items / med:,.0f}", f"{n_items / q3:,.0f} - {n_items / q1:,.0f}"


def fmt_us(n_items, stats_):
    med, q1, q3 = stats_
    scale = 1e6 / n_items
    return f"{med * scale:.2f}", f"{q1 * scale:.2f} - {q3 * scale:.2f}"


def grid(quick):
    spots = [80.0, 100.0, 120.0]
    strikes = [70.0, 85.0, 100.0, 115.0, 130.0] if quick else list(range(70, 131, 10))
    days = [7, 30, 91, 182, 365, 730]
    rates = [0.0, 0.03, 0.06]
    divs = [0.0, 0.02]
    vols = [0.1, 0.25, 0.5, 1.0]
    kinds = ["call", "put"]
    return list(itertools.product(spots, strikes, days, rates, divs, vols, kinds))


def ql_env():
    today = ql.Date(15, 1, 2025)
    ql.Settings.instance().evaluationDate = today
    return today, ql.Actual365Fixed()


def ql_option(today, dc, S, K, days, r, q, sigma, kind, american=False):
    spot = ql.SimpleQuote(S)
    vol = ql.SimpleQuote(sigma)
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(spot),
        ql.YieldTermStructureHandle(ql.FlatForward(today, q, dc)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc)),
        ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(today, ql.NullCalendar(), ql.QuoteHandle(vol), dc)
        ),
    )
    maturity = today + days
    flag = ql.Option.Call if kind == "call" else ql.Option.Put
    exercise = (
        ql.AmericanExercise(today, maturity)
        if american
        else ql.EuropeanExercise(maturity)
    )
    option = ql.VanillaOption(ql.PlainVanillaPayoff(flag, K), exercise)
    return option, process, vol, dc.yearFraction(today, maturity)


def max_errors(ours, ref, floor=1e-6):
    ours, ref = np.asarray(ours), np.asarray(ref)
    abs_err = np.abs(ours - ref)
    mask = np.abs(ref) > floor
    rel = abs_err[mask] / np.abs(ref[mask])
    return abs_err.max(), (rel.max() if rel.size else float("nan"))


# ------------------------------------------------------------------ accuracy
def accuracy(quick):
    points = grid(quick)
    print("## Accuracy\n")
    print(
        f"Grid: {len(points)} European options (S in 80-120, K in 70-130, "
        "expiry 7-730 days, r 0-6%, q 0-2%, sigma 10-100%, calls and puts). "
        "T = days / 365 for every library. Relative errors use points where "
        "the reference is above 1e-6 in absolute value: below that, far "
        "out-of-the-money prices are the difference of two nearly equal small "
        "numbers in every library and agree to fewer digits.\n"
    )
    names = ["price", "delta", "gamma", "theta", "vega", "rho"]
    S, K, D, R, Q, V, KIND = (np.array(c) for c in zip(*points, strict=True))
    T = D / 365.0
    ours = black_scholes_greeks(S, K, T, R, V, KIND, Q)

    print("| Quantity | Reference | Max abs error | Max rel error |")
    print("|---|---|---|---|")
    if ql is not None:
        today, dc = ql_env()
        ref = {n: [] for n in names}
        for s, k, d, r, q, v, kind in points:
            opt, proc, _, _ = ql_option(today, dc, s, k, d, r, q, v, kind)
            opt.setPricingEngine(ql.AnalyticEuropeanEngine(proc))
            ref["price"].append(opt.NPV())
            ref["delta"].append(opt.delta())
            ref["gamma"].append(opt.gamma())
            ref["theta"].append(opt.theta())
            ref["vega"].append(opt.vega() / 100)
            ref["rho"].append(opt.rho() / 100)
        for n in names:
            a, rel = max_errors(ours[n], ref[n])
            print(f"| {n} | QuantLib AnalyticEuropeanEngine | {a:.1e} | {rel:.1e} |")
    else:
        print("| all | QuantLib | not installed | |")
    if vl_price is not None:
        ref = [
            vl_price(kind[0], s, k, d / 365.0, r, v, q)
            for s, k, d, r, q, v, kind in points
        ]
        a, rel = max_errors(ours["price"], ref)
        print(f"| price | vollib black_scholes_merton | {a:.1e} | {rel:.1e} |")
    print()

    # Implied volatility: solve from exact prices, compare with the true sigma.
    print("### Implied volatility\n")
    cases = []
    for s, k, d, r, q, v, kind in points:
        t = d / 365.0
        vega = black_scholes_greeks(s, k, t, r, v, kind, q)["vega"] * 100
        if vega >= 1e-3:
            cases.append((s, k, t, r, q, v, kind))
    print(
        f"{len(cases)} of the {len(points)} grid options have vega >= 1e-3 "
        "(per unit of volatility); the rest have too little time value for "
        "any solver to recover sigma. Each solver starts from the exact model "
        "price and its error is |sigma_solved - sigma_true|.\n"
    )
    print("| Solver | Max abs vol error | Median abs vol error | Failures |")
    print("|---|---|---|---|")

    def report(label, solve):
        errs, fails = [], 0
        for s, k, t, r, q, v, kind in cases:
            price = black_scholes_price(s, k, t, r, v, kind, q)
            try:
                errs.append(abs(solve(price, s, k, t, r, q, v, kind) - v))
            except Exception:
                fails += 1
        errs = np.array(errs)
        print(f"| {label} | {errs.max():.1e} | {np.median(errs):.1e} | {fails} |")

    report(
        "quantflow (Brent, tol 1e-10)",
        lambda p, s, k, t, r, q, v, kind: BlackScholes(
            s, k, t, r, 0.3, kind, q
        ).implied_volatility(p),
    )
    if vl_iv is not None:
        report(
            "vollib (Let's Be Rational)",
            lambda p, s, k, t, r, q, v, kind: vl_iv(p, s, k, t, r, q, kind[0]),
        )
    if ql is not None:
        today, dc = ql_env()

        def ql_solve(p, s, k, t, r, q, v, kind):
            opt, proc, _, _ = ql_option(
                today, dc, s, k, round(t * 365), r, q, 0.3, kind
            )
            opt.setPricingEngine(ql.AnalyticEuropeanEngine(proc))
            return opt.impliedVolatility(p, proc, 1e-12, 1000, 1e-4, 5.0)

        report("QuantLib impliedVolatility (accuracy 1e-12)", ql_solve)
    print()


# ------------------------------------------------------------------- timings
def bs_numpy(S, K, T, r, sigma, q, is_call):
    """Minimal NumPy/SciPy Black-Scholes-Merton with no input checks."""
    vst = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vst
    d2 = d1 - vst
    fs, pk = S * np.exp(-q * T), K * np.exp(-r * T)
    call = fs * ndtr(d1) - pk * ndtr(d2)
    return np.where(is_call, call, call - fs + pk)


def timings(quick):
    repeat = 3 if quick else 7
    n_vec = 100_000 if quick else 1_000_000
    n_scalar = 500 if quick else 5_000
    rng = np.random.default_rng(0)
    S = rng.uniform(80, 120, n_vec)
    K = rng.uniform(70, 130, n_vec)
    T = rng.integers(7, 731, n_vec) / 365  # whole days, as QuantLib needs
    r = rng.uniform(0, 0.06, n_vec)
    q = rng.uniform(0, 0.02, n_vec)
    v = rng.uniform(0.1, 0.8, n_vec)
    kind = np.where(rng.random(n_vec) < 0.5, "call", "put")
    is_call = kind == "call"
    scalar = [
        (float(S[i]), float(K[i]), float(T[i]), float(r[i]), float(q[i]), float(v[i]), str(kind[i]))
        for i in range(n_scalar)
    ]  # fmt: skip

    print("## Speed\n")
    print(
        f"Inputs drawn uniformly (S 80-120, K 70-130, T 7-730 days, r 0-6%, "
        f"q 0-2%, sigma 10-80%, half calls). Vectorised rows price {n_vec:,} "
        f"options per call; scalar rows loop over {n_scalar:,} options in "
        f"Python. {repeat} repetitions; median and IQR.\n"
    )
    print("| Task | Implementation | Options per second (median) | IQR | Note |")
    print("|---|---|---|---|---|")

    def row(task, impl, n, fn, note=""):
        rate, spread = fmt_rate(n, timed(fn, repeat))
        print(f"| {task} | {impl} | {rate} | {spread} | {note} |")

    row(
        "Price, vectorised",
        "quantflow black_scholes_price",
        n_vec,
        lambda: black_scholes_price(S, K, T, r, v, kind, q),
        "includes input validation",
    )
    row(
        "Price, vectorised",
        "plain NumPy/SciPy (ndtr)",
        n_vec,
        lambda: bs_numpy(S, K, T, r, v, q, is_call),
        "same formula, no validation",
    )
    row(
        "Price + 5 Greeks, vectorised",
        "quantflow black_scholes_greeks",
        n_vec,
        lambda: black_scholes_greeks(S, K, T, r, v, kind, q),
    )
    row(
        "Price, scalar loop",
        "quantflow BlackScholes(...).price()",
        n_scalar,
        lambda: [
            BlackScholes(s, k, t, rr, vv, kk, qq).price()
            for s, k, t, rr, qq, vv, kk in scalar
        ],
        "object per option",
    )
    row(
        "Price + 5 Greeks, scalar loop",
        "quantflow BlackScholes(...).greeks()",
        n_scalar,
        lambda: [
            BlackScholes(s, k, t, rr, vv, kk, qq).greeks()
            for s, k, t, rr, qq, vv, kk in scalar
        ],
    )
    if vl_price is not None:
        row(
            "Price, scalar loop",
            "vollib black_scholes_merton",
            n_scalar,
            lambda: [
                vl_price(kk[0], s, k, t, rr, vv, qq)
                for s, k, t, rr, qq, vv, kk in scalar
            ],
            "pure Python",
        )

        def vl_all():
            for s, k, t, rr, qq, vv, kk in scalar:
                f = kk[0]
                vl_price(f, s, k, t, rr, vv, qq)
                vl_greeks.delta(f, s, k, t, rr, vv, qq)
                vl_greeks.gamma(f, s, k, t, rr, vv, qq)
                vl_greeks.theta(f, s, k, t, rr, vv, qq)
                vl_greeks.vega(f, s, k, t, rr, vv, qq)
                vl_greeks.rho(f, s, k, t, rr, vv, qq)

        row(
            "Price + 5 Greeks, scalar loop",
            "vollib analytical greeks",
            n_scalar,
            vl_all,
            "pure Python, six calls per option",
        )
    if ql is not None:
        today, dc = ql_env()
        ql_objs = []
        for s, k, t, rr, qq, vv, kk in scalar:
            opt, proc, vol, _ = ql_option(
                today, dc, s, k, max(1, round(t * 365)), rr, qq, vv, kk
            )
            opt.setPricingEngine(ql.AnalyticEuropeanEngine(proc))
            ql_objs.append((opt, vol, vv))

        def ql_reprice():
            # Changing the vol quote invalidates the cached result, so each
            # NPV() call recomputes the price.
            for opt, vol, vv in ql_objs:
                vol.setValue(vv * 1.0001)
                opt.NPV()
                vol.setValue(vv)
                opt.NPV()

        row(
            "Price, scalar loop",
            "QuantLib AnalyticEuropeanEngine",
            2 * n_scalar,
            ql_reprice,
            "C++ via SWIG; objects built once, quote bumped",
        )

        def ql_build():
            for s, k, t, rr, qq, vv, kk in scalar:
                opt, proc, _, _ = ql_option(
                    today, dc, s, k, max(1, round(t * 365)), rr, qq, vv, kk
                )
                opt.setPricingEngine(ql.AnalyticEuropeanEngine(proc))
                opt.NPV()

        row(
            "Price, scalar loop",
            "QuantLib, objects built per option",
            n_scalar,
            ql_build,
            "C++ via SWIG; includes curve and process set-up",
        )
    print()

    # Implied volatility timing (scalar only in every library).
    n_iv = 300 if quick else 2_000
    iv_cases = []
    for s, k, t, rr, qq, vv, kk in scalar:
        if len(iv_cases) == n_iv:
            break
        if black_scholes_greeks(s, k, t, rr, vv, kk, qq)["vega"] * 100 >= 1e-3:
            iv_cases.append(
                (s, k, t, rr, qq, vv, kk, black_scholes_price(s, k, t, rr, vv, kk, qq))
            )
    print(
        f"Implied volatility: {len(iv_cases):,} solves per repetition from exact "
        f"model prices (options with vega >= 1e-3 from the inputs above).\n"
    )
    print("| Solver | Solves per second (median) | IQR | Note |")
    print("|---|---|---|---|")

    def iv_row(impl, fn, note=""):
        rate, spread = fmt_rate(len(iv_cases), timed(fn, repeat))
        print(f"| {impl} | {rate} | {spread} | {note} |")

    iv_row(
        "quantflow implied_volatility",
        lambda: [
            BlackScholes(s, k, t, rr, 0.3, kk, qq).implied_volatility(p)
            for s, k, t, rr, qq, vv, kk, p in iv_cases
        ],
        "Brent's method, Python, tol 1e-10",
    )
    if vl_iv is not None:
        iv_row(
            "vollib implied_volatility",
            lambda: [
                vl_iv(p, s, k, t, rr, qq, kk[0])
                for s, k, t, rr, qq, vv, kk, p in iv_cases
            ],
            "Let's Be Rational, pure Python",
        )
    if ql is not None:
        today, dc = ql_env()
        objs = []
        for s, k, t, rr, qq, _vv, kk, p in iv_cases:
            opt, proc, _, _ = ql_option(
                today, dc, s, k, max(1, round(t * 365)), rr, qq, 0.3, kk
            )
            opt.setPricingEngine(ql.AnalyticEuropeanEngine(proc))
            objs.append((opt, proc, p))

        iv_row(
            "QuantLib impliedVolatility",
            lambda: [
                opt.impliedVolatility(p, proc, 1e-12, 1000, 1e-4, 5.0)
                for opt, proc, p in objs
            ],
            "C++ via SWIG, accuracy 1e-12",
        )
    print()


# --------------------------------------------------------------- Monte Carlo
def monte_carlo(quick):
    args = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type="call")
    exact = BlackScholes(**args).price()
    sizes = [1_000, 10_000, 100_000, 1_000_000] + ([] if quick else [10_000_000])
    print("## Monte Carlo\n")
    print(
        f"At-the-money call, S=K=100, T=1, r=5%, sigma=20%; Black-Scholes price "
        f"{exact:.6f}. Seed 42 for every run. Error is MC price minus the "
        "closed form.\n"
    )
    print("| Paths | Price | Std error | Error | Error / std error | Time (s) |")
    print("|---|---|---|---|---|---|")
    for n in sizes:
        mc = MonteCarlo(**args, n_paths=n, seed=42)
        start = time.perf_counter()
        res = mc.simulate()
        elapsed = time.perf_counter() - start
        err = res.price - exact
        print(
            f"| {n:,} | {res.price:.4f} | {res.std_error:.5f} | {err:+.5f} | "
            f"{err / res.std_error:+.2f} | {elapsed:.3f} |"
        )
    print()
    n = 1_000_000
    print(
        f"Variance reduction at {n:,} paths. Efficiency is "
        "1 / (std error^2 x time), relative to plain Monte Carlo (higher is "
        "better).\n"
    )
    print("| Method | Std error | Time (s) | Relative efficiency |")
    print("|---|---|---|---|")
    base = None
    for label, anti, cv in [
        ("plain", False, False),
        ("antithetic", True, False),
        ("control variate", False, True),
        ("antithetic + control variate", True, True),
    ]:
        mc = MonteCarlo(**args, n_paths=n, antithetic=anti, control_variate=cv, seed=42)
        st = timed(mc.simulate, 5 if not quick else 3)
        se = mc.simulate().std_error
        eff = 1.0 / (se * se * st[0])
        base = base or eff
        print(f"| {label} | {se:.5f} | {st[0]:.3f} | {eff / base:.1f} |")
    print()


# ------------------------------------------------------------------ American
def american(quick):
    if ql is None:
        print("## American put\n\nQuantLib not installed.\n")
        return
    today, dc = ql_env()
    S, K, r, q, sigma, days = 100.0, 105.0, 0.04, 0.0, 0.25, 365
    opt, proc, _, T = ql_option(
        today, dc, S, K, days, r, q, sigma, "put", american=True
    )
    opt.setPricingEngine(ql.FdBlackScholesVanillaEngine(proc, 4000, 4000))
    reference = opt.NPV()
    repeat = 3 if quick else 5
    print("## American put\n")
    print(
        f"S=100, K=105, T=1 year, r=4%, sigma=25%, no dividends. Reference: "
        f"QuantLib FdBlackScholesVanillaEngine on a 4000 x 4000 grid, "
        f"{reference:.6f}. Time per option, median of {repeat}.\n"
    )
    print("| Method | Steps | Price | Difference from reference | Time (ms) |")
    print("|---|---|---|---|---|")
    for steps in (100, 500, 2000):
        tree = BinomialTree(S, K, T, r, sigma, "put", "american", steps, q)
        st = timed(tree.price, repeat)
        print(
            f"| quantflow BinomialTree (CRR) | {steps} | {tree.price():.6f} | "
            f"{tree.price() - reference:+.6f} | {st[0] * 1e3:.2f} |"
        )
        opt.setPricingEngine(ql.BinomialVanillaEngine(proc, "crr", steps))
        st = timed(lambda: (opt.recalculate(), opt.NPV()), repeat)
        print(
            f"| QuantLib BinomialVanillaEngine crr | {steps} | {opt.NPV():.6f} | "
            f"{opt.NPV() - reference:+.6f} | {st[0] * 1e3:.2f} |"
        )
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--quick", action="store_true", help="smaller run")
    args = parser.parse_args()
    warnings.simplefilter("ignore")
    print(f"# quantflow benchmarks{' (quick)' if args.quick else ''}\n")
    print(
        f"Command: `python benchmarks/run_benchmarks.py{' --quick' if args.quick else ''}`\n"
    )
    environment()
    accuracy(args.quick)
    timings(args.quick)
    monte_carlo(args.quick)
    american(args.quick)


if __name__ == "__main__":
    main()
