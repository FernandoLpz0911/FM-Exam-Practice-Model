"""Generators for annuity topics: immediate, due, perpetuity, deferred, varying."""
from __future__ import annotations

import math

import numpy as np

from engine.generation.base import Problem, make_mc_choices, register


def _a_n(i: float, n: int) -> float:
    """Present value annuity-immediate: a_n|i = (1 - v^n)/i."""
    v = 1 / (1 + i)
    return (1 - v ** n) / i


def _s_n(i: float, n: int) -> float:
    """Future value annuity-immediate: s_n|i = ((1+i)^n - 1)/i."""
    return ((1 + i) ** n - 1) / i


def _a_due(i: float, n: int) -> float:
    """Present value annuity-due: a_due_n|i = (1+i)*a_n|i."""
    return (1 + i) * _a_n(i, n)


def _s_due(i: float, n: int) -> float:
    """Future value annuity-due: s_due_n|i = (1+i)*s_n|i."""
    return (1 + i) * _s_n(i, n)


@register("annuity_immediate")
def gen_annuity_immediate(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    i = round(float(rng.uniform(*ranges["i_range"])), 4)
    n = int(rng.integers(*ranges["n_range"]))
    payment = round(float(rng.uniform(*ranges["payment_range"])), 2)
    an = _a_n(i, n)
    sn = _s_n(i, n)

    if ask == "pv_annuity_imm":
        ans = round(payment * an, 2)
        stmt = (f"An annuity-immediate pays ${payment:,.2f} at the end of each year "
                f"for {n} years at {i*100:.2f}% annual effective interest. Find the present value.")
        wrongs = [round(payment * _a_due(i, n), 2),
                  round(payment * an * (1 + i), 2),
                  round(payment * n / (1 + i) ** (n / 2), 2)]

    elif ask == "fv_annuity_imm":
        ans = round(payment * sn, 2)
        stmt = (f"An annuity-immediate pays ${payment:,.2f} at the end of each year "
                f"for {n} years at {i*100:.2f}%. Find the accumulated value at the end of year {n}.")
        wrongs = [round(payment * _s_due(i, n), 2),
                  round(payment * n * (1 + i) ** (n / 2), 2),
                  round(payment * sn / (1 + i), 2)]

    elif ask == "payment_from_pv":
        pv = round(float(rng.uniform(10000, 200000)), 2)
        ans = round(pv / an, 2)
        stmt = (f"A loan of ${pv:,.2f} is repaid with level annual payments at the end of each "
                f"year for {n} years at {i*100:.2f}%. Find the annual payment.")
        wrongs = [round(pv / (an * (1 + i)), 2),
                  round(pv / n, 2),
                  round(pv * i + pv / n, 2)]

    elif ask == "n_from_pv_imm":
        pv = round(float(rng.uniform(1000, 20000)), 2)
        ans = n
        # back-calculate consistent payment from n
        k = round(pv / an, 2)
        stmt = (f"A loan of ${pv:,.2f} is repaid by payments of ${k:,.2f} at the end of each year "
                f"at {i*100:.2f}%. How many payments are needed?")
        wrongs = [n - 1, n + 1, round(math.log(k / (k - pv * i)) / math.log(1 + i) + 0.5)]

    else:
        raise ValueError(f"Unknown ask '{ask}' for annuity_immediate")

    return Problem("annuity_immediate", ask, stmt, ans,
                   make_mc_choices(ans, wrongs, rng),
                   params={"i": i, "n": n, "payment": payment}, seed=seed)


@register("annuity_due")
def gen_annuity_due(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    i = round(float(rng.uniform(*ranges["i_range"])), 4)
    n = int(rng.integers(*ranges["n_range"]))
    payment = round(float(rng.uniform(*ranges["payment_range"])), 2)
    a_due = _a_due(i, n)
    s_due = _s_due(i, n)

    if ask == "pv_annuity_due":
        ans = round(payment * a_due, 2)
        stmt = (f"An annuity-due pays ${payment:,.2f} at the beginning of each year "
                f"for {n} years at {i*100:.2f}%. Find the present value.")
        wrongs = [round(payment * _a_n(i, n), 2),
                  round(payment * a_due / (1 + i), 2),
                  round(payment * a_due * (1 + i), 2)]

    elif ask == "fv_annuity_due":
        ans = round(payment * s_due, 2)
        stmt = (f"An annuity-due pays ${payment:,.2f} at the beginning of each year "
                f"for {n} years at {i*100:.2f}%. Find the accumulated value at end of year {n}.")
        wrongs = [round(payment * _s_n(i, n), 2),
                  round(payment * s_due / (1 + i), 2),
                  round(payment * s_due * (1 + i), 2)]

    elif ask == "payment_from_pv_due":
        pv = round(float(rng.uniform(10000, 200000)), 2)
        ans = round(pv / a_due, 2)
        stmt = (f"A lease requires level payments at the beginning of each year for {n} years. "
                f"At {i*100:.2f}%, the present value of all payments is ${pv:,.2f}. Find the annual payment.")
        wrongs = [round(pv / _a_n(i, n), 2),
                  round(pv / n, 2),
                  round(pv / a_due * (1 + i), 2)]

    else:
        raise ValueError(f"Unknown ask '{ask}' for annuity_due")

    return Problem("annuity_due", ask, stmt, ans,
                   make_mc_choices(ans, wrongs, rng),
                   params={"i": i, "n": n, "payment": payment}, seed=seed)


@register("perpetuity")
def gen_perpetuity(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    i = round(float(rng.uniform(*ranges["i_range"])), 4)
    payment = round(float(rng.uniform(*ranges["payment_range"])), 2)
    d = i / (1 + i)

    if ask == "pv_perp_imm":
        ans = round(payment / i, 2)
        stmt = (f"A perpetuity-immediate pays ${payment:,.2f} at the end of each year forever. "
                f"At {i*100:.2f}% annual effective interest, find the present value.")
        wrongs = [round(payment / d, 2),
                  round(payment / i + payment, 2),
                  round(payment / (i + 0.01), 2)]

    elif ask == "pv_perp_due":
        ans = round(payment / d, 2)
        stmt = (f"A perpetuity-due pays ${payment:,.2f} at the beginning of each year forever. "
                f"At {i*100:.2f}%, find the present value.")
        wrongs = [round(payment / i, 2),
                  round(payment / d / (1 + i), 2),
                  round(payment * (1 + i) / i, 2)]

    elif ask == "payment_from_perp_pv":
        pv = round(float(rng.uniform(10000, 200000)), 2)
        ans = round(pv * i, 2)
        stmt = (f"A perpetuity-immediate has present value ${pv:,.2f} at {i*100:.2f}%. "
                f"Find the annual payment.")
        wrongs = [round(pv * d, 2),
                  round(pv * i / (1 + i), 2),
                  round(pv / i, 2)]

    else:
        raise ValueError(f"Unknown ask '{ask}' for perpetuity")

    return Problem("perpetuity", ask, stmt, ans,
                   make_mc_choices(ans, wrongs, rng),
                   params={"i": i, "payment": payment, "d": d}, seed=seed)


@register("deferred_annuity")
def gen_deferred_annuity(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    i = round(float(rng.uniform(*ranges["i_range"])), 4)
    n = int(rng.integers(*ranges["n_range"]))
    m = int(rng.integers(*ranges["m_range"]))
    payment = round(float(rng.uniform(*ranges["payment_range"])), 2)
    v = 1 / (1 + i)
    an = _a_n(i, n)

    if ask == "pv_deferred_imm":
        ans = round(payment * (v ** m) * an, 2)
        stmt = (f"An annuity-immediate pays ${payment:,.2f} per year for {n} years, "
                f"with the first payment {m+1} years from now. "
                f"At {i*100:.2f}%, find the present value today.")
        wrongs = [round(payment * _a_n(i, n + m), 2),
                  round(payment * an, 2),
                  round(payment * (v ** (m + 1)) * an, 2)]

    elif ask == "pv_deferred_due":
        a_due = _a_due(i, n)
        ans = round(payment * (v ** m) * a_due, 2)
        stmt = (f"An annuity-due pays ${payment:,.2f} per year for {n} years, "
                f"deferred {m} years (first payment at time {m}). "
                f"At {i*100:.2f}%, find the present value today.")
        wrongs = [round(payment * (v ** m) * an, 2),
                  round(payment * a_due, 2),
                  round(payment * (v ** (m - 1)) * a_due, 2)]

    else:
        raise ValueError(f"Unknown ask '{ask}' for deferred_annuity")

    return Problem("deferred_annuity", ask, stmt, ans,
                   make_mc_choices(ans, wrongs, rng),
                   params={"i": i, "n": n, "m": m, "payment": payment}, seed=seed)


@register("annuity_varying")
def gen_annuity_varying(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    i = round(float(rng.uniform(*ranges["i_range"])), 4)
    n = int(rng.integers(*ranges["n_range"]))
    base = round(float(rng.uniform(*ranges["base_payment_range"])), 2)
    v = 1 / (1 + i)
    an = _a_n(i, n)
    a_due = _a_due(i, n)

    if ask == "pv_arithmetic_inc":
        # Payments: base, 2*base, ..., n*base at times 1,...,n
        # PV = base * (Ia)_n|i = base * (a_due - n*v^n) / i
        Ia = (a_due - n * v ** n) / i
        ans = round(base * Ia, 2)
        stmt = (f"An annuity pays ${base:,.2f} at end of year 1, ${2*base:,.2f} at end of year 2, "
                f"..., ${n*base:,.2f} at end of year {n}. At {i*100:.2f}%, find the present value.")
        wrongs = [round(base * an, 2),
                  round(base * n * v ** (n / 2), 2),
                  round(base * (a_due - n * v ** n) / (i * (1 + i)), 2)]
        params = {"i": i, "n": n, "base": base, "type": "inc"}

    elif ask == "pv_arithmetic_dec":
        # Payments: n*base, (n-1)*base, ..., base at times 1,...,n
        # PV = base * (Da)_n|i = base * (n - a_n|i) / i  ... wait
        # (Da)_n|i: payment at time t is (n+1-t)
        # PV = Σ_{t=1}^{n} (n+1-t)*v^t = (n+1)*a_n| - (Ia)_n|
        # where (Ia)_n| = (a_due - n*v^n)/i
        Ia = (a_due - n * v ** n) / i
        Da = (n + 1) * an - Ia
        ans = round(base * Da, 2)
        stmt = (f"An annuity pays ${n*base:,.2f} at end of year 1, decreasing by ${base:,.2f} each year, "
                f"with final payment of ${base:,.2f} at end of year {n}. "
                f"At {i*100:.2f}%, find the present value.")
        wrongs = [round(base * an, 2),
                  round(base * Ia, 2),
                  round(base * Da * (1 + i), 2)]
        params = {"i": i, "n": n, "base": base, "type": "dec"}

    elif ask == "pv_geometric":
        g = round(float(rng.uniform(0.01, 0.05)), 3)  # geometric growth rate
        if abs(i - g) < 0.001:
            g = g + 0.01
        # PV = base * (1 - ((1+g)/(1+i))^n) / (i - g)
        ratio = (1 + g) / (1 + i)
        pv_geo = base * (1 - ratio ** n) / (i - g)
        ans = round(pv_geo, 2)
        stmt = (f"An annuity pays ${base:,.2f} at end of year 1, increasing by {g*100:.1f}% each year "
                f"for {n} years total. At {i*100:.2f}%, find the present value.")
        wrongs = [round(base * an, 2),
                  round(base / (i - g), 2),
                  round(pv_geo * (1 + g), 2)]
        params = {"i": i, "n": n, "base": base, "g": g, "type": "geo"}

    else:
        raise ValueError(f"Unknown ask '{ask}' for annuity_varying")

    return Problem("annuity_varying", ask, stmt, ans,
                   make_mc_choices(ans, wrongs, rng),
                   params=params, seed=seed)
