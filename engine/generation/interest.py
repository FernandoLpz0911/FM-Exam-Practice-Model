"""Generators for interest theory: TVM basics, nominal rates, force of interest, discount rate."""
from __future__ import annotations

import math

import numpy as np

from engine.generation.base import Problem, make_mc_choices, register


@register("interest_tvm")
def gen_interest_tvm(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    i = round(float(rng.uniform(*ranges["i_range"])), 4)
    n = int(rng.integers(*ranges["n_range"]))
    pv = round(float(rng.uniform(*ranges["pv_range"])), 2)
    v = 1 / (1 + i)

    if ask == "future_value":
        ans = round(pv * (1 + i) ** n, 2)
        stmt = (f"An account earns {i*100:.1f}% annual effective interest. "
                f"A deposit of ${pv:,.2f} is made today. Find the account value in {n} years.")
        wrongs = [round(pv * (1 + i * n), 2),
                  round(pv * (1 + i) ** (n - 1), 2),
                  round(pv * (1 + i / 2) ** (2 * n), 2)]

    elif ask == "present_value":
        fv = round(pv * (1 + i) ** n, 2)
        ans = round(fv * v ** n, 2)
        stmt = (f"An account earns {i*100:.1f}% annual effective interest. "
                f"Find the present value of ${fv:,.2f} due in {n} years.")
        wrongs = [round(fv / (1 + i * n), 2),
                  round(fv * v ** (n + 1), 2),
                  round(fv * v ** (n - 1), 2)]

    elif ask == "interest_rate_solve":
        fv = round(pv * (1 + i) ** n, 2)
        ans = round(i, 4)
        stmt = (f"An investment of ${pv:,.2f} grows to ${fv:,.2f} in {n} years "
                f"under annual effective interest. Find the annual effective interest rate.")
        wrongs = [round((fv - pv) / (pv * n), 4),
                  round((fv / pv) ** (1 / (n + 1)) - 1, 4),
                  round(i * 1.1, 4)]

    elif ask == "periods_solve":
        fv = round(pv * (1 + i) ** n, 2)
        ans = n
        stmt = (f"At {i*100:.1f}% annual effective interest, ${pv:,.2f} grows to ${fv:,.2f}. "
                f"How many years does this take?")
        wrongs = [n - 1, n + 1, round(math.log(fv / pv) / (i), 1)]

    else:
        raise ValueError(f"Unknown ask '{ask}' for interest_tvm")

    return Problem("interest_tvm", ask, stmt, ans,
                   make_mc_choices(ans, wrongs, rng),
                   params={"i": i, "n": n, "pv": pv}, seed=seed)


@register("interest_nominal")
def gen_interest_nominal(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    i_eff = round(float(rng.uniform(*ranges["i_range"])), 4)
    m = int(rng.choice(ranges["m_choices"]))

    if ask == "nominal_to_effective":
        i_nom_m = round(m * ((1 + i_eff) ** (1 / m) - 1), 6)
        ans = round(i_eff, 4)
        stmt = (f"The nominal interest rate compounded {m}timely is {i_nom_m*100:.4f}%. "
                f"Find the annual effective interest rate.")
        wrongs = [round(i_nom_m, 4),
                  round((1 + i_nom_m) ** m - 1, 4),
                  round(i_nom_m / m, 4)]

    elif ask == "effective_to_nominal":
        ans = round(m * ((1 + i_eff) ** (1 / m) - 1), 6)
        period_label = {2: "semi-annually", 4: "quarterly", 12: "monthly", 365: "daily"}.get(m, f"{m}timely")
        stmt = (f"The annual effective interest rate is {i_eff*100:.2f}%. "
                f"Find the nominal rate compounded {period_label}.")
        wrongs = [round(i_eff * m, 4),
                  round(i_eff, 4),
                  round(m * ((1 + i_eff) ** (1 / (m + 1)) - 1), 4)]

    elif ask == "equivalent_rate":
        m2 = int(rng.choice([x for x in ranges["m_choices"] if x != m]))
        i_nom_m = round(m * ((1 + i_eff) ** (1 / m) - 1), 6)
        ans = round(m2 * ((1 + i_eff) ** (1 / m2) - 1), 6)
        stmt = (f"The nominal rate compounded {m}timely is {i_nom_m*100:.4f}%. "
                f"Find the equivalent nominal rate compounded {m2}timely.")
        wrongs = [round(i_nom_m * m2 / m, 4),
                  round(ans * 1.05, 4),
                  round(i_eff, 4)]

    else:
        raise ValueError(f"Unknown ask '{ask}' for interest_nominal")

    return Problem("interest_nominal", ask, stmt, ans,
                   make_mc_choices(ans, wrongs, rng),
                   params={"i_eff": i_eff, "m": m}, seed=seed)


@register("interest_force")
def gen_interest_force(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    i = round(float(rng.uniform(*ranges["i_range"])), 4)
    delta = round(math.log(1 + i), 6)
    t = int(rng.integers(*ranges["t_range"]))

    if ask == "force_from_rate":
        ans = round(delta, 6)
        stmt = (f"The annual effective interest rate is {i*100:.2f}%. "
                f"Find the force of interest delta = ln(1+i).")
        wrongs = [round(i, 4),
                  round(i / (1 + i), 4),
                  round(math.exp(i) - 1, 6)]

    elif ask == "rate_from_force":
        ans = round(i, 4)
        stmt = (f"The force of interest is delta = {delta:.6f}. "
                f"Find the annual effective interest rate.")
        wrongs = [round(delta, 4),
                  round(math.exp(delta) ** 2 - 1, 4),
                  round(delta + delta**2 / 2, 4)]

    elif ask == "accumulation_continuous":
        pv = round(float(rng.uniform(1000, 10000)), 2)
        ans = round(pv * math.exp(delta * t), 2)
        stmt = (f"At force of interest delta = {delta:.6f}, find the accumulated value "
                f"of ${pv:,.2f} after {t} years under continuous compounding.")
        wrongs = [round(pv * (1 + delta * t), 2),
                  round(pv * math.exp(delta * (t - 1)), 2),
                  round(pv * math.exp(delta * (t + 1)), 2)]

    else:
        raise ValueError(f"Unknown ask '{ask}' for interest_force")

    return Problem("interest_force", ask, stmt, ans,
                   make_mc_choices(ans, wrongs, rng),
                   params={"i": i, "delta": delta, "t": t}, seed=seed)


@register("interest_discount")
def gen_interest_discount(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    i = round(float(rng.uniform(*ranges["i_range"])), 4)
    n = int(rng.integers(*ranges["n_range"]))
    d = round(i / (1 + i), 6)
    v = round(1 / (1 + i), 6)

    if ask == "discount_from_interest":
        ans = round(d, 6)
        stmt = (f"The annual effective interest rate is {i*100:.2f}%. "
                f"Find the annual effective discount rate d = i/(1+i).")
        wrongs = [round(i, 4), round(1 - v, 4) if abs(round(1 - v, 4) - ans) > 1e-6 else round(i * 0.9, 4),
                  round(i / (1 + 2 * i), 4)]

    elif ask == "interest_from_discount":
        ans = round(i, 4)
        stmt = (f"The annual effective discount rate is d = {d*100:.4f}%. "
                f"Find the annual effective interest rate i = d/(1-d).")
        wrongs = [round(d, 4),
                  round(d * (1 + d), 4),
                  round(i * 1.1, 4)]

    elif ask == "pv_using_discount":
        fv = round(float(rng.uniform(1000, 20000)), 2)
        ans = round(fv * v ** n, 2)
        stmt = (f"Using discount rate d = {d*100:.4f}%, find the present value "
                f"of ${fv:,.2f} due in {n} years. [Hint: v = 1-d]")
        wrongs = [round(fv * (1 - d * n), 2),
                  round(fv * (1 - d) ** (n + 1), 2),
                  round(fv * (1 - d) ** (n - 1), 2)]

    else:
        raise ValueError(f"Unknown ask '{ask}' for interest_discount")

    return Problem("interest_discount", ask, stmt, ans,
                   make_mc_choices(ans, wrongs, rng),
                   params={"i": i, "d": d, "v": v, "n": n}, seed=seed)
