"""Generators for Macaulay/modified duration, convexity, and Redington immunization."""
from __future__ import annotations

import numpy as np

from engine.generation.base import Problem, make_mc_choices, register


def _a_n(i: float, n: int) -> float:
    v = 1 / (1 + i)
    return (1 - v ** n) / i


def _bond_cashflows(face: float, coupon_rate: float, n: int,
                    redemption: float | None = None) -> list[tuple[int, float]]:
    """[(t, cashflow)] — coupon at t=1..n, redemption at t=n."""
    C = redemption if redemption is not None else face
    Fr = face * coupon_rate
    cashflows = [(t, Fr) for t in range(1, n + 1)]
    cashflows[-1] = (n, Fr + C)   # last period: coupon + redemption
    return cashflows


def _macaulay_duration(cashflows: list[tuple[int, float]], i: float) -> float:
    """D_mac = Σ t·PV(CF_t) / P."""
    v = 1 / (1 + i)
    pv_total = sum(cf * v ** t for t, cf in cashflows)
    weighted = sum(t * cf * v ** t for t, cf in cashflows)
    return weighted / pv_total


@register("macaulay_duration")
def gen_macaulay_duration(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    face = float(rng.choice([1000, 2000, 5000]))
    r = round(float(rng.uniform(*ranges["coupon_rate_range"])), 4)
    i = round(float(rng.uniform(*ranges["yield_range"])), 4)
    n = int(rng.integers(*ranges["n_range"]))
    cashflows = _bond_cashflows(face, r, n)
    d_mac = round(_macaulay_duration(cashflows, i), 4)

    if ask == "macaulay_duration_bond":
        ans = d_mac
        stmt = (f"A ${face:,.0f} par bond has coupon rate {r*100:.2f}%, yields {i*100:.2f}%, "
                f"and matures in {n} years. Compute the Macaulay duration "
                f"(D = Σ t·PV(CF_t) / P).")
        wrongs = [round(d_mac * (1 + i), 4),
                  round(d_mac / (1 + i), 4),
                  round(n * 0.9, 4)]

    elif ask == "macaulay_perpetuity":
        # D_mac of perpetuity-immediate = (1+i)/i
        d_perp = round((1 + i) / i, 4)
        ans = d_perp
        stmt = (f"Find the Macaulay duration of a perpetuity-immediate at yield {i*100:.2f}%. "
                f"Use D = (1+i)/i.")
        wrongs = [round(1 / i, 4),
                  round(1 / (1 + i), 4),
                  round((1 + i) / i * 1.05, 4)]

    else:
        raise ValueError(f"Unknown ask '{ask}' for macaulay_duration")

    return Problem("macaulay_duration", ask, stmt, ans,
                   make_mc_choices(ans, wrongs, rng),
                   params={"face": face, "r": r, "i": i, "n": n, "d_mac": d_mac}, seed=seed)


@register("modified_duration")
def gen_modified_duration(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    face = float(rng.choice([1000, 2000, 5000]))
    r = round(float(rng.uniform(*ranges["coupon_rate_range"])), 4)
    i = round(float(rng.uniform(*ranges["yield_range"])), 4)
    n = int(rng.integers(*ranges["n_range"]))
    cashflows = _bond_cashflows(face, r, n)
    d_mac = _macaulay_duration(cashflows, i)
    d_mod = round(d_mac / (1 + i), 4)

    if ask == "modified_duration_from_mac":
        ans = d_mod
        stmt = (f"A bond has Macaulay duration {round(d_mac, 4)} at yield {i*100:.2f}%. "
                f"Find the modified duration: D_mod = D_mac / (1+i).")
        wrongs = [round(d_mac * (1 + i), 4),
                  round(d_mac, 4),
                  round(d_mod * 1.05, 4)]

    elif ask == "price_change_approx":
        delta_i = round(float(rng.choice([-0.01, 0.01, -0.005, 0.005])), 3)
        v = 1 / (1 + i)
        price = sum(cf * v ** t for t, cf in cashflows)
        delta_p = round(-d_mod * price * delta_i, 2)
        ans = delta_p
        stmt = (f"A ${face:,.0f} par bond with coupon rate {r*100:.2f}% has "
                f"modified duration {d_mod} at yield {i*100:.2f}%. "
                f"Estimate the price change if yield {'rises' if delta_i > 0 else 'falls'} "
                f"by {abs(delta_i)*100:.2f}% using ΔP ≈ -D_mod · P · Δi.")
        wrongs = [round(-d_mod * price * delta_i * (1 + i), 2),
                  round(d_mod * price * delta_i, 2),
                  round(-d_mod * price * delta_i * 0.9, 2)]

    else:
        raise ValueError(f"Unknown ask '{ask}' for modified_duration")

    return Problem("modified_duration", ask, stmt, ans,
                   make_mc_choices(ans, wrongs, rng),
                   params={"face": face, "r": r, "i": i, "n": n,
                           "d_mac": round(d_mac, 4), "d_mod": d_mod}, seed=seed)


@register("convexity")
def gen_convexity(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    face = float(rng.choice([1000, 2000]))
    r = round(float(rng.uniform(*ranges["coupon_rate_range"])), 4)
    i = round(float(rng.uniform(*ranges["yield_range"])), 4)
    n = int(rng.integers(*ranges["n_range"]))
    cashflows = _bond_cashflows(face, r, n)
    v = 1 / (1 + i)
    price = sum(cf * v ** t for t, cf in cashflows)
    # Convexity = Σ t(t+1)·PV(CF_t) / (P·(1+i)^2)
    convex = sum(t * (t + 1) * cf * v ** t for t, cf in cashflows) / (price * (1 + i) ** 2)
    convex = round(convex, 4)

    if ask == "convexity_bond":
        ans = convex
        stmt = (f"A ${face:,.0f} par bond has coupon rate {r*100:.2f}%, yields {i*100:.2f}%, "
                f"and matures in {n} years. Compute convexity: "
                f"C = Σ t(t+1)·PV(CF_t) / (P·(1+i)²).")
        wrongs = [round(convex * (1 + i), 4),
                  round(convex / (1 + i), 4),
                  round(convex * 1.1, 4)]

    elif ask == "second_order_price_approx":
        d_mac = _macaulay_duration(cashflows, i)
        d_mod = d_mac / (1 + i)
        delta_i = round(float(rng.choice([-0.01, 0.01, 0.02])), 2)
        delta_p = round(-d_mod * price * delta_i + 0.5 * convex * price * delta_i ** 2, 2)
        ans = delta_p
        stmt = (f"A bond has price ${round(price,2):,.2f}, modified duration {round(d_mod,4)}, "
                f"and convexity {convex} at yield {i*100:.2f}%. "
                f"Estimate the price change for a Δi = {delta_i:+.3f} shift "
                f"using ΔP ≈ -D_mod·P·Δi + ½·C·P·(Δi)².")
        wrongs = [round(-d_mod * price * delta_i, 2),
                  round(delta_p * 1.05, 2),
                  round(-d_mod * price * delta_i + convex * price * delta_i ** 2, 2)]

    else:
        raise ValueError(f"Unknown ask '{ask}' for convexity")

    return Problem("convexity", ask, stmt, ans,
                   make_mc_choices(ans, wrongs, rng),
                   params={"face": face, "r": r, "i": i, "n": n,
                           "price": round(price, 2), "convexity": convex}, seed=seed)


@register("immunization")
def gen_immunization(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    i = round(float(rng.uniform(*ranges["yield_range"])), 4)

    if ask == "redington_conditions":
        # Present a scenario and ask which condition is violated
        liability_pv = round(float(rng.uniform(5000, 20000)), 2)
        # Asset mix: two bonds — short and long
        w = round(float(rng.uniform(0.3, 0.7)), 2)
        d_short = round(float(rng.uniform(2, 5)), 2)
        d_long = round(float(rng.uniform(8, 15)), 2)
        d_liability = round(float(rng.uniform(5, 7)), 2)
        d_asset = round(w * d_short + (1 - w) * d_long, 4)
        # PV condition met, duration condition not necessarily
        ans = round(d_asset, 4)   # asset duration
        stmt = (f"A portfolio has ${liability_pv:,.2f} in liabilities with Macaulay duration "
                f"{d_liability} at yield {i*100:.2f}%. "
                f"Assets: weight {w:.2f} in a bond with D={d_short} and weight "
                f"{1-w:.2f} in a bond with D={d_long}. "
                f"Compute the asset portfolio Macaulay duration (for Redington condition 2).")
        wrongs = [round(w * d_long + (1 - w) * d_short, 4),
                  round(d_liability, 4),
                  round(d_asset * (1 + i), 4)]

    elif ask == "full_immunization_weight":
        # Find weights so asset duration matches liability duration H
        H = round(float(rng.uniform(4, 8)), 2)
        d1 = round(float(rng.uniform(1, H - 0.5)), 2)
        d2 = round(float(rng.uniform(H + 0.5, 15)), 2)
        # w*d1 + (1-w)*d2 = H  =>  w = (d2 - H) / (d2 - d1)
        w1 = round((d2 - H) / (d2 - d1), 4)
        ans = w1
        stmt = (f"Immunize a liability with Macaulay duration H={H} using two bonds "
                f"with durations D1={d1} and D2={d2}. "
                f"Find the weight w1 in bond 1 so that w1·D1 + (1-w1)·D2 = H.")
        wrongs = [round(1 - w1, 4),
                  round((H - d1) / (d2 - d1), 4),
                  round(w1 * 1.1, 4)]

    else:
        raise ValueError(f"Unknown ask '{ask}' for immunization")

    return Problem("immunization", ask, stmt, ans,
                   make_mc_choices(ans, wrongs, rng),
                   params={"i": i}, seed=seed)
