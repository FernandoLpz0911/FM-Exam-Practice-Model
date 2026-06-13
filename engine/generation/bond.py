"""Generators for bond pricing, Makeham formula, premium/discount amortization."""
from __future__ import annotations

import numpy as np

from engine.generation.base import Problem, make_mc_choices, register


def _a_n(i: float, n: int) -> float:
    v = 1 / (1 + i)
    return (1 - v ** n) / i


def _bond_price(face: float, coupon_rate: float, n: int, yield_rate: float,
                redemption: float | None = None) -> float:
    """P = C*v^n + Fr*a_n|i where C = redemption, F = face, r = coupon_rate."""
    C = redemption if redemption is not None else face
    Fr = face * coupon_rate  # coupon payment per period
    v = 1 / (1 + yield_rate)
    return C * v ** n + Fr * _a_n(yield_rate, n)


@register("bond_price")
def gen_bond_price(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    face = round(float(rng.choice([1000, 2000, 5000, 10000])), 2)
    r = round(float(rng.uniform(*ranges["coupon_rate_range"])), 4)   # coupon rate
    i = round(float(rng.uniform(*ranges["yield_range"])), 4)          # yield rate
    n = int(rng.integers(*ranges["n_range"]))
    Fr = round(face * r, 2)
    v = 1 / (1 + i)
    price = _bond_price(face, r, n, i)

    if ask == "price_from_yield":
        ans = round(price, 2)
        stmt = (f"A ${face:,.0f} par bond pays {r*100:.2f}% annual coupons and matures in {n} years. "
                f"Find the price to yield {i*100:.2f}% annual effective.")
        wrongs = [round(_bond_price(face, r, n, i + 0.01), 2),
                  round(_bond_price(face, r, n, i - 0.01), 2),
                  round(Fr * _a_n(i, n), 2)]

    elif ask == "current_yield":
        ans = round(Fr / price, 6)
        stmt = (f"A ${face:,.0f} par bond with {r*100:.2f}% annual coupons trades at "
                f"${round(price,2):,.2f}. Find the current yield (annual coupon / price).")
        wrongs = [round(r, 4),
                  round(i, 4),
                  round(Fr / face, 4)]

    elif ask == "yield_approx":
        # Approximate yield: i ≈ (Fr + (C-P)/n) / ((C+P)/2)
        C = face
        approx_yield = round((Fr + (C - price) / n) / ((C + price) / 2), 4)
        ans = approx_yield
        stmt = (f"A ${face:,.0f} par bond with {r*100:.2f}% annual coupons matures in {n} years "
                f"and is priced at ${round(price,2):,.2f}. "
                f"Approximate the yield using the bond yield approximation formula: "
                f"i ≈ (Fr + (C-P)/n) / ((C+P)/2).")
        wrongs = [round(i, 4),
                  round(Fr / price, 4),
                  round(approx_yield * 1.05, 4)]

    else:
        raise ValueError(f"Unknown ask '{ask}' for bond_price")

    return Problem("bond_price", ask, stmt, ans,
                   make_mc_choices(ans, wrongs, rng),
                   params={"face": face, "r": r, "i": i, "n": n, "Fr": Fr, "price": round(price, 2)}, seed=seed)


@register("bond_makeham")
def gen_bond_makeham(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    face = round(float(rng.choice([1000, 2000, 5000])), 2)
    r = round(float(rng.uniform(*ranges["coupon_rate_range"])), 4)
    i = round(float(rng.uniform(*ranges["yield_range"])), 4)
    n = int(rng.integers(*ranges["n_range"]))
    C = face  # redemption = face
    Fr = face * r
    g = round(r, 4)   # modified coupon rate g = Fr/C = r (when C=F)
    v = 1 / (1 + i)
    K = round(C * v ** n, 4)   # PV of redemption
    price = round(K + (g / i) * (C - K), 2)

    if ask == "makeham_price":
        ans = round(price, 2)
        stmt = (f"A ${face:,.0f} par bond pays {r*100:.2f}% annual coupons, matures in {n} years, "
                f"yielding {i*100:.2f}%. Use the Makeham formula: "
                f"P = K + (g/i)(C-K) where K = C*v^n and g = Fr/C.")
        wrongs = [round(_bond_price(face, r, n, i) * 1.01, 2),
                  round(K + (g / i) * C, 2),
                  round(K + (r / i) * (C - K) * 1.02, 2)]

    elif ask == "makeham_modified_coupon_g":
        ans = round(g, 4)
        stmt = (f"A ${face:,.0f} par bond pays annual coupons of ${Fr:,.2f}. "
                f"In the Makeham formula, find g = Fr/C, the modified coupon rate.")
        wrongs = [round(r * 1.05, 4),
                  round(Fr / (face * (1 + i)), 4),
                  round(r / (1 + i), 4)]

    else:
        raise ValueError(f"Unknown ask '{ask}' for bond_makeham")

    return Problem("bond_makeham", ask, stmt, ans,
                   make_mc_choices(ans, wrongs, rng),
                   params={"face": face, "r": r, "i": i, "n": n, "g": g, "K": K, "price": price}, seed=seed)


@register("bond_prem_disc")
def gen_bond_prem_disc(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    face = round(float(rng.choice([1000, 2000, 5000])), 2)
    r = round(float(rng.uniform(*ranges["coupon_rate_range"])), 4)
    i = round(float(rng.uniform(*ranges["yield_range"])), 4)
    n = int(rng.integers(*ranges["n_range"]))
    Fr = face * r
    an = _a_n(i, n)
    price = round(_bond_price(face, r, n, i), 2)
    premium = round((r - i) * face * an, 2)   # P - C = (g-i)*C*a_n| (positive = premium)

    if ask == "premium_discount_amount":
        is_premium = r > i
        ans = round(abs(premium), 2)
        label = "premium" if is_premium else "discount"
        stmt = (f"A ${face:,.0f} par bond has coupon rate {r*100:.2f}% and yield {i*100:.2f}%, "
                f"maturing in {n} years. Find the {'premium' if is_premium else 'discount'} "
                f"using P - C = (g-i)*C*a_n|i.")
        wrongs = [round(abs(premium) * (1 + i), 2),
                  round(abs(Fr - face * i) * n, 2),
                  round(abs(premium) / an, 2)]

    elif ask == "book_value_tth":
        t = int(rng.integers(1, n))
        # Book value at t: BV_t = face + (r-i)*face*a_{n-t}|i
        bv_t = round(face + (r - i) * face * _a_n(i, n - t), 2)
        ans = bv_t
        stmt = (f"A ${face:,.0f} par bond has coupon rate {r*100:.2f}% and yield {i*100:.2f}%, "
                f"maturing in {n} years. Find the book value (amortized cost) "
                f"immediately after the {t}th coupon.")
        wrongs = [round(face + (r - i) * face * _a_n(i, n - t + 1), 2),
                  round(face + (r - i) * face * _a_n(i, n - t - 1), 2),
                  round(price + t * premium / n, 2)]

    else:
        raise ValueError(f"Unknown ask '{ask}' for bond_prem_disc")

    return Problem("bond_prem_disc", ask, stmt, ans,
                   make_mc_choices(ans, wrongs, rng),
                   params={"face": face, "r": r, "i": i, "n": n, "price": price,
                           "premium": premium}, seed=seed)
