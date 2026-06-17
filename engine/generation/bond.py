"""Generators for bond pricing, Makeham formula, premium/discount amortization."""
from __future__ import annotations

import numpy as np

from engine.generation.base import Problem, make_mc_choices, register


def _a_n(i: float, n: int) -> float:
    """Present value annuity-immediate factor a_n|i = (1 - v^n)/i."""
    discount_factor = 1 / (1 + i)
    return (1 - discount_factor ** n) / i


def _bond_price(face: float, coupon_rate: float, n: int, yield_rate: float,
                 redemption: float | None = None) -> float:
    """P = C*v^n + Fr*a_n|i where C = redemption, F = face, r = coupon_rate."""
    redemption_value = redemption if redemption is not None else face
    coupon_payment = face * coupon_rate
    discount_factor = 1 / (1 + yield_rate)
    return redemption_value * discount_factor ** n + coupon_payment * _a_n(yield_rate, n)


@register("bond_price")
def gen_bond_price(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    face = round(float(rng.choice([1000, 2000, 5000, 10000])), 2)
    coupon_rate = round(float(rng.uniform(*ranges["coupon_rate_range"])), 4)
    yield_rate = round(float(rng.uniform(*ranges["yield_range"])), 4)
    n_years = int(rng.integers(*ranges["n_range"]))
    coupon_payment = round(face * coupon_rate, 2)
    price = _bond_price(face, coupon_rate, n_years, yield_rate)

    if ask == "price_from_yield":
        answer = round(price, 2)
        question_text = (
            f"A ${face:,.0f} par bond pays {coupon_rate*100:.2f}% annual coupons "
            f"and matures in {n_years} years. "
            f"Find the price to yield {yield_rate*100:.2f}% annual effective."
        )
        wrong_answers = [
            round(_bond_price(face, coupon_rate, n_years, yield_rate + 0.01), 2),
            round(_bond_price(face, coupon_rate, n_years, yield_rate - 0.01), 2),
            round(coupon_payment * _a_n(yield_rate, n_years), 2),
        ]

    elif ask == "current_yield":
        answer = round(coupon_payment / price, 6)
        question_text = (
            f"A ${face:,.0f} par bond with {coupon_rate*100:.2f}% annual coupons "
            f"trades at ${round(price,2):,.2f}. "
            f"Find the current yield (annual coupon / price)."
        )
        wrong_answers = [
            round(coupon_rate, 4),
            round(yield_rate, 4),
            round(coupon_payment / face, 4),
        ]

    elif ask == "yield_approx":
        # Bond yield approximation: blends the coupon income with the average
        # straight-line gain/loss to redemption, divided by the average of
        # redemption and price — a quick estimate that avoids solving for i
        # in the exact price equation.
        redemption_value = face
        approx_yield = round(
            (coupon_payment + (redemption_value - price) / n_years)
            / ((redemption_value + price) / 2), 4
        )
        answer = approx_yield
        question_text = (
            f"A ${face:,.0f} par bond with {coupon_rate*100:.2f}% annual coupons "
            f"matures in {n_years} years and is priced at ${round(price,2):,.2f}. "
            f"Approximate the yield using the bond yield approximation formula: "
            f"i ≈ (Fr + (C-P)/n) / ((C+P)/2)."
        )
        wrong_answers = [
            round(yield_rate, 4),
            round(coupon_payment / price, 4),
            round(approx_yield * 1.05, 4),
        ]

    else:
        raise ValueError(f"Unknown ask '{ask}' for bond_price")

    return Problem(
        "bond_price", ask, question_text, answer,
        make_mc_choices(answer, wrong_answers, rng),
        params={
            "face": face, "r": coupon_rate, "i": yield_rate, "n": n_years,
            "Fr": coupon_payment, "price": round(price, 2),
        },
        seed=seed,
    )


@register("bond_makeham")
def gen_bond_makeham(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    face = round(float(rng.choice([1000, 2000, 5000])), 2)
    coupon_rate = round(float(rng.uniform(*ranges["coupon_rate_range"])), 4)
    yield_rate = round(float(rng.uniform(*ranges["yield_range"])), 4)
    n_years = int(rng.integers(*ranges["n_range"]))
    redemption_value = face  # redeemed at par
    coupon_payment = face * coupon_rate
    # Makeham's modified coupon rate g = Fr/C; reduces to the coupon rate r
    # itself whenever the bond redeems at par (C == F), as it does here.
    modified_coupon_rate = round(coupon_rate, 4)
    discount_factor = 1 / (1 + yield_rate)
    redemption_pv = round(redemption_value * discount_factor ** n_years, 4)
    price = round(
        redemption_pv
        + (modified_coupon_rate / yield_rate) * (redemption_value - redemption_pv), 2
    )

    if ask == "makeham_price":
        answer = round(price, 2)
        question_text = (
            f"A ${face:,.0f} par bond pays {coupon_rate*100:.2f}% annual coupons, "
            f"matures in {n_years} years, yielding {yield_rate*100:.2f}%. "
            f"Use the Makeham formula: P = K + (g/i)(C-K) where K = C*v^n and g = Fr/C."
        )
        wrong_answers = [
            round(_bond_price(face, coupon_rate, n_years, yield_rate) * 1.01, 2),
            round(redemption_pv + (modified_coupon_rate / yield_rate) * redemption_value, 2),
            round(redemption_pv + (coupon_rate / yield_rate)
                  * (redemption_value - redemption_pv) * 1.02, 2),
        ]

    elif ask == "makeham_modified_coupon_g":
        answer = round(modified_coupon_rate, 4)
        question_text = (
            f"A ${face:,.0f} par bond pays annual coupons of ${coupon_payment:,.2f}. "
            f"In the Makeham formula, find g = Fr/C, the modified coupon rate."
        )
        wrong_answers = [
            round(coupon_rate * 1.05, 4),
            round(coupon_payment / (face * (1 + yield_rate)), 4),
            round(coupon_rate / (1 + yield_rate), 4),
        ]

    else:
        raise ValueError(f"Unknown ask '{ask}' for bond_makeham")

    return Problem(
        "bond_makeham", ask, question_text, answer,
        make_mc_choices(answer, wrong_answers, rng),
        params={
            "face": face, "r": coupon_rate, "i": yield_rate, "n": n_years,
            "g": modified_coupon_rate, "K": redemption_pv, "price": price,
        },
        seed=seed,
    )


@register("bond_prem_disc")
def gen_bond_prem_disc(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    face = round(float(rng.choice([1000, 2000, 5000])), 2)
    coupon_rate = round(float(rng.uniform(*ranges["coupon_rate_range"])), 4)
    yield_rate = round(float(rng.uniform(*ranges["yield_range"])), 4)
    n_years = int(rng.integers(*ranges["n_range"]))
    pv_factor = _a_n(yield_rate, n_years)
    price = round(_bond_price(face, coupon_rate, n_years, yield_rate), 2)
    # P - C = (g-i)*C*a_n|: positive when coupon_rate > yield_rate (bond sells
    # at a premium), negative when coupon_rate < yield_rate (sells at a discount).
    premium_or_discount = round((coupon_rate - yield_rate) * face * pv_factor, 2)

    if ask == "premium_discount_amount":
        is_premium = coupon_rate > yield_rate
        answer = round(abs(premium_or_discount), 2)
        label = "premium" if is_premium else "discount"
        question_text = (
            f"A ${face:,.0f} par bond has coupon rate {coupon_rate*100:.2f}% "
            f"and yield {yield_rate*100:.2f}%, maturing in {n_years} years. "
            f"Find the {label} using P - C = (g-i)*C*a_n|i."
        )
        wrong_answers = [
            round(abs(premium_or_discount) * (1 + yield_rate), 2),
            round(abs(face * coupon_rate - face * yield_rate) * n_years, 2),
            round(abs(premium_or_discount) / pv_factor, 2),
        ]

    elif ask == "book_value_tth":
        coupon_number = int(rng.integers(1, n_years))
        # Book value at t: BV_t = face + (g-i)*face*a_{n-t}|i — the remaining
        # premium/discount still left to amortize over the n-t outstanding coupons.
        book_value = round(
            face + (coupon_rate - yield_rate) * face * _a_n(yield_rate, n_years - coupon_number), 2
        )
        answer = book_value
        question_text = (
            f"A ${face:,.0f} par bond has coupon rate {coupon_rate*100:.2f}% and "
            f"yield {yield_rate*100:.2f}%, maturing in {n_years} years. "
            f"Find the book value (amortized cost) immediately after the "
            f"{coupon_number}th coupon."
        )
        wrong_answers = [
            round(face + (coupon_rate - yield_rate) * face
                  * _a_n(yield_rate, n_years - coupon_number + 1), 2),
            round(face + (coupon_rate - yield_rate) * face
                  * _a_n(yield_rate, n_years - coupon_number - 1), 2),
            round(price + coupon_number * premium_or_discount / n_years, 2),
        ]
        # NOTE: coupon_number is randomly chosen above but is NOT saved into
        # `params` below — the solver (_s_bond_prem_disc in solve.py) falls
        # back to t=1 via params.get("t", 1), so the worked solution it shows
        # will only match this question when coupon_number happens to be 1.
        # This is a pre-existing generator/solver param mismatch, not
        # something introduced by this pass.

    else:
        raise ValueError(f"Unknown ask '{ask}' for bond_prem_disc")

    return Problem(
        "bond_prem_disc", ask, question_text, answer,
        make_mc_choices(answer, wrong_answers, rng),
        params={
            "face": face, "r": coupon_rate, "i": yield_rate, "n": n_years,
            "price": price, "premium": premium_or_discount,
        },
        seed=seed,
    )
