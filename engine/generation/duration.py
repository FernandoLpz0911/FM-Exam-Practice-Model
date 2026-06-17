"""Generators for Macaulay/modified duration, convexity, and Redington immunization."""
from __future__ import annotations

import numpy as np

from engine.generation.base import Problem, make_mc_choices, register


def _bond_cashflows(face: float, coupon_rate: float, n: int,
                     redemption: float | None = None) -> list[tuple[int, float]]:
    """[(t, cashflow)] — coupon at t=1..n, redemption at t=n."""
    redemption_value = redemption if redemption is not None else face
    coupon_payment = face * coupon_rate
    cashflows = [(t, coupon_payment) for t in range(1, n + 1)]
    cashflows[-1] = (n, coupon_payment + redemption_value)
    return cashflows


def _macaulay_duration(cashflows: list[tuple[int, float]], yield_rate: float) -> float:
    """D_mac = Σ t·PV(CF_t) / P — the cash-flow-weighted average time to receipt."""
    discount_factor = 1 / (1 + yield_rate)
    price = sum(cf * discount_factor ** t for t, cf in cashflows)
    time_weighted_pv = sum(t * cf * discount_factor ** t for t, cf in cashflows)
    return time_weighted_pv / price


@register("macaulay_duration")
def gen_macaulay_duration(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    face = float(rng.choice([1000, 2000, 5000]))
    coupon_rate = round(float(rng.uniform(*ranges["coupon_rate_range"])), 4)
    yield_rate = round(float(rng.uniform(*ranges["yield_range"])), 4)
    n_years = int(rng.integers(*ranges["n_range"]))
    cashflows = _bond_cashflows(face, coupon_rate, n_years)
    macaulay_duration = round(_macaulay_duration(cashflows, yield_rate), 4)

    if ask == "macaulay_duration_bond":
        answer = macaulay_duration
        question_text = (
            f"A ${face:,.0f} par bond has coupon rate {coupon_rate*100:.2f}%, "
            f"yields {yield_rate*100:.2f}%, and matures in {n_years} years. "
            f"Compute the Macaulay duration (D = Σ t·PV(CF_t) / P)."
        )
        wrong_answers = [
            round(macaulay_duration * (1 + yield_rate), 4),
            round(macaulay_duration / (1 + yield_rate), 4),
            round(n_years * 0.9, 4),
        ]

    elif ask == "macaulay_perpetuity":
        # Closed form for a perpetuity-immediate: D_mac = (1+i)/i (derived by
        # taking the limit of the level-annuity duration formula as n → ∞).
        perpetuity_duration = round((1 + yield_rate) / yield_rate, 4)
        answer = perpetuity_duration
        question_text = (
            f"Find the Macaulay duration of a perpetuity-immediate at yield "
            f"{yield_rate*100:.2f}%. Use D = (1+i)/i."
        )
        wrong_answers = [
            round(1 / yield_rate, 4),
            round(1 / (1 + yield_rate), 4),
            round((1 + yield_rate) / yield_rate * 1.05, 4),
        ]

    else:
        raise ValueError(f"Unknown ask '{ask}' for macaulay_duration")

    return Problem(
        "macaulay_duration", ask, question_text, answer,
        make_mc_choices(answer, wrong_answers, rng),
        params={
            "face": face, "r": coupon_rate, "i": yield_rate, "n": n_years,
            "d_mac": macaulay_duration,
        },
        seed=seed,
    )


@register("modified_duration")
def gen_modified_duration(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    face = float(rng.choice([1000, 2000, 5000]))
    coupon_rate = round(float(rng.uniform(*ranges["coupon_rate_range"])), 4)
    yield_rate = round(float(rng.uniform(*ranges["yield_range"])), 4)
    n_years = int(rng.integers(*ranges["n_range"]))
    cashflows = _bond_cashflows(face, coupon_rate, n_years)
    macaulay_duration = _macaulay_duration(cashflows, yield_rate)
    # Modified duration discounts Macaulay duration by one period of growth —
    # it measures price sensitivity directly (dP/di / -P), whereas Macaulay
    # duration measures the weighted-average timing of cash flows.
    modified_duration = round(macaulay_duration / (1 + yield_rate), 4)

    if ask == "modified_duration_from_mac":
        answer = modified_duration
        question_text = (
            f"A bond has Macaulay duration {round(macaulay_duration, 4)} at yield "
            f"{yield_rate*100:.2f}%. Find the modified duration: "
            f"D_mod = D_mac / (1+i)."
        )
        wrong_answers = [
            round(macaulay_duration * (1 + yield_rate), 4),
            round(macaulay_duration, 4),
            round(modified_duration * 1.05, 4),
        ]

    elif ask == "price_change_approx":
        yield_shift = round(float(rng.choice([-0.01, 0.01, -0.005, 0.005])), 3)
        discount_factor = 1 / (1 + yield_rate)
        price = sum(cf * discount_factor ** t for t, cf in cashflows)
        # First-order Taylor approximation of price sensitivity to yield: the
        # negative sign reflects the inverse price/yield relationship.
        price_change = round(-modified_duration * price * yield_shift, 2)
        answer = price_change
        question_text = (
            f"A ${face:,.0f} par bond with coupon rate {coupon_rate*100:.2f}% has "
            f"modified duration {modified_duration} at yield {yield_rate*100:.2f}%. "
            f"Estimate the price change if yield "
            f"{'rises' if yield_shift > 0 else 'falls'} by "
            f"{abs(yield_shift)*100:.2f}% using ΔP ≈ -D_mod · P · Δi."
        )
        wrong_answers = [
            round(-modified_duration * price * yield_shift * (1 + yield_rate), 2),
            round(modified_duration * price * yield_shift, 2),
            round(-modified_duration * price * yield_shift * 0.9, 2),
        ]

    else:
        raise ValueError(f"Unknown ask '{ask}' for modified_duration")

    return Problem(
        "modified_duration", ask, question_text, answer,
        make_mc_choices(answer, wrong_answers, rng),
        params={
            "face": face, "r": coupon_rate, "i": yield_rate, "n": n_years,
            "d_mac": round(macaulay_duration, 4), "d_mod": modified_duration,
        },
        seed=seed,
    )


@register("convexity")
def gen_convexity(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    face = float(rng.choice([1000, 2000]))
    coupon_rate = round(float(rng.uniform(*ranges["coupon_rate_range"])), 4)
    yield_rate = round(float(rng.uniform(*ranges["yield_range"])), 4)
    n_years = int(rng.integers(*ranges["n_range"]))
    cashflows = _bond_cashflows(face, coupon_rate, n_years)
    discount_factor = 1 / (1 + yield_rate)
    price = sum(cf * discount_factor ** t for t, cf in cashflows)
    # Convexity measures the curvature of the price/yield relationship —
    # the second derivative term that duration (a linear approximation) misses.
    convexity = sum(
        t * (t + 1) * cf * discount_factor ** t for t, cf in cashflows
    ) / (price * (1 + yield_rate) ** 2)
    convexity = round(convexity, 4)

    if ask == "convexity_bond":
        answer = convexity
        question_text = (
            f"A ${face:,.0f} par bond has coupon rate {coupon_rate*100:.2f}%, "
            f"yields {yield_rate*100:.2f}%, and matures in {n_years} years. "
            f"Compute convexity: C = Σ t(t+1)·PV(CF_t) / (P·(1+i)²)."
        )
        wrong_answers = [
            round(convexity * (1 + yield_rate), 4),
            round(convexity / (1 + yield_rate), 4),
            round(convexity * 1.1, 4),
        ]

    elif ask == "second_order_price_approx":
        macaulay_duration = _macaulay_duration(cashflows, yield_rate)
        modified_duration = macaulay_duration / (1 + yield_rate)
        yield_shift = round(float(rng.choice([-0.01, 0.01, 0.02])), 2)
        # Second-order Taylor expansion: duration captures the linear (slope)
        # term, convexity's ½·C·P·(Δi)² term corrects for the curve's bend.
        price_change = round(
            -modified_duration * price * yield_shift
            + 0.5 * convexity * price * yield_shift ** 2, 2
        )
        answer = price_change
        question_text = (
            f"A bond has price ${round(price,2):,.2f}, modified duration "
            f"{round(modified_duration,4)}, and convexity {convexity} at yield "
            f"{yield_rate*100:.2f}%. Estimate the price change for a "
            f"Δi = {yield_shift:+.3f} shift using "
            f"ΔP ≈ -D_mod·P·Δi + ½·C·P·(Δi)²."
        )
        wrong_answers = [
            round(-modified_duration * price * yield_shift, 2),
            round(price_change * 1.05, 2),
            round(
                -modified_duration * price * yield_shift
                + convexity * price * yield_shift ** 2, 2
            ),
        ]

    else:
        raise ValueError(f"Unknown ask '{ask}' for convexity")

    return Problem(
        "convexity", ask, question_text, answer,
        make_mc_choices(answer, wrong_answers, rng),
        params={
            "face": face, "r": coupon_rate, "i": yield_rate, "n": n_years,
            "price": round(price, 2), "convexity": convexity,
        },
        seed=seed,
    )


@register("immunization")
def gen_immunization(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    yield_rate = round(float(rng.uniform(*ranges["yield_range"])), 4)

    if ask == "redington_conditions":
        liability_pv = round(float(rng.uniform(5000, 20000)), 2)
        short_bond_weight = round(float(rng.uniform(0.3, 0.7)), 2)
        short_bond_duration = round(float(rng.uniform(2, 5)), 2)
        long_bond_duration = round(float(rng.uniform(8, 15)), 2)
        liability_duration = round(float(rng.uniform(5, 7)), 2)
        asset_portfolio_duration = round(
            short_bond_weight * short_bond_duration
            + (1 - short_bond_weight) * long_bond_duration, 4
        )
        answer = round(asset_portfolio_duration, 4)
        question_text = (
            f"A portfolio has ${liability_pv:,.2f} in liabilities with Macaulay "
            f"duration {liability_duration} at yield {yield_rate*100:.2f}%. "
            f"Assets: weight {short_bond_weight:.2f} in a bond with "
            f"D={short_bond_duration} and weight {1-short_bond_weight:.2f} in a "
            f"bond with D={long_bond_duration}. Compute the asset portfolio "
            f"Macaulay duration (for Redington condition 2)."
        )
        wrong_answers = [
            round(short_bond_weight * long_bond_duration
                  + (1 - short_bond_weight) * short_bond_duration, 4),
            round(liability_duration, 4),
            round(asset_portfolio_duration * (1 + yield_rate), 4),
        ]
        # NOTE: liability_pv, short_bond_weight, short_bond_duration,
        # long_bond_duration, and liability_duration are all randomly
        # generated above but only `yield_rate` is saved into params below —
        # the solver (_s_immunization in solve.py) can't reproduce this exact
        # scenario's numbers from params alone; it always returns the
        # theoretical NaN placeholder for this ask variant instead.

    elif ask == "full_immunization_weight":
        # Solve w1*d1 + (1-w1)*d2 = target_duration for w1 — the classic
        # two-bond duration-matching setup from Redington immunization.
        target_duration = round(float(rng.uniform(4, 8)), 2)
        bond1_duration = round(float(rng.uniform(1, target_duration - 0.5)), 2)
        bond2_duration = round(float(rng.uniform(target_duration + 0.5, 15)), 2)
        bond1_weight = round(
            (bond2_duration - target_duration) / (bond2_duration - bond1_duration), 4
        )
        answer = bond1_weight
        question_text = (
            f"Immunize a liability with Macaulay duration H={target_duration} "
            f"using two bonds with durations D1={bond1_duration} and "
            f"D2={bond2_duration}. Find the weight w1 in bond 1 so that "
            f"w1·D1 + (1-w1)·D2 = H."
        )
        wrong_answers = [
            round(1 - bond1_weight, 4),
            round((target_duration - bond1_duration) / (bond2_duration - bond1_duration), 4),
            round(bond1_weight * 1.1, 4),
        ]
        # NOTE: same issue as above — target_duration, bond1_duration, and
        # bond2_duration are randomly generated but not saved into params, so
        # the solver falls back to its hardcoded defaults (H=6, d1=3, d2=10)
        # rather than reproducing this question's actual numbers.

    else:
        raise ValueError(f"Unknown ask '{ask}' for immunization")

    return Problem(
        "immunization", ask, question_text, answer,
        make_mc_choices(answer, wrong_answers, rng),
        params={"i": yield_rate}, seed=seed,
    )
