"""Generators for annuity topics: immediate, due, perpetuity, deferred, varying."""
from __future__ import annotations

import math

import numpy as np

from engine.generation.base import Problem, make_mc_choices, register


def _a_n(i: float, n: int) -> float:
    """Present value annuity-immediate: a_n|i = (1 - v^n)/i."""
    discount_factor = 1 / (1 + i)
    return (1 - discount_factor ** n) / i


def _s_n(i: float, n: int) -> float:
    """Future value annuity-immediate: s_n|i = ((1+i)^n - 1)/i."""
    return ((1 + i) ** n - 1) / i


def _a_due(i: float, n: int) -> float:
    """Present value annuity-due: a_due_n|i = (1+i)*a_n|i (payments shifted one period earlier)."""
    return (1 + i) * _a_n(i, n)


def _s_due(i: float, n: int) -> float:
    """Future value annuity-due: s_due_n|i = (1+i)*s_n|i."""
    return (1 + i) * _s_n(i, n)


@register("annuity_immediate")
def gen_annuity_immediate(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    annual_rate = round(float(rng.uniform(*ranges["i_range"])), 4)
    n_years = int(rng.integers(*ranges["n_range"]))
    payment = round(float(rng.uniform(*ranges["payment_range"])), 2)
    pv_factor = _a_n(annual_rate, n_years)
    fv_factor = _s_n(annual_rate, n_years)

    if ask == "pv_annuity_imm":
        answer = round(payment * pv_factor, 2)
        question_text = (
            f"An annuity-immediate pays ${payment:,.2f} at the end of each year "
            f"for {n_years} years at {annual_rate*100:.2f}% annual effective interest. "
            f"Find the present value."
        )
        # Distractors encode: using the due (start-of-period) factor, applying
        # an extra period of growth, and a naive average-discounting shortcut.
        wrong_answers = [
            round(payment * _a_due(annual_rate, n_years), 2),
            round(payment * pv_factor * (1 + annual_rate), 2),
            round(payment * n_years / (1 + annual_rate) ** (n_years / 2), 2),
        ]

    elif ask == "fv_annuity_imm":
        answer = round(payment * fv_factor, 2)
        question_text = (
            f"An annuity-immediate pays ${payment:,.2f} at the end of each year "
            f"for {n_years} years at {annual_rate*100:.2f}%. "
            f"Find the accumulated value at the end of year {n_years}."
        )
        wrong_answers = [
            round(payment * _s_due(annual_rate, n_years), 2),
            round(payment * n_years * (1 + annual_rate) ** (n_years / 2), 2),
            round(payment * fv_factor / (1 + annual_rate), 2),
        ]

    elif ask == "payment_from_pv":
        loan_amount = round(float(rng.uniform(10000, 200000)), 2)
        answer = round(loan_amount / pv_factor, 2)
        question_text = (
            f"A loan of ${loan_amount:,.2f} is repaid with level annual payments at the "
            f"end of each year for {n_years} years at {annual_rate*100:.2f}%. "
            f"Find the annual payment."
        )
        wrong_answers = [
            round(loan_amount / (pv_factor * (1 + annual_rate)), 2),
            round(loan_amount / n_years, 2),
            round(loan_amount * annual_rate + loan_amount / n_years, 2),
        ]

    elif ask == "n_from_pv_imm":
        loan_amount = round(float(rng.uniform(1000, 20000)), 2)
        answer = n_years
        # Back-calculate the payment that's consistent with the chosen n_years
        # so the question's numbers (loan, payment, rate) all agree.
        level_payment = round(loan_amount / pv_factor, 2)
        question_text = (
            f"A loan of ${loan_amount:,.2f} is repaid by payments of "
            f"${level_payment:,.2f} at the end of each year at {annual_rate*100:.2f}%. "
            f"How many payments are needed?"
        )
        wrong_answers = [
            n_years - 1,
            n_years + 1,
            round(math.log(
                level_payment / (level_payment - loan_amount * annual_rate)
            ) / math.log(1 + annual_rate) + 0.5),
        ]

    else:
        raise ValueError(f"Unknown ask '{ask}' for annuity_immediate")

    return Problem(
        "annuity_immediate", ask, question_text, answer,
        make_mc_choices(answer, wrong_answers, rng),
        params={"i": annual_rate, "n": n_years, "payment": payment}, seed=seed,
    )


@register("annuity_due")
def gen_annuity_due(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    annual_rate = round(float(rng.uniform(*ranges["i_range"])), 4)
    n_years = int(rng.integers(*ranges["n_range"]))
    payment = round(float(rng.uniform(*ranges["payment_range"])), 2)
    pv_factor_due = _a_due(annual_rate, n_years)
    fv_factor_due = _s_due(annual_rate, n_years)

    if ask == "pv_annuity_due":
        answer = round(payment * pv_factor_due, 2)
        question_text = (
            f"An annuity-due pays ${payment:,.2f} at the beginning of each year "
            f"for {n_years} years at {annual_rate*100:.2f}%. Find the present value."
        )
        wrong_answers = [
            round(payment * _a_n(annual_rate, n_years), 2),
            round(payment * pv_factor_due / (1 + annual_rate), 2),
            round(payment * pv_factor_due * (1 + annual_rate), 2),
        ]

    elif ask == "fv_annuity_due":
        answer = round(payment * fv_factor_due, 2)
        question_text = (
            f"An annuity-due pays ${payment:,.2f} at the beginning of each year "
            f"for {n_years} years at {annual_rate*100:.2f}%. "
            f"Find the accumulated value at end of year {n_years}."
        )
        wrong_answers = [
            round(payment * _s_n(annual_rate, n_years), 2),
            round(payment * fv_factor_due / (1 + annual_rate), 2),
            round(payment * fv_factor_due * (1 + annual_rate), 2),
        ]

    elif ask == "payment_from_pv_due":
        present_value = round(float(rng.uniform(10000, 200000)), 2)
        answer = round(present_value / pv_factor_due, 2)
        question_text = (
            f"A lease requires level payments at the beginning of each year for "
            f"{n_years} years. At {annual_rate*100:.2f}%, the present value of all "
            f"payments is ${present_value:,.2f}. Find the annual payment."
        )
        wrong_answers = [
            round(present_value / _a_n(annual_rate, n_years), 2),
            round(present_value / n_years, 2),
            round(present_value / pv_factor_due * (1 + annual_rate), 2),
        ]

    else:
        raise ValueError(f"Unknown ask '{ask}' for annuity_due")

    return Problem(
        "annuity_due", ask, question_text, answer,
        make_mc_choices(answer, wrong_answers, rng),
        params={"i": annual_rate, "n": n_years, "payment": payment}, seed=seed,
    )


@register("perpetuity")
def gen_perpetuity(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    annual_rate = round(float(rng.uniform(*ranges["i_range"])), 4)
    payment = round(float(rng.uniform(*ranges["payment_range"])), 2)
    # Perpetuities never terminate, so a_n| → 1/i as n → ∞ (the v^n term vanishes).
    discount_rate = annual_rate / (1 + annual_rate)

    if ask == "pv_perp_imm":
        answer = round(payment / annual_rate, 2)
        question_text = (
            f"A perpetuity-immediate pays ${payment:,.2f} at the end of each year "
            f"forever. At {annual_rate*100:.2f}% annual effective interest, "
            f"find the present value."
        )
        wrong_answers = [
            round(payment / discount_rate, 2),
            round(payment / annual_rate + payment, 2),
            round(payment / (annual_rate + 0.01), 2),
        ]

    elif ask == "pv_perp_due":
        answer = round(payment / discount_rate, 2)
        question_text = (
            f"A perpetuity-due pays ${payment:,.2f} at the beginning of each year "
            f"forever. At {annual_rate*100:.2f}%, find the present value."
        )
        wrong_answers = [
            round(payment / annual_rate, 2),
            round(payment / discount_rate / (1 + annual_rate), 2),
            round(payment * (1 + annual_rate) / annual_rate, 2),
        ]

    elif ask == "payment_from_perp_pv":
        present_value = round(float(rng.uniform(10000, 200000)), 2)
        answer = round(present_value * annual_rate, 2)
        question_text = (
            f"A perpetuity-immediate has present value ${present_value:,.2f} "
            f"at {annual_rate*100:.2f}%. Find the annual payment."
        )
        wrong_answers = [
            round(present_value * discount_rate, 2),
            round(present_value * annual_rate / (1 + annual_rate), 2),
            round(present_value / annual_rate, 2),
        ]

    else:
        raise ValueError(f"Unknown ask '{ask}' for perpetuity")

    return Problem(
        "perpetuity", ask, question_text, answer,
        make_mc_choices(answer, wrong_answers, rng),
        params={"i": annual_rate, "payment": payment, "d": discount_rate}, seed=seed,
    )


@register("deferred_annuity")
def gen_deferred_annuity(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    annual_rate = round(float(rng.uniform(*ranges["i_range"])), 4)
    n_years = int(rng.integers(*ranges["n_range"]))
    deferral_years = int(rng.integers(*ranges["m_range"]))
    payment = round(float(rng.uniform(*ranges["payment_range"])), 2)
    discount_factor = 1 / (1 + annual_rate)
    pv_factor = _a_n(annual_rate, n_years)

    if ask == "pv_deferred_imm":
        # Discount the ordinary annuity's value back through the deferral
        # window with v^m before applying the usual a_n| factor.
        answer = round(payment * (discount_factor ** deferral_years) * pv_factor, 2)
        question_text = (
            f"An annuity-immediate pays ${payment:,.2f} per year for {n_years} years, "
            f"with the first payment {deferral_years+1} years from now. "
            f"At {annual_rate*100:.2f}%, find the present value today."
        )
        wrong_answers = [
            round(payment * _a_n(annual_rate, n_years + deferral_years), 2),
            round(payment * pv_factor, 2),
            round(payment * (discount_factor ** (deferral_years + 1)) * pv_factor, 2),
        ]

    elif ask == "pv_deferred_due":
        pv_factor_due = _a_due(annual_rate, n_years)
        answer = round(payment * (discount_factor ** deferral_years) * pv_factor_due, 2)
        question_text = (
            f"An annuity-due pays ${payment:,.2f} per year for {n_years} years, "
            f"deferred {deferral_years} years (first payment at time {deferral_years}). "
            f"At {annual_rate*100:.2f}%, find the present value today."
        )
        wrong_answers = [
            round(payment * (discount_factor ** deferral_years) * pv_factor, 2),
            round(payment * pv_factor_due, 2),
            round(payment * (discount_factor ** (deferral_years - 1)) * pv_factor_due, 2),
        ]

    else:
        raise ValueError(f"Unknown ask '{ask}' for deferred_annuity")

    return Problem(
        "deferred_annuity", ask, question_text, answer,
        make_mc_choices(answer, wrong_answers, rng),
        params={
            "i": annual_rate, "n": n_years, "m": deferral_years, "payment": payment,
        },
        seed=seed,
    )


@register("annuity_varying")
def gen_annuity_varying(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    annual_rate = round(float(rng.uniform(*ranges["i_range"])), 4)
    n_years = int(rng.integers(*ranges["n_range"]))
    base_payment = round(float(rng.uniform(*ranges["base_payment_range"])), 2)
    discount_factor = 1 / (1 + annual_rate)
    pv_factor = _a_n(annual_rate, n_years)
    pv_factor_due = _a_due(annual_rate, n_years)

    if ask == "pv_arithmetic_inc":
        # Payments: base, 2*base, ..., n*base at times 1,...,n.
        # PV = base * (Ia)_n|i, where (Ia)_n|i = (ä_n| - n*v^n) / i
        # is the standard increasing-annuity present-value factor.
        increasing_factor = (pv_factor_due - n_years * discount_factor ** n_years) / annual_rate
        answer = round(base_payment * increasing_factor, 2)
        question_text = (
            f"An annuity pays ${base_payment:,.2f} at end of year 1, "
            f"${2*base_payment:,.2f} at end of year 2, ..., "
            f"${n_years*base_payment:,.2f} at end of year {n_years}. "
            f"At {annual_rate*100:.2f}%, find the present value."
        )
        wrong_answers = [
            round(base_payment * pv_factor, 2),
            round(base_payment * n_years * discount_factor ** (n_years / 2), 2),
            round(base_payment * increasing_factor / (1 + annual_rate), 2),
        ]
        problem_params = {"i": annual_rate, "n": n_years, "base": base_payment, "type": "inc"}

    elif ask == "pv_arithmetic_dec":
        # Payments: n*base, (n-1)*base, ..., base at times 1,...,n.
        # The decreasing-annuity factor (Da)_n| is derived from the increasing
        # one via (Da)_n| = (n+1)*a_n| - (Ia)_n|.
        increasing_factor = (pv_factor_due - n_years * discount_factor ** n_years) / annual_rate
        decreasing_factor = (n_years + 1) * pv_factor - increasing_factor
        answer = round(base_payment * decreasing_factor, 2)
        question_text = (
            f"An annuity pays ${n_years*base_payment:,.2f} at end of year 1, "
            f"decreasing by ${base_payment:,.2f} each year, "
            f"with final payment of ${base_payment:,.2f} at end of year {n_years}. "
            f"At {annual_rate*100:.2f}%, find the present value."
        )
        wrong_answers = [
            round(base_payment * pv_factor, 2),
            round(base_payment * increasing_factor, 2),
            round(base_payment * decreasing_factor * (1 + annual_rate), 2),
        ]
        problem_params = {"i": annual_rate, "n": n_years, "base": base_payment, "type": "dec"}

    elif ask == "pv_geometric":
        growth_rate = round(float(rng.uniform(0.01, 0.05)), 3)
        # The closed-form PV formula divides by (i - g); nudge g away from i
        # so this generator never produces a near-zero denominator (which
        # would blow the answer up to an unusably large/unstable value).
        if abs(annual_rate - growth_rate) < 0.001:
            growth_rate = growth_rate + 0.01
        geometric_discount_ratio = (1 + growth_rate) / (1 + annual_rate)
        present_value = base_payment * (
            1 - geometric_discount_ratio ** n_years
        ) / (annual_rate - growth_rate)
        answer = round(present_value, 2)
        question_text = (
            f"An annuity pays ${base_payment:,.2f} at end of year 1, increasing by "
            f"{growth_rate*100:.1f}% each year for {n_years} years total. "
            f"At {annual_rate*100:.2f}%, find the present value."
        )
        wrong_answers = [
            round(base_payment * pv_factor, 2),
            round(base_payment / (annual_rate - growth_rate), 2),
            round(present_value * (1 + growth_rate), 2),
        ]
        problem_params = {
            "i": annual_rate, "n": n_years, "base": base_payment,
            "g": growth_rate, "type": "geo",
        }

    else:
        raise ValueError(f"Unknown ask '{ask}' for annuity_varying")

    return Problem(
        "annuity_varying", ask, question_text, answer,
        make_mc_choices(answer, wrong_answers, rng),
        params=problem_params, seed=seed,
    )
