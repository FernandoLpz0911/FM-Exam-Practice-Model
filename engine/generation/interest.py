"""Generators for interest theory: TVM basics, nominal rates, force of interest, discount rate."""
from __future__ import annotations

import math

import numpy as np

from engine.generation.base import Problem, make_mc_choices, register


@register("interest_tvm")
def gen_interest_tvm(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    annual_rate = round(float(rng.uniform(*ranges["i_range"])), 4)
    n_years = int(rng.integers(*ranges["n_range"]))
    present_value = round(float(rng.uniform(*ranges["pv_range"])), 2)
    discount_factor = 1 / (1 + annual_rate)

    if ask == "future_value":
        answer = round(present_value * (1 + annual_rate) ** n_years, 2)
        question_text = (
            f"An account earns {annual_rate*100:.1f}% annual effective interest. "
            f"A deposit of ${present_value:,.2f} is made today. "
            f"Find the account value in {n_years} years."
        )
        # Each distractor encodes a common compounding mistake, so the
        # misconception note shown after an incorrect pick is informative:
        # 1) simple interest instead of compound, 2) off-by-one exponent,
        # 3) right rate misapplied at semi-annual compounding.
        wrong_answers = [
            round(present_value * (1 + annual_rate * n_years), 2),
            round(present_value * (1 + annual_rate) ** (n_years - 1), 2),
            round(present_value * (1 + annual_rate / 2) ** (2 * n_years), 2),
        ]

    elif ask == "present_value":
        future_value = round(present_value * (1 + annual_rate) ** n_years, 2)
        answer = round(future_value * discount_factor ** n_years, 2)
        question_text = (
            f"An account earns {annual_rate*100:.1f}% annual effective interest. "
            f"Find the present value of ${future_value:,.2f} due in {n_years} years."
        )
        wrong_answers = [
            round(future_value / (1 + annual_rate * n_years), 2),
            round(future_value * discount_factor ** (n_years + 1), 2),
            round(future_value * discount_factor ** (n_years - 1), 2),
        ]

    elif ask == "interest_rate_solve":
        future_value = round(present_value * (1 + annual_rate) ** n_years, 2)
        answer = round(annual_rate, 4)
        question_text = (
            f"An investment of ${present_value:,.2f} grows to ${future_value:,.2f} "
            f"in {n_years} years under annual effective interest. "
            f"Find the annual effective interest rate."
        )
        wrong_answers = [
            round((future_value - present_value) / (present_value * n_years), 4),
            round((future_value / present_value) ** (1 / (n_years + 1)) - 1, 4),
            round(annual_rate * 1.1, 4),
        ]

    elif ask == "periods_solve":
        future_value = round(present_value * (1 + annual_rate) ** n_years, 2)
        answer = n_years
        question_text = (
            f"At {annual_rate*100:.1f}% annual effective interest, "
            f"${present_value:,.2f} grows to ${future_value:,.2f}. "
            f"How many years does this take?"
        )
        wrong_answers = [
            n_years - 1,
            n_years + 1,
            round(math.log(future_value / present_value) / annual_rate, 1),
        ]

    else:
        raise ValueError(f"Unknown ask '{ask}' for interest_tvm")

    return Problem(
        "interest_tvm", ask, question_text, answer,
        make_mc_choices(answer, wrong_answers, rng),
        params={"i": annual_rate, "n": n_years, "pv": present_value}, seed=seed,
    )


@register("interest_nominal")
def gen_interest_nominal(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    effective_annual_rate = round(float(rng.uniform(*ranges["i_range"])), 4)
    compounding_freq = int(rng.choice(ranges["m_choices"]))

    if ask == "nominal_to_effective":
        # i^(m) = m * [(1+i)^(1/m) - 1]: the nominal rate compounded m times a
        # year that is equivalent in growth to the chosen effective annual rate.
        nominal_rate = round(
            compounding_freq * ((1 + effective_annual_rate) ** (1 / compounding_freq) - 1), 6
        )
        answer = round(effective_annual_rate, 4)
        question_text = (
            f"The nominal interest rate compounded {compounding_freq}timely is "
            f"{nominal_rate*100:.4f}%. Find the annual effective interest rate."
        )
        wrong_answers = [
            round(nominal_rate, 4),
            round((1 + nominal_rate) ** compounding_freq - 1, 4),
            round(nominal_rate / compounding_freq, 4),
        ]

    elif ask == "effective_to_nominal":
        answer = round(
            compounding_freq * ((1 + effective_annual_rate) ** (1 / compounding_freq) - 1), 6
        )
        period_labels = {2: "semi-annually", 4: "quarterly", 12: "monthly", 365: "daily"}
        period_label = period_labels.get(compounding_freq, f"{compounding_freq}timely")
        question_text = (
            f"The annual effective interest rate is {effective_annual_rate*100:.2f}%. "
            f"Find the nominal rate compounded {period_label}."
        )
        wrong_answers = [
            round(effective_annual_rate * compounding_freq, 4),
            round(effective_annual_rate, 4),
            round(compounding_freq * (
                (1 + effective_annual_rate) ** (1 / (compounding_freq + 1)) - 1
            ), 4),
        ]

    elif ask == "equivalent_rate":
        # Two nominal rates are "equivalent" when they compound to the same
        # effective annual rate — convert old→effective→new rather than
        # scaling the nominal rate directly (that's the misconception this
        # ask variant is designed to catch).
        new_compounding_freq = int(
            rng.choice([freq for freq in ranges["m_choices"] if freq != compounding_freq])
        )
        old_nominal_rate = round(
            compounding_freq * ((1 + effective_annual_rate) ** (1 / compounding_freq) - 1), 6
        )
        answer = round(
            new_compounding_freq * (
                (1 + effective_annual_rate) ** (1 / new_compounding_freq) - 1
            ), 6
        )
        question_text = (
            f"The nominal rate compounded {compounding_freq}timely is "
            f"{old_nominal_rate*100:.4f}%. Find the equivalent nominal rate "
            f"compounded {new_compounding_freq}timely."
        )
        wrong_answers = [
            round(old_nominal_rate * new_compounding_freq / compounding_freq, 4),
            round(answer * 1.05, 4),
            round(effective_annual_rate, 4),
        ]

    else:
        raise ValueError(f"Unknown ask '{ask}' for interest_nominal")

    return Problem(
        "interest_nominal", ask, question_text, answer,
        make_mc_choices(answer, wrong_answers, rng),
        params={"i_eff": effective_annual_rate, "m": compounding_freq}, seed=seed,
    )


@register("interest_force")
def gen_interest_force(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    annual_rate = round(float(rng.uniform(*ranges["i_range"])), 4)
    # Force of interest δ = ln(1+i): the instantaneous (continuously
    # compounded) equivalent of the annual effective rate i.
    force_of_interest = round(math.log(1 + annual_rate), 6)
    n_years = int(rng.integers(*ranges["t_range"]))

    if ask == "force_from_rate":
        answer = round(force_of_interest, 6)
        question_text = (
            f"The annual effective interest rate is {annual_rate*100:.2f}%. "
            f"Find the force of interest delta = ln(1+i)."
        )
        wrong_answers = [
            round(annual_rate, 4),
            round(annual_rate / (1 + annual_rate), 4),
            round(math.exp(annual_rate) - 1, 6),
        ]

    elif ask == "rate_from_force":
        answer = round(annual_rate, 4)
        question_text = (
            f"The force of interest is delta = {force_of_interest:.6f}. "
            f"Find the annual effective interest rate."
        )
        wrong_answers = [
            round(force_of_interest, 4),
            round(math.exp(force_of_interest) ** 2 - 1, 4),
            round(force_of_interest + force_of_interest**2 / 2, 4),
        ]

    elif ask == "accumulation_continuous":
        present_value = round(float(rng.uniform(1000, 10000)), 2)
        answer = round(present_value * math.exp(force_of_interest * n_years), 2)
        question_text = (
            f"At force of interest delta = {force_of_interest:.6f}, find the "
            f"accumulated value of ${present_value:,.2f} after {n_years} years "
            f"under continuous compounding."
        )
        wrong_answers = [
            round(present_value * (1 + force_of_interest * n_years), 2),
            round(present_value * math.exp(force_of_interest * (n_years - 1)), 2),
            round(present_value * math.exp(force_of_interest * (n_years + 1)), 2),
        ]

    else:
        raise ValueError(f"Unknown ask '{ask}' for interest_force")

    return Problem(
        "interest_force", ask, question_text, answer,
        make_mc_choices(answer, wrong_answers, rng),
        params={"i": annual_rate, "delta": force_of_interest, "t": n_years}, seed=seed,
    )


@register("interest_discount")
def gen_interest_discount(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    annual_rate = round(float(rng.uniform(*ranges["i_range"])), 4)
    n_years = int(rng.integers(*ranges["n_range"]))
    # Standard interest/discount-rate relationship: d = i/(1+i), v = 1/(1+i) = 1-d.
    discount_rate = round(annual_rate / (1 + annual_rate), 6)
    discount_factor = round(1 / (1 + annual_rate), 6)

    if ask == "discount_from_interest":
        answer = round(discount_rate, 6)
        question_text = (
            f"The annual effective interest rate is {annual_rate*100:.2f}%. "
            f"Find the annual effective discount rate d = i/(1+i)."
        )
        # If "1 - v" happens to round to the same value as the correct answer
        # (it's mathematically identical), substitute a different wrong value
        # so make_mc_choices doesn't have to dedupe an exact answer match.
        alt_wrong = round(1 - discount_factor, 4)
        if abs(alt_wrong - answer) <= 1e-6:
            alt_wrong = round(annual_rate * 0.9, 4)
        wrong_answers = [
            round(annual_rate, 4),
            alt_wrong,
            round(annual_rate / (1 + 2 * annual_rate), 4),
        ]

    elif ask == "interest_from_discount":
        answer = round(annual_rate, 4)
        question_text = (
            f"The annual effective discount rate is d = {discount_rate*100:.4f}%. "
            f"Find the annual effective interest rate i = d/(1-d)."
        )
        wrong_answers = [
            round(discount_rate, 4),
            round(discount_rate * (1 + discount_rate), 4),
            round(annual_rate * 1.1, 4),
        ]

    elif ask == "pv_using_discount":
        future_value = round(float(rng.uniform(1000, 20000)), 2)
        answer = round(future_value * discount_factor ** n_years, 2)
        question_text = (
            f"Using discount rate d = {discount_rate*100:.4f}%, find the present "
            f"value of ${future_value:,.2f} due in {n_years} years. [Hint: v = 1-d]"
        )
        wrong_answers = [
            round(future_value * (1 - discount_rate * n_years), 2),
            round(future_value * (1 - discount_rate) ** (n_years + 1), 2),
            round(future_value * (1 - discount_rate) ** (n_years - 1), 2),
        ]

    else:
        raise ValueError(f"Unknown ask '{ask}' for interest_discount")

    return Problem(
        "interest_discount", ask, question_text, answer,
        make_mc_choices(answer, wrong_answers, rng),
        params={
            "i": annual_rate, "d": discount_rate, "v": discount_factor, "n": n_years,
        },
        seed=seed,
    )
