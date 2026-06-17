"""Generators for loan amortization, interest/principal breakdown, sinking fund."""
from __future__ import annotations

import numpy as np

from engine.generation.base import Problem, make_mc_choices, register


def _a_n(i: float, n: int) -> float:
    """Present value annuity-immediate factor a_n|i = (1 - v^n)/i.

    NOTE: divides by i, so a rate range that includes 0 would raise
    ZeroDivisionError here.
    """
    discount_factor = 1 / (1 + i)
    return (1 - discount_factor ** n) / i


def _s_n(i: float, n: int) -> float:
    """Future value annuity-immediate factor s_n|i = ((1+i)^n - 1)/i. Same i=0 caveat as _a_n."""
    return ((1 + i) ** n - 1) / i


@register("loan_amort")
def gen_loan_amort(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    annual_rate = round(float(rng.uniform(*ranges["i_range"])), 4)
    n_years = int(rng.integers(*ranges["n_range"]))
    loan_amount = round(float(rng.uniform(*ranges["loan_range"])), 2)
    pv_factor = _a_n(annual_rate, n_years)
    payment = round(loan_amount / pv_factor, 2)

    # "t" (the payment number being asked about) is only defined for the
    # outstanding-balance variants below; payment_t stays 0 for payment_amount.
    payment_t = 0

    if ask == "payment_amount":
        answer = payment
        question_text = (
            f"A loan of ${loan_amount:,.2f} is repaid with level annual payments at the "
            f"end of each year for {n_years} years at {annual_rate*100:.2f}% effective. "
            f"Find the annual payment amount."
        )
        wrong_answers = [
            round(loan_amount / n_years, 2),
            round(loan_amount * annual_rate + loan_amount / n_years, 2),
            round(loan_amount / _a_n(annual_rate, n_years + 1), 2),
        ]

    elif ask == "outstanding_prospective":
        # rng.integers(1, n_years) excludes n_years itself, so remaining_years
        # is always >= 1 (a balance "after the final payment" isn't asked here).
        payment_t = int(rng.integers(1, n_years))
        remaining_years = n_years - payment_t
        answer = round(payment * _a_n(annual_rate, remaining_years), 2)
        question_text = (
            f"A loan of ${loan_amount:,.2f} at {annual_rate*100:.2f}% is repaid over "
            f"{n_years} years with level annual payments of ${payment:,.2f}. "
            f"Find the outstanding balance immediately after the {payment_t}th payment "
            f"(prospective method: PV of remaining payments)."
        )
        wrong_answers = [
            round(loan_amount * (1 + annual_rate) ** payment_t
                  - payment * _s_n(annual_rate, payment_t), 2),
            round(payment * _a_n(annual_rate, remaining_years + 1), 2),
            round(payment * _a_n(annual_rate, remaining_years - 1), 2),
        ]

    elif ask == "outstanding_retrospective":
        payment_t = int(rng.integers(1, n_years))
        answer = round(loan_amount * (1 + annual_rate) ** payment_t
                        - payment * _s_n(annual_rate, payment_t), 2)
        question_text = (
            f"A loan of ${loan_amount:,.2f} at {annual_rate*100:.2f}% is repaid over "
            f"{n_years} years with level annual payments of ${payment:,.2f}. "
            f"Find the outstanding balance after {payment_t} payments "
            f"(retrospective method: accumulated loan minus accumulated payments)."
        )
        wrong_answers = [
            round(payment * _a_n(annual_rate, n_years - payment_t), 2),
            round(loan_amount * (1 + annual_rate) ** payment_t
                  + payment * _s_n(annual_rate, payment_t), 2),
            round(loan_amount * (1 + annual_rate) ** payment_t
                  - payment * _s_n(annual_rate, payment_t - 1), 2),
        ]

    else:
        raise ValueError(f"Unknown ask '{ask}' for loan_amort")

    return Problem(
        "loan_amort", ask, question_text, answer,
        make_mc_choices(answer, wrong_answers, rng),
        params={
            "i": annual_rate, "n": n_years, "loan": loan_amount,
            "payment": payment, "t": payment_t,
        },
        seed=seed,
    )


@register("loan_split")
def gen_loan_split(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    annual_rate = round(float(rng.uniform(*ranges["i_range"])), 4)
    n_years = int(rng.integers(*ranges["n_range"]))
    loan_amount = round(float(rng.uniform(*ranges["loan_range"])), 2)
    pv_factor = _a_n(annual_rate, n_years)
    payment = round(loan_amount / pv_factor, 2)
    discount_factor = 1 / (1 + annual_rate)
    payment_t = int(rng.integers(1, n_years + 1))

    if ask == "interest_tth_payment":
        # I_t = PMT·(1 - v^(n-t+1)): the interest owed on the balance just
        # before payment t, expressed directly in terms of remaining payments
        # so we never have to compute the outstanding balance itself.
        interest_portion = round(payment * (1 - discount_factor ** (n_years - payment_t + 1)), 2)
        answer = interest_portion
        question_text = (
            f"A loan of ${loan_amount:,.2f} at {annual_rate*100:.2f}% is amortized "
            f"over {n_years} years with level annual payments of ${payment:,.2f}. "
            f"Find the interest portion of the {payment_t}th payment."
        )
        wrong_answers = [
            round(loan_amount * annual_rate * discount_factor ** (payment_t - 1), 2),
            round(payment * (1 - discount_factor ** (n_years - payment_t)), 2),
            round(payment * (1 - discount_factor ** (n_years - payment_t + 2)), 2),
        ]

    elif ask == "principal_tth_payment":
        # P_t = PMT·v^(n-t+1): the principal repaid grows geometrically as the
        # remaining-term exponent shrinks each period.
        principal_portion = round(payment * discount_factor ** (n_years - payment_t + 1), 2)
        answer = principal_portion
        question_text = (
            f"A loan of ${loan_amount:,.2f} at {annual_rate*100:.2f}% is amortized "
            f"over {n_years} years with level annual payments of ${payment:,.2f}. "
            f"Find the principal repaid in the {payment_t}th payment."
        )
        wrong_answers = [
            round(loan_amount / n_years, 2),
            round(payment * discount_factor ** (n_years - payment_t), 2),
            round(payment * discount_factor ** (n_years - payment_t + 2), 2),
        ]

    elif ask == "total_interest_paid":
        answer = round(n_years * payment - loan_amount, 2)
        question_text = (
            f"A loan of ${loan_amount:,.2f} at {annual_rate*100:.2f}% is amortized "
            f"over {n_years} years with level annual payments of ${payment:,.2f}. "
            f"Find the total interest paid over the life of the loan."
        )
        wrong_answers = [
            round(loan_amount * annual_rate * n_years, 2),
            round(n_years * payment, 2),
            round((n_years - 1) * payment - loan_amount, 2),
        ]

    else:
        raise ValueError(f"Unknown ask '{ask}' for loan_split")

    return Problem(
        "loan_split", ask, question_text, answer,
        make_mc_choices(answer, wrong_answers, rng),
        params={
            "i": annual_rate, "n": n_years, "loan": loan_amount,
            "payment": payment, "t": payment_t,
        },
        seed=seed,
    )


@register("sinking_fund")
def gen_sinking_fund(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    loan_rate = round(float(rng.uniform(*ranges["j_range"])), 4)
    fund_rate = round(float(rng.uniform(*ranges["i_range"])), 4)
    n_years = int(rng.integers(*ranges["n_range"]))
    loan_amount = round(float(rng.uniform(*ranges["loan_range"])), 2)
    # NOTE: fund_rate == 0 would raise ZeroDivisionError here — callers must
    # supply an i_range that excludes 0.
    fund_fv_factor = ((1 + fund_rate) ** n_years - 1) / fund_rate
    annual_deposit = round(loan_amount / fund_fv_factor, 2)
    annual_interest_payment = round(loan_amount * loan_rate, 2)

    if ask == "sinking_fund_deposit":
        answer = annual_deposit
        question_text = (
            f"A borrower takes a ${loan_amount:,.2f} loan at {loan_rate*100:.2f}% "
            f"annual interest (interest-only), repaying the principal via a "
            f"sinking fund earning {fund_rate*100:.2f}% over {n_years} years. "
            f"Find the annual sinking fund deposit."
        )
        wrong_answers = [
            round(loan_amount / _a_n(fund_rate, n_years), 2),
            round(loan_amount / fund_fv_factor * (1 + fund_rate), 2),
            round(loan_amount / (n_years * (1 + fund_rate) ** (n_years / 2)), 2),
        ]

    elif ask == "total_periodic_outlay":
        answer = round(annual_deposit + annual_interest_payment, 2)
        question_text = (
            f"A borrower takes a ${loan_amount:,.2f} loan at {loan_rate*100:.2f}% "
            f"annual interest (interest-only), repaying principal via a sinking "
            f"fund earning {fund_rate*100:.2f}% over {n_years} years. "
            f"Find the total annual outlay (interest + sinking fund deposit)."
        )
        wrong_answers = [
            round(annual_deposit, 2),
            round(annual_interest_payment, 2),
            round(loan_amount / _a_n(loan_rate, n_years), 2),
        ]

    else:
        raise ValueError(f"Unknown ask '{ask}' for sinking_fund")

    return Problem(
        "sinking_fund", ask, question_text, answer,
        make_mc_choices(answer, wrong_answers, rng),
        params={
            "i_loan": loan_rate, "i_fund": fund_rate, "n": n_years,
            "loan": loan_amount, "deposit": annual_deposit,
        },
        seed=seed,
    )
