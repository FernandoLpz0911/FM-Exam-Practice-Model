"""Generators for loan amortization, interest/principal breakdown, sinking fund."""
from __future__ import annotations

import numpy as np

from engine.generation.base import Problem, make_mc_choices, register


def _a_n(i: float, n: int) -> float:
    v = 1 / (1 + i)
    return (1 - v ** n) / i


def _s_n(i: float, n: int) -> float:
    return ((1 + i) ** n - 1) / i


@register("loan_amort")
def gen_loan_amort(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    i = round(float(rng.uniform(*ranges["i_range"])), 4)
    n = int(rng.integers(*ranges["n_range"]))
    loan = round(float(rng.uniform(*ranges["loan_range"])), 2)
    an = _a_n(i, n)
    payment = round(loan / an, 2)
    v = 1 / (1 + i)

    if ask == "payment_amount":
        ans = payment
        stmt = (f"A loan of ${loan:,.2f} is repaid with level annual payments at the "
                f"end of each year for {n} years at {i*100:.2f}% effective. "
                f"Find the annual payment amount.")
        wrongs = [round(loan / n, 2),
                  round(loan * i + loan / n, 2),
                  round(loan / _a_n(i, n + 1), 2)]

    elif ask == "outstanding_prospective":
        t = int(rng.integers(1, n))
        remaining = n - t
        ans = round(payment * _a_n(i, remaining), 2)
        stmt = (f"A loan of ${loan:,.2f} at {i*100:.2f}% is repaid over {n} years with "
                f"level annual payments of ${payment:,.2f}. "
                f"Find the outstanding balance immediately after the {t}th payment "
                f"(prospective method: PV of remaining payments).")
        wrongs = [round(loan * (1 + i) ** t - payment * _s_n(i, t), 2),
                  round(payment * _a_n(i, remaining + 1), 2),
                  round(payment * _a_n(i, remaining - 1), 2)]

    elif ask == "outstanding_retrospective":
        t = int(rng.integers(1, n))
        ans = round(loan * (1 + i) ** t - payment * _s_n(i, t), 2)
        stmt = (f"A loan of ${loan:,.2f} at {i*100:.2f}% is repaid over {n} years with "
                f"level annual payments of ${payment:,.2f}. "
                f"Find the outstanding balance after {t} payments "
                f"(retrospective method: accumulated loan minus accumulated payments).")
        wrongs = [round(payment * _a_n(i, n - t), 2),
                  round(loan * (1 + i) ** t + payment * _s_n(i, t), 2),
                  round(loan * (1 + i) ** t - payment * _s_n(i, t - 1), 2)]

    else:
        raise ValueError(f"Unknown ask '{ask}' for loan_amort")

    # Store t so solver can reproduce the step-by-step work
    t_stored = locals().get("t", 0)
    return Problem("loan_amort", ask, stmt, ans,
                   make_mc_choices(ans, wrongs, rng),
                   params={"i": i, "n": n, "loan": loan, "payment": payment,
                           "t": t_stored}, seed=seed)


@register("loan_split")
def gen_loan_split(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    i = round(float(rng.uniform(*ranges["i_range"])), 4)
    n = int(rng.integers(*ranges["n_range"]))
    loan = round(float(rng.uniform(*ranges["loan_range"])), 2)
    an = _a_n(i, n)
    payment = round(loan / an, 2)
    v = 1 / (1 + i)
    t = int(rng.integers(1, n + 1))

    if ask == "interest_tth_payment":
        # I_t = payment * (1 - v^{n-t+1})
        I_t = round(payment * (1 - v ** (n - t + 1)), 2)
        ans = I_t
        stmt = (f"A loan of ${loan:,.2f} at {i*100:.2f}% is amortized over {n} years "
                f"with level annual payments of ${payment:,.2f}. "
                f"Find the interest portion of the {t}th payment.")
        wrongs = [round(loan * i * v ** (t - 1), 2),
                  round(payment * (1 - v ** (n - t)), 2),
                  round(payment * (1 - v ** (n - t + 2)), 2)]

    elif ask == "principal_tth_payment":
        # PR_t = payment * v^{n-t+1}
        PR_t = round(payment * v ** (n - t + 1), 2)
        ans = PR_t
        stmt = (f"A loan of ${loan:,.2f} at {i*100:.2f}% is amortized over {n} years "
                f"with level annual payments of ${payment:,.2f}. "
                f"Find the principal repaid in the {t}th payment.")
        wrongs = [round(loan / n, 2),
                  round(payment * v ** (n - t), 2),
                  round(payment * v ** (n - t + 2), 2)]

    elif ask == "total_interest_paid":
        ans = round(n * payment - loan, 2)
        stmt = (f"A loan of ${loan:,.2f} at {i*100:.2f}% is amortized over {n} years "
                f"with level annual payments of ${payment:,.2f}. "
                f"Find the total interest paid over the life of the loan.")
        wrongs = [round(loan * i * n, 2),
                  round(n * payment, 2),
                  round((n - 1) * payment - loan, 2)]

    else:
        raise ValueError(f"Unknown ask '{ask}' for loan_split")

    return Problem("loan_split", ask, stmt, ans,
                   make_mc_choices(ans, wrongs, rng),
                   params={"i": i, "n": n, "loan": loan, "payment": payment, "t": t}, seed=seed)


@register("sinking_fund")
def gen_sinking_fund(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    i_loan = round(float(rng.uniform(*ranges["j_range"])), 4)  # loan interest rate
    i_fund = round(float(rng.uniform(*ranges["i_range"])), 4)  # sinking fund rate
    n = int(rng.integers(*ranges["n_range"]))
    loan = round(float(rng.uniform(*ranges["loan_range"])), 2)
    s_n = ((1 + i_fund) ** n - 1) / i_fund
    deposit = round(loan / s_n, 2)  # annual sinking fund deposit
    interest_payment = round(loan * i_loan, 2)  # annual interest on loan

    if ask == "sinking_fund_deposit":
        ans = deposit
        stmt = (f"A borrower takes a ${loan:,.2f} loan at {i_loan*100:.2f}% annual interest "
                f"(interest-only), repaying the principal via a sinking fund earning {i_fund*100:.2f}% "
                f"over {n} years. Find the annual sinking fund deposit.")
        wrongs = [round(loan / _a_n(i_fund, n), 2),
                  round(loan / s_n * (1 + i_fund), 2),
                  round(loan / (n * (1 + i_fund) ** (n / 2)), 2)]

    elif ask == "total_periodic_outlay":
        ans = round(deposit + interest_payment, 2)
        stmt = (f"A borrower takes a ${loan:,.2f} loan at {i_loan*100:.2f}% annual interest "
                f"(interest-only), repaying principal via a sinking fund earning {i_fund*100:.2f}% "
                f"over {n} years. Find the total annual outlay (interest + sinking fund deposit).")
        wrongs = [round(deposit, 2),
                  round(interest_payment, 2),
                  round(loan / _a_n(i_loan, n), 2)]

    else:
        raise ValueError(f"Unknown ask '{ask}' for sinking_fund")

    return Problem("sinking_fund", ask, stmt, ans,
                   make_mc_choices(ans, wrongs, rng),
                   params={"i_loan": i_loan, "i_fund": i_fund, "n": n, "loan": loan,
                           "deposit": deposit}, seed=seed)
