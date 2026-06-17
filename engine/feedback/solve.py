"""
Worked-solution generator — same math as generators, step-by-step solutions.

Entry point: solve(kind, ask, params) → Solved
Each solver mirrors its gen_* counterpart's arithmetic exactly.
Param keys must match what the generator stores in Problem.params.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Solved:
    answer: float
    steps: list[str]
    intermediates: dict = field(default_factory=dict)


_solvers: dict[str, object] = {}


def _reg(kind: str):
    def decorator(fn):
        _solvers[kind] = fn
        return fn
    return decorator


def solve(kind: str, ask: str, params: dict) -> Solved:
    """Dispatch to the right solver. Returns Solved with answer, steps, intermediates."""
    fn = _solvers.get(kind)
    if fn is None:
        return Solved(float("nan"), [f"No worked solution available for '{kind}'."], {})
    return fn(ask, params)


def _unknown(ask: str) -> Solved:
    return Solved(float("nan"), [f"Unknown ask variant: '{ask}'."], {})


def _a_n(i: float, n: int) -> float:
    v = 1 / (1 + i)
    return (1 - v ** n) / i


def _s_n(i: float, n: int) -> float:
    return ((1 + i) ** n - 1) / i


@_reg("interest_tvm")
def _s_interest_tvm(ask: str, params: dict) -> Solved:
    annual_rate = params["i"]
    n_years = params["n"]
    present_value = params["pv"]
    future_value = round(present_value * (1 + annual_rate) ** n_years, 2)

    if ask == "future_value":
        accumulation_factor = round((1 + annual_rate) ** n_years, 6)
        return Solved(future_value, [
            f"Accumulation factor: (1+i)^n = (1+{annual_rate})^{n_years} = {accumulation_factor}",
            f"FV = PV · (1+i)^n = {present_value} × {accumulation_factor} = {future_value}",
        ], {"accumulation_factor": accumulation_factor})

    if ask == "present_value":
        discount_factor = round(1 / (1 + annual_rate) ** n_years, 6)
        return Solved(present_value, [
            f"Discount factor: v^n = 1/(1+i)^n = 1/(1+{annual_rate})^{n_years} = {discount_factor}",
            f"PV = FV · v^n = {future_value} × {discount_factor} = {present_value}",
        ], {"discount_factor": discount_factor})

    if ask == "interest_rate_solve":
        # Solve (1+i)^n = FV/PV for i by taking the nth root of both sides.
        growth_ratio = round(future_value / present_value, 6)
        rate_answer = round(growth_ratio ** (1 / n_years) - 1, 6)
        nth_root = round(growth_ratio ** (1 / n_years), 6)
        return Solved(rate_answer, [
            f"FV = PV·(1+i)^n  ⟹  (1+i)^n = FV/PV = {future_value}/{present_value} "
            f"= {growth_ratio}",
            f"1+i = (FV/PV)^(1/n) = {growth_ratio}^(1/{n_years}) = {nth_root}",
            f"i = {nth_root} - 1 = {rate_answer}",
        ], {"ratio": growth_ratio})

    if ask == "periods_solve":
        # Solve (1+i)^n = FV/PV for n by taking logs of both sides.
        years_answer = round(
            math.log(future_value / present_value) / math.log(1 + annual_rate), 4
        )
        return Solved(n_years, [
            "FV = PV·(1+i)^n  ⟹  n = ln(FV/PV) / ln(1+i)",
            f"n = ln({future_value}/{present_value}) / ln(1+{annual_rate})",
            f"  = {round(math.log(future_value/present_value),6)} / "
            f"{round(math.log(1+annual_rate),6)}",
            f"  = {years_answer}  (rounds to {n_years} years)",
        ], {})

    return _unknown(ask)


@_reg("interest_nominal")
def _s_interest_nominal(ask: str, params: dict) -> Solved:
    # Params stored: {"i_eff": i_eff, "m": m}  (all three ask variants)
    effective_annual_rate = params["i_eff"]
    compounding_freq = params["m"]

    if ask == "nominal_to_effective":
        nominal_rate = round(
            compounding_freq * ((1 + effective_annual_rate) ** (1 / compounding_freq) - 1), 6
        )
        periodic_rate = round(nominal_rate / compounding_freq, 6)
        return Solved(effective_annual_rate, [
            f"Given nominal rate i^({compounding_freq}) = {nominal_rate*100:.4f}%, "
            f"compounded {compounding_freq}x/year.",
            "Effective annual rate: (1 + i^(m)/m)^m - 1",
            f"= (1 + {periodic_rate})^{compounding_freq} - 1",
            f"= {round((1+periodic_rate)**compounding_freq,6)} - 1 = {effective_annual_rate}",
        ], {"i_nom": nominal_rate, "periodic_rate": periodic_rate})

    if ask == "effective_to_nominal":
        nominal_rate = round(
            compounding_freq * ((1 + effective_annual_rate) ** (1 / compounding_freq) - 1), 6
        )
        return Solved(nominal_rate, [
            f"Effective annual rate i = {effective_annual_rate}, "
            f"want nominal compounded {compounding_freq}x/year.",
            f"i^({compounding_freq}) = m·[(1+i)^(1/m) - 1]",
            f"= {compounding_freq}·[(1+{effective_annual_rate})^(1/{compounding_freq}) - 1]",
            f"= {compounding_freq}·"
            f"[{round((1+effective_annual_rate)**(1/compounding_freq),6)} - 1]",
            f"= {compounding_freq} × "
            f"{round((1+effective_annual_rate)**(1/compounding_freq)-1,6)} = {nominal_rate}",
        ], {})

    if ask == "equivalent_rate":
        # Bridge through the effective annual rate rather than scaling the
        # nominal rate by the frequency ratio directly — that shortcut is the
        # exact misconception this ask variant is meant to surface.
        old_freq = params.get("m_old", compounding_freq)
        new_freq = params.get("m_new", compounding_freq)
        old_nominal_rate = params.get(
            "i_nom_old",
            round(compounding_freq * ((1 + effective_annual_rate)**(1/compounding_freq)-1), 6),
        )
        new_nominal_rate = round(
            new_freq * ((1 + effective_annual_rate) ** (1 / new_freq) - 1), 6
        )
        return Solved(new_nominal_rate, [
            f"Step 1 — convert i^({old_freq})={old_nominal_rate} to effective: "
            f"{effective_annual_rate}",
            f"Step 2 — convert effective to i^({new_freq}): "
            f"{new_freq}·[(1+{effective_annual_rate})^(1/{new_freq})-1] = {new_nominal_rate}",
        ], {"i_eff": effective_annual_rate})

    return _unknown(ask)


@_reg("interest_force")
def _s_interest_force(ask: str, params: dict) -> Solved:
    # Params stored: {"i": i, "delta": delta, "t": t}
    if ask == "force_from_rate":
        annual_rate = params["i"]
        force_of_interest = round(math.log(1 + annual_rate), 6)
        return Solved(force_of_interest, [
            f"Force of interest: δ = ln(1+i) = ln(1+{annual_rate})",
            f"= ln({round(1+annual_rate,6)}) = {force_of_interest}",
        ], {})

    if ask == "rate_from_force":
        force_of_interest = params["delta"]
        rate_answer = round(math.exp(force_of_interest) - 1, 6)
        return Solved(rate_answer, [
            f"Effective rate from force: i = e^δ - 1 = e^{force_of_interest} - 1",
            f"= {round(math.exp(force_of_interest),6)} - 1 = {rate_answer}",
        ], {})

    if ask == "accumulation_continuous":
        present_value = params.get("pv", params.get("PV", 1000))
        force_of_interest = params["delta"]
        n_years = params["t"]
        future_value = round(present_value * math.exp(force_of_interest * n_years), 2)
        exponent = round(force_of_interest * n_years, 6)
        return Solved(future_value, [
            "Continuous accumulation: FV = PV · e^(δt)",
            f"= {present_value} · e^({force_of_interest}×{n_years})",
            f"= {present_value} · e^{exponent}",
            f"= {present_value} × {round(math.exp(exponent),6)} = {future_value}",
        ], {"exponent": exponent})

    return _unknown(ask)


@_reg("interest_discount")
def _s_interest_discount(ask: str, params: dict) -> Solved:
    # Params stored: {"i": i, "d": d, "v": v, "n": n}
    if ask == "discount_from_interest":
        annual_rate = params["i"]
        discount_rate = round(annual_rate / (1 + annual_rate), 6)
        return Solved(discount_rate, [
            f"Discount rate: d = i/(1+i) = {annual_rate}/(1+{annual_rate})",
            f"= {annual_rate}/{round(1+annual_rate,6)} = {discount_rate}",
        ], {})

    if ask == "interest_from_discount":
        discount_rate = params["d"]
        rate_answer = round(discount_rate / (1 - discount_rate), 6)
        return Solved(rate_answer, [
            f"Effective rate: i = d/(1-d) = {discount_rate}/(1-{discount_rate})",
            f"= {discount_rate}/{round(1-discount_rate,6)} = {rate_answer}",
        ], {})

    if ask == "pv_using_discount":
        discount_rate = params["d"]
        n_years = params["n"]
        future_value = params.get("fv", params.get("FV", 1000))
        discount_factor = round((1 - discount_rate) ** n_years, 6)
        present_value = round(future_value * discount_factor, 2)
        return Solved(present_value, [
            "PV using discount rate: PV = FV·(1-d)^n",
            f"= {future_value}·(1-{discount_rate})^{n_years}",
            f"= {future_value}·{discount_factor}",
            f"= {present_value}",
        ], {"discount_factor": discount_factor})

    return _unknown(ask)


@_reg("annuity_immediate")
def _s_annuity_immediate(ask: str, params: dict) -> Solved:
    annual_rate = params["i"]
    n_years = params["n"]
    payment = params["payment"]
    discount_factor = round(1 / (1 + annual_rate), 6)
    pv_factor = round(_a_n(annual_rate, n_years), 6)
    fv_factor = round(_s_n(annual_rate, n_years), 6)

    if ask == "pv_annuity_imm":
        present_value = round(payment * pv_factor, 2)
        return Solved(present_value, [
            f"Annuity-immediate PV factor: a_{n_years}| = (1 - v^n)/i",
            f"v = 1/(1+{annual_rate}) = {discount_factor}",
            f"a_{n_years}| = (1 - {discount_factor}^{n_years}) / {annual_rate} "
            f"= (1 - {round(discount_factor**n_years,6)}) / {annual_rate} = {pv_factor}",
            f"PV = PMT · a_{n_years}| = {payment} × {pv_factor} = {present_value}",
        ], {"a_n": pv_factor, "v": discount_factor})

    if ask == "fv_annuity_imm":
        future_value = round(payment * fv_factor, 2)
        return Solved(future_value, [
            f"Annuity-immediate FV factor: s_{n_years}| = ((1+i)^n - 1)/i",
            f"s_{n_years}| = ({round(1+annual_rate,4)}^{n_years} - 1)/{annual_rate} "
            f"= ({round((1+annual_rate)**n_years,6)} - 1)/{annual_rate} = {fv_factor}",
            f"FV = PMT · s_{n_years}| = {payment} × {fv_factor} = {future_value}",
        ], {"s_n": fv_factor})

    if ask == "payment_from_pv":
        loan_amount = round(payment * pv_factor, 2)   # recompute for display
        payment_answer = round(loan_amount / pv_factor, 2)
        return Solved(payment_answer, [
            f"PV = PMT · a_{n_years}|  ⟹  PMT = PV / a_{n_years}|",
            f"PV = {payment} × {pv_factor} = {loan_amount},  a_{n_years}| = {pv_factor}",
            f"PMT = {loan_amount} / {pv_factor} = {payment_answer}",
        ], {"a_n": pv_factor})

    if ask == "n_from_pv_imm":
        loan_amount = round(payment * pv_factor, 2)
        # Rearranged a_n| = (1-v^n)/i to isolate v^n, then solve for n via logs.
        remaining_factor = 1 - loan_amount * annual_rate / payment
        years_answer = round(-math.log(remaining_factor) / math.log(1 + annual_rate), 4)
        return Solved(n_years, [
            f"PV = PMT · a_n|  ⟹  a_n| = PV/PMT = {loan_amount}/{payment} "
            f"= {round(loan_amount/payment,6)}",
            f"(1 - v^n)/i = {round(loan_amount/payment,6)}  ⟹  "
            f"v^n = 1 - {round(loan_amount/payment,6)}·{annual_rate} = {round(remaining_factor,6)}",
            f"n = -ln({round(remaining_factor,6)}) / ln(1+{annual_rate}) = {years_answer}",
        ], {"target_a_n": round(loan_amount / payment, 6)})

    return _unknown(ask)


@_reg("annuity_due")
def _s_annuity_due(ask: str, params: dict) -> Solved:
    annual_rate = params["i"]
    n_years = params["n"]
    payment = params["payment"]
    pv_factor = _a_n(annual_rate, n_years)
    pv_factor_due = round((1 + annual_rate) * pv_factor, 6)
    fv_factor = _s_n(annual_rate, n_years)
    fv_factor_due = round((1 + annual_rate) * fv_factor, 6)

    if ask == "pv_annuity_due":
        present_value = round(payment * pv_factor_due, 2)
        return Solved(present_value, [
            "Annuity-due: payments at start of each period.",
            f"ä_{n_years}| = (1+i)·a_{n_years}| = (1+{annual_rate})·{round(pv_factor,6)} "
            f"= {pv_factor_due}",
            f"PV = PMT · ä_{n_years}| = {payment} × {pv_factor_due} = {present_value}",
        ], {"a_due": pv_factor_due, "a_n": round(pv_factor, 6)})

    if ask == "fv_annuity_due":
        future_value = round(payment * fv_factor_due, 2)
        return Solved(future_value, [
            f"FV factor for annuity-due: s̈_{n_years}| = (1+i)·s_{n_years}|",
            f"s_{n_years}| = {round(fv_factor,6)},  "
            f"s̈_{n_years}| = (1+{annual_rate})·{round(fv_factor,6)} = {fv_factor_due}",
            f"FV = PMT · s̈_{n_years}| = {payment} × {fv_factor_due} = {future_value}",
        ], {"s_due": fv_factor_due})

    if ask == "payment_from_pv_due":
        present_value = round(payment * pv_factor_due, 2)
        payment_answer = round(present_value / pv_factor_due, 2)
        return Solved(payment_answer, [
            f"PV = PMT · ä_{n_years}|  ⟹  PMT = PV / ä_{n_years}|",
            f"PV = {payment} × {pv_factor_due} = {present_value},  "
            f"ä_{n_years}| = {pv_factor_due}",
            f"PMT = {present_value} / {pv_factor_due} = {payment_answer}",
        ], {"a_due": pv_factor_due})

    return _unknown(ask)


@_reg("perpetuity")
def _s_perpetuity(ask: str, params: dict) -> Solved:
    # Params stored: {"i": i, "payment": payment, "d": d}
    annual_rate = params["i"]
    payment = params["payment"]

    if ask == "pv_perp_imm":
        present_value = round(payment / annual_rate, 2)
        return Solved(present_value, [
            "Perpetuity-immediate PV: a_∞| = 1/i",
            f"PV = PMT/i = {payment}/{annual_rate} = {present_value}",
        ], {})

    if ask == "pv_perp_due":
        present_value = round(payment * (1 + annual_rate) / annual_rate, 2)
        return Solved(present_value, [
            "Perpetuity-due PV: ä_∞| = (1+i)/i",
            f"PV = PMT·(1+i)/i = {payment}·(1+{annual_rate})/{annual_rate} "
            f"= {payment}·{round((1+annual_rate)/annual_rate,6)} = {present_value}",
        ], {"a_due_inf": round((1 + annual_rate) / annual_rate, 6)})

    if ask == "payment_from_perp_pv":
        present_value = round(payment / annual_rate, 2)   # actual PV from generator math
        payment_answer = round(present_value * annual_rate, 2)
        return Solved(payment_answer, [
            "PV = PMT/i  ⟹  PMT = PV·i",
            f"PV = {payment}/{annual_rate} = {present_value}",
            f"PMT = {present_value} × {annual_rate} = {payment_answer}",
        ], {})

    return _unknown(ask)


@_reg("deferred_annuity")
def _s_deferred_annuity(ask: str, params: dict) -> Solved:
    # Params stored: {"i": i, "n": n, "m": m, "payment": payment}
    annual_rate = params["i"]
    n_years = params["n"]
    deferral_years = params["m"]
    payment = params["payment"]
    deferral_discount = round(1 / (1 + annual_rate) ** deferral_years, 6)
    pv_factor = round(_a_n(annual_rate, n_years), 6)

    if ask == "pv_deferred_imm":
        present_value = round(payment * deferral_discount * pv_factor, 2)
        return Solved(present_value, [
            f"Deferred annuity-immediate: deferred {deferral_years} periods, "
            f"then {n_years} payments.",
            "PV = PMT · v^m · a_n|",
            f"v^m = 1/(1+{annual_rate})^{deferral_years} = {deferral_discount}",
            f"a_{n_years}| = {pv_factor}",
            f"PV = {payment} × {deferral_discount} × {pv_factor} = {present_value}",
        ], {"v_m": deferral_discount, "a_n": pv_factor})

    if ask == "pv_deferred_due":
        pv_factor_due = round((1 + annual_rate) * pv_factor, 6)
        present_value = round(payment * deferral_discount * pv_factor_due, 2)
        return Solved(present_value, [
            f"Deferred annuity-due: deferred {deferral_years} periods, "
            f"then {n_years} payments at start of period.",
            f"ä_{n_years}| = (1+{annual_rate}) · a_{n_years}| = (1+{annual_rate}) × "
            f"{pv_factor} = {pv_factor_due}",
            f"v^m = {deferral_discount}",
            f"PV = {payment} × {deferral_discount} × {pv_factor_due} = {present_value}",
        ], {"v_m": deferral_discount, "a_due": pv_factor_due})

    return _unknown(ask)


@_reg("annuity_varying")
def _s_annuity_varying(ask: str, params: dict) -> Solved:
    # Params stored: {"i": i, "n": n, "base": base, "type": "inc"/"dec"/"geo"}
    annual_rate = params["i"]
    n_years = params["n"]
    base_payment = params.get("base", params.get("base_payment", 100))

    if ask == "pv_arithmetic_inc":
        pv_factor = round(_a_n(annual_rate, n_years), 6)
        discount_factor = 1 / (1 + annual_rate)
        pv_factor_due = round((1 + annual_rate) * pv_factor, 6)
        n_times_discount = round(n_years * discount_factor**n_years, 6)
        increasing_factor = round((pv_factor_due - n_times_discount) / annual_rate, 6)
        present_value = round(
            base_payment * pv_factor + base_payment * increasing_factor, 2
        )
        return Solved(present_value, [
            f"Increasing annuity: payment at time t = {base_payment} + (t-1)·{base_payment}.",
            f"PV = base·a_{n_years}| + base·(Iä)_{n_years}|/i",
            f"a_{n_years}| = {pv_factor},  ä_{n_years}| = {pv_factor_due}",
            f"n·v^n = {n_years}·{round(discount_factor**n_years,6)} = {n_times_discount}",
            f"(Iä)_{n_years}| = ({pv_factor_due} - {n_times_discount})/{annual_rate} "
            f"= {increasing_factor}",
            f"PV = {base_payment}·{pv_factor} + {base_payment}·{increasing_factor} "
            f"= {present_value}",
        ], {"a_n": pv_factor, "ia_term": increasing_factor})

    if ask == "pv_arithmetic_dec":
        pv_factor = round(_a_n(annual_rate, n_years), 6)
        discount_factor = 1 / (1 + annual_rate)
        pv_factor_due = round((1 + annual_rate) * pv_factor, 6)
        increasing_factor = round(
            (pv_factor_due - n_years * discount_factor**n_years) / annual_rate, 6
        )
        present_value = round((base_payment + 1) * pv_factor - increasing_factor, 2)
        return Solved(present_value, [
            f"Decreasing annuity: payment at time t = {base_payment} - (t-1)·1.",
            f"PV = (base+1)·a_{n_years}| - (Ia)_{n_years}|",
            f"a_{n_years}| = {pv_factor},  (Ia)_{n_years}| = {increasing_factor}",
            f"PV = {base_payment+1}·{pv_factor} - {increasing_factor} = {present_value}",
        ], {"a_n": pv_factor, "ia": increasing_factor})

    if ask == "pv_geometric":
        growth_rate = params.get("g", 0.03)
        # A geometric annuity at rate i with growth g is equivalent to a level
        # annuity at the adjusted rate j = (i-g)/(1+g) — but j is undefined
        # (division by zero in a_n|_j) when i == g, hence the separate branch.
        equivalent_rate = round((annual_rate - growth_rate) / (1 + growth_rate), 6)
        if abs(annual_rate - growth_rate) < 1e-9:
            present_value = round(base_payment * n_years / (1 + annual_rate), 2)
            return Solved(present_value, [
                f"Geometric annuity with g = i = {annual_rate}: special case.",
                f"PV = PMT·n/(1+i) = {base_payment}·{n_years}/(1+{annual_rate}) "
                f"= {present_value}",
            ], {})
        pv_factor_at_j = round(_a_n(equivalent_rate, n_years), 6)
        present_value = round(base_payment * pv_factor_at_j / (1 + growth_rate), 2)
        return Solved(present_value, [
            f"Geometric annuity with g={growth_rate}: "
            f"equivalent rate j = (i-g)/(1+g) = {equivalent_rate}",
            f"a_{n_years}|_j = {pv_factor_at_j}",
            f"PV = PMT·a_{n_years}|_j / (1+g) = {base_payment}·{pv_factor_at_j}/"
            f"{round(1+growth_rate,4)} = {present_value}",
        ], {"j": equivalent_rate})

    return _unknown(ask)


@_reg("loan_amort")
def _s_loan_amort(ask: str, params: dict) -> Solved:
    # Params stored: {"i": i, "n": n, "loan": loan, "payment": payment, "t": t}
    annual_rate = params["i"]
    n_years = params["n"]
    loan_amount = params["loan"]
    payment = params["payment"]
    pv_factor = round(_a_n(annual_rate, n_years), 6)
    discount_factor = round(1 / (1 + annual_rate), 6)

    if ask == "payment_amount":
        return Solved(payment, [
            f"Amortization formula: PMT = Loan / a_{n_years}|",
            f"a_{n_years}|_i = (1 - v^n)/i,  v = 1/(1+{annual_rate}) = {discount_factor}",
            f"a_{n_years}| = (1 - {discount_factor}^{n_years}) / {annual_rate} = {pv_factor}",
            f"PMT = {loan_amount} / {pv_factor} = {payment}",
        ], {"a_n": pv_factor, "v": discount_factor})

    if ask == "outstanding_prospective":
        payment_t = params.get("t", 0)
        remaining_pv_factor = round(_a_n(annual_rate, n_years - payment_t), 6)
        outstanding_balance = round(payment * remaining_pv_factor, 2)
        return Solved(outstanding_balance, [
            f"Prospective outstanding balance after period {payment_t}:",
            f"OB_{payment_t} = PMT · a_{{n-t}}|",
            f"PMT = {payment},  "
            f"a_{{n-t={n_years-payment_t}}}| = {remaining_pv_factor}",
            f"OB_{payment_t} = {payment} × {remaining_pv_factor} = {outstanding_balance}",
        ], {"payment": payment, "a_n_remain": remaining_pv_factor})

    if ask == "outstanding_retrospective":
        payment_t = params.get("t", 0)
        fv_factor_t = round(_s_n(annual_rate, payment_t), 6)
        accumulated_loan = round(loan_amount * (1 + annual_rate) ** payment_t, 4)
        accumulated_payments = round(payment * fv_factor_t, 4)
        outstanding_balance = round(accumulated_loan - accumulated_payments, 2)
        return Solved(outstanding_balance, [
            f"Retrospective outstanding balance after period {payment_t}:",
            f"OB_{payment_t} = Loan·(1+i)^{payment_t} - PMT·s_{payment_t}|",
            f"Loan·(1+i)^{payment_t} = {loan_amount}·"
            f"{round((1+annual_rate)**payment_t,6)} = {accumulated_loan}",
            f"s_{payment_t}| = {fv_factor_t}",
            f"PMT·s_{payment_t}| = {payment}×{fv_factor_t} = {accumulated_payments}",
            f"OB_{payment_t} = {accumulated_loan} - {accumulated_payments} "
            f"= {outstanding_balance}",
        ], {"payment": payment, "s_n_t": fv_factor_t})

    return _unknown(ask)


@_reg("loan_split")
def _s_loan_split(ask: str, params: dict) -> Solved:
    # Params stored: {"i": i, "n": n, "loan": loan, "payment": payment, "t": t}
    annual_rate = params["i"]
    n_years = params["n"]
    loan_amount = params["loan"]
    payment = params["payment"]
    payment_t = params.get("t", 1)
    discount_factor = round(1 / (1 + annual_rate), 6)

    if ask == "interest_tth_payment":
        remaining_discount = round(discount_factor ** (n_years - payment_t + 1), 6)
        interest_portion = round(payment * (1 - remaining_discount), 2)
        return Solved(interest_portion, [
            f"Interest portion of payment {payment_t}:",
            "I_t = PMT·(1 - v^{n-t+1})",
            f"n-t+1 = {n_years}-{payment_t}+1 = {n_years-payment_t+1}",
            f"v^{n_years-payment_t+1} = {remaining_discount}",
            f"I_{payment_t} = {payment}·(1 - {remaining_discount}) "
            f"= {payment}·{round(1-remaining_discount,6)} = {interest_portion}",
        ], {"payment": payment, "v": discount_factor})

    if ask == "principal_tth_payment":
        remaining_discount = round(discount_factor ** (n_years - payment_t + 1), 6)
        principal_portion = round(payment * remaining_discount, 2)
        return Solved(principal_portion, [
            f"Principal portion of payment {payment_t}:",
            "P_t = PMT·v^{n-t+1}",
            f"n-t+1 = {n_years-payment_t+1},  v^{n_years-payment_t+1} = {remaining_discount}",
            f"P_{payment_t} = {payment} × {remaining_discount} = {principal_portion}",
        ], {"payment": payment, "v": discount_factor})

    if ask == "total_interest_paid":
        total_payments = round(payment * n_years, 4)
        total_interest = round(payment * n_years - loan_amount, 2)
        return Solved(total_interest, [
            "Total interest = Total payments - Loan",
            f"Total payments = PMT × n = {payment} × {n_years} = {total_payments}",
            f"Total interest = {total_payments} - {loan_amount} = {total_interest}",
        ], {"payment": payment})

    return _unknown(ask)


@_reg("sinking_fund")
def _s_sinking_fund(ask: str, params: dict) -> Solved:
    # Params stored: {"i_loan": i_loan, "i_fund": i_fund, "n": n, "loan": loan, "deposit": deposit}
    loan_amount = params["loan"]
    loan_rate = params["i_loan"]
    fund_rate = params["i_fund"]
    n_years = params["n"]
    annual_deposit = params["deposit"]
    fund_fv_factor = round(_s_n(fund_rate, n_years), 6)
    annual_interest_payment = round(loan_amount * loan_rate, 2)

    if ask == "sinking_fund_deposit":
        return Solved(annual_deposit, [
            f"Sinking fund deposit D = Loan / s_{n_years}|_i_fund",
            f"s_{n_years}|_{fund_rate} = ((1+{fund_rate})^{n_years}-1)/{fund_rate} "
            f"= {fund_fv_factor}",
            f"D = {loan_amount} / {fund_fv_factor} = {annual_deposit}",
        ], {"s_n": fund_fv_factor})

    if ask == "total_periodic_outlay":
        total_outlay = round(annual_interest_payment + annual_deposit, 2)
        return Solved(total_outlay, [
            "Sinking fund method: pay interest each period AND deposit to fund.",
            f"Interest payment = Loan × i_loan = {loan_amount} × {loan_rate} "
            f"= {annual_interest_payment}",
            f"Sinking fund deposit = {annual_deposit}",
            f"Total outlay = {annual_interest_payment} + {annual_deposit} = {total_outlay}",
        ], {"int_pmt": annual_interest_payment, "deposit": annual_deposit})

    return _unknown(ask)


@_reg("bond_price")
def _s_bond_price(ask: str, params: dict) -> Solved:
    # Params stored: {"face": face, "r": r, "i": i, "n": n, "Fr": Fr, "price": price}
    face = params["face"]
    yield_rate = params["i"]
    n_years = params["n"]
    coupon_payment = params["Fr"]
    price = params["price"]
    discount_factor = round(1 / (1 + yield_rate), 6)
    redemption_discount = round(discount_factor ** n_years, 6)
    pv_factor = round(_a_n(yield_rate, n_years), 6)

    if ask == "price_from_yield":
        redemption_pv = round(face * redemption_discount, 4)
        coupon_pv = round(coupon_payment * pv_factor, 4)
        return Solved(price, [
            f"Bond price: P = C·v^n + Fr·a_{n_years}|",
            f"C = {face}, Fr = {coupon_payment}, yield i = {yield_rate}",
            f"v = 1/(1+{yield_rate}) = {discount_factor}",
            f"C·v^n = {face}·{redemption_discount} = {redemption_pv}",
            f"Fr·a_{n_years}| = {coupon_payment}·{pv_factor} = {coupon_pv}",
            f"P = {redemption_pv} + {coupon_pv} = {price}",
        ], {"v": discount_factor, "vn": redemption_discount, "a_n": pv_factor})

    if ask == "current_yield":
        current_yield = round(coupon_payment / price, 6)
        return Solved(current_yield, [
            "Current yield = Annual coupon / Price = Fr/P",
            f"Fr = {coupon_payment},  P = {price}",
            f"Current yield = {coupon_payment}/{price} = {current_yield}",
        ], {})

    if ask == "yield_approx":
        redemption_value = face
        numerator = round(coupon_payment + (redemption_value - price) / n_years, 4)
        denominator = round((redemption_value + price) / 2, 4)
        approx_yield = round(numerator / denominator, 4)
        return Solved(approx_yield, [
            "Bond yield approximation: i ≈ (Fr + (C-P)/n) / ((C+P)/2)",
            f"Fr = {coupon_payment},  C = {redemption_value},  P = {price},  n = {n_years}",
            f"Numerator: {coupon_payment} + ({redemption_value}-{price})/{n_years} "
            f"= {numerator}",
            f"Denominator: ({redemption_value}+{price})/2 = {denominator}",
            f"i ≈ {numerator} / {denominator} = {approx_yield}",
        ], {})

    return _unknown(ask)


@_reg("bond_makeham")
def _s_bond_makeham(ask: str, params: dict) -> Solved:
    face = params["face"]
    coupon_rate = params["r"]
    yield_rate = params["i"]
    modified_coupon_rate = params["g"]
    redemption_pv = params["K"]
    price = params["price"]

    if ask == "makeham_price":
        g_over_i = round(modified_coupon_rate / yield_rate, 6)
        redemption_gap = round(face - redemption_pv, 4)
        return Solved(price, [
            "Makeham formula: P = K + (g/i)·(C - K)",
            f"C = {face},  K = C·v^n = {redemption_pv}",
            f"g = Fr/C = {modified_coupon_rate},  g/i = {modified_coupon_rate}/{yield_rate} "
            f"= {g_over_i}",
            f"C - K = {face} - {redemption_pv} = {redemption_gap}",
            f"P = {redemption_pv} + {g_over_i} × {redemption_gap} = {price}",
        ], {"K": redemption_pv, "g": modified_coupon_rate})

    if ask == "makeham_modified_coupon_g":
        coupon_payment = face * coupon_rate
        g_answer = round(coupon_payment / face, 4)
        return Solved(g_answer, [
            "Modified coupon rate g = Fr/C",
            f"Fr = F·r = {face}·{coupon_rate} = {round(coupon_payment,4)}",
            f"C = {face}",
            f"g = {round(coupon_payment,4)}/{face} = {g_answer}",
        ], {})

    return _unknown(ask)


@_reg("bond_prem_disc")
def _s_bond_prem_disc(ask: str, params: dict) -> Solved:
    face = params["face"]
    coupon_rate = params["r"]
    yield_rate = params["i"]
    n_years = params["n"]
    premium_or_discount = params["premium"]

    if ask == "premium_discount_amount":
        answer = round(abs(premium_or_discount), 2)
        label = "premium" if coupon_rate > yield_rate else "discount"
        pv_factor = round(_a_n(yield_rate, n_years), 6)
        return Solved(answer, [
            f"Bond premium/discount: P - C = (g-i)·C·a_{n_years}|",
            f"g - i = {coupon_rate} - {yield_rate} = {round(coupon_rate-yield_rate,4)}",
            f"a_{n_years}| = {pv_factor}",
            f"P - C = {round(coupon_rate-yield_rate,4)} × {face} × {pv_factor} "
            f"= {round(premium_or_discount,4)}",
            f"|P - C| = {answer}  ({label})",
        ], {"a_n": pv_factor, "premium": premium_or_discount})

    if ask == "book_value_tth":
        # NOTE: the generator (gen_bond_prem_disc in engine/generation/bond.py)
        # picks coupon_number randomly but doesn't persist it into params, so
        # this defaults to t=1 — see the comment there for the same caveat.
        coupon_number = params.get("t", 1)
        remaining_pv_factor = round(_a_n(yield_rate, n_years - coupon_number), 6)
        book_value = round(face + (coupon_rate - yield_rate) * face * remaining_pv_factor, 2)
        rate_gap_times_face = round((coupon_rate - yield_rate) * face, 4)
        return Solved(book_value, [
            f"Book value after period {coupon_number}: "
            f"BV_t = C + (g-i)·C·a_{{n-t}}|",
            f"n - t = {n_years-coupon_number},  a_{{n-t}}| = {remaining_pv_factor}",
            f"(g-i)·C = ({coupon_rate}-{yield_rate})·{face} = {rate_gap_times_face}",
            f"BV_{coupon_number} = {face} + {rate_gap_times_face} × {remaining_pv_factor} "
            f"= {book_value}",
        ], {"a_n_remain": remaining_pv_factor})

    return _unknown(ask)


@_reg("macaulay_duration")
def _s_macaulay_duration(ask: str, params: dict) -> Solved:
    # Params stored: {"face": face, "r": r, "i": i, "n": n, "d_mac": d_mac}
    yield_rate = params["i"]
    stored_duration = params.get("d_mac")

    if ask == "macaulay_duration_bond":
        face = params["face"]
        coupon_rate = params["r"]
        n_years = params["n"]
        discount_factor = round(1 / (1 + yield_rate), 6)
        cashflows = [(t, face * coupon_rate) for t in range(1, n_years + 1)]
        cashflows[-1] = (n_years, face * coupon_rate + face)
        price = sum(cf * discount_factor ** t for t, cf in cashflows)
        time_weighted_pv = sum(t * cf * discount_factor ** t for t, cf in cashflows)
        recomputed_duration = round(time_weighted_pv / price, 4)
        # Prefer the value the generator already computed (and rounded once);
        # only recompute from scratch if the caller didn't supply d_mac.
        duration = stored_duration or recomputed_duration
        return Solved(duration, [
            "Macaulay duration: D = Σ t·PV(CF_t) / P",
            f"Bond: coupon = {face}·{coupon_rate} = {round(face*coupon_rate,4)} per period, "
            f"redemption = {face}",
            f"Price P = Σ PV(CF_t) = {round(price,4)}",
            f"Σ t·PV(CF_t) = {round(time_weighted_pv,4)}",
            f"D_mac = {round(time_weighted_pv,4)} / {round(price,4)} = {duration}",
        ], {"price": round(price, 4)})

    if ask == "macaulay_perpetuity":
        perpetuity_duration = round((1 + yield_rate) / yield_rate, 4)
        return Solved(perpetuity_duration, [
            "Macaulay duration of perpetuity-immediate: D = (1+i)/i",
            f"D = (1+{yield_rate})/{yield_rate} = {round(1+yield_rate,4)}/{yield_rate} "
            f"= {perpetuity_duration}",
        ], {})

    return _unknown(ask)


@_reg("modified_duration")
def _s_modified_duration(ask: str, params: dict) -> Solved:
    yield_rate = params["i"]
    modified_duration = params["d_mod"]
    macaulay_duration = params["d_mac"]

    if ask == "modified_duration_from_mac":
        return Solved(modified_duration, [
            "Modified duration: D_mod = D_mac / (1+i)",
            f"D_mac = {macaulay_duration},  1+i = {round(1+yield_rate,4)}",
            f"D_mod = {macaulay_duration} / {round(1+yield_rate,4)} = {modified_duration}",
        ], {})

    if ask == "price_change_approx":
        face = params.get("face", 1000)
        coupon_rate = params.get("r", 0)
        n_years = params.get("n", 1)
        discount_factor = 1 / (1 + yield_rate)
        cashflows = [(t, face * coupon_rate) for t in range(1, n_years + 1)]
        if cashflows:
            cashflows[-1] = (n_years, face * coupon_rate + face)
        price = sum(cf * discount_factor ** t for t, cf in cashflows)
        yield_shift = params.get("delta_i", 0.01)
        price_change = round(-modified_duration * price * yield_shift, 2)
        return Solved(price_change, [
            "Price change approximation: ΔP ≈ -D_mod · P · Δi",
            f"D_mod = {modified_duration},  P = {round(price,4)},  Δi = {yield_shift}",
            f"ΔP ≈ -{modified_duration} × {round(price,4)} × {yield_shift} = {price_change}",
        ], {"price": round(price, 4)})

    return _unknown(ask)


@_reg("convexity")
def _s_convexity(ask: str, params: dict) -> Solved:
    yield_rate = params["i"]
    convexity = params["convexity"]
    price = params.get("price", 1000)

    if ask == "convexity_bond":
        face = params["face"]
        coupon_rate = params["r"]
        n_years = params["n"]
        discount_factor = 1 / (1 + yield_rate)
        cashflows = [(t, face * coupon_rate) for t in range(1, n_years + 1)]
        cashflows[-1] = (n_years, face * coupon_rate + face)
        price_total = sum(cf * discount_factor ** t for t, cf in cashflows)
        weighted_sum = sum(t * (t + 1) * cf * discount_factor ** t for t, cf in cashflows)
        convexity_answer = round(weighted_sum / (price_total * (1 + yield_rate) ** 2), 4)
        return Solved(convexity_answer, [
            "Convexity: C = Σ t(t+1)·PV(CF_t) / (P·(1+i)²)",
            f"P = {round(price_total,4)},  (1+i)² = {round((1+yield_rate)**2,6)}",
            f"Numerator = {round(weighted_sum,4)}",
            f"Convexity = {round(weighted_sum,4)} / "
            f"({round(price_total,4)}·{round((1+yield_rate)**2,6)}) = {convexity_answer}",
        ], {"price": round(price_total, 4)})

    if ask == "second_order_price_approx":
        modified_duration = params.get("d_mod", 0)
        yield_shift = params.get("delta_i", 0.01)
        price_change = round(
            -modified_duration * price * yield_shift
            + 0.5 * convexity * price * yield_shift ** 2, 2
        )
        return Solved(price_change, [
            "Second-order price approximation:",
            "ΔP ≈ -D_mod·P·Δi + ½·C·P·(Δi)²",
            f"= -{modified_duration}·{price}·{yield_shift} "
            f"+ 0.5·{convexity}·{price}·{yield_shift}²",
            f"= {round(-modified_duration*price*yield_shift,4)} + "
            f"{round(0.5*convexity*price*yield_shift**2,4)}",
            f"= {price_change}",
        ], {})

    return _unknown(ask)


@_reg("immunization")
def _s_immunization(ask: str, params: dict) -> Solved:
    if ask == "redington_conditions":
        # The generator doesn't persist its randomly-chosen liability/asset
        # numbers into params (see the NOTE in gen_immunization), so there's
        # no scenario data to plug into a worked numeric answer here — this
        # variant is intentionally NaN and only explains the theory.
        return Solved(float("nan"), [
            "Redington immunization requires three conditions:",
            "1. PV(assets) = PV(liabilities) at current yield i",
            "2. D_mac(assets) = D_mac(liabilities)  [duration match]",
            "3. Convexity(assets) > Convexity(liabilities)  [dispersion condition]",
            "Compute asset portfolio duration as w₁·D₁ + w₂·D₂.",
        ], {})

    if ask == "full_immunization_weight":
        # NOTE: target_duration/bond1_duration/bond2_duration aren't persisted
        # by the generator either (same caveat as above), so this falls back
        # to fixed example numbers rather than the actual generated scenario.
        target_duration = params.get("H", 6)
        bond1_duration = params.get("d1", 3)
        bond2_duration = params.get("d2", 10)
        bond1_weight = round(
            (bond2_duration - target_duration) / (bond2_duration - bond1_duration), 4
        )
        return Solved(bond1_weight, [
            "Duration match: w₁·D₁ + (1-w₁)·D₂ = H",
            f"w₁·{bond1_duration} + (1-w₁)·{bond2_duration} = {target_duration}",
            f"w₁({bond1_duration}-{bond2_duration}) = {target_duration} - {bond2_duration}",
            f"w₁ = ({bond2_duration}-{target_duration}) / ({bond2_duration}-{bond1_duration}) "
            f"= {round(bond2_duration-target_duration,4)}/"
            f"{round(bond2_duration-bond1_duration,4)} = {bond1_weight}",
        ], {"d1": bond1_duration, "d2": bond2_duration, "H": target_duration})

    return _unknown(ask)


@_reg("spot_forward_rates")
def _s_spot_forward_rates(ask: str, params: dict) -> Solved:
    spot_rate_1yr = params["s1"]
    spot_rate_2yr = params["s2"]

    if ask == "implied_forward_rate":
        growth_ratio = round((1 + spot_rate_2yr) ** 2 / (1 + spot_rate_1yr), 6)
        forward_rate = round(growth_ratio - 1, 6)
        return Solved(forward_rate, [
            "Implied forward rate f₁,₂: (1+s₁)·(1+f₁,₂) = (1+s₂)²",
            f"1+f₁,₂ = (1+s₂)²/(1+s₁) = (1+{spot_rate_2yr})²/(1+{spot_rate_1yr})",
            f"= {round((1+spot_rate_2yr)**2,6)}/{round(1+spot_rate_1yr,6)} = {growth_ratio}",
            f"f₁,₂ = {forward_rate}",
        ], {})

    if ask == "pv_using_spot_rates":
        spot_rate_3yr = params.get("s3", spot_rate_2yr)
        cash_flow_1 = params.get("cf1", 100)
        cash_flow_2 = params.get("cf2", 100)
        cash_flow_3 = params.get("cf3", 1000)
        pv_1 = round(cash_flow_1 / (1 + spot_rate_1yr), 4)
        pv_2 = round(cash_flow_2 / (1 + spot_rate_2yr) ** 2, 4)
        pv_3 = round(cash_flow_3 / (1 + spot_rate_3yr) ** 3, 4)
        present_value = round(pv_1 + pv_2 + pv_3, 2)
        return Solved(present_value, [
            "Discount each cash flow at its spot rate:",
            f"PV(CF₁) = {cash_flow_1}/(1+{spot_rate_1yr}) = {pv_1}",
            f"PV(CF₂) = {cash_flow_2}/(1+{spot_rate_2yr})² = {pv_2}",
            f"PV(CF₃) = {cash_flow_3}/(1+{spot_rate_3yr})³ = {pv_3}",
            f"Total PV = {pv_1} + {pv_2} + {pv_3} = {present_value}",
        ], {})

    return _unknown(ask)


@_reg("forward_contract")
def _s_forward_contract(ask: str, params: dict) -> Solved:
    spot_price = params["S0"]
    risk_free_rate = params["r"]
    maturity_years = params["T"]

    if ask == "forward_price_no_dividend":
        forward_price = round(spot_price * (1 + risk_free_rate) ** maturity_years, 2)
        return Solved(forward_price, [
            "Forward price (no dividends): F = S₀·(1+r)^T",
            f"= {spot_price}·(1+{risk_free_rate})^{maturity_years}",
            f"= {spot_price}·{round((1+risk_free_rate)**maturity_years,6)} = {forward_price}",
        ], {})

    if ask == "forward_price_with_dividend":
        dividend_yield = params.get("q", 0)
        forward_price = round(
            spot_price * (1 + risk_free_rate - dividend_yield) ** maturity_years, 2
        )
        return Solved(forward_price, [
            f"Forward price (dividend yield q={dividend_yield}): F = S₀·(1+r-q)^T",
            f"= {spot_price}·{round(1+risk_free_rate-dividend_yield,6)}^{maturity_years} "
            f"= {forward_price}",
        ], {})

    if ask == "forward_payoff":
        # NOTE: the generator (gen_forward_contract, forward_payoff branch in
        # engine/generation/derivatives.py) doesn't persist its randomly
        # generated price_at_expiry, so the default below (= spot_price) will
        # generally not match the question actually shown to the user.
        delivery_price = params.get(
            "K", round(spot_price * (1 + risk_free_rate) ** maturity_years, 2)
        )
        price_at_expiry = params.get("ST", spot_price)
        payoff = round(price_at_expiry - delivery_price, 2)
        return Solved(payoff, [
            "Long forward payoff at expiry: ST - K",
            f"ST = {price_at_expiry},  K = {delivery_price}",
            f"Payoff = {price_at_expiry} - {delivery_price} = {payoff}",
        ], {})

    return _unknown(ask)


@_reg("option_payoff")
def _s_option_payoff(ask: str, params: dict) -> Solved:
    strike_price = params["K"]
    price_at_expiry = params["ST"]
    premium = params["premium"]
    call_payoff = round(max(price_at_expiry - strike_price, 0), 2)
    put_payoff = round(max(strike_price - price_at_expiry, 0), 2)

    if ask == "call_payoff":
        spread = round(price_at_expiry - strike_price, 2)
        return Solved(call_payoff, [
            "Call payoff = max(ST - K, 0)",
            f"ST = {price_at_expiry},  K = {strike_price}",
            f"ST - K = {spread},  payoff = max({spread}, 0) = {call_payoff}",
        ], {})

    if ask == "put_payoff":
        spread = round(strike_price - price_at_expiry, 2)
        return Solved(put_payoff, [
            "Put payoff = max(K - ST, 0)",
            f"K = {strike_price},  ST = {price_at_expiry}",
            f"K - ST = {spread},  payoff = max({spread}, 0) = {put_payoff}",
        ], {})

    if ask == "call_profit":
        profit = round(call_payoff - premium, 2)
        spread = round(price_at_expiry - strike_price, 2)
        return Solved(profit, [
            "Call profit = payoff - premium",
            f"Payoff = max(ST-K,0) = max({spread},0) = {call_payoff}",
            f"Profit = {call_payoff} - {premium} = {profit}",
        ], {"payoff": call_payoff})

    if ask == "put_profit":
        profit = round(put_payoff - premium, 2)
        spread = round(strike_price - price_at_expiry, 2)
        return Solved(profit, [
            "Put profit = payoff - premium",
            f"Payoff = max(K-ST,0) = max({spread},0) = {put_payoff}",
            f"Profit = {put_payoff} - {premium} = {profit}",
        ], {"payoff": put_payoff})

    return _unknown(ask)


@_reg("put_call_parity")
def _s_put_call_parity(ask: str, params: dict) -> Solved:
    spot_price = params["S0"]
    strike_price = params["K"]
    risk_free_rate = params["r"]
    maturity_years = params["T"]
    call_price = params["call_price"]
    put_price = params["put_price"]
    strike_pv = params["pv_K"]

    if ask == "find_put_from_call":
        return Solved(put_price, [
            "Put-call parity: C + PV(K) = P + S₀  ⟹  P = C + PV(K) - S₀",
            f"PV(K) = {strike_price}/(1+{risk_free_rate})^{maturity_years} = {strike_pv}",
            f"P = {call_price} + {strike_pv} - {spot_price} = {put_price}",
        ], {"pv_K": strike_pv})

    if ask == "find_call_from_put":
        call_answer = round(put_price + spot_price - strike_pv, 2)
        return Solved(call_answer, [
            "Put-call parity: C = P + S₀ - PV(K)",
            f"PV(K) = {strike_pv}",
            f"C = {put_price} + {spot_price} - {strike_pv} = {call_answer}",
        ], {"pv_K": strike_pv})

    if ask == "arbitrage_check":
        fair_call_price = round(put_price + spot_price - strike_pv, 2)
        arbitrage_profit = round(abs(call_price - fair_call_price), 2)
        return Solved(arbitrage_profit, [
            f"Fair call = P + S₀ - PV(K) = {put_price}+{spot_price}-{strike_pv} "
            f"= {fair_call_price}",
            f"Market call = {call_price}",
            f"Arbitrage profit = |{call_price} - {fair_call_price}| = {arbitrage_profit}",
        ], {})

    return _unknown(ask)


@_reg("swap_rate")
def _s_swap_rate(ask: str, params: dict) -> Solved:
    spot_rates = params["spot_rates"]
    swap_rate = params["R"]
    n_periods = params["n"]
    notional = params["notional"]
    discount_factors = [1 / (1 + spot_rates[t]) ** (t + 1) for t in range(n_periods)]
    notional_pv_factor = round(discount_factors[-1], 6)
    sum_discount_factors = round(sum(discount_factors), 6)

    if ask == "fixed_swap_rate":
        return Solved(swap_rate, [
            "Swap rate R: R = (1 - v_n) / Σ vₜ",
            "vₜ = 1/(1+sₜ)^t for each period t",
            f"Σ vₜ = {sum_discount_factors},  v_n = {notional_pv_factor}",
            f"R = (1 - {notional_pv_factor}) / {sum_discount_factors} = "
            f"{round(1-notional_pv_factor,6)} / {sum_discount_factors} = {swap_rate}",
        ], {"sum_v": sum_discount_factors, "pv_K": notional_pv_factor})

    if ask == "fixed_payment":
        fixed_payment = round(swap_rate * notional, 2)
        return Solved(fixed_payment, [
            "Fixed payment = R × Notional",
            f"= {swap_rate} × {notional:,.0f} = {fixed_payment}",
        ], {})

    return _unknown(ask)
