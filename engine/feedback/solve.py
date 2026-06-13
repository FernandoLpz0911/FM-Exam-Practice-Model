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


# ── Interest ──────────────────────────────────────────────────────────────────

@_reg("interest_tvm")
def _s_interest_tvm(ask: str, params: dict) -> Solved:
    i = params["i"]
    n = params["n"]
    pv = params["pv"]
    fv = round(pv * (1 + i) ** n, 2)

    if ask == "future_value":
        return Solved(fv, [
            f"Accumulation factor: (1+i)^n = (1+{i})^{n} = {round((1+i)**n, 6)}",
            f"FV = PV · (1+i)^n = {pv} × {round((1+i)**n, 6)} = {fv}",
        ], {"accumulation_factor": round((1+i)**n, 6)})

    if ask == "present_value":
        v_factor = round(1 / (1 + i) ** n, 6)
        return Solved(pv, [
            f"Discount factor: v^n = 1/(1+i)^n = 1/(1+{i})^{n} = {v_factor}",
            f"PV = FV · v^n = {fv} × {v_factor} = {pv}",
        ], {"discount_factor": v_factor})

    if ask == "interest_rate_solve":
        i_ans = round((fv / pv) ** (1 / n) - 1, 6)
        return Solved(i_ans, [
            f"FV = PV·(1+i)^n  ⟹  (1+i)^n = FV/PV = {fv}/{pv} = {round(fv/pv, 6)}",
            f"1+i = (FV/PV)^(1/n) = {round(fv/pv,6)}^(1/{n}) = {round((fv/pv)**(1/n),6)}",
            f"i = {round((fv/pv)**(1/n),6)} - 1 = {i_ans}",
        ], {"ratio": round(fv/pv, 6)})

    if ask == "periods_solve":
        n_ans = round(math.log(fv / pv) / math.log(1 + i), 4)
        return Solved(n, [
            f"FV = PV·(1+i)^n  ⟹  n = ln(FV/PV) / ln(1+i)",
            f"n = ln({fv}/{pv}) / ln(1+{i})",
            f"  = {round(math.log(fv/pv),6)} / {round(math.log(1+i),6)}",
            f"  = {n_ans}  (rounds to {n} years)",
        ], {})

    return _unknown(ask)


@_reg("interest_nominal")
def _s_interest_nominal(ask: str, params: dict) -> Solved:
    # Params stored: {"i_eff": i_eff, "m": m}  (both ask variants)
    i_eff = params["i_eff"]
    m = params["m"]

    if ask == "nominal_to_effective":
        i_nom = round(m * ((1 + i_eff) ** (1 / m) - 1), 6)
        return Solved(i_eff, [
            f"Given nominal rate i^({m}) = {i_nom*100:.4f}%, compounded {m}x/year.",
            f"Effective annual rate: (1 + i^(m)/m)^m - 1",
            f"= (1 + {round(i_nom/m,6)})^{m} - 1",
            f"= {round((1+i_nom/m)**m,6)} - 1 = {i_eff}",
        ], {"i_nom": i_nom, "periodic_rate": round(i_nom/m, 6)})

    if ask == "effective_to_nominal":
        i_nom = round(m * ((1 + i_eff) ** (1 / m) - 1), 6)
        return Solved(i_nom, [
            f"Effective annual rate i = {i_eff}, want nominal compounded {m}x/year.",
            f"i^({m}) = m·[(1+i)^(1/m) - 1]",
            f"= {m}·[(1+{i_eff})^(1/{m}) - 1]",
            f"= {m}·[{round((1+i_eff)**(1/m),6)} - 1]",
            f"= {m} × {round((1+i_eff)**(1/m)-1,6)} = {i_nom}",
        ], {})

    if ask == "equivalent_rate":
        m_old = params.get("m_old", m)
        m_new = params.get("m_new", m)
        i_nom_old = params.get("i_nom_old", round(m * ((1+i_eff)**(1/m)-1), 6))
        i_nom_new = round(m_new * ((1 + i_eff) ** (1 / m_new) - 1), 6)
        return Solved(i_nom_new, [
            f"Step 1 — convert i^({m_old})={i_nom_old} to effective: {i_eff}",
            f"Step 2 — convert effective to i^({m_new}): {m_new}·[(1+{i_eff})^(1/{m_new})-1] = {i_nom_new}",
        ], {"i_eff": i_eff})

    return _unknown(ask)


@_reg("interest_force")
def _s_interest_force(ask: str, params: dict) -> Solved:
    # Params stored: {"i": i, "delta": delta, "t": t}
    if ask == "force_from_rate":
        i = params["i"]
        delta = round(math.log(1 + i), 6)
        return Solved(delta, [
            f"Force of interest: δ = ln(1+i) = ln(1+{i})",
            f"= ln({round(1+i,6)}) = {delta}",
        ], {})

    if ask == "rate_from_force":
        delta = params["delta"]
        i = round(math.exp(delta) - 1, 6)
        return Solved(i, [
            f"Effective rate from force: i = e^δ - 1 = e^{delta} - 1",
            f"= {round(math.exp(delta),6)} - 1 = {i}",
        ], {})

    if ask == "accumulation_continuous":
        pv = params.get("pv", params.get("PV", 1000))
        delta = params["delta"]
        t = params["t"]
        fv = round(pv * math.exp(delta * t), 2)
        return Solved(fv, [
            f"Continuous accumulation: FV = PV · e^(δt)",
            f"= {pv} · e^({delta}×{t})",
            f"= {pv} · e^{round(delta*t,6)}",
            f"= {pv} × {round(math.exp(delta*t),6)} = {fv}",
        ], {"exponent": round(delta * t, 6)})

    return _unknown(ask)


@_reg("interest_discount")
def _s_interest_discount(ask: str, params: dict) -> Solved:
    # Params stored: {"i": i, "d": d, "v": v, "n": n}
    if ask == "discount_from_interest":
        i = params["i"]
        d = round(i / (1 + i), 6)
        return Solved(d, [
            f"Discount rate: d = i/(1+i) = {i}/(1+{i})",
            f"= {i}/{round(1+i,6)} = {d}",
        ], {})

    if ask == "interest_from_discount":
        d = params["d"]
        i = round(d / (1 - d), 6)
        return Solved(i, [
            f"Effective rate: i = d/(1-d) = {d}/(1-{d})",
            f"= {d}/{round(1-d,6)} = {i}",
        ], {})

    if ask == "pv_using_discount":
        d = params["d"]
        n = params["n"]
        fv = params.get("fv", params.get("FV", 1000))
        pv = round(fv * (1 - d) ** n, 2)
        return Solved(pv, [
            f"PV using discount rate: PV = FV·(1-d)^n",
            f"= {fv}·(1-{d})^{n}",
            f"= {fv}·{round((1-d)**n,6)}",
            f"= {pv}",
        ], {"discount_factor": round((1-d)**n, 6)})

    return _unknown(ask)


# ── Annuity ───────────────────────────────────────────────────────────────────

@_reg("annuity_immediate")
def _s_annuity_immediate(ask: str, params: dict) -> Solved:
    i = params["i"]
    n = params["n"]
    pmt = params["payment"]
    v = round(1 / (1 + i), 6)
    an = round(_a_n(i, n), 6)
    sn = round(_s_n(i, n), 6)

    if ask == "pv_annuity_imm":
        pv = round(pmt * an, 2)
        return Solved(pv, [
            f"Annuity-immediate PV factor: a_{n}| = (1 - v^n)/i",
            f"v = 1/(1+{i}) = {v}",
            f"a_{n}| = (1 - {v}^{n}) / {i} = (1 - {round(v**n,6)}) / {i} = {an}",
            f"PV = PMT · a_{n}| = {pmt} × {an} = {pv}",
        ], {"a_n": an, "v": v})

    if ask == "fv_annuity_imm":
        fv = round(pmt * sn, 2)
        return Solved(fv, [
            f"Annuity-immediate FV factor: s_{n}| = ((1+i)^n - 1)/i",
            f"s_{n}| = ({round(1+i,4)}^{n} - 1)/{i} = ({round((1+i)**n,6)} - 1)/{i} = {sn}",
            f"FV = PMT · s_{n}| = {pmt} × {sn} = {fv}",
        ], {"s_n": sn})

    if ask == "payment_from_pv":
        pv = round(pmt * an, 2)   # recompute for display
        pmt_ans = round(pv / an, 2)
        return Solved(pmt_ans, [
            f"PV = PMT · a_{n}|  ⟹  PMT = PV / a_{n}|",
            f"PV = {pmt} × {an} = {pv},  a_{n}| = {an}",
            f"PMT = {pv} / {an} = {pmt_ans}",
        ], {"a_n": an})

    if ask == "n_from_pv_imm":
        pv = round(pmt * an, 2)
        arg = 1 - pv * i / pmt
        n_ans = round(-math.log(arg) / math.log(1 + i), 4)
        return Solved(n, [
            f"PV = PMT · a_n|  ⟹  a_n| = PV/PMT = {pv}/{pmt} = {round(pv/pmt,6)}",
            f"(1 - v^n)/i = {round(pv/pmt,6)}  ⟹  v^n = 1 - {round(pv/pmt,6)}·{i} = {round(arg,6)}",
            f"n = -ln({round(arg,6)}) / ln(1+{i}) = {n_ans}",
        ], {"target_a_n": round(pv/pmt, 6)})

    return _unknown(ask)


@_reg("annuity_due")
def _s_annuity_due(ask: str, params: dict) -> Solved:
    i = params["i"]
    n = params["n"]
    pmt = params["payment"]
    an = _a_n(i, n)
    a_due = round((1 + i) * an, 6)
    sn = _s_n(i, n)
    s_due = round((1 + i) * sn, 6)

    if ask == "pv_annuity_due":
        pv = round(pmt * a_due, 2)
        return Solved(pv, [
            f"Annuity-due: payments at start of each period.",
            f"ä_{n}| = (1+i)·a_{n}| = (1+{i})·{round(an,6)} = {a_due}",
            f"PV = PMT · ä_{n}| = {pmt} × {a_due} = {pv}",
        ], {"a_due": a_due, "a_n": round(an,6)})

    if ask == "fv_annuity_due":
        fv = round(pmt * s_due, 2)
        return Solved(fv, [
            f"FV factor for annuity-due: s̈_{n}| = (1+i)·s_{n}|",
            f"s_{n}| = {round(sn,6)},  s̈_{n}| = (1+{i})·{round(sn,6)} = {s_due}",
            f"FV = PMT · s̈_{n}| = {pmt} × {s_due} = {fv}",
        ], {"s_due": s_due})

    if ask == "payment_from_pv_due":
        pv = round(pmt * a_due, 2)
        pmt_ans = round(pv / a_due, 2)
        return Solved(pmt_ans, [
            f"PV = PMT · ä_{n}|  ⟹  PMT = PV / ä_{n}|",
            f"PV = {pmt} × {a_due} = {pv},  ä_{n}| = {a_due}",
            f"PMT = {pv} / {a_due} = {pmt_ans}",
        ], {"a_due": a_due})

    return _unknown(ask)


@_reg("perpetuity")
def _s_perpetuity(ask: str, params: dict) -> Solved:
    # Params stored: {"i": i, "payment": payment, "d": d}
    i = params["i"]
    pmt = params["payment"]

    if ask == "pv_perp_imm":
        pv = round(pmt / i, 2)
        return Solved(pv, [
            f"Perpetuity-immediate PV: a_∞| = 1/i",
            f"PV = PMT/i = {pmt}/{i} = {pv}",
        ], {})

    if ask == "pv_perp_due":
        pv = round(pmt * (1 + i) / i, 2)
        return Solved(pv, [
            f"Perpetuity-due PV: ä_∞| = (1+i)/i",
            f"PV = PMT·(1+i)/i = {pmt}·(1+{i})/{i} = {pmt}·{round((1+i)/i,6)} = {pv}",
        ], {"a_due_inf": round((1+i)/i,6)})

    if ask == "payment_from_perp_pv":
        pv = round(pmt / i, 2)   # actual PV from generator math
        pmt_ans = round(pv * i, 2)
        return Solved(pmt_ans, [
            f"PV = PMT/i  ⟹  PMT = PV·i",
            f"PV = {pmt}/{i} = {pv}",
            f"PMT = {pv} × {i} = {pmt_ans}",
        ], {})

    return _unknown(ask)


@_reg("deferred_annuity")
def _s_deferred_annuity(ask: str, params: dict) -> Solved:
    # Params stored: {"i": i, "n": n, "m": m, "payment": payment}
    i = params["i"]
    n = params["n"]
    m = params["m"]
    pmt = params["payment"]
    v_m = round(1 / (1 + i) ** m, 6)
    an = round(_a_n(i, n), 6)

    if ask == "pv_deferred_imm":
        pv = round(pmt * v_m * an, 2)
        return Solved(pv, [
            f"Deferred annuity-immediate: deferred {m} periods, then {n} payments.",
            f"PV = PMT · v^m · a_n|",
            f"v^m = 1/(1+{i})^{m} = {v_m}",
            f"a_{n}| = {an}",
            f"PV = {pmt} × {v_m} × {an} = {pv}",
        ], {"v_m": v_m, "a_n": an})

    if ask == "pv_deferred_due":
        a_due = round((1 + i) * an, 6)
        pv = round(pmt * v_m * a_due, 2)
        return Solved(pv, [
            f"Deferred annuity-due: deferred {m} periods, then {n} payments at start of period.",
            f"ä_{n}| = (1+{i}) · a_{n}| = (1+{i}) × {an} = {a_due}",
            f"v^m = {v_m}",
            f"PV = {pmt} × {v_m} × {a_due} = {pv}",
        ], {"v_m": v_m, "a_due": a_due})

    return _unknown(ask)


@_reg("annuity_varying")
def _s_annuity_varying(ask: str, params: dict) -> Solved:
    # Params stored: {"i": i, "n": n, "base": base, "type": "inc"/"dec"/"geo"}
    i = params["i"]
    n = params["n"]
    base = params.get("base", params.get("base_payment", 100))

    if ask == "pv_arithmetic_inc":
        an = round(_a_n(i, n), 6)
        v = 1/(1+i)
        a_due = round((1+i)*an, 6)
        nvn = round(n * v**n, 6)
        ia_term = round((a_due - nvn)/i, 6)
        pv = round(base * an + base * ia_term, 2)
        return Solved(pv, [
            f"Increasing annuity: payment at time t = {base} + (t-1)·{base}.",
            f"PV = base·a_{n}| + base·(Iä)_{n}|/i",
            f"a_{n}| = {an},  ä_{n}| = {a_due}",
            f"n·v^n = {n}·{round(v**n,6)} = {nvn}",
            f"(Iä)_{n}| = ({a_due} - {nvn})/{i} = {ia_term}",
            f"PV = {base}·{an} + {base}·{ia_term} = {pv}",
        ], {"a_n": an, "ia_term": ia_term})

    if ask == "pv_arithmetic_dec":
        an = round(_a_n(i, n), 6)
        v = 1/(1+i)
        a_due = round((1+i)*an, 6)
        ia = round((a_due - n*v**n)/i, 6)
        pv = round((base+1)*an - ia, 2)
        return Solved(pv, [
            f"Decreasing annuity: payment at time t = {base} - (t-1)·1.",
            f"PV = (base+1)·a_{n}| - (Ia)_{n}|",
            f"a_{n}| = {an},  (Ia)_{n}| = {ia}",
            f"PV = {base+1}·{an} - {ia} = {pv}",
        ], {"a_n": an, "ia": ia})

    if ask == "pv_geometric":
        g = params.get("g", 0.03)
        j = round((i - g) / (1 + g), 6)
        if abs(i - g) < 1e-9:
            pv = round(base * n / (1 + i), 2)
            return Solved(pv, [
                f"Geometric annuity with g = i = {i}: special case.",
                f"PV = PMT·n/(1+i) = {base}·{n}/(1+{i}) = {pv}",
            ], {})
        an_j = round(_a_n(j, n), 6)
        pv = round(base * an_j / (1 + g), 2)
        return Solved(pv, [
            f"Geometric annuity with g={g}: equivalent rate j = (i-g)/(1+g) = {j}",
            f"a_{n}|_j = {an_j}",
            f"PV = PMT·a_{n}|_j / (1+g) = {base}·{an_j}/{round(1+g,4)} = {pv}",
        ], {"j": j})

    return _unknown(ask)


# ── Loan ──────────────────────────────────────────────────────────────────────

@_reg("loan_amort")
def _s_loan_amort(ask: str, params: dict) -> Solved:
    # Params stored: {"i": i, "n": n, "loan": loan, "payment": payment, "t": t}
    i = params["i"]
    n = params["n"]
    loan = params["loan"]
    payment = params["payment"]
    an = round(_a_n(i, n), 6)
    v = round(1 / (1 + i), 6)

    if ask == "payment_amount":
        return Solved(payment, [
            f"Amortization formula: PMT = Loan / a_{n}|",
            f"a_{n}|_i = (1 - v^n)/i,  v = 1/(1+{i}) = {v}",
            f"a_{n}| = (1 - {v}^{n}) / {i} = {an}",
            f"PMT = {loan} / {an} = {payment}",
        ], {"a_n": an, "v": v})

    if ask == "outstanding_prospective":
        t = params.get("t", 0)
        an_remain = round(_a_n(i, n - t), 6)
        ob = round(payment * an_remain, 2)
        return Solved(ob, [
            f"Prospective outstanding balance after period {t}:",
            f"OB_{t} = PMT · a_{{n-t}}|",
            f"PMT = {payment},  a_{{n-t={n-t}}}| = {an_remain}",
            f"OB_{t} = {payment} × {an_remain} = {ob}",
        ], {"payment": payment, "a_n_remain": an_remain})

    if ask == "outstanding_retrospective":
        t = params.get("t", 0)
        sn_t = round(_s_n(i, t), 6)
        ob = round(loan * (1 + i) ** t - payment * sn_t, 2)
        return Solved(ob, [
            f"Retrospective outstanding balance after period {t}:",
            f"OB_{t} = Loan·(1+i)^{t} - PMT·s_{t}|",
            f"Loan·(1+i)^{t} = {loan}·{round((1+i)**t,6)} = {round(loan*(1+i)**t,4)}",
            f"s_{t}| = {sn_t}",
            f"PMT·s_{t}| = {payment}×{sn_t} = {round(payment*sn_t,4)}",
            f"OB_{t} = {round(loan*(1+i)**t,4)} - {round(payment*sn_t,4)} = {ob}",
        ], {"payment": payment, "s_n_t": sn_t})

    return _unknown(ask)


@_reg("loan_split")
def _s_loan_split(ask: str, params: dict) -> Solved:
    # Params stored: {"i": i, "n": n, "loan": loan, "payment": payment, "t": t}
    i = params["i"]
    n = params["n"]
    loan = params["loan"]
    payment = params["payment"]
    t = params.get("t", 1)
    v = round(1 / (1 + i), 6)

    if ask == "interest_tth_payment":
        int_t = round(payment * (1 - v ** (n - t + 1)), 2)
        return Solved(int_t, [
            f"Interest portion of payment {t}:",
            f"I_t = PMT·(1 - v^{{n-t+1}})",
            f"n-t+1 = {n}-{t}+1 = {n-t+1}",
            f"v^{n-t+1} = {round(v**(n-t+1),6)}",
            f"I_{t} = {payment}·(1 - {round(v**(n-t+1),6)}) = {payment}·{round(1-v**(n-t+1),6)} = {int_t}",
        ], {"payment": payment, "v": v})

    if ask == "principal_tth_payment":
        prin_t = round(payment * v ** (n - t + 1), 2)
        return Solved(prin_t, [
            f"Principal portion of payment {t}:",
            f"P_t = PMT·v^{{n-t+1}}",
            f"n-t+1 = {n-t+1},  v^{n-t+1} = {round(v**(n-t+1),6)}",
            f"P_{t} = {payment} × {round(v**(n-t+1),6)} = {prin_t}",
        ], {"payment": payment, "v": v})

    if ask == "total_interest_paid":
        total_int = round(payment * n - loan, 2)
        return Solved(total_int, [
            f"Total interest = Total payments - Loan",
            f"Total payments = PMT × n = {payment} × {n} = {round(payment*n,4)}",
            f"Total interest = {round(payment*n,4)} - {loan} = {total_int}",
        ], {"payment": payment})

    return _unknown(ask)


@_reg("sinking_fund")
def _s_sinking_fund(ask: str, params: dict) -> Solved:
    # Params stored: {"i_loan": i_loan, "i_fund": i_fund, "n": n, "loan": loan, "deposit": deposit}
    loan = params["loan"]
    i_loan = params["i_loan"]
    i_fund = params["i_fund"]
    n = params["n"]
    deposit = params["deposit"]
    sn = round(_s_n(i_fund, n), 6)
    int_pmt = round(loan * i_loan, 2)

    if ask == "sinking_fund_deposit":
        return Solved(deposit, [
            f"Sinking fund deposit D = Loan / s_{n}|_i_fund",
            f"s_{n}|_{i_fund} = ((1+{i_fund})^{n}-1)/{i_fund} = {sn}",
            f"D = {loan} / {sn} = {deposit}",
        ], {"s_n": sn})

    if ask == "total_periodic_outlay":
        total = round(int_pmt + deposit, 2)
        return Solved(total, [
            f"Sinking fund method: pay interest each period AND deposit to fund.",
            f"Interest payment = Loan × i_loan = {loan} × {i_loan} = {int_pmt}",
            f"Sinking fund deposit = {deposit}",
            f"Total outlay = {int_pmt} + {deposit} = {total}",
        ], {"int_pmt": int_pmt, "deposit": deposit})

    return _unknown(ask)


# ── Bond ──────────────────────────────────────────────────────────────────────

@_reg("bond_price")
def _s_bond_price(ask: str, params: dict) -> Solved:
    # Params stored: {"face": face, "r": r, "i": i, "n": n, "Fr": Fr, "price": price}
    face = params["face"]
    r = params["r"]
    i = params["i"]
    n = params["n"]
    Fr = params["Fr"]
    price = params["price"]
    v = round(1 / (1 + i), 6)
    vn = round(v ** n, 6)
    an = round(_a_n(i, n), 6)

    if ask == "price_from_yield":
        return Solved(price, [
            f"Bond price: P = C·v^n + Fr·a_{n}|",
            f"C = {face}, Fr = {Fr}, yield i = {i}",
            f"v = 1/(1+{i}) = {v}",
            f"C·v^n = {face}·{vn} = {round(face*vn,4)}",
            f"Fr·a_{n}| = {Fr}·{an} = {round(Fr*an,4)}",
            f"P = {round(face*vn,4)} + {round(Fr*an,4)} = {price}",
        ], {"v": v, "vn": vn, "a_n": an})

    if ask == "current_yield":
        cy = round(Fr / price, 6)
        return Solved(cy, [
            f"Current yield = Annual coupon / Price = Fr/P",
            f"Fr = {Fr},  P = {price}",
            f"Current yield = {Fr}/{price} = {cy}",
        ], {})

    if ask == "yield_approx":
        C = face
        approx = round((Fr + (C - price) / n) / ((C + price) / 2), 4)
        return Solved(approx, [
            f"Bond yield approximation: i ≈ (Fr + (C-P)/n) / ((C+P)/2)",
            f"Fr = {Fr},  C = {C},  P = {price},  n = {n}",
            f"Numerator: {Fr} + ({C}-{price})/{n} = {round(Fr+(C-price)/n,4)}",
            f"Denominator: ({C}+{price})/2 = {round((C+price)/2,4)}",
            f"i ≈ {round(Fr+(C-price)/n,4)} / {round((C+price)/2,4)} = {approx}",
        ], {})

    return _unknown(ask)


@_reg("bond_makeham")
def _s_bond_makeham(ask: str, params: dict) -> Solved:
    face = params["face"]
    r = params["r"]
    i = params["i"]
    n = params["n"]
    g = params["g"]
    K = params["K"]
    price = params["price"]

    if ask == "makeham_price":
        return Solved(price, [
            f"Makeham formula: P = K + (g/i)·(C - K)",
            f"C = {face},  K = C·v^n = {K}",
            f"g = Fr/C = {g},  g/i = {g}/{i} = {round(g/i,6)}",
            f"C - K = {face} - {K} = {round(face-K,4)}",
            f"P = {K} + {round(g/i,6)} × {round(face-K,4)} = {price}",
        ], {"K": K, "g": g})

    if ask == "makeham_modified_coupon_g":
        Fr = face * r
        ans = round(Fr / face, 4)
        return Solved(ans, [
            f"Modified coupon rate g = Fr/C",
            f"Fr = F·r = {face}·{r} = {round(Fr,4)}",
            f"C = {face}",
            f"g = {round(Fr,4)}/{face} = {ans}",
        ], {})

    return _unknown(ask)


@_reg("bond_prem_disc")
def _s_bond_prem_disc(ask: str, params: dict) -> Solved:
    face = params["face"]
    r = params["r"]
    i = params["i"]
    n = params["n"]
    price = params["price"]
    premium = params["premium"]

    if ask == "premium_discount_amount":
        ans = round(abs(premium), 2)
        label = "premium" if r > i else "discount"
        an = round(_a_n(i, n), 6)
        return Solved(ans, [
            f"Bond premium/discount: P - C = (g-i)·C·a_{n}|",
            f"g - i = {r} - {i} = {round(r-i,4)}",
            f"a_{n}| = {an}",
            f"P - C = {round(r-i,4)} × {face} × {an} = {round(premium,4)}",
            f"|P - C| = {ans}  ({label})",
        ], {"a_n": an, "premium": premium})

    if ask == "book_value_tth":
        t = params.get("t", 1)
        an_remain = round(_a_n(i, n - t), 6)
        bv_t = round(face + (r - i) * face * an_remain, 2)
        return Solved(bv_t, [
            f"Book value after period {t}: BV_t = C + (g-i)·C·a_{{n-t}}|",
            f"n - t = {n-t},  a_{{n-t}}| = {an_remain}",
            f"(g-i)·C = ({r}-{i})·{face} = {round((r-i)*face,4)}",
            f"BV_{t} = {face} + {round((r-i)*face,4)} × {an_remain} = {bv_t}",
        ], {"a_n_remain": an_remain})

    return _unknown(ask)


# ── Duration ──────────────────────────────────────────────────────────────────

@_reg("macaulay_duration")
def _s_macaulay_duration(ask: str, params: dict) -> Solved:
    # Params stored: {"face": face, "r": r, "i": i, "n": n, "d_mac": d_mac}
    i = params["i"]
    d_mac = params.get("d_mac")

    if ask == "macaulay_duration_bond":
        face = params["face"]
        r = params["r"]
        n = params["n"]
        v = round(1 / (1 + i), 6)
        cashflows = [(t, face * r) for t in range(1, n + 1)]
        cashflows[-1] = (n, face * r + face)
        pv_total = sum(cf * v ** t for t, cf in cashflows)
        weighted = sum(t * cf * v ** t for t, cf in cashflows)
        d = round(weighted / pv_total, 4)
        return Solved(d_mac or d, [
            f"Macaulay duration: D = Σ t·PV(CF_t) / P",
            f"Bond: coupon = {face}·{r} = {round(face*r,4)} per period, redemption = {face}",
            f"Price P = Σ PV(CF_t) = {round(pv_total,4)}",
            f"Σ t·PV(CF_t) = {round(weighted,4)}",
            f"D_mac = {round(weighted,4)} / {round(pv_total,4)} = {d_mac or d}",
        ], {"price": round(pv_total, 4)})

    if ask == "macaulay_perpetuity":
        d = round((1 + i) / i, 4)
        return Solved(d, [
            f"Macaulay duration of perpetuity-immediate: D = (1+i)/i",
            f"D = (1+{i})/{i} = {round(1+i,4)}/{i} = {d}",
        ], {})

    return _unknown(ask)


@_reg("modified_duration")
def _s_modified_duration(ask: str, params: dict) -> Solved:
    i = params["i"]
    d_mod = params["d_mod"]
    d_mac = params["d_mac"]

    if ask == "modified_duration_from_mac":
        return Solved(d_mod, [
            f"Modified duration: D_mod = D_mac / (1+i)",
            f"D_mac = {d_mac},  1+i = {round(1+i,4)}",
            f"D_mod = {d_mac} / {round(1+i,4)} = {d_mod}",
        ], {})

    if ask == "price_change_approx":
        face = params.get("face", 1000)
        r = params.get("r", 0)
        n = params.get("n", 1)
        v = 1/(1+i)
        cashflows = [(t, face*r) for t in range(1, n+1)]
        if cashflows:
            cashflows[-1] = (n, face*r + face)
        price = sum(cf * v**t for t, cf in cashflows)
        delta_i = params.get("delta_i", 0.01)
        delta_p = round(-d_mod * price * delta_i, 2)
        return Solved(delta_p, [
            f"Price change approximation: ΔP ≈ -D_mod · P · Δi",
            f"D_mod = {d_mod},  P = {round(price,4)},  Δi = {delta_i}",
            f"ΔP ≈ -{d_mod} × {round(price,4)} × {delta_i} = {delta_p}",
        ], {"price": round(price,4)})

    return _unknown(ask)


@_reg("convexity")
def _s_convexity(ask: str, params: dict) -> Solved:
    i = params["i"]
    convexity = params["convexity"]
    price = params.get("price", 1000)

    if ask == "convexity_bond":
        face = params["face"]
        r = params["r"]
        n = params["n"]
        v = 1/(1+i)
        cashflows = [(t, face*r) for t in range(1, n+1)]
        cashflows[-1] = (n, face*r+face)
        pv_total = sum(cf*v**t for t,cf in cashflows)
        num = sum(t*(t+1)*cf*v**t for t,cf in cashflows)
        C_val = round(num / (pv_total*(1+i)**2), 4)
        return Solved(C_val, [
            f"Convexity: C = Σ t(t+1)·PV(CF_t) / (P·(1+i)²)",
            f"P = {round(pv_total,4)},  (1+i)² = {round((1+i)**2,6)}",
            f"Numerator = {round(num,4)}",
            f"Convexity = {round(num,4)} / ({round(pv_total,4)}·{round((1+i)**2,6)}) = {C_val}",
        ], {"price": round(pv_total,4)})

    if ask == "second_order_price_approx":
        d_mod = params.get("d_mod", 0)
        delta_i = params.get("delta_i", 0.01)
        delta_p = round(-d_mod*price*delta_i + 0.5*convexity*price*delta_i**2, 2)
        return Solved(delta_p, [
            f"Second-order price approximation:",
            f"ΔP ≈ -D_mod·P·Δi + ½·C·P·(Δi)²",
            f"= -{d_mod}·{price}·{delta_i} + 0.5·{convexity}·{price}·{delta_i}²",
            f"= {round(-d_mod*price*delta_i,4)} + {round(0.5*convexity*price*delta_i**2,4)}",
            f"= {delta_p}",
        ], {})

    return _unknown(ask)


@_reg("immunization")
def _s_immunization(ask: str, params: dict) -> Solved:
    if ask == "redington_conditions":
        return Solved(float("nan"), [
            "Redington immunization requires three conditions:",
            "1. PV(assets) = PV(liabilities) at current yield i",
            "2. D_mac(assets) = D_mac(liabilities)  [duration match]",
            "3. Convexity(assets) > Convexity(liabilities)  [dispersion condition]",
            "Compute asset portfolio duration as w₁·D₁ + w₂·D₂.",
        ], {})

    if ask == "full_immunization_weight":
        H = params.get("H", 6)
        d1 = params.get("d1", 3)
        d2 = params.get("d2", 10)
        w1 = round((d2 - H) / (d2 - d1), 4)
        return Solved(w1, [
            f"Duration match: w₁·D₁ + (1-w₁)·D₂ = H",
            f"w₁·{d1} + (1-w₁)·{d2} = {H}",
            f"w₁({d1}-{d2}) = {H} - {d2}",
            f"w₁ = ({d2}-{H}) / ({d2}-{d1}) = {round(d2-H,4)}/{round(d2-d1,4)} = {w1}",
        ], {"d1": d1, "d2": d2, "H": H})

    return _unknown(ask)


# ── Derivatives ───────────────────────────────────────────────────────────────

@_reg("spot_forward_rates")
def _s_spot_forward_rates(ask: str, params: dict) -> Solved:
    s1 = params["s1"]
    s2 = params["s2"]

    if ask == "implied_forward_rate":
        f12 = round((1 + s2) ** 2 / (1 + s1) - 1, 6)
        return Solved(f12, [
            f"Implied forward rate f₁,₂: (1+s₁)·(1+f₁,₂) = (1+s₂)²",
            f"1+f₁,₂ = (1+s₂)²/(1+s₁) = (1+{s2})²/(1+{s1})",
            f"= {round((1+s2)**2,6)}/{round(1+s1,6)} = {round((1+s2)**2/(1+s1),6)}",
            f"f₁,₂ = {f12}",
        ], {})

    if ask == "pv_using_spot_rates":
        s3 = params.get("s3", s2)
        cf1 = params.get("cf1", 100)
        cf2 = params.get("cf2", 100)
        cf3 = params.get("cf3", 1000)
        pv1 = round(cf1/(1+s1), 4)
        pv2 = round(cf2/(1+s2)**2, 4)
        pv3 = round(cf3/(1+s3)**3, 4)
        pv = round(pv1 + pv2 + pv3, 2)
        return Solved(pv, [
            f"Discount each cash flow at its spot rate:",
            f"PV(CF₁) = {cf1}/(1+{s1}) = {pv1}",
            f"PV(CF₂) = {cf2}/(1+{s2})² = {pv2}",
            f"PV(CF₃) = {cf3}/(1+{s3})³ = {pv3}",
            f"Total PV = {pv1} + {pv2} + {pv3} = {pv}",
        ], {})

    return _unknown(ask)


@_reg("forward_contract")
def _s_forward_contract(ask: str, params: dict) -> Solved:
    S0 = params["S0"]
    r = params["r"]
    T = params["T"]

    if ask == "forward_price_no_dividend":
        F = round(S0 * (1 + r) ** T, 2)
        return Solved(F, [
            f"Forward price (no dividends): F = S₀·(1+r)^T",
            f"= {S0}·(1+{r})^{T}",
            f"= {S0}·{round((1+r)**T,6)} = {F}",
        ], {})

    if ask == "forward_price_with_dividend":
        q = params.get("q", 0)
        F = round(S0 * (1 + r - q) ** T, 2)
        return Solved(F, [
            f"Forward price (dividend yield q={q}): F = S₀·(1+r-q)^T",
            f"= {S0}·{round(1+r-q,6)}^{T} = {F}",
        ], {})

    if ask == "forward_payoff":
        K = params.get("K", round(S0*(1+r)**T, 2))
        ST = params.get("ST", S0)
        payoff = round(ST - K, 2)
        return Solved(payoff, [
            f"Long forward payoff at expiry: ST - K",
            f"ST = {ST},  K = {K}",
            f"Payoff = {ST} - {K} = {payoff}",
        ], {})

    return _unknown(ask)


@_reg("option_payoff")
def _s_option_payoff(ask: str, params: dict) -> Solved:
    K = params["K"]
    ST = params["ST"]
    premium = params["premium"]
    call_payoff = round(max(ST - K, 0), 2)
    put_payoff = round(max(K - ST, 0), 2)

    if ask == "call_payoff":
        return Solved(call_payoff, [
            f"Call payoff = max(ST - K, 0)",
            f"ST = {ST},  K = {K}",
            f"ST - K = {round(ST-K,2)},  payoff = max({round(ST-K,2)}, 0) = {call_payoff}",
        ], {})

    if ask == "put_payoff":
        return Solved(put_payoff, [
            f"Put payoff = max(K - ST, 0)",
            f"K = {K},  ST = {ST}",
            f"K - ST = {round(K-ST,2)},  payoff = max({round(K-ST,2)}, 0) = {put_payoff}",
        ], {})

    if ask == "call_profit":
        profit = round(call_payoff - premium, 2)
        return Solved(profit, [
            f"Call profit = payoff - premium",
            f"Payoff = max(ST-K,0) = max({round(ST-K,2)},0) = {call_payoff}",
            f"Profit = {call_payoff} - {premium} = {profit}",
        ], {"payoff": call_payoff})

    if ask == "put_profit":
        profit = round(put_payoff - premium, 2)
        return Solved(profit, [
            f"Put profit = payoff - premium",
            f"Payoff = max(K-ST,0) = max({round(K-ST,2)},0) = {put_payoff}",
            f"Profit = {put_payoff} - {premium} = {profit}",
        ], {"payoff": put_payoff})

    return _unknown(ask)


@_reg("put_call_parity")
def _s_put_call_parity(ask: str, params: dict) -> Solved:
    S0 = params["S0"]
    K = params["K"]
    r = params["r"]
    T = params["T"]
    call_price = params["call_price"]
    put_price = params["put_price"]
    pv_K = params["pv_K"]

    if ask == "find_put_from_call":
        return Solved(put_price, [
            f"Put-call parity: C + PV(K) = P + S₀  ⟹  P = C + PV(K) - S₀",
            f"PV(K) = {K}/(1+{r})^{T} = {pv_K}",
            f"P = {call_price} + {pv_K} - {S0} = {put_price}",
        ], {"pv_K": pv_K})

    if ask == "find_call_from_put":
        call_ans = round(put_price + S0 - pv_K, 2)
        return Solved(call_ans, [
            f"Put-call parity: C = P + S₀ - PV(K)",
            f"PV(K) = {pv_K}",
            f"C = {put_price} + {S0} - {pv_K} = {call_ans}",
        ], {"pv_K": pv_K})

    if ask == "arbitrage_check":
        fair_call = round(put_price + S0 - pv_K, 2)
        arb = round(abs(call_price - fair_call), 2)
        return Solved(arb, [
            f"Fair call = P + S₀ - PV(K) = {put_price}+{S0}-{pv_K} = {fair_call}",
            f"Market call = {call_price}",
            f"Arbitrage profit = |{call_price} - {fair_call}| = {arb}",
        ], {})

    return _unknown(ask)


@_reg("swap_rate")
def _s_swap_rate(ask: str, params: dict) -> Solved:
    spot_rates = params["spot_rates"]
    R = params["R"]
    n = params["n"]
    notional = params["notional"]
    v = [1 / (1 + spot_rates[t]) ** (t + 1) for t in range(n)]
    pv_K = round(v[-1], 6)
    sum_v = round(sum(v), 6)

    if ask == "fixed_swap_rate":
        return Solved(R, [
            f"Swap rate R: R = (1 - v_n) / Σ vₜ",
            f"vₜ = 1/(1+sₜ)^t for each period t",
            f"Σ vₜ = {sum_v},  v_n = {pv_K}",
            f"R = (1 - {pv_K}) / {sum_v} = {round(1-pv_K,6)} / {sum_v} = {R}",
        ], {"sum_v": sum_v, "pv_K": pv_K})

    if ask == "fixed_payment":
        pmt = round(R * notional, 2)
        return Solved(pmt, [
            f"Fixed payment = R × Notional",
            f"= {R} × {notional:,.0f} = {pmt}",
        ], {})

    return _unknown(ask)
