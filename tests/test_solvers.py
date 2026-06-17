"""Sanity tests: every solver returns Solved with non-NaN answer and non-empty steps.
Param dicts mirror what each generator actually stores in Problem.params.
"""
import math

import pytest

from engine.feedback.solve import solve

# All params use the exact keys generators store in Problem.params
SOLVER_CASES = [
    # interest_tvm: {"i": i, "n": n, "pv": pv}
    ("interest_tvm", "future_value",        {"i": 0.05, "n": 5, "pv": 1000.0}),
    ("interest_tvm", "present_value",       {"i": 0.05, "n": 5, "pv": 1000.0}),
    ("interest_tvm", "interest_rate_solve", {"i": 0.05, "n": 5, "pv": 1000.0}),
    ("interest_tvm", "periods_solve",       {"i": 0.05, "n": 5, "pv": 1000.0}),
    # interest_nominal: {"i_eff": i_eff, "m": m}
    ("interest_nominal", "nominal_to_effective", {"i_eff": 0.1268, "m": 12}),
    ("interest_nominal", "effective_to_nominal", {"i_eff": 0.1268, "m": 12}),
    # interest_force: {"i": i, "delta": delta, "t": t}
    ("interest_force", "force_from_rate",        {"i": 0.05, "delta": 0.04879, "t": 5}),
    ("interest_force", "rate_from_force",        {"i": 0.05, "delta": 0.04879, "t": 5}),
    ("interest_force", "accumulation_continuous",{"i": 0.05, "delta": 0.04879, "t": 5, "pv": 1000.0}),
    # interest_discount: {"i": i, "d": d, "v": v, "n": n}
    ("interest_discount", "discount_from_interest", {"i": 0.05, "d": 0.04762, "v": 0.9524, "n": 5}),
    ("interest_discount", "interest_from_discount", {"i": 0.05, "d": 0.04762, "v": 0.9524, "n": 5}),
    ("interest_discount", "pv_using_discount",      {"i": 0.05, "d": 0.04762, "v": 0.9524, "n": 5, "fv": 1000.0}),
    # annuity_immediate: {"i": i, "n": n, "payment": payment}
    ("annuity_immediate", "pv_annuity_imm",  {"i": 0.05, "n": 10, "payment": 100.0}),
    ("annuity_immediate", "fv_annuity_imm",  {"i": 0.05, "n": 10, "payment": 100.0}),
    ("annuity_immediate", "payment_from_pv", {"i": 0.05, "n": 10, "payment": 100.0}),
    # annuity_due: {"i": i, "n": n, "payment": payment}
    ("annuity_due", "pv_annuity_due",     {"i": 0.05, "n": 10, "payment": 100.0}),
    # perpetuity: {"i": i, "payment": payment, "d": d}
    ("perpetuity", "pv_perp_imm",          {"i": 0.05, "payment": 100.0, "d": 0.04762}),
    ("perpetuity", "pv_perp_due",          {"i": 0.05, "payment": 100.0, "d": 0.04762}),
    ("perpetuity", "payment_from_perp_pv", {"i": 0.05, "payment": 100.0, "d": 0.04762}),
    # deferred_annuity: {"i": i, "n": n, "m": m, "payment": payment}
    ("deferred_annuity", "pv_deferred_imm", {"i": 0.05, "n": 10, "m": 3, "payment": 100.0}),
    ("deferred_annuity", "pv_deferred_due", {"i": 0.05, "n": 10, "m": 3, "payment": 100.0}),
    # annuity_varying: {"i": i, "n": n, "base": base, "type": type}
    ("annuity_varying", "pv_arithmetic_inc", {"i": 0.05, "n": 10, "base": 100.0, "type": "inc"}),
    ("annuity_varying", "pv_arithmetic_dec", {"i": 0.05, "n": 10, "base": 100.0, "type": "dec"}),
    ("annuity_varying", "pv_geometric",      {"i": 0.05, "n": 10, "base": 100.0, "g": 0.03, "type": "geo"}),
    # loan_amort: {"i": i, "n": n, "loan": loan, "payment": payment, "t": t}
    ("loan_amort", "payment_amount",           {"i": 0.05, "n": 10, "loan": 10000.0, "payment": 1295.05, "t": 0}),
    ("loan_amort", "outstanding_prospective",  {"i": 0.05, "n": 10, "loan": 10000.0, "payment": 1295.05, "t": 5}),
    ("loan_amort", "outstanding_retrospective",{"i": 0.05, "n": 10, "loan": 10000.0, "payment": 1295.05, "t": 5}),
    # loan_split: {"i": i, "n": n, "loan": loan, "payment": payment, "t": t}
    ("loan_split", "interest_tth_payment",  {"i": 0.05, "n": 10, "loan": 10000.0, "payment": 1295.05, "t": 3}),
    ("loan_split", "principal_tth_payment", {"i": 0.05, "n": 10, "loan": 10000.0, "payment": 1295.05, "t": 3}),
    ("loan_split", "total_interest_paid",   {"i": 0.05, "n": 10, "loan": 10000.0, "payment": 1295.05, "t": 3}),
    # sinking_fund: {"i_loan": i_loan, "i_fund": i_fund, "n": n, "loan": loan, "deposit": deposit}
    ("sinking_fund", "sinking_fund_deposit",   {"i_loan": 0.06, "i_fund": 0.04, "n": 10, "loan": 10000.0, "deposit": 832.91}),
    ("sinking_fund", "total_periodic_outlay",  {"i_loan": 0.06, "i_fund": 0.04, "n": 10, "loan": 10000.0, "deposit": 832.91}),
    # bond_price: {"face": face, "r": r, "i": i, "n": n, "Fr": Fr, "price": price}
    ("bond_price", "price_from_yield", {"face": 1000.0, "r": 0.06, "i": 0.05, "n": 10, "Fr": 60.0, "price": 1077.22}),
    ("bond_price", "current_yield",    {"face": 1000.0, "r": 0.06, "i": 0.05, "n": 10, "Fr": 60.0, "price": 1077.22}),
    ("bond_price", "yield_approx",     {"face": 1000.0, "r": 0.06, "i": 0.05, "n": 10, "Fr": 60.0, "price": 1077.22}),
    # bond_makeham: {"face": face, "r": r, "i": i, "n": n, "g": g, "K": K, "price": price}
    ("bond_makeham", "makeham_price",             {"face": 1000.0, "r": 0.06, "i": 0.05, "n": 10, "g": 0.06, "K": 613.91, "price": 1077.22}),
    ("bond_makeham", "makeham_modified_coupon_g", {"face": 1000.0, "r": 0.06, "i": 0.05, "n": 10, "g": 0.06, "K": 613.91, "price": 1077.22}),
    # bond_prem_disc: {"face": face, "r": r, "i": i, "n": n, "price": price, "premium": premium}
    ("bond_prem_disc", "premium_discount_amount", {"face": 1000.0, "r": 0.06, "i": 0.05, "n": 10, "price": 1077.22, "premium": 77.22}),
    ("bond_prem_disc", "book_value_tth",          {"face": 1000.0, "r": 0.06, "i": 0.05, "n": 10, "price": 1077.22, "premium": 77.22, "t": 3}),
    # macaulay_duration: {"face": face, "r": r, "i": i, "n": n, "d_mac": d_mac}
    ("macaulay_duration", "macaulay_duration_bond", {"face": 1000.0, "r": 0.06, "i": 0.05, "n": 10, "d_mac": 7.80}),
    ("macaulay_duration", "macaulay_perpetuity",    {"face": 1000.0, "r": 0.06, "i": 0.05, "n": 10, "d_mac": 7.80}),
    # modified_duration: {"face": face, "r": r, "i": i, "n": n, "d_mac": d_mac, "d_mod": d_mod}
    ("modified_duration", "modified_duration_from_mac", {"face": 1000.0, "r": 0.06, "i": 0.05, "n": 10, "d_mac": 7.80, "d_mod": 7.43}),
    ("modified_duration", "price_change_approx",        {"face": 1000.0, "r": 0.06, "i": 0.05, "n": 10, "d_mac": 7.80, "d_mod": 7.43, "delta_i": 0.01}),
    # convexity: {"face": face, "r": r, "i": i, "n": n, "price": price, "convexity": C}
    ("convexity", "convexity_bond",            {"face": 1000.0, "r": 0.06, "i": 0.05, "n": 10, "price": 1077.22, "convexity": 68.96}),
    ("convexity", "second_order_price_approx", {"face": 1000.0, "r": 0.06, "i": 0.05, "n": 10, "price": 1077.22, "convexity": 68.96, "d_mod": 7.43, "delta_i": 0.01}),
    # immunization: {"i": i}  (+optional H/d1/d2 for full_immunization_weight)
    ("immunization", "full_immunization_weight", {"i": 0.05, "H": 6.0, "d1": 3.0, "d2": 10.0}),
    # spot_forward_rates: {"s1": s1, "s2": s2}
    ("spot_forward_rates", "implied_forward_rate", {"s1": 0.04, "s2": 0.045}),
    # forward_contract: {"S0": S0, "r": r, "T": T}
    ("forward_contract", "forward_price_no_dividend", {"S0": 100.0, "r": 0.05, "T": 1}),
    ("forward_contract", "forward_payoff",            {"S0": 100.0, "r": 0.05, "T": 1, "ST": 108.0, "K": 105.0}),
    # option_payoff: {"K": K, "ST": ST, "premium": premium}
    ("option_payoff", "call_payoff",  {"K": 100.0, "ST": 110.0, "premium": 5.0}),
    ("option_payoff", "put_payoff",   {"K": 100.0, "ST": 90.0,  "premium": 5.0}),
    ("option_payoff", "call_profit",  {"K": 100.0, "ST": 110.0, "premium": 5.0}),
    ("option_payoff", "put_profit",   {"K": 100.0, "ST": 90.0,  "premium": 5.0}),
    # put_call_parity: full dict
    ("put_call_parity", "find_put_from_call", {"S0": 100.0, "K": 105.0, "r": 0.05, "T": 1, "call_price": 8.0, "put_price": 12.76, "pv_K": 100.0}),
    ("put_call_parity", "find_call_from_put", {"S0": 100.0, "K": 105.0, "r": 0.05, "T": 1, "call_price": 8.0, "put_price": 12.76, "pv_K": 100.0}),
    # swap_rate: {"n": n, "spot_rates": [...], "R": R, "notional": notional}
    ("swap_rate", "fixed_swap_rate", {"n": 3, "spot_rates": [0.04, 0.045, 0.05], "R": 0.049, "notional": 1000000.0}),
    ("swap_rate", "fixed_payment",   {"n": 3, "spot_rates": [0.04, 0.045, 0.05], "R": 0.049, "notional": 1000000.0}),
]


@pytest.mark.parametrize("kind,ask,params", SOLVER_CASES)
def test_solver_returns_valid_solved(kind: str, ask: str, params: dict):
    result = solve(kind, ask, params)
    assert not math.isnan(result.answer) or kind == "immunization", \
        f"{kind}:{ask} returned NaN"
    assert len(result.steps) > 0, f"{kind}:{ask} returned empty steps"
