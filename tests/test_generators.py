"""Sanity tests: every registered generator produces a valid Problem with correct answer in choices."""
import math
import pytest
import engine.generation  # noqa: F401
from engine.generation.base import generate, list_kinds

# (kind, ask, ranges)  — ranges match actual key names used by each generator
ALL_KINDS_ASKS = [
    ("interest_tvm", "future_value",       {"i_range": [0.05, 0.06], "n_range": [5, 6], "pv_range": [1000, 1001]}),
    ("interest_tvm", "present_value",      {"i_range": [0.05, 0.06], "n_range": [5, 6], "pv_range": [1000, 1001]}),
    ("interest_tvm", "interest_rate_solve",{"i_range": [0.05, 0.06], "n_range": [5, 6], "pv_range": [1000, 1001]}),
    ("interest_tvm", "periods_solve",      {"i_range": [0.05, 0.06], "n_range": [5, 6], "pv_range": [1000, 1001]}),
    ("interest_nominal", "nominal_to_effective", {"i_range": [0.05, 0.06], "m_choices": [4, 12]}),
    ("interest_nominal", "effective_to_nominal", {"i_range": [0.05, 0.06], "m_choices": [4, 12]}),
    ("interest_nominal", "equivalent_rate",      {"i_range": [0.05, 0.06], "m_choices": [4, 12]}),
    ("interest_force", "force_from_rate",         {"i_range": [0.05, 0.06], "delta_range": [0.04, 0.06], "t_range": [3, 6], "pv_range": [1000, 1001]}),
    ("interest_force", "rate_from_force",         {"i_range": [0.05, 0.06], "delta_range": [0.04, 0.06], "t_range": [3, 6], "pv_range": [1000, 1001]}),
    ("interest_force", "accumulation_continuous", {"i_range": [0.05, 0.06], "delta_range": [0.04, 0.06], "t_range": [3, 6], "pv_range": [1000, 1001]}),
    ("interest_discount", "discount_from_interest", {"i_range": [0.05, 0.06], "d_range": [0.04, 0.06], "n_range": [5, 6], "fv_range": [1000, 1001]}),
    ("interest_discount", "interest_from_discount", {"i_range": [0.05, 0.06], "d_range": [0.04, 0.06], "n_range": [5, 6], "fv_range": [1000, 1001]}),
    ("interest_discount", "pv_using_discount",      {"i_range": [0.05, 0.06], "d_range": [0.04, 0.06], "n_range": [5, 6], "fv_range": [1000, 1001]}),
    ("annuity_immediate", "pv_annuity_imm",   {"i_range": [0.05, 0.06], "n_range": [10, 11], "payment_range": [100, 101]}),
    ("annuity_immediate", "fv_annuity_imm",   {"i_range": [0.05, 0.06], "n_range": [10, 11], "payment_range": [100, 101]}),
    ("annuity_immediate", "payment_from_pv",  {"i_range": [0.05, 0.06], "n_range": [10, 11], "payment_range": [100, 101]}),
    ("annuity_immediate", "n_from_pv_imm",    {"i_range": [0.05, 0.06], "n_range": [10, 11], "payment_range": [100, 101]}),
    ("annuity_due", "pv_annuity_due",      {"i_range": [0.05, 0.06], "n_range": [10, 11], "payment_range": [100, 101]}),
    ("annuity_due", "fv_annuity_due",      {"i_range": [0.05, 0.06], "n_range": [10, 11], "payment_range": [100, 101]}),
    ("annuity_due", "payment_from_pv_due", {"i_range": [0.05, 0.06], "n_range": [10, 11], "payment_range": [100, 101]}),
    ("perpetuity", "pv_perp_imm",         {"i_range": [0.05, 0.06], "payment_range": [100, 101]}),
    ("perpetuity", "pv_perp_due",         {"i_range": [0.05, 0.06], "payment_range": [100, 101]}),
    ("perpetuity", "payment_from_perp_pv",{"i_range": [0.05, 0.06], "payment_range": [100, 101]}),
    ("deferred_annuity", "pv_deferred_imm", {"i_range": [0.05, 0.06], "n_range": [10, 11], "m_range": [3, 4], "payment_range": [100, 101]}),
    ("deferred_annuity", "pv_deferred_due", {"i_range": [0.05, 0.06], "n_range": [10, 11], "m_range": [3, 4], "payment_range": [100, 101]}),
    ("annuity_varying", "pv_arithmetic_inc", {"i_range": [0.05, 0.06], "n_range": [10, 11], "base_payment_range": [100, 101]}),
    ("annuity_varying", "pv_arithmetic_dec", {"i_range": [0.05, 0.06], "n_range": [10, 11], "base_payment_range": [100, 101]}),
    ("annuity_varying", "pv_geometric",      {"i_range": [0.05, 0.06], "n_range": [10, 11], "base_payment_range": [100, 101]}),
    ("loan_amort", "payment_amount",           {"i_range": [0.05, 0.06], "n_range": [10, 11], "loan_range": [10000, 10001]}),
    ("loan_amort", "outstanding_prospective",  {"i_range": [0.05, 0.06], "n_range": [10, 11], "loan_range": [10000, 10001]}),
    ("loan_amort", "outstanding_retrospective",{"i_range": [0.05, 0.06], "n_range": [10, 11], "loan_range": [10000, 10001]}),
    ("loan_split", "interest_tth_payment",  {"i_range": [0.05, 0.06], "n_range": [10, 11], "loan_range": [10000, 10001]}),
    ("loan_split", "principal_tth_payment", {"i_range": [0.05, 0.06], "n_range": [10, 11], "loan_range": [10000, 10001]}),
    ("loan_split", "total_interest_paid",   {"i_range": [0.05, 0.06], "n_range": [10, 11], "loan_range": [10000, 10001]}),
    ("sinking_fund", "sinking_fund_deposit",    {"j_range": [0.06, 0.07], "i_range": [0.04, 0.05], "n_range": [10, 11], "loan_range": [10000, 10001]}),
    ("sinking_fund", "total_periodic_outlay",   {"j_range": [0.06, 0.07], "i_range": [0.04, 0.05], "n_range": [10, 11], "loan_range": [10000, 10001]}),
    ("bond_price", "price_from_yield", {"coupon_rate_range": [0.06, 0.07], "yield_range": [0.05, 0.06], "n_range": [10, 11]}),
    ("bond_price", "current_yield",    {"coupon_rate_range": [0.06, 0.07], "yield_range": [0.05, 0.06], "n_range": [10, 11]}),
    ("bond_price", "yield_approx",     {"coupon_rate_range": [0.06, 0.07], "yield_range": [0.05, 0.06], "n_range": [10, 11]}),
    ("bond_makeham", "makeham_price",              {"coupon_rate_range": [0.06, 0.07], "yield_range": [0.05, 0.06], "n_range": [10, 11]}),
    ("bond_makeham", "makeham_modified_coupon_g",  {"coupon_rate_range": [0.06, 0.07], "yield_range": [0.05, 0.06], "n_range": [10, 11]}),
    ("bond_prem_disc", "premium_discount_amount", {"coupon_rate_range": [0.06, 0.07], "yield_range": [0.05, 0.06], "n_range": [10, 11]}),
    ("bond_prem_disc", "book_value_tth",          {"coupon_rate_range": [0.06, 0.07], "yield_range": [0.05, 0.06], "n_range": [10, 11]}),
    ("macaulay_duration", "macaulay_duration_bond", {"coupon_rate_range": [0.06, 0.07], "yield_range": [0.05, 0.06], "n_range": [5, 6]}),
    ("macaulay_duration", "macaulay_perpetuity",    {"coupon_rate_range": [0.06, 0.07], "yield_range": [0.05, 0.06], "n_range": [5, 6]}),
    ("modified_duration", "modified_duration_from_mac", {"coupon_rate_range": [0.06, 0.07], "yield_range": [0.05, 0.06], "n_range": [5, 6]}),
    ("modified_duration", "price_change_approx",        {"coupon_rate_range": [0.06, 0.07], "yield_range": [0.05, 0.06], "n_range": [5, 6]}),
    ("convexity", "convexity_bond",           {"coupon_rate_range": [0.06, 0.07], "yield_range": [0.05, 0.06], "n_range": [5, 6]}),
    ("convexity", "second_order_price_approx",{"coupon_rate_range": [0.06, 0.07], "yield_range": [0.05, 0.06], "n_range": [5, 6]}),
    ("immunization", "redington_conditions",     {"yield_range": [0.04, 0.06]}),
    ("immunization", "full_immunization_weight", {"yield_range": [0.04, 0.06]}),
    ("spot_forward_rates", "implied_forward_rate", {"rate_range": [0.04, 0.06]}),
    ("spot_forward_rates", "pv_using_spot_rates",  {"rate_range": [0.04, 0.06]}),
    ("forward_contract", "forward_price_no_dividend",  {"spot_price_range": [95, 105], "rate_range": [0.04, 0.06], "maturity_range": [1, 3]}),
    ("forward_contract", "forward_price_with_dividend",{"spot_price_range": [95, 105], "rate_range": [0.04, 0.06], "maturity_range": [1, 3]}),
    ("forward_contract", "forward_payoff",             {"spot_price_range": [95, 105], "rate_range": [0.04, 0.06], "maturity_range": [1, 3]}),
    ("option_payoff", "call_payoff",  {"strike_range": [95, 105]}),
    ("option_payoff", "put_payoff",   {"strike_range": [95, 105]}),
    ("option_payoff", "call_profit",  {"strike_range": [95, 105]}),
    ("option_payoff", "put_profit",   {"strike_range": [95, 105]}),
    ("put_call_parity", "find_put_from_call", {"spot_price_range": [95, 105], "rate_range": [0.04, 0.06], "maturity_range": [1, 2]}),
    ("put_call_parity", "find_call_from_put", {"spot_price_range": [95, 105], "rate_range": [0.04, 0.06], "maturity_range": [1, 2]}),
    ("put_call_parity", "arbitrage_check",    {"spot_price_range": [95, 105], "rate_range": [0.04, 0.06], "maturity_range": [1, 2]}),
    ("swap_rate", "fixed_swap_rate",  {"n_range": [3, 4], "rate_range": [0.04, 0.06]}),
    ("swap_rate", "fixed_payment",    {"n_range": [3, 4], "rate_range": [0.04, 0.06]}),
]


@pytest.mark.parametrize("kind,ask,ranges", ALL_KINDS_ASKS)
def test_generator_produces_valid_problem(kind: str, ask: str, ranges: dict):
    problem = generate(kind, ask, ranges, seed=42)
    assert problem.kind == kind
    assert problem.ask == ask
    assert problem.statement
    assert isinstance(problem.correct_answer, (int, float))
    assert not math.isnan(problem.correct_answer)
    if problem.choices is not None:
        assert len(problem.choices) == 4
        # Correct answer must appear as one of the choices (formatted to 4 dp)
        ca_fmt = f"{problem.correct_answer:.4f}"
        assert any(ca_fmt == c for c in problem.choices), \
            f"Correct answer {ca_fmt} not found in choices: {problem.choices}"


def test_all_kinds_registered():
    kinds = list_kinds()
    expected = {
        "interest_tvm", "interest_nominal", "interest_force", "interest_discount",
        "annuity_immediate", "annuity_due", "perpetuity", "deferred_annuity", "annuity_varying",
        "loan_amort", "loan_split", "sinking_fund",
        "bond_price", "bond_makeham", "bond_prem_disc",
        "macaulay_duration", "modified_duration", "convexity", "immunization",
        "spot_forward_rates", "forward_contract", "option_payoff", "put_call_parity", "swap_rate",
    }
    missing = expected - set(kinds)
    assert not missing, f"Missing from registry: {missing}"
