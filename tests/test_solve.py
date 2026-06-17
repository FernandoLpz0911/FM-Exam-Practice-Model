"""Tests for engine/feedback/solve.py — covering all uncovered paths."""
from __future__ import annotations

import math

import pytest

from engine.feedback.solve import solve


class TestDispatch:
    def test_unknown_kind_returns_nan(self):
        result = solve("nonexistent_kind", "any_ask", {})
        assert math.isnan(result.answer)
        assert "nonexistent_kind" in result.steps[0]

    def test_unknown_ask_calls_unknown_helper(self):
        result = solve("interest_tvm", "__bogus__", {"i": 0.05, "n": 5, "pv": 1000})
        assert math.isnan(result.answer)
        assert "__bogus__" in result.steps[0]



class TestInterestNominal:
    def test_equivalent_rate(self):
        result = solve("interest_nominal", "equivalent_rate",
                       {"i_eff": 0.05, "m": 12, "m_old": 4, "m_new": 2})
        assert not math.isnan(result.answer)
        assert result.answer == pytest.approx(
            2 * ((1 + 0.05) ** (1 / 2) - 1), abs=1e-4
        )

    def test_unknown_ask(self):
        result = solve("interest_nominal", "__bogus__", {"i_eff": 0.05, "m": 12})
        assert math.isnan(result.answer)


class TestInterestForceUnknown:
    def test_unknown_ask(self):
        result = solve("interest_force", "__bogus__", {"i": 0.05, "delta": 0.04879, "t": 3})
        assert math.isnan(result.answer)


class TestInterestDiscountUnknown:
    def test_unknown_ask(self):
        result = solve("interest_discount", "__bogus__", {"i": 0.05, "d": 0.04762, "v": 0.9524, "n": 5})
        assert math.isnan(result.answer)



class TestAnnuityImmediate:
    def test_n_from_pv_imm(self):
        result = solve("annuity_immediate", "n_from_pv_imm",
                       {"i": 0.05, "n": 10, "payment": 500})
        assert not math.isnan(result.answer)
        assert result.answer == 10

    def test_unknown_ask(self):
        result = solve("annuity_immediate", "__bogus__",
                       {"i": 0.05, "n": 10, "payment": 500})
        assert math.isnan(result.answer)


class TestAnnuityDue:
    def test_fv_annuity_due(self):
        result = solve("annuity_due", "fv_annuity_due",
                       {"i": 0.05, "n": 10, "payment": 500})
        assert not math.isnan(result.answer)
        assert result.answer > 0

    def test_payment_from_pv_due(self):
        result = solve("annuity_due", "payment_from_pv_due",
                       {"i": 0.05, "n": 10, "payment": 500})
        assert not math.isnan(result.answer)

    def test_unknown_ask(self):
        result = solve("annuity_due", "__bogus__",
                       {"i": 0.05, "n": 10, "payment": 500})
        assert math.isnan(result.answer)


class TestPerpetuityUnknown:
    def test_unknown_ask(self):
        result = solve("perpetuity", "__bogus__", {"i": 0.05, "payment": 100})
        assert math.isnan(result.answer)


class TestDeferredAnnuityUnknown:
    def test_unknown_ask(self):
        result = solve("deferred_annuity", "__bogus__",
                       {"i": 0.05, "n": 10, "m": 3, "payment": 500})
        assert math.isnan(result.answer)


class TestAnnuityVarying:
    def test_pv_geometric_g_equals_i(self):
        i = 0.05
        result = solve("annuity_varying", "pv_geometric",
                       {"i": i, "n": 10, "base": 100, "g": i})
        assert not math.isnan(result.answer)
        assert result.answer == pytest.approx(100 * 10 / (1 + i), abs=0.01)

    def test_unknown_ask(self):
        result = solve("annuity_varying", "__bogus__",
                       {"i": 0.05, "n": 10, "base": 100})
        assert math.isnan(result.answer)



class TestLoanAmortUnknown:
    def test_unknown_ask(self):
        result = solve("loan_amort", "__bogus__",
                       {"i": 0.05, "n": 10, "loan": 10000, "payment": 1295.05})
        assert math.isnan(result.answer)


class TestLoanSplitUnknown:
    def test_unknown_ask(self):
        result = solve("loan_split", "__bogus__",
                       {"i": 0.05, "n": 10, "loan": 10000, "payment": 1295.05})
        assert math.isnan(result.answer)


class TestSinkingFundUnknown:
    def test_unknown_ask(self):
        result = solve("sinking_fund", "__bogus__",
                       {"loan": 10000, "i_loan": 0.07, "i_fund": 0.05,
                        "n": 10, "deposit": 795.05})
        assert math.isnan(result.answer)



class TestBondPriceUnknown:
    def test_unknown_ask(self):
        result = solve("bond_price", "__bogus__",
                       {"face": 1000, "r": 0.06, "i": 0.07, "n": 10,
                        "Fr": 60, "price": 929.76})
        assert math.isnan(result.answer)


class TestBondMakehamUnknown:
    def test_unknown_ask(self):
        result = solve("bond_makeham", "__bogus__",
                       {"face": 1000, "r": 0.06, "i": 0.07, "g": 0.06,
                        "K": 508.35, "price": 929.76})
        assert math.isnan(result.answer)


class TestBondPremDiscUnknown:
    def test_unknown_ask(self):
        result = solve("bond_prem_disc", "__bogus__",
                       {"face": 1000, "r": 0.06, "i": 0.07, "n": 10,
                        "premium": -70.24})
        assert math.isnan(result.answer)



class TestMacaulayDurationUnknown:
    def test_unknown_ask(self):
        result = solve("macaulay_duration", "__bogus__",
                       {"i": 0.06, "d_mac": 7.8, "face": 1000, "r": 0.06, "n": 10})
        assert math.isnan(result.answer)


class TestModifiedDuration:
    def test_price_change_approx_empty_cashflows(self):
        """n=0 → cashflows list is empty → `if cashflows:` branch is False."""
        result = solve("modified_duration", "price_change_approx",
                       {"i": 0.05, "d_mod": 7.0, "d_mac": 7.35,
                        "n": 0, "face": 1000, "r": 0.05, "delta_i": 0.01})
        assert not math.isnan(result.answer)
        assert result.answer == pytest.approx(0.0, abs=0.01)

    def test_unknown_ask(self):
        result = solve("modified_duration", "__bogus__",
                       {"i": 0.05, "d_mod": 7.0, "d_mac": 7.35})
        assert math.isnan(result.answer)


class TestConvexityUnknown:
    def test_unknown_ask(self):
        result = solve("convexity", "__bogus__",
                       {"i": 0.06, "convexity": 68.9, "price": 950})
        assert math.isnan(result.answer)


class TestImmunization:
    def test_redington_conditions(self):
        result = solve("immunization", "redington_conditions", {})
        assert math.isnan(result.answer)  # answer is nan by design
        assert len(result.steps) >= 4

    def test_unknown_ask(self):
        result = solve("immunization", "__bogus__", {})
        assert math.isnan(result.answer)



class TestSpotForwardRates:
    def test_pv_using_spot_rates(self):
        result = solve("spot_forward_rates", "pv_using_spot_rates",
                       {"s1": 0.04, "s2": 0.045, "s3": 0.05,
                        "cf1": 50, "cf2": 50, "cf3": 1050})
        assert not math.isnan(result.answer)
        expected = 50 / 1.04 + 50 / 1.045**2 + 1050 / 1.05**3
        assert result.answer == pytest.approx(expected, abs=0.01)

    def test_unknown_ask(self):
        result = solve("spot_forward_rates", "__bogus__",
                       {"s1": 0.04, "s2": 0.045})
        assert math.isnan(result.answer)


class TestForwardContract:
    def test_forward_price_with_dividend(self):
        result = solve("forward_contract", "forward_price_with_dividend",
                       {"S0": 100, "r": 0.05, "T": 1, "q": 0.02})
        assert not math.isnan(result.answer)
        assert result.answer == pytest.approx(100 * (1.05 - 0.02) ** 1, abs=0.01)

    def test_unknown_ask(self):
        result = solve("forward_contract", "__bogus__",
                       {"S0": 100, "r": 0.05, "T": 1})
        assert math.isnan(result.answer)


class TestOptionPayoffUnknown:
    def test_unknown_ask(self):
        result = solve("option_payoff", "__bogus__",
                       {"K": 100, "ST": 105, "premium": 5})
        assert math.isnan(result.answer)


class TestPutCallParity:
    def test_arbitrage_check(self):
        result = solve("put_call_parity", "arbitrage_check",
                       {"S0": 100, "K": 100, "r": 0.05, "T": 1,
                        "call_price": 10, "put_price": 5.24, "pv_K": 95.24})
        assert not math.isnan(result.answer)
        assert result.answer >= 0  # arb profit is non-negative absolute value

    def test_unknown_ask(self):
        result = solve("put_call_parity", "__bogus__",
                       {"S0": 100, "K": 100, "r": 0.05, "T": 1,
                        "call_price": 10, "put_price": 5.24, "pv_K": 95.24})
        assert math.isnan(result.answer)


class TestSwapRateUnknown:
    def test_unknown_ask(self):
        result = solve("swap_rate", "__bogus__",
                       {"spot_rates": [0.04, 0.045], "R": 0.044,
                        "n": 2, "notional": 1_000_000})
        assert math.isnan(result.answer)
