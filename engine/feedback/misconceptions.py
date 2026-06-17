"""FM misconception notes keyed by (kind, ask, distractor_index)."""
from __future__ import annotations

_notes: dict[tuple[str, str], list[str]] = {
    ("interest_tvm", "future_value"): [
        "Used simple interest (PV·(1+i·n)) instead of compound: (1+i)^n.",
        "Forgot to raise the whole factor to the power n — check order of operations.",
        "Divided instead of multiplied by the accumulation factor.",
    ],
    ("interest_tvm", "present_value"): [
        "Multiplied instead of divided by the accumulation factor.",
        "Used simple discount (FV·(1-d·n)) instead of compound discounting.",
        "Applied discount for wrong number of periods.",
    ],
    ("interest_tvm", "interest_rate_solve"): [
        "Took a linear (simple-interest) root: (FV/PV-1)/n instead of (FV/PV)^(1/n)-1.",
        "Forgot to subtract 1 after taking the nth root.",
    ],
    ("interest_tvm", "periods_solve"): [
        "Used simple interest formula n=(FV-PV)/(PV·i) — correct only for simple interest.",
        "Applied ln to wrong ratio: used PV/FV instead of FV/PV.",
    ],
    ("interest_nominal", "nominal_to_effective"): [
        "Added i^(m) directly to effective rate without converting — ignores compounding.",
        "Divided effective by m instead of converting via (1+i/m)^m.",
    ],
    ("interest_nominal", "effective_to_nominal"): [
        "Multiplied effective rate by m — ignores compounding within the year.",
    ],
    ("interest_force", "force_from_rate"): [
        "Used δ = i/(1+i) (discount rate formula) instead of δ = ln(1+i).",
    ],
    ("interest_force", "rate_from_force"): [
        "Used i = 1-e^(-δ) (present-value discount) instead of i = e^δ-1.",
    ],
    ("interest_discount", "discount_from_interest"): [
        "Confused discount rate with interest rate: wrote d = i instead of d = i/(1+i).",
    ],
    ("annuity_immediate", "pv_annuity_imm"): [
        "Used annuity-due factor (1+i)·a_n| — annuity-immediate has payments at END of period.",
        "Forgot to divide by i in the factor formula.",
        "Used (1+i)^n in numerator instead of denominator.",
    ],
    ("annuity_due", "pv_annuity_due"): [
        "Used annuity-immediate factor a_n| — annuity-due shifts payments one period earlier.",
    ],
    ("perpetuity", "pv_perp_imm"): [
        "Used PMT·(1+i)/i (perpetuity-due formula) instead of PMT/i.",
    ],
    ("perpetuity", "pv_perp_due"): [
        "Used PMT/i (perpetuity-immediate) instead of PMT·(1+i)/i.",
    ],
    ("deferred_annuity", "pv_deferred_imm"): [
        "Forgot to discount back the m deferral periods — omitted v^m factor.",
        "Added deferral and annuity periods: used a_{m+n}| instead of v^m·a_n|.",
    ],
    ("loan_amort", "payment_amount"): [
        "Divided by s_n| (FV factor) instead of a_n| (PV factor).",
        "Used simple interest to compute payment: Loan·(1+i·n)/n.",
    ],
    ("loan_amort", "outstanding_prospective"): [
        "Used retrospective formula instead: Loan·(1+i)^t - PMT·s_t|.",
    ],
    ("loan_amort", "outstanding_retrospective"): [
        "Used prospective formula instead: PMT·a_{n-t}|.",
        "Forgot to accumulate the original loan: missed (1+i)^t factor.",
    ],
    ("loan_split", "interest_tth_payment"): [
        "Took I_t = OB_{t-1}·i which is correct but requires computing OB first; "
        "shortcut I_t = PMT·(1-v^{n-t+1}) avoids that intermediate step.",
        "Used v^t instead of v^{n-t+1}.",
    ],
    ("loan_split", "principal_tth_payment"): [
        "Used P_t = PMT·v^t instead of PMT·v^{n-t+1}.",
    ],
    ("sinking_fund", "sinking_fund_deposit"): [
        "Divided loan by a_n| (amortization factor) instead of s_n| (accumulation factor).",
    ],
    ("sinking_fund", "total_periodic_outlay"): [
        "Only added interest payment, forgot the sinking fund deposit.",
        "Used amortization PMT instead of interest + sinking fund components.",
    ],
    ("bond_price", "price_from_yield"): [
        "Forgot the redemption term C·v^n — only computed Fr·a_n|.",
        "Used simple interest instead of compound: Fr·n·v instead of Fr·a_n|.",
    ],
    ("bond_price", "current_yield"): [
        "Confused current yield (coupon/price) with yield-to-maturity.",
        "Used coupon rate r instead of coupon amount Fr in numerator.",
    ],
    ("bond_makeham", "makeham_price"): [
        "Forgot the Makeham adjustment; just used P = K + Fr·a_n| instead.",
        "Used i instead of g in the (g/i) multiplier.",
    ],
    ("bond_prem_disc", "premium_discount_amount"): [
        "Forgot the a_n| factor: wrote (r-i)·C instead of (r-i)·C·a_n|.",
        "Confused premium/discount direction: premium when r > i, discount when r < i.",
    ],
    ("bond_prem_disc", "book_value_tth"): [
        "Used a_{n-t+1}| (off-by-one) instead of a_{n-t}|.",
        "Used a_n| (original) instead of remaining a_{n-t}|.",
    ],
    ("macaulay_duration", "macaulay_duration_bond"): [
        "Forgot to divide by price P: computed Σ t·PV(CF_t) without normalizing.",
        "Confused with modified duration — those differ by a (1+i) factor.",
    ],
    ("modified_duration", "modified_duration_from_mac"): [
        "Multiplied D_mac by (1+i) instead of dividing: D_mod = D_mac/(1+i).",
    ],
    ("modified_duration", "price_change_approx"): [
        "Forgot the negative sign: ΔP ≈ -D_mod·P·Δi.",
        "Used D_mac instead of D_mod in the approximation.",
    ],
    ("convexity", "second_order_price_approx"): [
        "Forgot the ½ in front of the convexity term.",
        "Only used first-order (duration) approximation.",
    ],
    ("immunization", "full_immunization_weight"): [
        "Used w₁ = (H-d₁)/(d₂-d₁) instead of (d₂-H)/(d₂-d₁) — wrong weight assignment.",
    ],
    ("spot_forward_rates", "implied_forward_rate"): [
        "Used simple arithmetic: f₁,₂ = 2·s₂ - s₁ instead of the compound formula.",
    ],
    ("forward_contract", "forward_price_no_dividend"): [
        "Used simple interest: F = S₀·(1+r·T) instead of compound F = S₀·(1+r)^T.",
    ],
    ("option_payoff", "call_payoff"): [
        "Included the premium in payoff — payoff is pre-cost; profit deducts the premium.",
        "Used put formula max(K-ST,0) instead of call max(ST-K,0).",
    ],
    ("option_payoff", "put_payoff"): [
        "Used call formula max(ST-K,0) instead of put max(K-ST,0).",
    ],
    ("put_call_parity", "find_put_from_call"): [
        "Used K instead of PV(K) = K/(1+r)^T — forgot time-value of strike.",
        "Rearranged parity incorrectly: P = C - S₀ + PV(K) is correct; check signs.",
    ],
    ("swap_rate", "fixed_swap_rate"): [
        "Averaged spot rates instead of using the proper discount-factor formula.",
        "Used K at face instead of present-valued notional v_n.",
    ],
}


def get_notes(kind: str, ask: str, distractor_index: int) -> str:
    """Return a misconception note for the given kind/ask and which wrong answer was chosen."""
    key = (kind, ask)
    notes = _notes.get(key, [])
    if not notes:
        return ""
    # Notes aren't written one-per-distractor-slot (make_mc_choices can
    # generate more wrong answers than a kind/ask has documented
    # misconceptions for), so wrap around with modulo instead of indexing
    # directly — this never raises even if distractor_index is out of range.
    return notes[distractor_index % len(notes)]
