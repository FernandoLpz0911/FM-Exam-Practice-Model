"""Generators for spot/forward rates, forward contracts, options, put-call parity, swaps."""
from __future__ import annotations

import numpy as np

from engine.generation.base import Problem, make_mc_choices, register


@register("spot_forward_rates")
def gen_spot_forward_rates(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    spot_rate_1yr = round(float(rng.uniform(*ranges["rate_range"])), 4)
    spot_rate_2yr = round(float(rng.uniform(*ranges["rate_range"])), 4)

    if ask == "implied_forward_rate":
        # No-arbitrage condition: investing for 2 years directly must equal
        # investing for 1 year then rolling into the implied forward rate.
        # (1+s1)^1 * (1+f_1,2) = (1+s2)^2  =>  f_1,2 = (1+s2)^2/(1+s1) - 1
        forward_rate_1to2 = round((1 + spot_rate_2yr) ** 2 / (1 + spot_rate_1yr) - 1, 6)
        answer = forward_rate_1to2
        question_text = (
            f"The 1-year spot rate is {spot_rate_1yr*100:.2f}% and the 2-year spot "
            f"rate is {spot_rate_2yr*100:.2f}%. Find the implied 1-year forward rate "
            f"one year from now using (1+s₁)(1+f₁,₂) = (1+s₂)²."
        )
        wrong_answers = [
            round(spot_rate_2yr * 2 - spot_rate_1yr, 4),
            round(spot_rate_2yr - spot_rate_1yr, 4),
            round(forward_rate_1to2 * 1.1, 4),
        ]

    elif ask == "pv_using_spot_rates":
        spot_rate_3yr = round(float(rng.uniform(*ranges["rate_range"])), 4)
        cash_flow_1 = round(float(rng.uniform(100, 500)), 2)
        cash_flow_2 = round(float(rng.uniform(100, 500)), 2)
        cash_flow_3 = round(float(rng.uniform(1000, 3000)), 2)
        present_value = round(
            cash_flow_1 / (1 + spot_rate_1yr)
            + cash_flow_2 / (1 + spot_rate_2yr) ** 2
            + cash_flow_3 / (1 + spot_rate_3yr) ** 3, 2
        )
        answer = present_value
        question_text = (
            f"Cash flows: CF₁=${cash_flow_1:.2f}, CF₂=${cash_flow_2:.2f}, "
            f"CF₃=${cash_flow_3:.2f}. Spot rates: s₁={spot_rate_1yr*100:.2f}%, "
            f"s₂={spot_rate_2yr*100:.2f}%, s₃={spot_rate_3yr*100:.2f}%. "
            f"Find the present value by discounting each cash flow at the "
            f"matching spot rate."
        )
        wrong_answers = [
            round(present_value * (1 + spot_rate_1yr), 2),
            round((cash_flow_1 + cash_flow_2 + cash_flow_3) / (1 + spot_rate_2yr) ** 2, 2),
            round(present_value * 1.05, 2),
        ]
        return Problem(
            "spot_forward_rates", ask, question_text, answer,
            make_mc_choices(answer, wrong_answers, rng),
            params={
                "s1": spot_rate_1yr, "s2": spot_rate_2yr, "s3": spot_rate_3yr,
                "cf1": cash_flow_1, "cf2": cash_flow_2, "cf3": cash_flow_3,
            },
            seed=seed,
        )

    else:
        raise ValueError(f"Unknown ask '{ask}' for spot_forward_rates")

    return Problem(
        "spot_forward_rates", ask, question_text, answer,
        make_mc_choices(answer, wrong_answers, rng),
        params={"s1": spot_rate_1yr, "s2": spot_rate_2yr}, seed=seed,
    )


@register("forward_contract")
def gen_forward_contract(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    spot_price = round(float(rng.uniform(*ranges["spot_price_range"])), 2)
    risk_free_rate = round(float(rng.uniform(*ranges["rate_range"])), 4)
    maturity_years = int(rng.integers(*ranges["maturity_range"]))

    if ask == "forward_price_no_dividend":
        forward_price = round(spot_price * (1 + risk_free_rate) ** maturity_years, 2)
        answer = forward_price
        question_text = (
            f"An asset has spot price ${spot_price:,.2f}. Risk-free rate is "
            f"{risk_free_rate*100:.2f}% per year. "
            f"Find the {maturity_years}-year forward price: F = S₀(1+r)^T."
        )
        wrong_answers = [
            round(spot_price * (1 + risk_free_rate * maturity_years), 2),
            round(spot_price + risk_free_rate * maturity_years * spot_price, 2),
            round(forward_price * 1.02, 2),
        ]

    elif ask == "forward_price_with_dividend":
        dividend_yield = round(float(rng.uniform(0.01, 0.05)), 4)
        # A continuous dividend yield q reduces the cost-of-carry rate from r
        # to (r-q), since holding the asset itself now pays a return too.
        forward_price = round(
            spot_price * (1 + risk_free_rate - dividend_yield) ** maturity_years, 2
        )
        answer = forward_price
        question_text = (
            f"An asset has spot price ${spot_price:,.2f}, risk-free rate "
            f"{risk_free_rate*100:.2f}%, and dividend yield "
            f"{dividend_yield*100:.2f}% (all annual). "
            f"Find the {maturity_years}-year forward price: F = S₀·(1+r-q)^T."
        )
        wrong_answers = [
            round(spot_price * (1 + risk_free_rate) ** maturity_years, 2),
            round(spot_price * (1 + risk_free_rate) ** maturity_years
                  * (1 - dividend_yield * maturity_years), 2),
            round(forward_price * 1.03, 2),
        ]
        return Problem(
            "forward_contract", ask, question_text, answer,
            make_mc_choices(answer, wrong_answers, rng),
            params={
                "S0": spot_price, "r": risk_free_rate, "T": maturity_years,
                "q": dividend_yield,
            },
            seed=seed,
        )

    elif ask == "forward_payoff":
        # Setting the delivery price equal to today's no-arbitrage forward
        # price means the contract costs nothing to enter (standard convention).
        delivery_price = round(spot_price * (1 + risk_free_rate) ** maturity_years, 2)
        price_at_expiry = round(float(rng.uniform(spot_price * 0.8, spot_price * 1.3)), 2)
        payoff = round(price_at_expiry - delivery_price, 2)
        answer = payoff
        question_text = (
            f"You entered a long forward contract with delivery price "
            f"K=${delivery_price:,.2f}. At expiry the asset price is "
            f"${price_at_expiry:,.2f}. Compute the payoff (ST - K)."
        )
        wrong_answers = [
            round(delivery_price - price_at_expiry, 2),
            round(abs(price_at_expiry - delivery_price), 2),
            round(payoff * (1 + risk_free_rate) ** maturity_years, 2),
        ]
        # NOTE: delivery_price and price_at_expiry are computed here but not
        # saved into params below — the solver (_s_forward_contract,
        # forward_payoff branch in solve.py) falls back to recomputing
        # delivery_price from S0/r/T (which happens to match) but defaults
        # price_at_expiry to S0, which will NOT match this question's actual
        # randomly-drawn expiry price.

    else:
        raise ValueError(f"Unknown ask '{ask}' for forward_contract")

    return Problem(
        "forward_contract", ask, question_text, answer,
        make_mc_choices(answer, wrong_answers, rng),
        params={"S0": spot_price, "r": risk_free_rate, "T": maturity_years}, seed=seed,
    )


@register("option_payoff")
def gen_option_payoff(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    strike_price = round(float(rng.uniform(*ranges["strike_range"])), 2)
    price_at_expiry = round(float(rng.uniform(strike_price * 0.7, strike_price * 1.4)), 2)
    premium = round(float(rng.uniform(1, 20)), 2)

    if ask == "call_payoff":
        payoff = round(max(price_at_expiry - strike_price, 0), 2)
        answer = payoff
        question_text = (
            f"A European call option has strike K=${strike_price:,.2f} and costs "
            f"${premium:.2f}. At expiry, the asset price is ${price_at_expiry:,.2f}. "
            f"Find the payoff (not profit): max(ST - K, 0)."
        )
        wrong_answers = [
            round(max(strike_price - price_at_expiry, 0), 2),
            round(max(price_at_expiry - strike_price - premium, 0), 2),
            round(price_at_expiry - strike_price, 2),
        ]

    elif ask == "put_payoff":
        payoff = round(max(strike_price - price_at_expiry, 0), 2)
        answer = payoff
        question_text = (
            f"A European put option has strike K=${strike_price:,.2f} and costs "
            f"${premium:.2f}. At expiry, the asset price is ${price_at_expiry:,.2f}. "
            f"Find the payoff: max(K - ST, 0)."
        )
        wrong_answers = [
            round(max(price_at_expiry - strike_price, 0), 2),
            round(max(strike_price - price_at_expiry - premium, 0), 2),
            round(strike_price - price_at_expiry, 2),
        ]

    elif ask == "call_profit":
        payoff = max(price_at_expiry - strike_price, 0)
        profit = round(payoff - premium, 2)
        answer = profit
        question_text = (
            f"A European call option has strike K=${strike_price:,.2f} and costs "
            f"${premium:.2f}. At expiry, the asset price is ${price_at_expiry:,.2f}. "
            f"Find the profit: payoff − premium."
        )
        wrong_answers = [
            round(payoff, 2),
            round(profit + premium, 2),
            round(-premium if payoff == 0 else profit * 1.1, 2),
        ]

    elif ask == "put_profit":
        payoff = max(strike_price - price_at_expiry, 0)
        profit = round(payoff - premium, 2)
        answer = profit
        question_text = (
            f"A European put option has strike K=${strike_price:,.2f} and costs "
            f"${premium:.2f}. At expiry, the asset price is ${price_at_expiry:,.2f}. "
            f"Find the profit: payoff − premium."
        )
        wrong_answers = [
            round(payoff, 2),
            round(profit + premium, 2),
            round(-premium if payoff == 0 else profit * 1.1, 2),
        ]

    else:
        raise ValueError(f"Unknown ask '{ask}' for option_payoff")

    return Problem(
        "option_payoff", ask, question_text, answer,
        make_mc_choices(answer, wrong_answers, rng),
        params={"K": strike_price, "ST": price_at_expiry, "premium": premium}, seed=seed,
    )


@register("put_call_parity")
def gen_put_call_parity(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    spot_price = round(float(rng.uniform(*ranges["spot_price_range"])), 2)
    strike_price = round(spot_price * float(rng.uniform(0.9, 1.1)), 2)
    risk_free_rate = round(float(rng.uniform(*ranges["rate_range"])), 4)
    maturity_years = int(rng.integers(*ranges["maturity_range"]))
    call_price = round(float(rng.uniform(1, 30)), 2)
    strike_pv = round(strike_price / (1 + risk_free_rate) ** maturity_years, 2)
    # Put-call parity: C + PV(K) = P + S0  =>  P = C + PV(K) - S0
    put_price = round(call_price + strike_pv - spot_price, 2)

    if ask == "find_put_from_call":
        answer = put_price
        question_text = (
            f"European call price C=${call_price:.2f}, spot S₀=${spot_price:,.2f}, "
            f"strike K=${strike_price:,.2f}, rate {risk_free_rate*100:.2f}%, "
            f"maturity T={maturity_years} yr. Find the put price using "
            f"put-call parity: P = C + PV(K) − S₀."
        )
        wrong_answers = [
            round(call_price - strike_pv + spot_price, 2),
            round(call_price + strike_price - spot_price, 2),
            round(put_price * 1.1, 2),
        ]

    elif ask == "find_call_from_put":
        call_answer = round(put_price + spot_price - strike_pv, 2)
        answer = call_answer
        question_text = (
            f"European put price P=${put_price:.2f}, spot S₀=${spot_price:,.2f}, "
            f"strike K=${strike_price:,.2f}, rate {risk_free_rate*100:.2f}%, "
            f"maturity T={maturity_years} yr. Find the call price using "
            f"put-call parity: C = P + S₀ − PV(K)."
        )
        wrong_answers = [
            round(call_answer + strike_pv - spot_price, 2),
            round(put_price + strike_pv - spot_price, 2),
            round(call_answer * 1.1, 2),
        ]

    elif ask == "arbitrage_check":
        # Quote the call above its fair parity price; the arbitrage profit is
        # simply the mispricing, since the parity-implied position is risk-free.
        market_call_price = round(call_price * float(rng.uniform(1.05, 1.2)), 2)
        arbitrage_profit = round(market_call_price - call_price, 2)
        answer = arbitrage_profit
        question_text = (
            f"Put-call parity implies call price = ${call_price:.2f}, but the "
            f"market quotes the call at ${market_call_price:.2f} "
            f"(spot=${spot_price:,.2f}, K=${strike_price:,.2f}, "
            f"r={risk_free_rate*100:.2f}%, T={maturity_years} yr, "
            f"put=${put_price:.2f}). Find the risk-free arbitrage profit per contract."
        )
        wrong_answers = [
            round(arbitrage_profit * (1 + risk_free_rate) ** maturity_years, 2),
            round(arbitrage_profit / 2, 2),
            round(arbitrage_profit * 0.9, 2),
        ]

    else:
        raise ValueError(f"Unknown ask '{ask}' for put_call_parity")

    return Problem(
        "put_call_parity", ask, question_text, answer,
        make_mc_choices(answer, wrong_answers, rng),
        params={
            "S0": spot_price, "K": strike_price, "r": risk_free_rate,
            "T": maturity_years, "call_price": call_price, "put_price": put_price,
            "pv_K": strike_pv,
        },
        seed=seed,
    )


@register("swap_rate")
def gen_swap_rate(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    n_periods = int(rng.integers(*ranges["n_range"]))
    base_rate = round(float(rng.uniform(*ranges["rate_range"])), 4)
    # Build a slightly upward-sloping term structure around base_rate so the
    # swap-rate formula has genuinely different spot rates per period to work with.
    spot_rates = [
        round(base_rate + 0.001 * period + float(rng.uniform(-0.002, 0.002)), 4)
        for period in range(n_periods)
    ]
    notional = round(float(rng.choice([1_000_000, 500_000, 100_000])), 0)

    # Swap rate R solves Σ R/(1+s_t)^t = 1 - 1/(1+s_n)^n (the fixed leg's PV
    # must equal the floating leg's PV for zero value at initiation):
    # R = (1 - v_n) / Σ v_t  where v_t = 1/(1+s_t)^t
    discount_factors = [1 / (1 + spot_rates[t]) ** (t + 1) for t in range(n_periods)]
    notional_pv_factor = discount_factors[-1]
    sum_discount_factors = sum(discount_factors)
    swap_rate = round((1 - notional_pv_factor) / sum_discount_factors, 6)

    if ask == "fixed_swap_rate":
        answer = swap_rate
        spot_rate_summary = ", ".join(
            f"s_{t+1}={spot_rates[t]*100:.2f}%" for t in range(n_periods)
        )
        question_text = (
            f"A {n_periods}-period interest rate swap uses spot rates: "
            f"{spot_rate_summary}. Find the fixed swap rate R so the swap has "
            f"zero value at initiation using R = (1 - v_n) / Σ vₜ."
        )
        wrong_answers = [
            round(sum(spot_rates) / n_periods, 6),
            round(swap_rate * (1 + spot_rates[-1]), 6),
            round(swap_rate * 1.05, 6),
        ]

    elif ask == "fixed_payment":
        answer = round(swap_rate * notional, 2)
        question_text = (
            f"A ${notional:,.0f} notional {n_periods}-period swap has fixed "
            f"rate {swap_rate*100:.4f}%. Find the fixed-leg payment each period."
        )
        wrong_answers = [
            round(swap_rate * notional * (1 + spot_rates[0]), 2),
            round(notional * spot_rates[-1], 2),
            round(swap_rate * notional * 1.05, 2),
        ]

    else:
        raise ValueError(f"Unknown ask '{ask}' for swap_rate")

    return Problem(
        "swap_rate", ask, question_text, answer,
        make_mc_choices(answer, wrong_answers, rng),
        params={
            "n": n_periods, "spot_rates": spot_rates, "R": swap_rate,
            "notional": notional,
        },
        seed=seed,
    )
