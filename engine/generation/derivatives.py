"""Generators for spot/forward rates, forward contracts, options, put-call parity, swaps."""
from __future__ import annotations

import numpy as np

from engine.generation.base import Problem, make_mc_choices, register


@register("spot_forward_rates")
def gen_spot_forward_rates(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    s1 = round(float(rng.uniform(*ranges["rate_range"])), 4)
    s2 = round(float(rng.uniform(*ranges["rate_range"])), 4)

    if ask == "implied_forward_rate":
        # (1+s1)^1 * (1+f_{1,2}) = (1+s2)^2  => f = (1+s2)^2/(1+s1) - 1
        f12 = round((1 + s2) ** 2 / (1 + s1) - 1, 6)
        ans = f12
        stmt = (f"The 1-year spot rate is {s1*100:.2f}% and the 2-year spot rate is "
                f"{s2*100:.2f}%. Find the implied 1-year forward rate one year from now "
                f"using (1+s₁)(1+f₁,₂) = (1+s₂)².")
        wrongs = [round((s2 * 2 - s1), 4),
                  round(s2 - s1, 4),
                  round(f12 * 1.1, 4)]

    elif ask == "pv_using_spot_rates":
        s3 = round(float(rng.uniform(*ranges["rate_range"])), 4)
        cf1 = round(float(rng.uniform(100, 500)), 2)
        cf2 = round(float(rng.uniform(100, 500)), 2)
        cf3 = round(float(rng.uniform(1000, 3000)), 2)
        pv = round(cf1 / (1 + s1) + cf2 / (1 + s2) ** 2 + cf3 / (1 + s3) ** 3, 2)
        ans = pv
        stmt = (f"Cash flows: CF₁=${cf1:.2f}, CF₂=${cf2:.2f}, CF₃=${cf3:.2f}. "
                f"Spot rates: s₁={s1*100:.2f}%, s₂={s2*100:.2f}%, s₃={s3*100:.2f}%. "
                f"Find the present value by discounting each cash flow at the matching spot rate.")
        wrongs = [round(pv * (1 + s1), 2),
                  round((cf1 + cf2 + cf3) / (1 + s2) ** 2, 2),
                  round(pv * 1.05, 2)]
        return Problem("spot_forward_rates", ask, stmt, ans,
                       make_mc_choices(ans, wrongs, rng),
                       params={"s1": s1, "s2": s2, "s3": s3, "cf1": cf1, "cf2": cf2, "cf3": cf3}, seed=seed)

    else:
        raise ValueError(f"Unknown ask '{ask}' for spot_forward_rates")

    return Problem("spot_forward_rates", ask, stmt, ans,
                   make_mc_choices(ans, wrongs, rng),
                   params={"s1": s1, "s2": s2}, seed=seed)


@register("forward_contract")
def gen_forward_contract(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    S0 = round(float(rng.uniform(*ranges["spot_price_range"])), 2)   # spot price
    r = round(float(rng.uniform(*ranges["rate_range"])), 4)           # risk-free rate
    T = int(rng.integers(*ranges["maturity_range"]))                   # years

    if ask == "forward_price_no_dividend":
        F = round(S0 * (1 + r) ** T, 2)
        ans = F
        stmt = (f"An asset has spot price ${S0:,.2f}. Risk-free rate is {r*100:.2f}% per year. "
                f"Find the {T}-year forward price: F = S₀(1+r)^T.")
        wrongs = [round(S0 * (1 + r * T), 2),
                  round(S0 + r * T * S0, 2),
                  round(F * 1.02, 2)]

    elif ask == "forward_price_with_dividend":
        q = round(float(rng.uniform(0.01, 0.05)), 4)   # dividend yield
        F = round(S0 * (1 + r - q) ** T, 2)
        ans = F
        stmt = (f"An asset has spot price ${S0:,.2f}, risk-free rate {r*100:.2f}%, "
                f"and dividend yield {q*100:.2f}% (all annual). "
                f"Find the {T}-year forward price: F = S₀·(1+r-q)^T.")
        wrongs = [round(S0 * (1 + r) ** T, 2),
                  round(S0 * (1 + r) ** T * (1 - q * T), 2),
                  round(F * 1.03, 2)]
        return Problem("forward_contract", ask, stmt, ans,
                       make_mc_choices(ans, wrongs, rng),
                       params={"S0": S0, "r": r, "T": T, "q": q}, seed=seed)

    elif ask == "forward_payoff":
        K = round(S0 * (1 + r) ** T, 2)   # set K = forward price (zero cost entry)
        ST = round(float(rng.uniform(S0 * 0.8, S0 * 1.3)), 2)
        long = round(ST - K, 2)
        ans = long
        stmt = (f"You entered a long forward contract with delivery price K=${K:,.2f}. "
                f"At expiry the asset price is ${ST:,.2f}. Compute the payoff (ST - K).")
        wrongs = [round(K - ST, 2),
                  round(abs(ST - K), 2),
                  round(long * (1 + r) ** T, 2)]

    else:
        raise ValueError(f"Unknown ask '{ask}' for forward_contract")

    return Problem("forward_contract", ask, stmt, ans,
                   make_mc_choices(ans, wrongs, rng),
                   params={"S0": S0, "r": r, "T": T}, seed=seed)


@register("option_payoff")
def gen_option_payoff(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    K = round(float(rng.uniform(*ranges["strike_range"])), 2)
    ST = round(float(rng.uniform(K * 0.7, K * 1.4)), 2)
    premium = round(float(rng.uniform(1, 20)), 2)

    if ask == "call_payoff":
        payoff = round(max(ST - K, 0), 2)
        ans = payoff
        stmt = (f"A European call option has strike K=${K:,.2f} and costs ${premium:.2f}. "
                f"At expiry, the asset price is ${ST:,.2f}. Find the payoff (not profit): "
                f"max(ST - K, 0).")
        wrongs = [round(max(K - ST, 0), 2),
                  round(max(ST - K - premium, 0), 2),
                  round(ST - K, 2)]

    elif ask == "put_payoff":
        payoff = round(max(K - ST, 0), 2)
        ans = payoff
        stmt = (f"A European put option has strike K=${K:,.2f} and costs ${premium:.2f}. "
                f"At expiry, the asset price is ${ST:,.2f}. Find the payoff: max(K - ST, 0).")
        wrongs = [round(max(ST - K, 0), 2),
                  round(max(K - ST - premium, 0), 2),
                  round(K - ST, 2)]

    elif ask == "call_profit":
        payoff = max(ST - K, 0)
        profit = round(payoff - premium, 2)
        ans = profit
        stmt = (f"A European call option has strike K=${K:,.2f} and costs ${premium:.2f}. "
                f"At expiry, the asset price is ${ST:,.2f}. Find the profit: payoff − premium.")
        wrongs = [round(payoff, 2),
                  round(profit + premium, 2),
                  round(-premium if payoff == 0 else profit * 1.1, 2)]

    elif ask == "put_profit":
        payoff = max(K - ST, 0)
        profit = round(payoff - premium, 2)
        ans = profit
        stmt = (f"A European put option has strike K=${K:,.2f} and costs ${premium:.2f}. "
                f"At expiry, the asset price is ${ST:,.2f}. Find the profit: payoff − premium.")
        wrongs = [round(payoff, 2),
                  round(profit + premium, 2),
                  round(-premium if payoff == 0 else profit * 1.1, 2)]

    else:
        raise ValueError(f"Unknown ask '{ask}' for option_payoff")

    return Problem("option_payoff", ask, stmt, ans,
                   make_mc_choices(ans, wrongs, rng),
                   params={"K": K, "ST": ST, "premium": premium}, seed=seed)


@register("put_call_parity")
def gen_put_call_parity(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    S0 = round(float(rng.uniform(*ranges["spot_price_range"])), 2)
    K = round(S0 * float(rng.uniform(0.9, 1.1)), 2)
    r = round(float(rng.uniform(*ranges["rate_range"])), 4)
    T = int(rng.integers(*ranges["maturity_range"]))
    call_price = round(float(rng.uniform(1, 30)), 2)
    pv_K = round(K / (1 + r) ** T, 2)
    # put-call parity: C + PV(K) = P + S0  =>  P = C + PV(K) - S0
    put_price = round(call_price + pv_K - S0, 2)

    if ask == "find_put_from_call":
        ans = put_price
        stmt = (f"European call price C=${call_price:.2f}, spot S₀=${S0:,.2f}, "
                f"strike K=${K:,.2f}, rate {r*100:.2f}%, maturity T={T} yr. "
                f"Find the put price using put-call parity: P = C + PV(K) − S₀.")
        wrongs = [round(call_price - pv_K + S0, 2),
                  round(call_price + K - S0, 2),
                  round(put_price * 1.1, 2)]

    elif ask == "find_call_from_put":
        call_ans = round(put_price + S0 - pv_K, 2)
        ans = call_ans
        stmt = (f"European put price P=${put_price:.2f}, spot S₀=${S0:,.2f}, "
                f"strike K=${K:,.2f}, rate {r*100:.2f}%, maturity T={T} yr. "
                f"Find the call price using put-call parity: C = P + S₀ − PV(K).")
        wrongs = [round(call_ans + pv_K - S0, 2),
                  round(put_price + pv_K - S0, 2),
                  round(call_ans * 1.1, 2)]

    elif ask == "arbitrage_check":
        # Give a mispriced call and ask for arbitrage profit
        call_mkt = round(call_price * float(rng.uniform(1.05, 1.2)), 2)
        arb = round(call_mkt - call_price, 2)
        ans = arb
        stmt = (f"Put-call parity implies call price = ${call_price:.2f}, but the market "
                f"quotes the call at ${call_mkt:.2f} (spot=${S0:,.2f}, K=${K:,.2f}, "
                f"r={r*100:.2f}%, T={T} yr, put=${put_price:.2f}). "
                f"Find the risk-free arbitrage profit per contract.")
        wrongs = [round(arb * (1 + r) ** T, 2),
                  round(arb / 2, 2),
                  round(arb * 0.9, 2)]

    else:
        raise ValueError(f"Unknown ask '{ask}' for put_call_parity")

    return Problem("put_call_parity", ask, stmt, ans,
                   make_mc_choices(ans, wrongs, rng),
                   params={"S0": S0, "K": K, "r": r, "T": T,
                           "call_price": call_price, "put_price": put_price, "pv_K": pv_K}, seed=seed)


@register("swap_rate")
def gen_swap_rate(ask: str, ranges: dict, seed: int) -> Problem:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(*ranges["n_range"]))   # number of settlement periods
    # Generate n spot rates
    base_rate = round(float(rng.uniform(*ranges["rate_range"])), 4)
    spot_rates = [round(base_rate + 0.001 * k + float(rng.uniform(-0.002, 0.002)), 4)
                  for k in range(n)]
    notional = round(float(rng.choice([1_000_000, 500_000, 100_000])), 0)

    # Swap rate R: Σ R/(1+s_t)^t = 1 - 1/(1+s_n)^n
    # R = (1 - v_n) / Σ v_t  where v_t = 1/(1+s_t)^t
    v = [1 / (1 + spot_rates[t]) ** (t + 1) for t in range(n)]
    pv_K = v[-1]      # PV of notional at maturity
    sum_v = sum(v)
    R = round((1 - pv_K) / sum_v, 6)

    if ask == "fixed_swap_rate":
        ans = R
        spot_str = ", ".join(f"s_{t+1}={spot_rates[t]*100:.2f}%" for t in range(n))
        stmt = (f"A {n}-period interest rate swap uses spot rates: {spot_str}. "
                f"Find the fixed swap rate R so the swap has zero value at initiation "
                f"using R = (1 - v_n) / Σ vₜ.")
        wrongs = [round(sum(spot_rates) / n, 6),
                  round(R * (1 + spot_rates[-1]), 6),
                  round(R * 1.05, 6)]

    elif ask == "fixed_payment":
        ans = round(R * notional, 2)
        stmt = (f"A ${notional:,.0f} notional {n}-period swap has fixed rate {R*100:.4f}%. "
                f"Find the fixed-leg payment each period.")
        wrongs = [round(R * notional * (1 + spot_rates[0]), 2),
                  round(notional * spot_rates[-1], 2),
                  round(R * notional * 1.05, 2)]

    else:
        raise ValueError(f"Unknown ask '{ask}' for swap_rate")

    return Problem("swap_rate", ask, stmt, ans,
                   make_mc_choices(ans, wrongs, rng),
                   params={"n": n, "spot_rates": spot_rates, "R": R, "notional": notional}, seed=seed)
