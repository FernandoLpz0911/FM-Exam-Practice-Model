"""Difficulty tiers for every (generator_kind, ask) combination.

Tier 1 — direct formula application: plug in values, single step.
Tier 2 — multi-step application: combine formulas, set up the calculation.
Tier 3 — novel / combined reasoning: memoryless property, LOTUS, Jacobians, etc.

Reps thresholds are in engine.config. Unknown (kind, ask) pairs default to tier 2.
"""
from __future__ import annotations

ASK_DIFFICULTY: dict[tuple[str, str], int] = {
    # ── General Probability ───────────────────────────────────────────────────
    ("set_inclusion_exclusion", "union_prob"): 1,
    ("set_inclusion_exclusion", "complement_prob"): 1,
    ("addition_rule", "union_prob"): 1,
    ("addition_rule", "either_or"): 1,
    ("bayes", "posterior"): 2,
    ("combinatorics", "nCr"): 1,
    ("combinatorics", "nPr"): 1,
    ("combinatorics", "multinomial"): 2,
    ("conditional", "cond_prob"): 2,
    ("conditional", "joint_from_cond"): 2,
    ("independence", "is_independent"): 1,
    ("independence", "joint_indep"): 2,
    ("counting_prob", "draw_prob"): 1,
    ("counting_prob", "arrangement_prob"): 2,
    # ── Univariate basics ─────────────────────────────────────────────────────
    ("density_basics", "normalize_constant"): 1,
    ("density_basics", "prob_interval"): 1,
    ("density_basics", "cdf_from_pdf"): 2,
    ("expectation_generic", "mean"): 1,
    ("expectation_generic", "lotus"): 3,
    ("variance_generic", "variance"): 1,
    ("variance_generic", "sd"): 1,
    ("moments", "variance_from_moments"): 2,
    ("moments", "ex_squared"): 2,
    ("mgf", "identify_dist_from_mgf"): 2,
    ("mgf", "moment_from_mgf"): 3,
    ("transformation_univariate", "cdf_method"): 3,
    ("transformation_univariate", "pdf_of_transform"): 3,
    ("percentile", "median"): 1,
    ("percentile", "percentile"): 2,
    # ── Discrete distributions ────────────────────────────────────────────────
    ("bernoulli", "mean"): 1,
    ("bernoulli", "variance"): 1,
    ("binomial", "mean"): 1,
    ("binomial", "variance"): 1,
    ("binomial", "pmf_eq"): 1,
    ("binomial", "cdf_le"): 2,
    ("binomial", "cdf_ge"): 2,
    ("geometric", "mean"): 1,
    ("geometric", "variance"): 1,
    ("geometric", "pmf_eq"): 1,
    ("geometric", "cdf_ge"): 2,
    ("negbinomial", "mean"): 1,
    ("negbinomial", "variance"): 1,
    ("negbinomial", "pmf_eq"): 2,
    ("hypergeometric", "mean"): 1,
    ("hypergeometric", "pmf_eq"): 2,
    ("poisson", "mean"): 1,
    ("poisson", "pmf_eq"): 1,
    ("poisson", "cdf_le"): 2,
    ("poisson", "cdf_ge"): 2,
    ("discrete_uniform", "mean"): 1,
    ("discrete_uniform", "variance"): 1,
    ("discrete_uniform", "pmf_eq"): 1,
    # ── Continuous distributions ──────────────────────────────────────────────
    ("continuous_uniform", "mean"): 1,
    ("continuous_uniform", "variance"): 1,
    ("continuous_uniform", "prob_interval"): 1,
    ("continuous_uniform", "percentile"): 2,
    ("exponential", "mean"): 1,
    ("exponential", "survival"): 1,
    ("exponential", "prob_interval"): 1,
    ("exponential", "percentile"): 2,
    ("exponential", "memoryless"): 3,
    ("gamma", "mean"): 1,
    ("gamma", "variance"): 1,
    ("gamma", "prob_interval"): 2,
    ("normal", "prob_interval"): 1,
    ("normal", "survival"): 1,
    ("normal", "standardize"): 2,
    ("normal", "percentile"): 2,
    ("beta", "mean"): 1,
    ("beta", "prob_interval"): 2,
    ("lognormal", "mean"): 1,
    ("lognormal", "prob_interval"): 2,
    ("chisquare", "mean"): 1,
    ("chisquare", "variance"): 1,
    ("pareto", "mean"): 1,
    ("pareto", "survival"): 1,
    ("pareto", "prob_interval"): 2,
    ("weibull", "cdf"): 1,
    ("weibull", "survival"): 1,
    ("weibull", "mean"): 2,
    # ── Multivariate ──────────────────────────────────────────────────────────
    ("chebyshev", "markov_bound"): 1,
    ("chebyshev", "chebyshev_bound"): 2,
    ("joint_basics", "normalize_constant"): 1,
    ("joint_basics", "joint_prob_region"): 2,
    ("marginal", "marginal_pdf"): 2,
    ("marginal", "marginal_prob"): 2,
    ("conditional_dist", "cond_pdf"): 2,
    ("conditional_dist", "cond_prob"): 3,
    ("independence_rv", "is_independent"): 2,
    ("covariance", "covariance"): 2,
    ("covariance", "var_of_sum"): 2,
    ("correlation", "correlation"): 2,
    ("expectation_joint", "E_sum"): 1,
    ("expectation_joint", "E_XY"): 2,
    ("conditional_expectation", "cond_expectation"): 2,
    ("conditional_expectation", "double_expectation"): 3,
    ("total_variance", "total_variance"): 3,
    ("sum_distribution", "sum_mean_var"): 1,
    ("sum_distribution", "identify_sum_dist"): 2,
    ("order_statistics", "min_cdf"): 2,
    ("order_statistics", "max_cdf"): 2,
    ("order_statistics", "min_max_prob"): 3,
    ("clt", "clt_prob"): 2,
    ("clt", "clt_sum_prob"): 3,
    ("transformations_multi", "min_of_exp"): 2,
    ("transformations_multi", "sum_uniform_prob"): 2,
    ("transformations_multi", "jacobian_abs"): 3,
}


def max_tier_for_reps(reps: int, tier2_threshold: int, tier3_threshold: int) -> int:
    """Difficulty ceiling based on how many times a concept has been reviewed."""
    if reps >= tier3_threshold:
        return 3
    if reps >= tier2_threshold:
        return 2
    return 1


def filter_asks(kind: str, ask_list: list[str], max_tier: int) -> list[str]:
    """Return asks with difficulty <= max_tier. Falls back to full list if none qualify."""
    eligible = [a for a in ask_list if ASK_DIFFICULTY.get((kind, a), 2) <= max_tier]
    return eligible if eligible else ask_list
