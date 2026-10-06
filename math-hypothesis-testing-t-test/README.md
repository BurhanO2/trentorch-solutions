# Hypothesis Testing

Advanced | probability

### The problem, from first principles

Group A (a new website layout) averages 52 seconds on-page, group B (the old layout) averages 48 seconds. Is the new layout genuinely better, or would two random samples from the SAME underlying population, with no real difference at all, plausibly show a 4-second gap just from ordinary sampling noise? `01-confidence-interval` quantified uncertainty around ONE group's mean; a hypothesis test directly answers the "is this difference real" question for TWO groups, by asking exactly how surprising the observed gap would be if there were actually no true difference at all.

This is the formal machinery behind essentially every "does this change actually help" decision made with data: an A/B test comparing two website variants, a clinical trial comparing a drug to a placebo, an ML experiment comparing a new model architecture against a baseline, they're all, underneath, this exact same two-sample comparison.

### From theory to code

Theory builds Welch's t-test (the version that doesn't assume equal variance between groups, the safer, more commonly recommended default) from two pieces: a t-statistic measuring how many standard errors apart the two means are, and the Welch-Satterthwaite equation for the (non-integer) effective degrees of freedom needed to convert that statistic into a p-value.

Implement `welch_t_statistic(a, b)` and `welch_degrees_of_freedom(a, b)` first, then `two_sample_t_test(a, b)`, which combines them into a `(t_statistic, p_value)` pair.

### Constraints

- Do not assume equal variance between `a` and `b` (this is specifically Welch's t-test, not the classic Student's equal-variance version).
- `two_sample_t_test` computes a two-sided p-value.
- Use the provided `_t_cdf(t, df)` for the p-value lookup (an exact Student's t-distribution CDF, built from `numpy`/`math` alone since the browser runtime doesn't have `scipy` available).

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

The t-statistic's numerator is just the difference of the two means; the denominator is a "combined standard error," `sqrt(var_a/n_a + var_b/n_b)`.

</details>

<details>
<summary>Hint 2</summary>

The two-sided p-value is `2 * (1 - stats.t.cdf(abs(t), df))`, the probability, under the "no real difference" assumption, of seeing a t-statistic at least this extreme in either direction.

</details>
