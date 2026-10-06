# Bootstrap Confidence Intervals

Intermediate | probability

### The problem, from first principles

`01-confidence-interval`'s t-based formula works specifically because the sample MEAN has a known, well-understood sampling distribution (the t-distribution). But what if you want a confidence interval for the MEDIAN? Or the standard deviation? Or some custom statistic entirely, like "the 90th percentile" or "the ratio of two other statistics"? None of these have the same convenient closed-form math the mean does. You'd need a different, hand-derived formula for every single statistic, if one even exists.

The bootstrap sidesteps this entirely with a strikingly simple idea: since you can't easily draw more real samples from the true population, resample from the one sample you DO have, treating it as a stand-in for the population itself. Do this thousands of times, computing your statistic of interest on each resample, and the SPREAD of those thousands of computed statistics directly tells you how uncertain your original estimate is, no closed-form formula required, for literally any statistic you can write a function for.

### From theory to code

Theory resamples the original data WITH replacement (same size as the original), computes the statistic of interest on each resample, and uses percentiles of the resulting distribution of statistics as the interval bounds.

Implement `bootstrap_resample(x, rng)` first, then `bootstrap_confidence_interval(x, statistic_fn, n_bootstrap=1000, confidence=0.95, seed=None)` on top of it.

### Constraints

- `bootstrap_resample` samples WITH replacement, same size as `x`.
- `bootstrap_confidence_interval` works for ANY `statistic_fn` (a function taking an array and returning a scalar), not just the mean.
- Uses `np.percentile` on the collected bootstrap statistics for the interval bounds.
- Same `seed` must produce the same interval (reproducibility).

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

`rng.choice(x, size=len(x), replace=True)` is exactly "with replacement, same size."

</details>

<details>
<summary>Hint 2</summary>

After collecting `n_bootstrap` computed statistics, the interval bounds are `np.percentile(statistics, (1-confidence)/2 * 100)` and `np.percentile(statistics, (1+confidence)/2 * 100)`.

</details>
