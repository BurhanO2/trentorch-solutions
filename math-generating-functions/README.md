# Moment & Probability Generating Functions

Advanced | calculus | series | probability

### The problem, from first principles

A distribution can be described by its probabilities, or by a handful of summary numbers such as the mean and variance. A generating function is a third description that packs the whole distribution into one function. It looks like a strange thing to do until it pays off: the mean, the variance and every higher moment fall out of the function by differentiating it, and the distribution of a sum of independent variables, which normally needs a convolution, falls out by multiplying two functions. This question builds both flavours for discrete variables and uses them to read off moments and to add random variables.

### From theory to code

Implement `mgf(values, probabilities, t)`, the moment generating function, then `pgf(probabilities, z)`, the probability generating function, then `mgf_moments(mgf_fn, step)`, the first two moments recovered by differentiating a moment generating function numerically at zero, then `pgf_mean_variance(probabilities)`, the mean and variance from the derivatives of a probability generating function at one, then `sum_distribution(p, q)`, the distribution of the sum of two independent variables. The signatures and docstrings are already in the editor.

### Constraints

- `values` is a 1D array-like of the outcomes a variable can take, and `probabilities` is a 1D array-like of the same length that is non-negative and sums to `1.0`.
- `mgf` accepts `t` as a float or an array and returns the same shape. `mgf(values, probabilities, 0)` is exactly `1.0`.
- For `pgf`, the variable takes the values `0, 1, 2, ...` and `probabilities[k]` is `P(X = k)`. `z` is a float or an array.
- `mgf_moments` receives a function `mgf_fn(t)` of one float, and returns `(first_moment, second_moment)` as floats, estimated with central finite differences around `t = 0` using the given `step`.
- `pgf_mean_variance` returns `(mean, variance)` as floats and must not first compute the mean as a plain weighted average of the values. It works from the derivatives of the generating function at `z = 1`.
- `sum_distribution` takes two probability lists as in `pgf` and returns the probability list for `X + Y`, of length `len(p) + len(q) - 1`.

### Hints

<details>
<summary>Hint 1</summary>

Both functions are expectations of something raised to the value: `e^{t x}` for the moment generating function and `z^x` for the probability generating function. An expectation is a probability-weighted sum.

</details>

<details>
<summary>Hint 2</summary>

Differentiating `E[e^{tX}]` once at `t = 0` brings down one factor of `X` and leaves `E[X]`. A central difference `(M(h) - M(-h)) / (2h)` approximates that derivative, and `(M(h) - 2M(0) + M(-h)) / h**2` approximates the second.

</details>

<details>
<summary>Hint 3</summary>

The first and second derivatives of a polynomial `sum(p_k z^k)` at `z = 1` are `sum(k p_k)` and `sum(k (k - 1) p_k)`. The variance needs the second derivative, plus the mean, minus the mean squared.

</details>

<details>
<summary>Hint 4</summary>

Multiplying two polynomials multiplies their coefficient lists as a convolution, and the product of two probability generating functions is the generating function of the sum.

</details>
