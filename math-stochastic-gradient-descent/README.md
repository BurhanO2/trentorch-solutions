# Stochastic

Intermediate | calculus | optimization

### The problem, from first principles

`02-mini-batch` showed that a gradient computed from a few rows is a cheap, noisy stand-in for the full gradient. Push that idea as far as it goes and use a single row per update. That is stochastic gradient descent: look at one training example, nudge the weights to fit it a little better, move on to the next one. The model updates `m` times per pass through the data instead of once, and every one of those updates is as cheap as an update can be. This question fits the same linear model as before, one example at a time.

### From theory to code

Implement `sample_gradient(x_i, y_i, w)`, the gradient of the squared error on a single example, then `stochastic_gradient_descent(X, y, w0, learning_rate, num_epochs, rng)`, which runs the full loop and returns every intermediate weight vector. The signatures and docstrings are already in the editor.

### Constraints

- `X` has shape `(m, d)`, `y` has shape `(m,)`, and `w0` has shape `(d,)`. `x_i` is one row of `X` with shape `(d,)` and `y_i` is its scalar target. The model predicts `x_i @ w`.
- `stochastic_gradient_descent` draws a fresh shuffle at the start of every epoch with `rng.permutation(m)` and makes one update per example, so each example is used exactly once per epoch.
- It returns a list of length `1 + num_epochs * m`: `w0` followed by the weights after each update.
- The inputs `X`, `y` and `w0` must not be modified.

### Hints

<details>
<summary>Hint 1</summary>

For one example the loss is `(x_i @ w - y_i) ** 2`. Its gradient is the scalar residual times the example's own feature vector, doubled.

</details>

<details>
<summary>Hint 2</summary>

Iterate directly over the shuffled index array. Each pass through the inner loop is one update, which is `01-standard`'s step with a single example's gradient.

</details>
