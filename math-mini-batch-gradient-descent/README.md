# Mini-Batch

Intermediate | calculus | optimization

### The problem, from first principles

`01-standard` computes every update from the gradient of the loss over the whole training set. That is exact, but on a dataset with millions of rows a single step means reading every row once, and the model only moves after all of them have been seen. Mini-batch gradient descent keeps the same walk-downhill idea and changes one thing: each update uses the gradient from a small random subset of the data, so the model moves many times per pass through the data. This question fits a linear model to a small dataset that way.

### From theory to code

Implement `mse_gradient(X, y, w)`, the gradient of the mean squared error for a linear model, then `make_batches(num_samples, batch_size, rng)`, which shuffles the sample indices and cuts them into batches, then `mini_batch_gradient_descent(X, y, w0, learning_rate, batch_size, num_epochs, rng)`, which runs the full loop and returns every intermediate weight vector. The signatures and docstrings are already in the editor.

### Constraints

- `X` has shape `(m, d)`, `y` has shape `(m,)`, and `w` has shape `(d,)`. The model predicts `X @ w` and the loss is the mean of the squared residuals.
- `mse_gradient` must average over the rows it is given, so it works unchanged for a full dataset, a batch, or a single row.
- `make_batches` draws exactly one shuffle per call with `rng.permutation(num_samples)`. Every index appears exactly once, every batch except possibly the last has `batch_size` indices, and the last batch holds the remainder.
- `mini_batch_gradient_descent` draws a fresh shuffle at the start of every epoch and makes one update per batch. It returns a list of length `1 + num_epochs * ceil(m / batch_size)`: `w0` followed by the weights after each update.
- The inputs `X`, `y` and `w0` must not be modified.

### Hints

<details>
<summary>Hint 1</summary>

The gradient of `mean((X @ w - y) ** 2)` with respect to `w` is a matrix-vector product: the transposed design matrix times the residual vector, scaled by `2 / len(y)`.

</details>

<details>
<summary>Hint 2</summary>

Slicing a shuffled index array with `order[start:start + batch_size]` already handles the shorter final batch, because a slice past the end just stops.

</details>

<details>
<summary>Hint 3</summary>

The loop is two nested loops: epochs on the outside, batches on the inside. Each inner iteration is one `01-standard` step, with the batch's gradient in place of the full gradient.

</details>
