# Standard (Batch)

Intermediate | calculus

### The problem, from first principles

`06-directional-derivatives` establishes that the gradient points in the direction of steepest **ascent** — but it stops there, never actually using that fact to find a minimum. Gradient descent is the direct, obvious consequence: if the gradient points toward the steepest increase, walking a small step in the **opposite** direction decreases the function, and repeating that over and over eventually reaches a minimum. This question implements the loop itself, not just the formula.

### From theory to code

Implement `gradient_descent_step(x, gradient_fn, learning_rate)`, performing a single update, then `gradient_descent(x0, gradient_fn, learning_rate, num_steps)`, running the full loop and returning every intermediate position. The signatures and docstrings are already in the editor.

### Constraints

- `x` may be a scalar float or a NumPy array (the same update rule applies elementwise either way).
- `gradient_fn` takes `x` and returns the gradient at that point, same shape as `x`.
- `gradient_descent` returns a list of length `num_steps + 1`: the starting point `x0`, followed by the position after each of the `num_steps` updates.

### Hints

<details>
<summary>Hint 1</summary>

`gradient_descent_step` is one line: `x - learning_rate * gradient_fn(x)` — the entire "walk downhill" idea, applied once.

</details>

<details>
<summary>Hint 2</summary>

`gradient_descent` just calls `gradient_descent_step` in a loop, appending each new position to a running list that starts with `x0` already in it.

</details>
