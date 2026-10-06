# QR Factorization using Reflections

Advanced | linear-algebra | orthogonalization | least-squares

### The problem, from first principles

`02-householder-reflections` builds one mirror that clears a vector down to its first entry. A whole matrix is cleared the same way, one column at a time: use a mirror to zero everything below the diagonal in column 1, then ignore that row and column and do it again on what is left. After a mirror per column the matrix has become upper triangular, and the mirrors multiplied together form an orthogonal matrix. That is the QR factorization, `A = QR`, and it is the standard stable way to solve least-squares problems. This question builds the factorization and uses it to fit a line to noisy data.

### From theory to code

Implement `householder_qr(a)`, which returns the orthogonal factor `Q` and the upper-triangular factor `R`, then `back_substitution(r, y)`, which solves an upper-triangular system, then `least_squares_qr(a, b)`, which uses the first two to minimise the squared error of `a @ x - b`. The helpers from `02-householder-reflections` are already imported in the editor. The signatures and docstrings are already there too.

### Constraints

- `a` is an `m x n` float array with `m >= n`. `householder_qr` returns `(Q, R)` with `Q` of shape `m x m` and orthogonal, and `R` of shape `m x n` with exact zeros below the diagonal, such that `Q @ R` equals `a`.
- A column that is already zero below the diagonal needs no mirror. A zero column must not cause a division by zero.
- `a` must not be modified.
- `back_substitution(r, y)` takes a square upper-triangular `r` with nonzero diagonal and a 1D `y`, and returns the 1D `x` with `r @ x == y`. Solve from the last row upward with a loop, not with `np.linalg.solve`.
- `least_squares_qr(a, b)` assumes `a` has full column rank and returns the 1D `x` of length `n` that minimises the squared error. It must not form `a.T @ a`.

### Hints

<details>
<summary>Hint 1</summary>

At step `k` the mirror only touches rows `k` and below. Take the part of column `k` from row `k` down, find its mirror direction, and reflect the whole block of rows `k` and below.

</details>

<details>
<summary>Hint 2</summary>

Each mirror is its own inverse, so `A = H1 H2 ... Hn R`. Start `Q` as the identity and multiply on the right by each mirror in turn. Because every mirror is symmetric, reflecting the transpose of the relevant block of `Q` and transposing back does the multiplication.

</details>

<details>
<summary>Hint 3</summary>

Since `Q` is orthogonal, `||a x - b||` equals `||R x - Q^T b||`. Only the top `n` rows of `R` are nonzero, so the best `x` solves the top `n x n` triangle against the top `n` entries of `Q^T b`.

</details>
