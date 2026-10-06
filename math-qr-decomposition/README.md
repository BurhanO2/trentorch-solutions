# QR Decomposition

Intermediate | linear-algebra

### The problem, from first principles

`12-gram-schmidt` turns a set of vectors into an orthonormal basis, one vector at a time, discarding the "how much of each original vector was removed" bookkeeping along the way. QR decomposition is Gram-Schmidt with that bookkeeping kept: the orthonormal basis becomes matrix `Q`, and exactly how each original column was built from it becomes matrix `R` — together, `Q` and `R` capture the same information as the original matrix, just split into an orthogonal part and a triangular part.

### From theory to code

Implement `qr_decompose(A)`, returning `(Q, R)` such that `A = QR`, built directly from `12-gram-schmidt`'s orthonormalization process applied column-by-column. The signature and docstring are already in the editor.

### Constraints

- `A` is an `m × n` NumPy array whose columns are linearly independent (a requirement Gram-Schmidt itself already assumes).
- `Q`'s columns are orthonormal (`12-gram-schmidt`'s output).
- `R` is upper-triangular.
- `Q @ R` must reconstruct `A` exactly (up to floating-point precision).

### Hints

<details>
<summary>Hint 1</summary>

Apply Gram-Schmidt to `A`'s columns one at a time to get `Q`'s columns. `R`'s entries are exactly the projection coefficients Gram-Schmidt computes and subtracts along the way — record them instead of throwing them away.

</details>

<details>
<summary>Hint 2</summary>

`R[i, j]` (for `i <= j`) is the dot product of `Q`'s `i`-th column with `A`'s original `j`-th column; `R[j, j]` specifically is the norm of what's left of column `j` after subtracting off its projections onto every earlier `Q` column, before that leftover gets normalized into `Q`'s `j`-th column.

</details>
