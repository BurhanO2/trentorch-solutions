# LU Decomposition

Intermediate | linear-algebra

### The problem, from first principles

`10-gaussian-elimination` solves `Ax = b` by row-reducing `A` down to an upper-triangular form. But that same row-reduction work is thrown away the moment you need to solve a **different** `b` with the **same** `A` — a common situation (simulating many right-hand sides against one fixed system matrix). LU decomposition captures the row-reduction work itself as two reusable matrices, so solving for a new `b` afterward is nearly free.

### From theory to code

Implement `lu_decompose(A)`, returning `(L, U)` such that `A = LU`, using the exact row-elimination steps from `10-gaussian-elimination` but recording each elimination multiplier into `L` instead of discarding it. The signature and docstring are already in the editor.

### Constraints

- `A` is a square `n × n` NumPy array that does not require row swaps (a **partial pivoting**-free case — real solvers handle the swap case too, but that's a refinement on top of this question's core idea, not covered here).
- `L` is lower-triangular with `1`s on its diagonal; `U` is upper-triangular.
- `L @ U` must reconstruct `A` exactly (up to floating-point precision).

### Hints

<details>
<summary>Hint 1</summary>

Start `U` as a copy of `A` and `L` as the identity matrix. Run the same elimination loop as `10-gaussian-elimination` on `U`, but every time you compute a multiplier to zero out an entry, also write that exact multiplier into the corresponding position of `L`.

</details>

<details>
<summary>Hint 2</summary>

The multiplier that zeros out `U[i, j]` using pivot row `j` is `U[i, j] / U[j, j]` — this is precisely the number that belongs at `L[i, j]`.

</details>
