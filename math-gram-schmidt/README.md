# Gram-Schmidt Process

Advanced | linear-algebra

### The problem, from first principles

Many useful properties (`13-qr-decomposition`'s stable solving, an orthonormal basis's trivially-invertible-by-transpose structure) only apply when a set of vectors is not just independent, but **orthonormal**: mutually perpendicular, each of unit length. Most real vector sets you're handed aren't — Gram-Schmidt is the direct procedure for turning any independent set into an orthonormal one, one vector at a time, using nothing but `09-vector-projection`'s "subtract off the overlapping part" idea repeated against every earlier result.

### From theory to code

Implement `gram_schmidt(vectors)`, taking a list of linearly independent vectors and returning an orthonormal basis spanning the same space. The signature and docstring are already in the editor.

### Constraints

- `vectors` is a list of 1D NumPy arrays, all the same length, guaranteed linearly independent.
- Return a list of the same length, each entry a unit vector (`‖v‖ = 1`), with every pair mutually orthogonal (`v_i · v_j = 0` for `i ≠ j`).
- Process vectors in the given order — the resulting basis depends on that order (a different input order generally produces a different, still valid, orthonormal basis).

### Hints

<details>
<summary>Hint 1</summary>

For each new vector, subtract its projection (`09-vector-projection`) onto **every** already-built basis vector, not just the most recent one — each earlier basis vector may capture a different direction of overlap.

</details>

<details>
<summary>Hint 2</summary>

Only normalize (divide by the norm) **after** all of that vector's projections onto earlier basis vectors have been subtracted — normalizing too early would rescale the vector before its remaining overlap has been fully removed.

</details>
