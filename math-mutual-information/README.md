# Mutual Information

Advanced | information-theory

### The problem, from first principles

`03-probability`'s correlation measures whether two variables move together LINEARLY, and can be exactly zero even when two variables are strongly, deterministically related in a curved way (`y = x^2`, say). What you actually want, much of the time, is a measure of "how much does knowing X tell you about Y," with no assumption about linearity at all. That's mutual information: it's zero if and only if X and Y are genuinely, completely independent (knowing one tells you literally nothing about the other), and it grows the more knowing X narrows down what Y could be, no matter what shape that relationship takes.

The elegant part, and the reason this question sits at the very end of both the Probability and Information Theory tracks: mutual information is built entirely from tools you've already implemented, it is literally `03-kl-divergence`'s KL divergence, applied to compare the real joint distribution against what that same joint distribution WOULD look like if X and Y had no relationship at all.

### From theory to code

Theory defines mutual information as `KL(P(X,Y) || P(X)*P(Y))`: the real joint distribution, compared against the "independent" joint you'd get by multiplying the two marginals together (`03-probability/04-conditional-probability`'s `marginal_x`/`marginal_y`). Build that hypothetical independent joint via an outer product, then measure the KL divergence between the two, flattened into matching 1D distributions.

Implement `mutual_information(joint, base=2.0)` against that reasoning. The signature and docstring are already in the editor.

### Constraints

- `joint` is a 2D array, `joint[i, j] = P(X=i, Y=j)`, summing to `1.0`.
- Must use `marginal_x`/`marginal_y` (already imported) to build the independent joint, not compute it any other way.
- Must reuse `kl_divergence` (already imported), not reimplement its formula.

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

`np.outer(px, py)` builds exactly the `(len(px), len(py))` table you'd expect from two independent variables with those marginals.

</details>

<details>
<summary>Hint 2</summary>

`kl_divergence` expects two flat distributions of matching shape. `.flatten()` both 2D tables before passing them in.

</details>
