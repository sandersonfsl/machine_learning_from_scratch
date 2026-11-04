# Step 14: Gradient descent with multiple features

The algorithm is the same as before, applied to every weight:

$$w_j := w_j - \alpha \frac{\partial J}{\partial w_j} \quad (j = 1, \dots, n) \qquad b := b - \alpha \frac{\partial J}{\partial b}$$

The derivatives keep the same shape as in the one-feature case:

$$\frac{\partial J}{\partial w_j} = \frac{1}{m}\sum_{i=1}^{m}\left(f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)}\right)x_j^{(i)} \qquad \frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^{m}\left(f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)}\right)$$

| | One feature | Several features |
|---|---|---|
| Weights | `w` | `w₁ … wₙ` |
| Input of the derivative | `x⁽ⁱ⁾` | `x⁽ⁱ⁾ⱼ`, feature `j` of example `i` |
| Error term | `f(x⁽ⁱ⁾) - y⁽ⁱ⁾` | the same for every weight |

The error of each example is the same for every weight; each derivative multiplies
it by its own feature. All weights and `b` are still updated **simultaneously**,
each computed from the old values.

In `src/machine_learning_from_scratch/gradient.py`, the change is one line:

```python
def compute_gradient(x, y, w, b):
    m = len(x)
    errors = predict(x, w, b) - y
    dj_dw = np.dot(x.T, errors) / m      # one derivative per weight
    dj_db = errors.sum() / m
    return dj_dw, dj_db
```

`x.T` is the table turned on its side, so each row becomes a feature.
`np.dot(x.T, errors)` multiplies every feature by the errors and adds the results,
which yields all the derivatives at once (the vectorized form of two nested loops).
`compute_cost` and `gradient_descent` do not change: they already work when `w` is
a vector.

## A reference: the normal equation

For linear regression there is a method that finds `w` and `b` directly, without
iterating, by solving a linear algebra problem. It works only for linear regression
and becomes slow with many features, so it is not the way the model is trained
here. It is useful as a **reference**: it gives the exact answer that gradient
descent should approach. `np.linalg.lstsq` computes it.

## Training on the raw features

`scripts/14_multiple_gradient_descent.py` trains on the deliveries exactly as they
are, with the distance in meters, starting from `w = 0` and `b = 0`.

```bash
uv run scripts/14_multiple_gradient_descent.py
```

```
true rule:      w = [3.0e-03 1.2e+00 7.0e+00], b = 10.0
exact solution: w = [3.0000e-03 1.0524e+00 6.8911e+00], b = 10.39
lowest possible cost: 4.241

alpha = 1e-07, 50 iterations
  cost: 6.666e+03 at the start -> 7.288e+51 at the end
  w = [-1.88731524e+22 -1.74884151e+19 -9.34910700e+17], b = -2634077247444417024.00

alpha = 1e-08, 100,000 iterations
  cost: 3.003e+02 at the start -> 5.733e+01 at the end
  w = [0.0058 0.0406 0.0034] b = 0.00
```

The exact solution is close to the rule that generated the data; the difference
comes from the noise. The lowest cost any line can reach is 4.241.

- With `alpha = 1e-7` the training **diverges**: the cost grows from 6.7 thousand to
  7 × 10⁵¹ in 50 steps. This `alpha` looks small, but it is already above the stability
  limit for these data, which is about 5e-8 (computed in [Step 15](15-feature-scaling.md)).
- With `alpha = 1e-8` the training is stable, but after **100,000 iterations** the cost
  is still 57.3, against a minimum of 4.24:

| | `w` distance | `w` items | `w` rush hour | `b` |
|---|---|---|---|---|
| Exact | 0.0030 | 1.052 | 6.891 | 10.39 |
| Learned | 0.0058 | 0.041 | 0.003 | 0.00 |

Only the weight of the distance moved. Its derivative is multiplied by values up to
10,000, so it is huge; the derivatives of the items (up to 12) and of the rush hour
(0 or 1) are tiny. One `alpha` is shared by all weights: it is too large for the
first and too small for the others. The features have very different scales, and
that is what slows gradient descent down. Fixing it is the subject of [Step 15](15-feature-scaling.md).

## Tests

Four tests in `tests/test_gradient_multiple.py`:

- the cost with several features, checked by hand (a two-row table where each
  example uses a different feature, giving a cost of 5);
- the gradient, checked by hand (`[-1, -2]` and `-3` for the same table);
- gradient checking: nudging each parameter and measuring the change in the cost
  must agree with `compute_gradient`;
- recovering the parameters `[2, -1, 0.5]` and `4` from noise-free data whose
  features have similar scales, which shows that the algorithm works and the raw
  deliveries are what makes it slow.

Together with the earlier tests, `uv run pytest` reports `17 passed`.

---

[← Step 13: Vectorization](13-vectorization.md) · [Step 15: Feature scaling →](15-feature-scaling.md)
