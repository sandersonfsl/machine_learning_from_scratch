# Step 12: Multiple features

So far each delivery was described by a single number, the distance. A delivery
has more information: how many items the order contains and whether it happens
during rush hour. With more features, the model can use all of them.

The notation grows accordingly:

| Symbol | Meaning | Here |
|---|---|---|
| `n` | Number of features | 3 |
| `m` | Number of examples | 100 |
| `x⃗` | The features of one example: a list of `n` numbers | `[6600, 3, 1]` |
| `x⁽ⁱ⁾` | The features of the `i`-th example | row `i` of the table |
| `x⁽ⁱ⁾ⱼ` | Feature `j` of the `i`-th example | one cell of the table |
| `w⃗` | One weight per feature | `[w₁, w₂, w₃]` |
| `b` | Still a single number | fixed minutes |

The data stops being a list and becomes a table: a matrix `X` with `m` rows
(deliveries) and `n` columns (features), in `src/machine_learning_from_scratch/data_multiple.py`.
The distance is in **meters** on purpose; why that matters appears in [Step 15](15-feature-scaling.md).

The model adds one term per feature:

$$f_{\vec{w},b}(\vec{x}) = w_1x_1 + w_2x_2 + \dots + w_nx_n + b$$

Each weight is the effect of one unit of its feature: minutes per meter, minutes
per item, and the extra minutes when it is rush hour. The same expression is
written with the **dot product**, which multiplies each weight by its feature
and adds the results:

$$f_{\vec{w},b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$$

This model is called **multiple linear regression**. It is not the same thing as
"multivariate regression", a different concept that is not used here.

In `src/machine_learning_from_scratch/model.py`, the same prediction in both forms:

```python
def predict_loop(x, w, b):
    total = 0.0
    for j in range(len(x)):
        total += w[j] * x[j]       # one term at a time
    return total + b


def predict(x, w, b):
    return np.dot(x, w) + b        # the dot product
```

```bash
uv run scripts/12_multiple_features.py
```

```
m = 100 examples, n = 3 features: ['distance_m', 'items', 'rush_hour']
X shape: (100, 3) | y shape: (100,)

x^(1) = [6.6e+03 3.0e+00 1.0e+00] -> y^(1) = 41.0 min
x^(1)_1 = 6600.0 m

prediction, term by term: 39.80 min
prediction, dot product:  39.80 min
real time:                41.00 min
```

In Python, counting starts at 0: `X[0]` is `x⁽¹⁾` and `X[0, 0]` is `x⁽¹⁾₁`. Both
forms of the prediction give the same number, 39.80 minutes against a real time
of 41.0. The weights here were picked by hand, so they are not the best ones yet;

`predict` also accepts the whole table at once and returns one prediction per
row. Three tests in `tests/test_model_multiple.py` cover it: a prediction checked
by hand (`0.003 × 2000 + 1.0 × 3 + 7.0 × 1 + 10 = 26`), the equivalence between
the loop and the dot product, and the shape of the result for many examples.

---

[← Step 11: Automated tests](11-tests.md) · [Step 13: Vectorization →](13-vectorization.md)
