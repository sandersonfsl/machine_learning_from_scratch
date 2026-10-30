# Step 4: The model

Other files reuse the data with
`from machine_learning_from_scratch.data import x, y`.

The **model** is the rule that turns an input into a prediction. The simplest
one is a straight line:

$$f_{w,b}(x) = w \cdot x + b$$

| Part | Name | Meaning here |
|---|---|---|
| `x` | input | distance in km |
| `f(x)` | prediction (written $\hat{y}$) | estimated time in minutes |
| `w` | weight (slope) | minutes added per extra km |
| `b` | bias (intercept) | fixed minutes, even at 0 km (preparing the order) |

`w` and `b` are the **parameters** of the model. Different values give
different lines, and the goal of learning is to find the best ones.

In `src/machine_learning_from_scratch/model.py`:

```python
def predict(x, w, b):
    return np.dot(x, w) + b
```

With a single weight, `np.dot(x, w)` is just `w * x`, so passing an array of
distances returns an array of predictions. The same function also handles several
features ([Step 12](12-multiple-features.md)).

```bash
uv run scripts/04_model.py
```

```
distances:       [0.5 0.6 0.6]
w=3, b=12:       [13.5 13.8 13.8]
w=1, b=5:        [5.5 5.6 5.6]
```

Same distances, different parameters, different predictions. The real times of
those three deliveries were 20.5, 16.5 and 9.6 minutes. From these numbers
alone, it is not possible to say which line is better. Measuring that is the
job of the cost function.

---

[← Step 3: The training set](03-training-set.md) · [Step 5: Drawing the model →](05-plot-model.md)
