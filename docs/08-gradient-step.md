# Step 8: Gradient descent, one step

**Gradient descent** works like walking down the side of the bowl from the
previous step: find the direction that goes downhill the fastest, take a small
step that way, and repeat from the new position until the bottom is reached.

The direction comes from the **derivative** of the cost: how much `J` changes
if a parameter grows a little. If the derivative is negative, increasing the
parameter lowers the cost. If it is positive, decreasing it does. Each
parameter is updated with:

$$w := w - \alpha \frac{\partial J}{\partial w} \qquad b := b - \alpha \frac{\partial J}{\partial b}$$

- `:=` means "assign the new value" (in Python, plain `=`).
- $\alpha$ (alpha) is the **learning rate**, a small number such as `0.01`
  that controls how big each step is.
- For the squared error cost function, the derivatives are:

$$\frac{\partial J}{\partial w} = \frac{1}{m}\sum_{i=1}^{m}\left(f_{w,b}(x^{(i)}) - y^{(i)}\right)x^{(i)} \qquad \frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^{m}\left(f_{w,b}(x^{(i)}) - y^{(i)}\right)$$

Calculus is not needed to use them. (The `2` in the cost cancels the exponent
that appears when differentiating, which is why it is there.)

In `src/machine_learning_from_scratch/gradient.py`:

```python
def compute_gradient(x, y, w, b):
    m = len(x)
    errors = predict(x, w, b) - y
    dj_dw = np.dot(x.T, errors) / m
    dj_db = errors.sum() / m
    return dj_dw, dj_db
```

With a single feature, `np.dot(x.T, errors)` is the same as `(errors * x).sum()`. With
several features it gives one derivative per weight ([Step 14](14-multiple-gradient-descent.md)).

One detail matters: **both parameters are updated simultaneously**. The new
`w` and the new `b` are both computed from the old values, and only then
assigned:

```python
new_w = w - alpha * dj_dw
new_b = b - alpha * dj_db
w, b = new_w, new_b
```

```bash
uv run scripts/08_gradient_step.py
```

```
start: w=0.0, b=0.0, cost=418.37
derivatives: dj_dw=-165.56, dj_db=-27.67
after 1 step: w=1.66, b=0.28, cost=186.49
```

Starting from `w = 0, b = 0`, both derivatives are negative: increasing `w` and
`b` lowers the cost. After a single step, `w` went from 0 to 1.66 and the cost
dropped from 418.37 to 186.49. Repeating the step many times leads to
the bottom of the bowl.

---

[← Step 7: The shape of the cost](07-plot-cost.md) · [Step 9: Gradient descent in a loop →](09-gradient-descent.md)
